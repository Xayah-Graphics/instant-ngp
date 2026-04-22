#include "instant-ngp.h"
#include <algorithm>
#include <concepts>
#include <cstdint>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/functional.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>
#include <limits>
#include <mma.h>
#include <source_location>
#include <sstream>
#include <type_traits>
#include <variant>

#if !defined(NGP_DENSITY_NETWORK_WIDTH)
#error "NGP_DENSITY_NETWORK_WIDTH must be provided by the build system."
#endif

#if !defined(NGP_RGB_NETWORK_WIDTH)
#error "NGP_RGB_NETWORK_WIDTH must be provided by the build system."
#endif

namespace ngp {

    constexpr bool supported_fully_fused_width(const std::uint32_t width) {
        return width == 16u || width == 32u || width == 64u || width == 128u;
    }

    static_assert(supported_fully_fused_width(NGP_DENSITY_NETWORK_WIDTH), "NGP_DENSITY_NETWORK_WIDTH must be one of 16, 32, 64, or 128.");
    static_assert(supported_fully_fused_width(NGP_RGB_NETWORK_WIDTH), "NGP_RGB_NETWORK_WIDTH must be one of 16, 32, 64, or 128.");

    inline constexpr std::uint32_t density_network_width = NGP_DENSITY_NETWORK_WIDTH;
    inline constexpr std::uint32_t rgb_network_width     = NGP_RGB_NETWORK_WIDTH;

} // namespace ngp

namespace ngp::network::detail {

    template <typename T>
    inline constexpr float default_loss_scale = 1.0f;

#ifdef __CUDACC__
    template <>
    inline constexpr float default_loss_scale<__half> = 128.0f;
#endif

#if defined(TCNN_PARAMS_UNALIGNED)
    inline constexpr bool params_aligned = false;
#else
    inline constexpr bool params_aligned = true;
#endif

    enum class GradientMode {
        Ignore,
        Overwrite,
        Accumulate,
    };

    enum class Activation {
        ReLU,
        LeakyReLU,
        Exponential,
        Sigmoid,
        Squareplus,
        Softplus,
        Tanh,
        None,
    };

    enum class GridType {
        Hash,
        Dense,
        Tiled,
    };

    template <typename T>
    struct IsVariant : std::false_type {};

    template <typename... Ts>
    struct IsVariant<std::variant<Ts...>> : std::true_type {};

    template <typename T>
    inline constexpr bool is_variant_v = IsVariant<std::remove_cvref_t<T>>::value;

    template <typename Module, typename Fn>
    decltype(auto) visit_module(Module&& module, Fn&& fn) {
        if constexpr (is_variant_v<Module>) {
            return std::visit([&](auto&& impl) -> decltype(auto) { return fn(std::forward<decltype(impl)>(impl)); }, std::forward<Module>(module));
        } else {
            return fn(std::forward<Module>(module));
        }
    }

    inline constexpr std::uint32_t batch_size_granularity = 256u;
    inline constexpr std::uint32_t n_threads_linear       = 128u;

    inline legacy::GpuAllocation allocate_workspace(cudaStream_t stream, const std::size_t n_bytes) {
        return legacy::GpuAllocation{n_bytes, stream};
    }

    template <typename T>
    __global__ void cast(const std::uint32_t num_elements, const float* __restrict__ full_precision, T* __restrict__ target) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;
        target[i] = static_cast<T>(full_precision[i]);
    }

    inline std::uint32_t powi(const std::uint32_t base, const std::uint32_t exponent) {
        std::uint32_t result = 1u;
        for (std::uint32_t i = 0; i < exponent; ++i) result *= base;
        return result;
    }

    template <std::uint32_t N_DIMS, std::uint32_t N_PRIMES>
    __device__ std::uint32_t lcg_hash(const legacy::math::uvec<N_DIMS>& pos_grid, const std::uint32_t primes[N_PRIMES]) {
        static_assert(N_DIMS <= N_PRIMES, "lcg_hash can only hash up to N_PRIMES dimensions.");

        std::uint32_t result = 0u;
        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0; i < N_DIMS; ++i) result ^= pos_grid[i] * primes[i];
        return result;
    }

    template <std::uint32_t N_DIMS>
    __device__ std::uint32_t coherent_prime_hash(const legacy::math::uvec<N_DIMS>& pos_grid) {
        constexpr std::uint32_t factors[7] = {1u, 2654435761u, 805459861u, 3674653429u, 2097192037u, 1434869437u, 2165219737u};
        return lcg_hash<N_DIMS, 7u>(pos_grid, factors);
    }

    template <std::uint32_t N_DIMS>
    __device__ std::uint32_t grid_index(const GridType grid_type, const std::uint32_t hashmap_size, const std::uint32_t grid_resolution_value, const legacy::math::uvec<N_DIMS>& pos_grid) {
        std::uint32_t stride = 1u;
        std::uint32_t index  = 0u;

        constexpr std::uint32_t max_bases[] = {0x0u, 0xFFFFFFFFu, 0xFFFFu, 0x659u, 0xFFu, 0x54u, 0x28u, 0x17u, 0xFu, 0xBu, 0x9u};
        static_assert(N_DIMS <= sizeof(max_bases) / sizeof(max_bases[0]), "grid_index can only be used for N_DIMS <= 10");

        if (grid_resolution_value <= max_bases[N_DIMS]) {
            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0; dim < N_DIMS; ++dim) {
                index += pos_grid[dim] * stride;
                stride *= grid_resolution_value;
            }
        } else
            stride = 0xFFFFFFFFu;

        if (grid_type == GridType::Hash && hashmap_size < stride) index = coherent_prime_hash<N_DIMS>(pos_grid);
        return index % hashmap_size;
    }

    __host__ __device__ inline float grid_scale(const std::uint32_t level, const float log2_per_level_scale, const std::uint32_t base_resolution) {
        return exp2f(static_cast<float>(level) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
    }

    __host__ __device__ inline std::uint32_t grid_resolution(const float scale) {
        return static_cast<std::uint32_t>(ceilf(scale)) + 1u;
    }

    __device__ inline std::uint32_t morton3D(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
        std::uint32_t xx = x;
        std::uint32_t yy = y;
        std::uint32_t zz = z;
        xx               = (xx * 0x00010001u) & 0xFF0000FFu;
        xx               = (xx * 0x00000101u) & 0x0F00F00Fu;
        xx               = (xx * 0x00000011u) & 0xC30C30C3u;
        xx               = (xx * 0x00000005u) & 0x49249249u;
        yy               = (yy * 0x00010001u) & 0xFF0000FFu;
        yy               = (yy * 0x00000101u) & 0x0F00F00Fu;
        yy               = (yy * 0x00000011u) & 0xC30C30C3u;
        yy               = (yy * 0x00000005u) & 0x49249249u;
        zz               = (zz * 0x00010001u) & 0xFF0000FFu;
        zz               = (zz * 0x00000101u) & 0x0F00F00Fu;
        zz               = (zz * 0x00000011u) & 0xC30C30C3u;
        zz               = (zz * 0x00000005u) & 0x49249249u;
        return xx | (yy << 1u) | (zz << 2u);
    }

    __device__ inline std::uint32_t morton3D_invert(std::uint32_t value) {
        value = value & 0x49249249u;
        value = (value | (value >> 2u)) & 0xC30C30C3u;
        value = (value | (value >> 4u)) & 0x0F00F00Fu;
        value = (value | (value >> 8u)) & 0xFF0000FFu;
        value = (value | (value >> 16u)) & 0x0000FFFFu;
        return value;
    }

    __device__ inline void pos_fract(const float input, float* pos, std::uint32_t* pos_grid, const float scale) {
        *pos            = fmaf(scale, input, 0.5f);
        const float tmp = floorf(*pos);
        *pos_grid       = static_cast<std::uint32_t>(static_cast<int>(tmp));
        *pos -= tmp;
    }

    template <typename T, typename RNG, std::size_t N_TO_GENERATE>
    __global__ void generate_random_uniform_kernel(const std::size_t n_elements, RNG rng, T* __restrict__ out, const T lower, const T upper) {
        const std::size_t i         = threadIdx.x + blockIdx.x * blockDim.x;
        const std::size_t n_threads = blockDim.x * gridDim.x;

        rng.advance(i * N_TO_GENERATE);

        TCNN_PRAGMA_UNROLL
        for (std::size_t j = 0; j < N_TO_GENERATE; ++j) {
            const std::size_t idx = i + n_threads * j;
            if (idx >= n_elements) return;
            out[idx] = static_cast<T>(rng.next_float()) * (upper - lower) + lower;
        }
    }

    template <typename T, typename RNG>
    void generate_random_uniform(RNG& rng, const std::size_t n_elements, T* out, const T lower = static_cast<T>(0.0f), const T upper = static_cast<T>(1.0f)) {
        static constexpr std::size_t n_to_generate = 4u;
        if (n_elements == 0u) return;

        const std::size_t n_threads = (n_elements + n_to_generate - 1u) / n_to_generate;
        const std::size_t blocks    = (n_threads + n_threads_linear - 1u) / n_threads_linear;
        generate_random_uniform_kernel<T, RNG, n_to_generate><<<blocks, n_threads_linear>>>(n_elements, rng, out, lower, upper);
        rng.advance(n_elements);
    }

    template <typename RNG>
    __device__ legacy::math::vec3 random_val_3d(RNG& rng) {
        return {rng.next_float(), rng.next_float(), rng.next_float()};
    }

    __device__ inline float random_val(const std::uint32_t seed, const std::uint32_t idx) {
        legacy::math::pcg32 rng{seed};
        rng.advance(idx);
        return rng.next_float();
    }

    inline Activation activation_from_config(const ngp::InstantNGP::ActivationMode activation) {
        switch (activation) {
        case ngp::InstantNGP::ActivationMode::None: return Activation::None;
        case ngp::InstantNGP::ActivationMode::ReLU: return Activation::ReLU;
        case ngp::InstantNGP::ActivationMode::Exponential: return Activation::Exponential;
        case ngp::InstantNGP::ActivationMode::Sigmoid: return Activation::Sigmoid;
        case ngp::InstantNGP::ActivationMode::Squareplus: return Activation::Squareplus;
        case ngp::InstantNGP::ActivationMode::Softplus: return Activation::Softplus;
        case ngp::InstantNGP::ActivationMode::Tanh: return Activation::Tanh;
        case ngp::InstantNGP::ActivationMode::LeakyReLU: return Activation::LeakyReLU;
        default: throw std::runtime_error{"Unsupported public activation mode."};
        }
    }

    __device__ inline float logistic(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    template <typename V>
    struct VectorFragment final {
        static constexpr std::uint32_t num_elements = V::size();
        V x                                         = {};
    };

    template <typename T>
    __device__ T relu(T value) {
        return static_cast<T>(cuda::std::max(static_cast<float>(value), 0.0f));
    }

    template <>
    inline __device__ half relu(half value) {
#if defined(__CUDA_ARCH__)
        return __hmax(value, static_cast<half>(0.0f));
#else
        return static_cast<half>(relu<float>(static_cast<float>(value)));
#endif
    }

    inline constexpr float k_act = 10.0f;

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::None)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        result = frag;
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::ReLU)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = relu(static_cast<T>(frag.x[t]));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::LeakyReLU)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(static_cast<T>(frag.x[t]) > static_cast<T>(0.0f) ? 1.0f : 0.01f);
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Exponential)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(expf(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Sigmoid)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(logistic(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Squareplus)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) {
            const float x = static_cast<float>(frag.x[t]) * k_act;
            result.x[t]   = static_cast<T>(0.5f * (x + sqrtf(x * x + 4.0f)) / k_act);
        }
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Softplus)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(logf(expf(static_cast<float>(frag.x[t]) * k_act) + 1.0f) / k_act);
    }

    template <typename T, typename Fragment, Activation activation>
        requires (activation == Activation::Tanh)
    __device__ void warp_activation(const Fragment& frag, Fragment& result) {
        TCNN_PRAGMA_UNROLL
        for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = static_cast<T>(tanhf(static_cast<float>(frag.x[t])));
    }

    template <typename T, typename Fragment>
    __device__ void warp_activation(const Activation activation, const Fragment& frag, Fragment& result) {
        switch (activation) {
        case Activation::ReLU: warp_activation<T, Fragment, Activation::ReLU>(frag, result); return;
        case Activation::LeakyReLU: warp_activation<T, Fragment, Activation::LeakyReLU>(frag, result); return;
        case Activation::Exponential: warp_activation<T, Fragment, Activation::Exponential>(frag, result); return;
        case Activation::Sigmoid: warp_activation<T, Fragment, Activation::Sigmoid>(frag, result); return;
        case Activation::Squareplus: warp_activation<T, Fragment, Activation::Squareplus>(frag, result); return;
        case Activation::Softplus: warp_activation<T, Fragment, Activation::Softplus>(frag, result); return;
        case Activation::Tanh: warp_activation<T, Fragment, Activation::Tanh>(frag, result); return;
        case Activation::None: warp_activation<T, Fragment, Activation::None>(frag, result); return;
        default: return;
        }
    }

    template <typename T, typename Fragment>
    __device__ Fragment warp_activation(const Activation activation, const Fragment& frag) {
        Fragment result = {};
        warp_activation<T>(activation, frag, result);
        return result;
    }

    template <typename T, typename Fragment, typename ForwardFragment>
    __device__ void warp_activation_backward(const Activation activation, const Fragment& frag, const ForwardFragment& forward_frag, Fragment& result) {
        switch (activation) {
        case Activation::ReLU:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] > static_cast<T>(0.0f));
            return;
        case Activation::LeakyReLU:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] > static_cast<T>(0.0f) ? 1.0f : 0.01f);
            return;
        case Activation::Exponential:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * forward_frag.x[t];
            return;
        case Activation::Sigmoid:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(forward_frag.x[t] * static_cast<T>(1.0f - static_cast<float>(forward_frag.x[t])));
            return;
        case Activation::Squareplus:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) {
                const float y = static_cast<float>(forward_frag.x[t]) * k_act;
                result.x[t]   = frag.x[t] * static_cast<T>(y * y / (y * y + 1.0f));
            }
            return;
        case Activation::Softplus:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(1.0f - expf(-static_cast<float>(forward_frag.x[t]) * k_act));
            return;
        case Activation::Tanh:
            TCNN_PRAGMA_UNROLL
            for (int t = 0; t < static_cast<int>(result.num_elements); ++t) result.x[t] = frag.x[t] * static_cast<T>(1.0f - (static_cast<float>(forward_frag.x[t]) * static_cast<float>(forward_frag.x[t])));
            return;
        case Activation::None: result = frag; return;
        default: return;
        }
    }

    template <typename T, typename Fragment, typename ForwardFragment>
    __device__ Fragment warp_activation_backward(const Activation activation, const Fragment& frag, const ForwardFragment& forward_frag) {
        Fragment result = {};
        warp_activation_backward<T>(activation, frag, forward_frag, result);
        return result;
    }

    template <typename T, typename Array>
    __device__ void sh_enc(const std::uint32_t degree, const float x, const float y, const float z, Array& data_out) {
        const float xy = x * y;
        const float xz = x * z;
        const float yz = y * z;
        const float x2 = x * x;
        const float y2 = y * y;
        const float z2 = z * z;
        const float x4 = x2 * x2;
        const float y4 = y2 * y2;
        const float z4 = z2 * z2;
        const float x6 = x4 * x2;
        const float y6 = y4 * y2;
        const float z6 = z4 * z2;

        data_out(0) = static_cast<T>(0.28209479177387814f);
        if (degree <= 1u) return;
        data_out(1) = static_cast<T>(-0.48860251190291987f * y);
        data_out(2) = static_cast<T>(0.48860251190291987f * z);
        data_out(3) = static_cast<T>(-0.48860251190291987f * x);
        if (degree <= 2u) return;
        data_out(4) = static_cast<T>(1.0925484305920792f * xy);
        data_out(5) = static_cast<T>(-1.0925484305920792f * yz);
        data_out(6) = static_cast<T>(0.94617469575755997f * z2 - 0.31539156525251999f);
        data_out(7) = static_cast<T>(-1.0925484305920792f * xz);
        data_out(8) = static_cast<T>(0.54627421529603959f * x2 - 0.54627421529603959f * y2);
        if (degree <= 3u) return;
        data_out(9)  = static_cast<T>(0.59004358992664352f * y * (-3.0f * x2 + y2));
        data_out(10) = static_cast<T>(2.8906114426405538f * xy * z);
        data_out(11) = static_cast<T>(0.45704579946446572f * y * (1.0f - 5.0f * z2));
        data_out(12) = static_cast<T>(0.3731763325901154f * z * (5.0f * z2 - 3.0f));
        data_out(13) = static_cast<T>(0.45704579946446572f * x * (1.0f - 5.0f * z2));
        data_out(14) = static_cast<T>(1.4453057213202769f * z * (x2 - y2));
        data_out(15) = static_cast<T>(0.59004358992664352f * x * (-x2 + 3.0f * y2));
        if (degree <= 4u) return;
        data_out(16) = static_cast<T>(2.5033429417967046f * xy * (x2 - y2));
        data_out(17) = static_cast<T>(1.7701307697799304f * yz * (-3.0f * x2 + y2));
        data_out(18) = static_cast<T>(0.94617469575756008f * xy * (7.0f * z2 - 1.0f));
        data_out(19) = static_cast<T>(0.66904654355728921f * yz * (3.0f - 7.0f * z2));
        data_out(20) = static_cast<T>(-3.1735664074561294f * z2 + 3.7024941420321507f * z4 + 0.31735664074561293f);
        data_out(21) = static_cast<T>(0.66904654355728921f * xz * (3.0f - 7.0f * z2));
        data_out(22) = static_cast<T>(0.47308734787878004f * (x2 - y2) * (7.0f * z2 - 1.0f));
        data_out(23) = static_cast<T>(1.7701307697799304f * xz * (-x2 + 3.0f * y2));
        data_out(24) = static_cast<T>(-3.7550144126950569f * x2 * y2 + 0.62583573544917614f * x4 + 0.62583573544917614f * y4);
        if (degree <= 5u) return;
        data_out(25) = static_cast<T>(0.65638205684017015f * y * (10.0f * x2 * y2 - 5.0f * x4 - y4));
        data_out(26) = static_cast<T>(8.3026492595241645f * xy * z * (x2 - y2));
        data_out(27) = static_cast<T>(-0.48923829943525038f * y * (3.0f * x2 - y2) * (9.0f * z2 - 1.0f));
        data_out(28) = static_cast<T>(4.7935367849733241f * xy * z * (3.0f * z2 - 1.0f));
        data_out(29) = static_cast<T>(0.45294665119569694f * y * (14.0f * z2 - 21.0f * z4 - 1.0f));
        data_out(30) = static_cast<T>(0.1169503224534236f * z * (-70.0f * z2 + 63.0f * z4 + 15.0f));
        data_out(31) = static_cast<T>(0.45294665119569694f * x * (14.0f * z2 - 21.0f * z4 - 1.0f));
        data_out(32) = static_cast<T>(2.3967683924866621f * z * (x2 - y2) * (3.0f * z2 - 1.0f));
        data_out(33) = static_cast<T>(-0.48923829943525038f * x * (x2 - 3.0f * y2) * (9.0f * z2 - 1.0f));
        data_out(34) = static_cast<T>(2.0756623148810411f * z * (-6.0f * x2 * y2 + x4 + y4));
        data_out(35) = static_cast<T>(0.65638205684017015f * x * (10.0f * x2 * y2 - x4 - 5.0f * y4));
        if (degree <= 6u) return;
        data_out(36) = static_cast<T>(1.3663682103838286f * xy * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
        data_out(37) = static_cast<T>(2.3666191622317521f * yz * (10.0f * x2 * y2 - 5.0f * x4 - y4));
        data_out(38) = static_cast<T>(2.0182596029148963f * xy * (x2 - y2) * (11.0f * z2 - 1.0f));
        data_out(39) = static_cast<T>(-0.92120525951492349f * yz * (3.0f * x2 - y2) * (11.0f * z2 - 3.0f));
        data_out(40) = static_cast<T>(0.92120525951492349f * xy * (-18.0f * z2 + 33.0f * z4 + 1.0f));
        data_out(41) = static_cast<T>(0.58262136251873131f * yz * (30.0f * z2 - 33.0f * z4 - 5.0f));
        data_out(42) = static_cast<T>(6.6747662381009842f * z2 - 20.024298714302954f * z4 + 14.684485723822165f * z6 - 0.31784601133814211f);
        data_out(43) = static_cast<T>(0.58262136251873131f * xz * (30.0f * z2 - 33.0f * z4 - 5.0f));
        data_out(44) = static_cast<T>(0.46060262975746175f * (x2 - y2) * (11.0f * z2 * (3.0f * z2 - 1.0f) - 7.0f * z2 + 1.0f));
        data_out(45) = static_cast<T>(-0.92120525951492349f * xz * (x2 - 3.0f * y2) * (11.0f * z2 - 3.0f));
        data_out(46) = static_cast<T>(0.50456490072872406f * (11.0f * z2 - 1.0f) * (-6.0f * x2 * y2 + x4 + y4));
        data_out(47) = static_cast<T>(2.3666191622317521f * xz * (10.0f * x2 * y2 - x4 - 5.0f * y4));
        data_out(48) = static_cast<T>(10.247761577878714f * x2 * y4 - 10.247761577878714f * x4 * y2 + 0.6831841051919143f * x6 - 0.6831841051919143f * y6);
        if (degree <= 7u) return;
        data_out(49) = static_cast<T>(0.70716273252459627f * y * (-21.0f * x2 * y4 + 35.0f * x4 * y2 - 7.0f * x6 + y6));
        data_out(50) = static_cast<T>(5.2919213236038001f * xy * z * (-10.0f * x2 * y2 + 3.0f * x4 + 3.0f * y4));
        data_out(51) = static_cast<T>(-0.51891557872026028f * y * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + 5.0f * x4 + y4));
        data_out(52) = static_cast<T>(4.1513246297620823f * xy * z * (x2 - y2) * (13.0f * z2 - 3.0f));
        data_out(53) = static_cast<T>(-0.15645893386229404f * y * (3.0f * x2 - y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f));
        data_out(54) = static_cast<T>(0.44253269244498261f * xy * z * (-110.0f * z2 + 143.0f * z4 + 15.0f));
        data_out(55) = static_cast<T>(0.090331607582517306f * y * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f));
        data_out(56) = static_cast<T>(0.068284276912004949f * z * (315.0f * z2 - 693.0f * z4 + 429.0f * z6 - 35.0f));
        data_out(57) = static_cast<T>(0.090331607582517306f * x * (-135.0f * z2 + 495.0f * z4 - 429.0f * z6 + 5.0f));
        data_out(58) = static_cast<T>(0.07375544874083044f * z * (x2 - y2) * (143.0f * z2 * (3.0f * z2 - 1.0f) - 187.0f * z2 + 45.0f));
        data_out(59) = static_cast<T>(-0.15645893386229404f * x * (x2 - 3.0f * y2) * (13.0f * z2 * (11.0f * z2 - 3.0f) - 27.0f * z2 + 3.0f));
        data_out(60) = static_cast<T>(1.0378311574405206f * z * (13.0f * z2 - 3.0f) * (-6.0f * x2 * y2 + x4 + y4));
        data_out(61) = static_cast<T>(-0.51891557872026028f * x * (13.0f * z2 - 1.0f) * (-10.0f * x2 * y2 + x4 + 5.0f * y4));
        data_out(62) = static_cast<T>(2.6459606618019f * z * (15.0f * x2 * y4 - 15.0f * x4 * y2 + x6 - y6));
        data_out(63) = static_cast<T>(0.70716273252459627f * x * (-35.0f * x2 * y4 + 21.0f * x4 * y2 - x6 + 7.0f * y6));
    }

    template <typename T, std::uint32_t N = 1u>
    __global__ void kernel_activation_backward_output(const std::uint32_t num_elements, const Activation activation, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;

        const auto frag_forward_out = reinterpret_cast<const VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(output_values)[i];
        auto frag                   = reinterpret_cast<const VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(gradients_out)[i];
        warp_activation_backward<T>(activation, frag, frag_forward_out, frag);
        reinterpret_cast<VectorFragment<legacy::math::tvec<T, N, sizeof(T)>>*>(gradients_in)[i] = frag;
    }

    struct AuxStreamSlot;
    std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>>>& aux_stream_pools();
    void free_aux_stream_pool(cudaStream_t parent_stream);

    struct SyncedStreamReservation final {
        SyncedStreamReservation() = default;
        SyncedStreamReservation(cudaStream_t stream, std::size_t n_streams);
        ~SyncedStreamReservation();
        SyncedStreamReservation& operator=(const SyncedStreamReservation&) = delete;
        SyncedStreamReservation(const SyncedStreamReservation&)            = delete;
        SyncedStreamReservation& operator=(SyncedStreamReservation&& other);
        SyncedStreamReservation(SyncedStreamReservation&& other);

        std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)> aux_stream_slot = {nullptr, nullptr};
        cudaStream_t aux_stream                                                  = nullptr;
        cudaStream_t main_stream                                                 = nullptr;
    };

} // namespace ngp::network::detail

namespace ngp::encoding {

    template <typename T>
    __global__ void transpose_encoded_position(const std::uint32_t n_elements, const T* __restrict__ encoded_positions, legacy::PitchedPtr<T> output) {
        const std::uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
        if (i >= n_elements) return;

        const std::uint32_t elem_idx = i;
        const std::uint32_t dim_idx  = threadIdx.x;
        output(elem_idx)[dim_idx]    = encoded_positions[elem_idx + n_elements * dim_idx];
    }

    template <typename T>
    __global__ void transpose_gradients(const std::uint32_t n_elements, T* __restrict__ transposed_dL_dy, legacy::PitchedPtr<const T> dL_dy) {
        const std::uint32_t i = threadIdx.y + blockIdx.x * blockDim.y;
        if (i >= n_elements) return;

        const std::uint32_t elem_idx                      = i;
        const std::uint32_t dim_idx                       = threadIdx.x;
        transposed_dL_dy[elem_idx + n_elements * dim_idx] = dL_dy(elem_idx)[dim_idx];
    }

    template <typename T>
    __global__ void zero_padded_output_aos(const std::uint32_t n_elements, const std::uint32_t n_output_dims, const std::uint32_t n_to_pad, legacy::PitchedPtr<T> output) {
        const std::uint32_t elem = threadIdx.y + blockIdx.x * blockDim.y;
        const std::uint32_t dim  = threadIdx.x;
        if (elem >= n_elements || dim >= n_to_pad) return;

        output(elem)[n_output_dims + dim] = static_cast<T>(0);
    }

    inline constexpr std::uint32_t max_n_levels = 128u;

    struct ParamsOffsetTable final {
        std::uint32_t data[max_n_levels + 1u] = {};
        std::uint32_t size                    = 0u;
    };

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    __global__ void kernel_grid(const std::uint32_t num_elements, const std::uint32_t num_grid_features, const ParamsOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, float max_level, const float* __restrict__ max_level_gpu, const network::detail::GridType grid_type, const T* __restrict__ grid, legacy::MatrixView<const float> positions_in, T* __restrict__ encoded_positions) {
        const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_elements) return;

        const std::uint32_t level = blockIdx.y;

        if (max_level_gpu)
            max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
        else
            max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

        if (level >= max_level + 1e-3f) {
            if (encoded_positions) {
                TCNN_PRAGMA_UNROLL
                for (std::uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = static_cast<T>(0.0f);
            }
            return;
        }

        grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
        const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
        const float scale                = network::detail::grid_scale(level, log2_per_level_scale, base_resolution);
        const std::uint32_t resolution   = network::detail::grid_resolution(scale);

        float pos[N_POS_DIMS];
        legacy::math::uvec<N_POS_DIMS> pos_grid = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t dim = 0; dim < N_POS_DIMS; ++dim) network::detail::pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale);

        auto grid_val = [&](const legacy::math::uvec<N_POS_DIMS>& local_pos) {
            const std::uint32_t index = network::detail::grid_index<N_POS_DIMS>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
            return *reinterpret_cast<const legacy::math::tvec<T, N_FEATURES_PER_LEVEL, network::detail::params_aligned ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)>*>(&grid[index]);
        };

        if (encoded_positions) {
            legacy::math::tvec<T, N_FEATURES_PER_LEVEL, network::detail::params_aligned ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)> result = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t idx = 0; idx < (1u << N_POS_DIMS); ++idx) {
                float weight                                  = 1.0f;
                legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

                TCNN_PRAGMA_UNROLL
                for (std::uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                    if ((idx & (1u << dim)) == 0u) {
                        weight *= 1.0f - pos[dim];
                        pos_grid_local[dim] = pos_grid[dim];
                    } else {
                        weight *= pos[dim];
                        pos_grid_local[dim] = pos_grid[dim] + 1u;
                    }
                }

                const auto grid_value = grid_val(pos_grid_local);
                TCNN_PRAGMA_UNROLL
                for (std::uint32_t feature_idx = 0; feature_idx < N_FEATURES_PER_LEVEL; ++feature_idx) {
                    if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, half>) {
#if defined(__CUDA_ARCH__)
                        result[feature_idx] = __hfma(static_cast<T>(weight), grid_value[feature_idx], result[feature_idx]);
#else
                        result[feature_idx] = static_cast<T>(static_cast<float>(weight) * static_cast<float>(grid_value[feature_idx]) + static_cast<float>(result[feature_idx]));
#endif
                    } else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
                        result[feature_idx] = cuda::std::fma(static_cast<T>(weight), grid_value[feature_idx], result[feature_idx]);
                    } else {
                        result[feature_idx] = static_cast<T>(weight) * grid_value[feature_idx] + result[feature_idx];
                    }
                }
            }

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t f = 0; f < N_FEATURES_PER_LEVEL; ++f) encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
        }
    }

    template <typename T, typename GradT, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL, std::uint32_t N_FEATURES_PER_THREAD>
    __global__ void kernel_grid_backward(const std::uint32_t num_elements, const std::uint32_t num_grid_features, const ParamsOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, float max_level, const float* __restrict__ max_level_gpu, const bool stochastic_interpolation, const network::detail::GridType grid_type, GradT* __restrict__ grid_gradient, legacy::MatrixView<const float> positions_in, const T* __restrict__ dL_dy) {
        const std::uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
        if (i >= num_elements) return;

        const std::uint32_t level   = blockIdx.y;
        const std::uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

        if (max_level_gpu)
            max_level = (max_level_gpu[i] * num_grid_features) / N_FEATURES_PER_LEVEL;
        else
            max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

        if (level > max_level + 1e-3f) return;

        grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
        const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
        const float scale                = network::detail::grid_scale(level, log2_per_level_scale, base_resolution);
        const std::uint32_t resolution   = network::detail::grid_resolution(scale);

        auto add_grid_gradient = [&](const legacy::math::uvec<N_POS_DIMS>& local_pos, const legacy::math::tvec<GradT, N_FEATURES_PER_THREAD>& grad, const float weight) {
            const std::uint32_t index = network::detail::grid_index<N_POS_DIMS>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL + feature;
            legacy::math::atomic_add_gmem(grid_gradient + index, static_cast<GradT>(weight) * grad);
        };

        float pos[N_POS_DIMS];
        legacy::math::uvec<N_POS_DIMS> pos_grid = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t dim = 0; dim < N_POS_DIMS; ++dim) network::detail::pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale);

        legacy::math::tvec<T, N_FEATURES_PER_THREAD> grad = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t f = 0; f < N_FEATURES_PER_THREAD; ++f) grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];

        if (stochastic_interpolation) {
            const float sample                            = network::detail::random_val(1337u, i + level * num_elements);
            legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if (sample >= pos[dim])
                    pos_grid_local[dim] = pos_grid[dim];
                else
                    pos_grid_local[dim] = pos_grid[dim] + 1u;
            }

            add_grid_gradient(pos_grid_local, grad, 1.0f);
            return;
        }

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t idx = 0; idx < (1u << N_POS_DIMS); ++idx) {
            float weight                                  = 1.0f;
            legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0; dim < N_POS_DIMS; ++dim) {
                if ((idx & (1u << dim)) == 0u) {
                    weight *= 1.0f - pos[dim];
                    pos_grid_local[dim] = pos_grid[dim];
                } else {
                    weight *= pos[dim];
                    pos_grid_local[dim] = pos_grid[dim] + 1u;
                }
            }

            add_grid_gradient(pos_grid_local, grad, weight);
        }
    }

    template <typename T, std::uint32_t N_POS_DIMS = 3u, std::uint32_t N_FEATURES_PER_LEVEL = 2u>
    class GridEncodingTemplated final {
    public:
        GridEncodingTemplated(const std::uint32_t n_features, const std::uint32_t log2_hashmap_size, const std::uint32_t base_resolution, const float per_level_scale, const bool stochastic_interpolation, const network::detail::GridType grid_type) : m_n_features{n_features}, m_base_resolution{base_resolution}, m_per_level_scale{per_level_scale}, m_stochastic_interpolation{stochastic_interpolation}, m_grid_type{grid_type} {
            m_n_levels           = (m_n_features + N_FEATURES_PER_LEVEL - 1u) / N_FEATURES_PER_LEVEL;
            std::uint32_t offset = 0u;

            if (m_n_levels > max_n_levels) {
                std::ostringstream stream;
                stream << "GridEncoding: m_n_levels=" << m_n_levels << " must be at most MAX_N_LEVELS=" << max_n_levels;
                throw std::runtime_error{stream.str()};
            }

            for (std::uint32_t i = 0; i < m_n_levels; ++i) {
                const std::uint32_t resolution = network::detail::grid_resolution(network::detail::grid_scale(i, std::log2(per_level_scale), base_resolution));
                const std::uint32_t max_params = std::numeric_limits<std::uint32_t>::max() / 2u;
                std::uint32_t params_in_level  = std::pow(static_cast<float>(resolution), N_POS_DIMS) > static_cast<float>(max_params) ? max_params : network::detail::powi(resolution, N_POS_DIMS);

                params_in_level = legacy::next_multiple(params_in_level, 8u);

                if (grid_type == network::detail::GridType::Dense) {
                } else if (grid_type == network::detail::GridType::Tiled)
                    params_in_level = std::min(params_in_level, network::detail::powi(base_resolution, N_POS_DIMS));
                else if (grid_type == network::detail::GridType::Hash)
                    params_in_level = std::min(params_in_level, 1u << log2_hashmap_size);
                else
                    throw std::runtime_error{"GridEncoding: invalid grid type."};

                m_offset_table.data[i] = offset;
                offset += params_in_level;
            }

            m_offset_table.data[m_n_levels] = offset;
            m_offset_table.size             = m_n_levels + 1u;
            m_n_params                      = static_cast<std::size_t>(m_offset_table.data[m_n_levels]) * N_FEATURES_PER_LEVEL;
            m_n_output_dims                 = m_n_features;

            if (n_features % N_FEATURES_PER_LEVEL != 0u) {
                std::ostringstream stream;
                stream << "GridEncoding: n_features=" << n_features << " must be a multiple of N_FEATURES_PER_LEVEL=" << N_FEATURES_PER_LEVEL;
                throw std::runtime_error{stream.str()};
            }
        }

        void set_params(T* params, T* gradients) {
            m_params    = params;
            m_gradients = gradients;
        }

        T* params() const {
            return m_params;
        }

        T* gradients() const {
            return m_gradients;
        }

        void encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output) {
            legacy::check_or_throw(input.m() == input_width());
            legacy::check_or_throw(output.m() == padded_output_width());
            legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
            legacy::check_or_throw(input.n() == output.n());
            if (n_params() > 0u) legacy::check_or_throw(params() != nullptr);

            const std::uint32_t num_elements = input.n();
            if (padded_output_width() == 0u || num_elements == 0u) return;

            network::detail::SyncedStreamReservation synced_streams{stream, m_n_to_pad > 0u ? 2u : 1u};
            const cudaStream_t main_stream = synced_streams.main_stream;
            const cudaStream_t aux_stream  = synced_streams.aux_stream ? synced_streams.aux_stream : synced_streams.main_stream;

            if (m_n_to_pad > 0u) {
                if (output.layout() == legacy::AoS) {
                    const dim3 threads         = {m_n_to_pad, (network::detail::n_threads_linear + m_n_to_pad - 1u) / m_n_to_pad, 1u};
                    const std::uint32_t blocks = (num_elements + threads.y - 1u) / threads.y;
                    zero_padded_output_aos<T><<<blocks, threads, 0, aux_stream>>>(num_elements, m_n_output_dims, m_n_to_pad, output.pitched_ptr());
                } else {
                    legacy::cuda_check(cudaMemsetAsync(output.data() + num_elements * m_n_output_dims, 0, sizeof(T) * num_elements * m_n_to_pad, aux_stream));
                }
            }

            static constexpr std::uint32_t n_threads_hashgrid = 512u;
            const dim3 blocks_hashgrid                        = {(num_elements + n_threads_hashgrid - 1u) / n_threads_hashgrid, m_n_levels, 1u};

            T* encoded_positions_soa        = output.data();
            legacy::GpuAllocation workspace = {};
            if (output.layout() == legacy::AoS) {
                workspace             = network::detail::allocate_workspace(main_stream, static_cast<std::size_t>(num_elements) * m_n_features * sizeof(T));
                encoded_positions_soa = reinterpret_cast<T*>(workspace.data());
            }

            kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, n_threads_hashgrid, 0, main_stream>>>(num_elements, m_n_features, m_offset_table, m_base_resolution, std::log2(m_per_level_scale), m_max_level, m_max_level_gpu, m_grid_type, params(), input.view(), encoded_positions_soa);

            if (output.layout() == legacy::AoS) {
                const dim3 threads_transpose         = {m_n_levels * N_FEATURES_PER_LEVEL, 8u, 1u};
                const std::uint32_t blocks_transpose = (num_elements + threads_transpose.y - 1u) / threads_transpose.y;
                transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, main_stream>>>(num_elements, encoded_positions_soa, output.pitched_ptr());
            }
        }

        void backward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, const legacy::GPUMatrixDynamic<T>& dL_doutput, const network::detail::GradientMode param_gradients_mode = network::detail::GradientMode::Overwrite) {
            legacy::check_or_throw(input.m() == input_width());
            legacy::check_or_throw(dL_doutput.m() == padded_output_width());
            legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
            legacy::check_or_throw(input.n() == dL_doutput.n());
            if (n_params() > 0u) {
                legacy::check_or_throw(params() != nullptr);
                if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(gradients() != nullptr);
            }

            const std::uint32_t num_elements = input.n();
            if (param_gradients_mode == network::detail::GradientMode::Ignore || num_elements == 0u) return;

            const T* dL_dy_rm               = dL_doutput.data();
            legacy::GpuAllocation workspace = {};
            if (dL_doutput.layout() == legacy::CM) {
                workspace = network::detail::allocate_workspace(stream, static_cast<std::size_t>(num_elements) * m_n_features * sizeof(T));

                const dim3 threads_transpose         = {m_n_levels * N_FEATURES_PER_LEVEL, 8u, 1u};
                const std::uint32_t blocks_transpose = (num_elements + threads_transpose.y - 1u) / threads_transpose.y;
                transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(num_elements, reinterpret_cast<T*>(workspace.data()), dL_doutput.pitched_ptr());

                dL_dy_rm = reinterpret_cast<const T*>(workspace.data());
            }

            typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type* grid_gradient = nullptr;
            legacy::GpuAllocation grid_gradient_tmp                                              = {};

            if constexpr (!std::is_same_v<typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type, T>) {
                grid_gradient_tmp = network::detail::allocate_workspace(stream, n_params() * sizeof(typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type));
                grid_gradient     = reinterpret_cast<typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type*>(grid_gradient_tmp.data());
            } else {
                grid_gradient = reinterpret_cast<typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type*>(gradients());
            }

            if (param_gradients_mode == network::detail::GradientMode::Overwrite) legacy::cuda_check(cudaMemsetAsync(grid_gradient, 0, n_params() * sizeof(typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type), stream));

            static constexpr std::uint32_t n_threads_hashgrid    = 256u;
            static constexpr std::uint32_t n_features_per_thread = std::min(2u, N_FEATURES_PER_LEVEL);

            const dim3 blocks_hashgrid = {((num_elements * N_FEATURES_PER_LEVEL / n_features_per_thread) + n_threads_hashgrid - 1u) / n_threads_hashgrid, m_n_levels, 1u};
            kernel_grid_backward<T, typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type, N_POS_DIMS, N_FEATURES_PER_LEVEL, n_features_per_thread><<<blocks_hashgrid, n_threads_hashgrid, 0, stream>>>(num_elements, m_n_features, m_offset_table, m_base_resolution, std::log2(m_per_level_scale), m_max_level, m_max_level_gpu, m_stochastic_interpolation, m_grid_type, grid_gradient, input.view(), dL_dy_rm);

            if constexpr (!std::is_same_v<typename std::conditional<N_FEATURES_PER_LEVEL == 1u, float, T>::type, T>) {
                if (n_params() > 0u) {
                    const std::uint32_t blocks = (static_cast<std::uint32_t>(n_params()) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    network::detail::cast<T><<<blocks, network::detail::n_threads_linear, 0, stream>>>(static_cast<std::uint32_t>(n_params()), reinterpret_cast<const float*>(grid_gradient), gradients());
                }
            }
        }

        std::uint32_t input_width() const {
            return N_POS_DIMS;
        }

        std::uint32_t padded_output_width() const {
            return m_n_output_dims + m_n_to_pad;
        }

        std::uint32_t output_width() const {
            return padded_output_width();
        }

        void set_padded_output_width(const std::uint32_t padded_output_width_value) {
            legacy::check_or_throw(padded_output_width_value >= m_n_output_dims);
            m_n_to_pad = padded_output_width_value - m_n_output_dims;
        }

        std::uint32_t required_output_alignment() const {
            return N_FEATURES_PER_LEVEL;
        }

        legacy::MatrixLayout preferred_output_layout() const {
            return legacy::SoA;
        }

        void initialize_params(legacy::math::pcg32& rng, float* params_full_precision, const float scale = 1.0f) {
            ngp::network::detail::generate_random_uniform<float>(rng, n_params(), params_full_precision, -1e-4f * scale, 1e-4f * scale);
        }

        std::size_t n_params() const {
            return m_n_params;
        }

    private:
        std::uint32_t m_n_features            = 0u;
        std::uint32_t m_n_levels              = 0u;
        std::size_t m_n_params                = 0u;
        ParamsOffsetTable m_offset_table      = {};
        std::uint32_t m_base_resolution       = 0u;
        std::uint32_t m_n_output_dims         = 0u;
        std::uint32_t m_n_to_pad              = 0u;
        float m_per_level_scale               = 0.0f;
        bool m_stochastic_interpolation       = false;
        network::detail::GridType m_grid_type = network::detail::GridType::Hash;
        float m_max_level                     = 1000.0f;
        float* m_max_level_gpu                = nullptr;
        T* m_params                           = nullptr;
        T* m_gradients                        = nullptr;
    };

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> make_hash_grid_encoding(const ngp::InstantNGP::NetworkConfig::HashGridConfig& config) {
        const network::detail::GridType grid_type = config.storage == ngp::InstantNGP::GridStorage::Hash ? network::detail::GridType::Hash : (config.storage == ngp::InstantNGP::GridStorage::Dense ? network::detail::GridType::Dense : (config.storage == ngp::InstantNGP::GridStorage::Tiled ? network::detail::GridType::Tiled : throw std::runtime_error{"Unsupported grid storage mode."}));
        const std::uint32_t n_features            = N_FEATURES_PER_LEVEL * config.n_levels;
        const float per_level_scale               = config.per_level_scale.has_value() ? *config.per_level_scale : (grid_type == network::detail::GridType::Dense && config.n_levels > 1u && config.base_resolution > 0u ? std::exp(std::log(256.0f / static_cast<float>(config.base_resolution)) / (config.n_levels - 1u)) : 2.0f);
        return {n_features, config.log2_hashmap_size, config.base_resolution, per_level_scale, config.stochastic_interpolation, grid_type};
    }

    template <typename T>
    std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>> create_position_encoding(const std::uint32_t n_dims_to_encode, const ngp::InstantNGP::NetworkConfig::HashGridConfig& config, const std::uint32_t alignment) {
        if (config.n_levels == 0u) throw std::runtime_error{"HashGrid encoding requires at least one level."};
        if (config.base_resolution == 0u) throw std::runtime_error{"HashGrid encoding base_resolution must be greater than zero."};
        if (n_dims_to_encode != 3u) throw std::runtime_error{"HashGrid encoding in this repository only supports 3D positions."};

        switch (config.n_features_per_level) {
        case 1u:
            {
                auto result = std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{make_hash_grid_encoding<T, 3u, 1u>(config)};
                if (alignment > 0u) network::detail::visit_module(result, [&](auto& encoding) { encoding.set_padded_output_width(legacy::next_multiple(encoding.output_width(), std::lcm(alignment, encoding.required_output_alignment()))); });
                return result;
            }
        case 2u:
            {
                auto result = std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{make_hash_grid_encoding<T, 3u, 2u>(config)};
                if (alignment > 0u) network::detail::visit_module(result, [&](auto& encoding) { encoding.set_padded_output_width(legacy::next_multiple(encoding.output_width(), std::lcm(alignment, encoding.required_output_alignment()))); });
                return result;
            }
        case 4u:
            {
                auto result = std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{make_hash_grid_encoding<T, 3u, 4u>(config)};
                if (alignment > 0u) network::detail::visit_module(result, [&](auto& encoding) { encoding.set_padded_output_width(legacy::next_multiple(encoding.output_width(), std::lcm(alignment, encoding.required_output_alignment()))); });
                return result;
            }
        case 8u:
            {
                auto result = std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{make_hash_grid_encoding<T, 3u, 8u>(config)};
                if (alignment > 0u) network::detail::visit_module(result, [&](auto& encoding) { encoding.set_padded_output_width(legacy::next_multiple(encoding.output_width(), std::lcm(alignment, encoding.required_output_alignment()))); });
                return result;
            }
        default: throw std::runtime_error{"HashGrid encoding n_features_per_level must be 1, 2, 4, or 8."};
        }
    }

    template <typename T>
    __global__ void kernel_sh(const std::uint32_t num_elements, const std::uint32_t degree, const std::uint32_t num_to_pad, legacy::MatrixView<const float> data_in, legacy::MatrixView<T> data_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;

        data_out.advance_cols(i);
        TCNN_PRAGMA_UNROLL
        for (std::uint32_t j = 0; j < num_to_pad; ++j) data_out(j) = static_cast<T>(1.0f);

        data_out.advance_rows(num_to_pad);
        network::detail::sh_enc<T, legacy::MatrixView<T>>(degree, data_in(0u, i) * 2.0f - 1.0f, data_in(1u, i) * 2.0f - 1.0f, data_in(2u, i) * 2.0f - 1.0f, data_out);
    }

    template <typename T>
    class SphericalHarmonicsEncoding final {
    public:
        SphericalHarmonicsEncoding(const std::uint32_t degree, const std::uint32_t n_dims_to_encode) : m_degree{degree} {
            m_n_output_dims = degree * degree;

            if (n_dims_to_encode != 3u) throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};
            if (m_degree <= 0u) throw std::runtime_error{"Spherical harmonics must have positive degree."};
            if (m_degree > 8u) throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
        }

        void encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output) {
            legacy::check_or_throw(input.m() == input_width());
            legacy::check_or_throw(output.m() == padded_output_width());
            legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
            legacy::check_or_throw(input.n() == output.n());
            if (padded_output_width() == 0u) return;

            const std::uint32_t num_elements = input.n();
            if (num_elements > 0u) {
                const std::uint32_t blocks = (num_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                kernel_sh<T><<<blocks, network::detail::n_threads_linear, 0, stream>>>(num_elements, m_degree, m_n_to_pad, input.view(), output.view());
            }
        }

        std::uint32_t input_width() const {
            return 3u;
        }

        std::uint32_t padded_output_width() const {
            return m_n_output_dims + m_n_to_pad;
        }

        std::uint32_t output_width() const {
            return padded_output_width();
        }

        void set_padded_output_width(const std::uint32_t padded_output_width_value) {
            legacy::check_or_throw(padded_output_width_value >= m_n_output_dims);
            m_n_to_pad = padded_output_width_value - m_n_output_dims;
        }

        std::uint32_t required_output_alignment() const {
            return 1u;
        }

        legacy::MatrixLayout preferred_output_layout() const {
            return legacy::SoA;
        }

    private:
        std::uint32_t m_degree        = 0u;
        std::uint32_t m_n_output_dims = 0u;
        std::uint32_t m_n_to_pad      = 0u;
    };

    template <typename T>
    SphericalHarmonicsEncoding<T> create_direction_encoding(const std::uint32_t n_dims_to_encode, const ngp::InstantNGP::NetworkConfig::DirectionEncodingConfig& config, const std::uint32_t alignment) {
        auto result = SphericalHarmonicsEncoding<T>{config.sh_degree, n_dims_to_encode};
        if (alignment > 0u) result.set_padded_output_width(legacy::next_multiple(result.output_width(), std::lcm(alignment, result.required_output_alignment())));
        return result;
    }

} // namespace ngp::encoding

namespace ngp::mlp {

    inline void cutlass_check(const cutlass::Status result, const std::source_location& location = std::source_location::current()) {
        if (result == cutlass::Status::kSuccess) return;

        std::ostringstream stream;
        stream << "CUTLASS call failed: " << cutlassGetStatusString(result);
        legacy::throw_runtime_error(stream.str(), location);
    }

    template <legacy::MatrixLayout Layout>
    struct CutlassLayout final {
        typedef cutlass::layout::ColumnMajor type;
    };

    template <>
    struct CutlassLayout<legacy::RM> final {
        typedef cutlass::layout::RowMajor type;
    };

    template <>
    struct CutlassLayout<legacy::CM> final {
        typedef cutlass::layout::ColumnMajor type;
    };

    template <typename T>
    struct CutlassElementType final {
        typedef cutlass::half_t type;
    };

    template <>
    struct CutlassElementType<float> final {
        typedef float type;
    };

    template <typename ThreadBlock, typename Warp>
    struct LayerConfig {
        typedef ThreadBlock thread_block_shape;
        typedef Warp warp_shape;
    };

    struct FullLayerK : LayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>> {};
    struct LastLayerK : FullLayerK {};

    struct FullLayer : LayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>> {};
    struct LastLayer : FullLayer {};

    template <typename V>
    struct CutlassFragmentWrapper final {
        static constexpr std::uint32_t num_elements = V::kElements;
        V x                                         = {};
    };

    template <typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_, cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
    class ActivationEpilogue {
    public:
        typedef ElementOutput_ ElementOutput;
        typedef ElementAccumulator_ ElementAccumulator;
        typedef ElementCompute_ ElementCompute;

        static constexpr int kCount = Count;

        typedef cutlass::Array<ElementOutput, kCount> FragmentOutput;
        typedef cutlass::Array<ElementAccumulator, kCount> FragmentAccumulator;
        typedef cutlass::Array<ElementCompute, kCount> ComputeFragment;

        static constexpr cutlass::FloatRoundStyle kRound = Round;

        struct Params {
            network::detail::Activation activation;
            bool sum_source;
        };

        CUTLASS_HOST_DEVICE
        explicit ActivationEpilogue(const Params& params) : m_activation{params.activation}, m_sum_source{params.sum_source} {}

        CUTLASS_HOST_DEVICE
        bool is_source_needed() const {
            return m_sum_source;
        }

        CUTLASS_HOST_DEVICE
        void set_k_partition(int k_partition, int k_partition_count) {}

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

            CutlassFragmentWrapper<ComputeFragment> intermediate{accumulator_converter(accumulator)};
            intermediate = network::detail::warp_activation<ElementCompute>(m_activation, intermediate);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(intermediate.x);
        }

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator, const FragmentOutput& source) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

            cutlass::plus<ComputeFragment> plus_op;
            CutlassFragmentWrapper<ComputeFragment> intermediate{accumulator_converter(accumulator)};
            if (m_sum_source) intermediate.x = plus_op(intermediate.x, source_converter(source));
            intermediate = network::detail::warp_activation<ElementCompute>(m_activation, intermediate);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(intermediate.x);
        }

    private:
        network::detail::Activation m_activation;
        bool m_sum_source = false;
    };

    template <typename ElementOutput_, int Count, typename ElementAccumulator_ = ElementOutput_, typename ElementCompute_ = ElementOutput_, cutlass::FloatRoundStyle Round = cutlass::FloatRoundStyle::round_to_nearest>
    class ActivationTransferEpilogue {
    public:
        typedef ElementOutput_ ElementOutput;
        typedef ElementAccumulator_ ElementAccumulator;
        typedef ElementCompute_ ElementCompute;

        static constexpr int kCount = Count;

        typedef cutlass::Array<ElementOutput, kCount> FragmentOutput;
        typedef cutlass::Array<ElementAccumulator, kCount> FragmentAccumulator;
        typedef cutlass::Array<ElementCompute, kCount> ComputeFragment;

        static constexpr cutlass::FloatRoundStyle kRound = Round;

        struct Params {
            network::detail::Activation activation;
        };

        CUTLASS_HOST_DEVICE
        explicit ActivationTransferEpilogue(const Params& params) : m_activation{params.activation} {}

        CUTLASS_HOST_DEVICE
        bool is_source_needed() const {
            return true;
        }

        CUTLASS_HOST_DEVICE
        void set_k_partition(int k_partition, int k_partition_count) {}

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator, const FragmentOutput& source) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementOutput, kCount, Round> source_converter;
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;

            CutlassFragmentWrapper<ComputeFragment> converted_source{source_converter(source)};
            CutlassFragmentWrapper<ComputeFragment> intermediate{accumulator_converter(accumulator)};
            intermediate = network::detail::warp_activation_backward<ElementCompute>(m_activation, intermediate, converted_source);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(intermediate.x);
        }

        CUTLASS_HOST_DEVICE
        FragmentOutput operator()(const FragmentAccumulator& accumulator) const {
            cutlass::NumericArrayConverter<ElementCompute, ElementAccumulator, kCount, Round> accumulator_converter;
            ComputeFragment converted_accumulator = accumulator_converter(accumulator);

            cutlass::NumericArrayConverter<ElementOutput, ElementCompute, kCount, Round> destination_converter;
            return destination_converter(converted_accumulator);
        }

    private:
        network::detail::Activation m_activation;
    };

    template <typename T>
    inline constexpr int n_vectorized_elements = 128 / cutlass::sizeof_bits<T>::value;

    template <class Gemm>
    void fc_multiply_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
        const std::size_t workspace_size = Gemm::get_workspace_size(args);
        Gemm gemm_op;

        auto workspace         = network::detail::allocate_workspace(stream, workspace_size);
        cutlass::Status status = gemm_op.initialize(args, workspace.data(), stream);
        cutlass_check(status);

        status = gemm_op(stream);
        cutlass_check(status);
    }

    template <class Gemm>
    void fc_multiply_split_k_impl(cudaStream_t stream, const typename Gemm::Arguments& args) {
        const std::size_t workspace_size = Gemm::get_workspace_size(args);
        Gemm gemm_op;

        auto workspace         = network::detail::allocate_workspace(stream, workspace_size);
        cutlass::Status status = gemm_op.initialize(args, workspace.data());
        cutlass_check(status);

        status = gemm_op(stream);
        cutlass_check(status);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, legacy::MatrixLayout LayoutC, typename TypeD, legacy::MatrixLayout LayoutD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, LayoutC>& C, const legacy::GPUMatrix<TypeD, LayoutD>& D, network::detail::Activation act = network::detail::Activation::None, bool transfer = false, bool sum_source = false) {
        static_assert(std::is_same_v<TypeA, TypeB>, "Type of matrix A and B must be equal");
        static_assert(std::is_same_v<TypeC, TypeD>, "Type of matrix C and D must be equal");
        static_assert(std::is_same_v<typename CutlassLayout<LayoutC>::type, typename CutlassLayout<LayoutD>::type>, "Layout of matrix C and D must be equal");

        if (A.n() != B.m()) throw std::runtime_error{"Matrices A and B can not be multiplied together"};

        const int M = static_cast<int>(A.m());
        const int K = static_cast<int>(A.n());
        const int N = static_cast<int>(B.n());

        if (C.m() != static_cast<std::uint32_t>(M) || C.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix C has incorrect size " << C.m() << 'x' << C.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        if (D.m() != static_cast<std::uint32_t>(M) || D.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix D has incorrect size " << D.m() << 'x' << D.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        typedef typename CutlassElementType<TypeA>::type MatmulTypeCompute;
        typedef typename CutlassElementType<TypeC>::type MatmulTypeAccumulator;

        if (transfer) {
            typedef cutlass::gemm::device::Gemm<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
                ActivationTransferEpilogue<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>
                Gemm;

            typename Gemm::Arguments arguments{{M, N, K}, {(MatmulTypeCompute*) A.data(), (int) A.stride()}, {(MatmulTypeCompute*) B.data(), (int) B.stride()}, {(MatmulTypeAccumulator*) C.data(), (int) C.stride()}, {(MatmulTypeAccumulator*) D.data(), (int) D.stride()}, {act}, 1};
            fc_multiply_impl<Gemm>(stream, arguments);
        } else {
            typedef cutlass::gemm::device::Gemm<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
                ActivationEpilogue<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>
                Gemm;

            typename Gemm::Arguments arguments{{M, N, K}, {(MatmulTypeCompute*) A.data(), (int) A.stride()}, {(MatmulTypeCompute*) B.data(), (int) B.stride()}, {(MatmulTypeAccumulator*) C.data(), (int) C.stride()}, {(MatmulTypeAccumulator*) D.data(), (int) D.stride()}, {act, sum_source}, 1};
            fc_multiply_impl<Gemm>(stream, arguments);
        }
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrixDynamic<TypeC>& C, const legacy::GPUMatrixDynamic<TypeD>& D, network::detail::Activation act = network::detail::Activation::None, bool transfer = false, bool sum_source = false) {
        if (C.layout() != D.layout()) throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
        if (D.layout() == legacy::CM)
            fc_multiply<Config>(stream, A, B, C.cm(), D.cm(), act, transfer, sum_source);
        else
            fc_multiply<Config>(stream, A, B, C.rm(), D.rm(), act, transfer, sum_source);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrixDynamic<TypeB>& B, const legacy::GPUMatrixDynamic<TypeC>& C, const legacy::GPUMatrixDynamic<TypeD>& D, network::detail::Activation act = network::detail::Activation::None, bool transfer = false, bool sum_source = false) {
        if (B.layout() == legacy::CM)
            fc_multiply<Config>(stream, A, B.cm(), C, D, act, transfer, sum_source);
        else
            fc_multiply<Config>(stream, A, B.rm(), C, D, act, transfer, sum_source);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeD>
    void fc_multiply(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrixDynamic<TypeB>& B, const legacy::GPUMatrixDynamic<TypeD>& D, network::detail::Activation act = network::detail::Activation::None) {
        fc_multiply<Config>(stream, A, B, D, D, act);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, legacy::MatrixLayout LayoutC, typename TypeD, legacy::MatrixLayout LayoutD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrix<TypeC, LayoutC>& C, const legacy::GPUMatrix<TypeD, LayoutD>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        static_assert(std::is_same_v<TypeA, TypeB>, "Type of matrix A and B must be equal");
        static_assert(std::is_same_v<TypeC, TypeD>, "Type of matrix C and D must be equal");
        static_assert(std::is_same_v<typename CutlassLayout<LayoutC>::type, typename CutlassLayout<LayoutD>::type>, "Layout of matrix C and D must be equal");

        if (A.n() != B.m()) throw std::runtime_error{"Matrices A and B can not be multiplied together"};

        const int M = static_cast<int>(A.m());
        const int K = static_cast<int>(A.n());
        const int N = static_cast<int>(B.n());

        if (C.m() != static_cast<std::uint32_t>(M) || C.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix C has incorrect size " << C.m() << 'x' << C.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        if (D.m() != static_cast<std::uint32_t>(M) || D.n() != static_cast<std::uint32_t>(N)) {
            std::ostringstream stream_message;
            stream_message << "Matrix D has incorrect size " << D.m() << 'x' << D.n() << " != " << M << 'x' << N;
            throw std::runtime_error{stream_message.str()};
        }

        typedef typename CutlassElementType<TypeA>::type MatmulTypeCompute;
        typedef typename CutlassElementType<TypeC>::type MatmulTypeAccumulator;
        typedef cutlass::gemm::device::GemmSplitKParallel<MatmulTypeCompute, typename CutlassLayout<LayoutA>::type, MatmulTypeCompute, typename CutlassLayout<LayoutB>::type, MatmulTypeAccumulator, typename CutlassLayout<LayoutC>::type, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>,
            cutlass::epilogue::thread::LinearCombination<MatmulTypeAccumulator, n_vectorized_elements<MatmulTypeAccumulator>, cutlass::half_t, cutlass::half_t>>
            Gemm;

        typename Gemm::Arguments arguments{{M, N, K}, {(MatmulTypeCompute*) A.data(), (int) A.stride()}, {(MatmulTypeCompute*) B.data(), (int) B.stride()}, {(MatmulTypeAccumulator*) C.data(), (int) C.stride()}, {(MatmulTypeAccumulator*) D.data(), (int) D.stride()}, {(cutlass::half_t) 1.0f, (cutlass::half_t) beta}, (int) split_k_slices};
        fc_multiply_split_k_impl<Gemm>(stream, arguments);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, legacy::MatrixLayout LayoutB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrix<TypeB, LayoutB>& B, const legacy::GPUMatrixDynamic<TypeC>& C, const legacy::GPUMatrixDynamic<TypeD>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (C.layout() != D.layout()) throw std::runtime_error{"fc_multiply: Layout of GPUMatrixDynamic C and D must be equal"};
        if (D.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A, B, C.cm(), D.cm(), split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A, B, C.rm(), D.rm(), split_k_slices, beta);
    }

    template <typename Config, typename TypeA, legacy::MatrixLayout LayoutA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrix<TypeA, LayoutA>& A, const legacy::GPUMatrixDynamic<TypeB>& B, const legacy::GPUMatrixDynamic<TypeC>& C, const legacy::GPUMatrixDynamic<TypeD>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (B.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A, B.cm(), C, D, split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A, B.rm(), C, D, split_k_slices, beta);
    }

    template <typename Config, typename TypeA, typename TypeB, typename TypeC, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrixDynamic<TypeA>& A, const legacy::GPUMatrixDynamic<TypeB>& B, const legacy::GPUMatrixDynamic<TypeC>& C, const legacy::GPUMatrixDynamic<TypeD>& D, std::uint32_t split_k_slices = 1u, float beta = 0.0f) {
        if (A.layout() == legacy::CM)
            fc_multiply_split_k<Config>(stream, A.cm(), B, C, D, split_k_slices, beta);
        else
            fc_multiply_split_k<Config>(stream, A.rm(), B, C, D, split_k_slices, beta);
    }

    template <typename Config, typename TypeA, typename TypeB, typename TypeD>
    void fc_multiply_split_k(cudaStream_t stream, const legacy::GPUMatrixDynamic<TypeA>& A, const legacy::GPUMatrixDynamic<TypeB>& B, const legacy::GPUMatrixDynamic<TypeD>& D, std::uint32_t split_k_slices, float beta) {
        fc_multiply_split_k<Config>(stream, A, B, D, D, split_k_slices, beta);
    }

    template <typename T>
    void activation_backward_output_gpu(cudaStream_t stream, const std::uint32_t num_elements, const network::detail::Activation act, const T* __restrict__ output_values, const T* gradients_out, T* gradients_in) {
        static constexpr std::uint32_t activation_vector_size = 16u / sizeof(T);
        if (num_elements % activation_vector_size != 0u) {
            std::ostringstream stream_message;
            stream_message << "activation_backward_output_gpu: number of elements must be a multiple of " << activation_vector_size;
            throw std::runtime_error{stream_message.str()};
        }

        if (act == network::detail::Activation::None && gradients_out == gradients_in) return;

        const std::uint32_t vector_count = num_elements / activation_vector_size;
        if (vector_count > 0u) {
            const std::uint32_t blocks = (vector_count + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            network::detail::kernel_activation_backward_output<T, activation_vector_size><<<blocks, network::detail::n_threads_linear, 0, stream>>>(vector_count, act, output_values, gradients_out, gradients_in);
        }
    }

    inline void check_shmem_error(const cudaError_t error) {
        if (error == cudaSuccess) return;
        throw std::runtime_error{"FullyFusedMLP: insufficient shared memory available on the GPU. Reduce the selected compile-time network width to fit the device."};
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, bool BACKWARD = false>
    __device__ void threadblock_layer(network::detail::Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const OUT_T* __restrict__ activation_aux = nullptr) {
        constexpr std::uint32_t SKEW     = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, std::conditional_t<BACKWARD, nvcuda::wmma::row_major, nvcuda::wmma::col_major>> weights_frag[N_BLOCKS];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;
        const std::uint32_t weights_col = 16u * wi;

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
            if constexpr (BACKWARD)
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * WIDTH + weights_col, WIDTH);
            else
                nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * WIDTH, WIDTH);
        }

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
            nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
                nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (WIDTH + SKEW), WIDTH + SKEW);
                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
            }

            if constexpr (BACKWARD) {
                nvcuda::wmma::load_matrix_sync(act_frag, activation_aux + weights_col + l * 16u * WIDTH, WIDTH);
                network::detail::warp_activation_backward<__half>(activation, result_frag[l], act_frag, result_frag[l]);
            } else {
                network::detail::warp_activation<__half>(activation, result_frag[l], result_frag[l]);
            }
        }

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + l * 16u * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

        if (out_intermediate_threadblock_this_layer != nullptr) {
            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) *(int4*) &out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * l) * WIDTH] = *(int4*) &act_shmem[lane_offset + (row + 16u * l) * (WIDTH + SKEW)];
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS>
    __device__ void threadblock_load_input_static(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock) {
        constexpr std::uint32_t SKEW = WIDTH % 16u == 0u ? 8u : 0u;

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_ITERS; ++i) *(int4*) &act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)] = *(int4*) &input_threadblock[lane_offset + (row + 16u * i) * WIDTH];
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, network::detail::Activation ACTIVATION, typename OUTPUT_LAYOUT>
    __global__ void kernel_mlp_fused_backward(const __half* __restrict__ dL_doutput, const __half* __restrict__ weights, __half* __restrict__ out_intermediate, const __half* __restrict__ forward, __half* __restrict__ dL_dinput, const __half* __restrict__ weights_first_layer, const std::uint32_t output_stride, const std::uint32_t batch_size, const std::uint32_t out_width, const std::uint32_t n_hidden_matmuls) {
        constexpr std::uint32_t SKEW = WIDTH % 16u == 0u ? 8u : 0u;

        const std::uint32_t wi            = threadIdx.y;
        const std::uint32_t bi            = blockIdx.x;
        const std::uint32_t elem_idx_base = 16u * bi * N_ITERS;
        const std::uint32_t elem_idx      = elem_idx_base;

        extern __shared__ __half shmem[];
        __half* act_shmem = shmem;

        const std::uint32_t weights_stride = WIDTH * WIDTH;
        const std::uint32_t layer_stride   = WIDTH * batch_size;

        if (out_width <= 16u) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, OUTPUT_LAYOUT> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[N_ITERS];

            const std::uint32_t weights_col = 16u * wi;
            nvcuda::wmma::load_matrix_sync(weights_frag, weights + weights_stride * n_hidden_matmuls + weights_col, WIDTH);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

                if constexpr (std::is_same_v<OUTPUT_LAYOUT, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16u * l) * output_stride, output_stride);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, dL_doutput + (elem_idx + 16u * l), output_stride);

                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward + layer_stride * n_hidden_matmuls + weights_col + (elem_idx + l * 16u) * WIDTH, WIDTH);
                network::detail::warp_activation_backward<__half>(ACTIVATION, result_frag[l], forward_frag, result_frag[l]);
            }

            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t lane_offset = (8u * li) % WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_ITERS; ++i) *(int4*) &out_intermediate[lane_offset + (row + elem_idx + i * 16u) * WIDTH] = *(int4*) &act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)];
        } else {
            threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
        }

        for (std::uint32_t k = 0u; k < n_hidden_matmuls; ++k) threadblock_layer<WIDTH, N_ITERS, __half, true>(ACTIVATION, act_shmem, weights + weights_stride * (n_hidden_matmuls - k - 1u), out_intermediate + layer_stride * (k + 1u) + elem_idx_base * WIDTH, forward + layer_stride * (n_hidden_matmuls - k - 1u) + elem_idx_base * WIDTH);

        if (dL_dinput != nullptr) threadblock_layer<WIDTH, N_ITERS, __half, true>(network::detail::Activation::None, act_shmem, weights_first_layer, dL_dinput + elem_idx_base * WIDTH);
    }

    template <std::uint32_t WIDTH, typename T, network::detail::Activation ACTIVATION>
    void mlp_fused_backward(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::RM>& weights_first_layer, const legacy::GPUMatrix<T, legacy::RM>& weights, const legacy::GPUMatrixDynamic<T>& dL_doutput, legacy::GPUMatrixDynamic<T>& temporaries, const legacy::GPUMatrixDynamic<T>& forward, legacy::GPUMatrixDynamic<T>* dL_dinput, const std::uint32_t n_hidden_matmuls) {
        static_assert(std::is_same_v<T, __half>, "The fully fused backward pass only supports __half precision.");
        const std::uint32_t batch_size   = dL_doutput.cols();
        const std::uint32_t out_width    = dL_doutput.rows();
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;
        const std::uint32_t N_ITERS      = WIDTH >= 256u ? 2u : 8u;

        legacy::check_or_throw(forward.cols() == batch_size);
        legacy::check_or_throw(batch_size % (16u * N_ITERS) == 0u);
        legacy::check_or_throw(!dL_dinput || dL_dinput->layout() == legacy::RM || dL_dinput->stride() == dL_dinput->m());

        const dim3 threads                    = {32u, N_BLOCKS, 1u};
        const std::uint32_t n_elems_per_block = 16u * N_ITERS;
        const std::uint32_t n_blocks          = (batch_size + n_elems_per_block - 1u) / n_elems_per_block;
        const int shmem_size                  = sizeof(__half) * ((16u * N_ITERS) * (WIDTH + (WIDTH % 16u == 0u ? 8u : 0u)));
        const dim3 blocks                     = {n_blocks, 1u, 1u};

        if (dL_doutput.layout() == legacy::RM) {
            check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::col_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
        } else {
            check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, shmem_size));
            kernel_mlp_fused_backward<WIDTH, N_ITERS, ACTIVATION, nvcuda::wmma::row_major><<<blocks, threads, shmem_size, stream>>>(dL_doutput.data(), weights.data(), temporaries.data(), forward.data(), dL_dinput ? dL_dinput->data() : nullptr, weights_first_layer.data(), dL_doutput.stride(), batch_size, out_width, n_hidden_matmuls);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, typename INPUT_LAYOUT>
    __device__ void threadblock_input_layer_forward_dynamic(network::detail::Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out_intermediate_threadblock_this_layer, const std::uint32_t in_width, const std::uint32_t batch_size) {
        constexpr std::uint32_t SKEW       = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t INPUT_SKEW = 8u;
        constexpr std::uint32_t N_BLOCKS   = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, INPUT_LAYOUT> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag[N_ITERS];

        const std::uint32_t li          = threadIdx.x;
        const std::uint32_t wi          = threadIdx.y;
        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;
        const std::uint32_t weights_col = 16u * wi;

        __half* __restrict__ weights_shmem   = act_shmem + 16u * (in_width + INPUT_SKEW);
        const std::uint32_t n_elems_per_load = N_BLOCKS * 32u * 8u;
        const std::uint32_t thread_elem_idx  = (li + wi * 32u) * 8u;
        const std::uint32_t n_elems_b        = WIDTH * in_width;

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t idx = thread_elem_idx; idx < n_elems_b; idx += n_elems_per_load) {
            const std::uint32_t idx_skewed      = idx + idx / in_width * INPUT_SKEW;
            *(int4*) &weights_shmem[idx_skewed] = *(int4*) &weights_this_layer[idx];
        }

        const std::uint32_t n_tensor_ops = in_width / 16u;
        if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::col_major>) __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) {
            if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>) {
                const std::uint32_t n_elems_a = 16u * in_width;

                TCNN_PRAGMA_UNROLL
                for (std::uint32_t idx = thread_elem_idx; idx < n_elems_a; idx += n_elems_per_load) {
                    const std::uint32_t idx_skewed  = idx + idx / in_width * INPUT_SKEW;
                    *(int4*) &act_shmem[idx_skewed] = *(int4*) &input_threadblock[l * n_elems_a + idx];
                }

                __syncthreads();
            }

            nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < n_tensor_ops; ++i) {
                if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i, in_width + INPUT_SKEW);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, input_threadblock + 16u * i * batch_size + 16u * l, batch_size);

                nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (in_width + INPUT_SKEW), in_width + INPUT_SKEW);
                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
            }

            if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::row_major>) __syncthreads();
            network::detail::warp_activation<__half>(activation, result_frag[l], result_frag[l]);
        }

        if constexpr (std::is_same_v<INPUT_LAYOUT, nvcuda::wmma::col_major>) __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t l = 0u; l < N_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (WIDTH + SKEW), result_frag[l], WIDTH + SKEW, nvcuda::wmma::mem_row_major);

        if (out_intermediate_threadblock_this_layer != nullptr) {
            __syncthreads();

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_ITERS; ++i) *(int4*) &out_intermediate_threadblock_this_layer[lane_offset + (row + 16u * i) * WIDTH] = *(int4*) &act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)];
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T>
    __device__ void threadblock_last_layer_forward(network::detail::Activation activation, __half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, OUT_T* __restrict__ out, const std::uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
        constexpr std::uint32_t SKEW     = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t N_BLOCKS = WIDTH / 16u;

        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[N_BLOCKS];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, OUT_T> result_frag;

        const std::uint32_t li = threadIdx.x;
        const std::uint32_t wi = threadIdx.y;

        __half* __restrict__ weights_shmem = act_shmem + N_ITERS * 16u * (WIDTH + SKEW);
        const std::uint32_t weights_row    = (8u * li) % WIDTH;
        const std::uint32_t weights_col    = (8u * li + 8u * 32u * wi) / WIDTH;

        *(int4*) &weights_shmem[weights_row + weights_col * (WIDTH + SKEW)] = *(int4*) &weights_this_layer[weights_row + weights_col * WIDTH];
        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, WIDTH + SKEW);

        for (std::uint32_t idx = wi; idx < N_ITERS; idx += N_BLOCKS) {
            nvcuda::wmma::fill_fragment(result_frag, 0.0f);

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t i = 0u; i < N_BLOCKS; ++i) {
                nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * idx) * (WIDTH + SKEW), WIDTH + SKEW);
                nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
            }

            network::detail::warp_activation<__half>(activation, result_frag, result_frag);
            if (output_layout == nvcuda::wmma::mem_row_major)
                nvcuda::wmma::store_matrix_sync(out + idx * 16u * output_stride, result_frag, output_stride, output_layout);
            else
                nvcuda::wmma::store_matrix_sync(out + idx * 16u, result_frag, output_stride, output_layout);
        }
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS>
    __device__ void threadblock_write_output_static(const __half* __restrict__ act_shmem, __half* __restrict__ output_threadblock) {
        constexpr std::uint32_t SKEW    = WIDTH % 16u == 0u ? 8u : 0u;
        const std::uint32_t li          = threadIdx.x;
        const std::uint32_t wi          = threadIdx.y;
        const std::uint32_t lane_offset = (8u * li) % WIDTH;
        const std::uint32_t row         = (8u * li + wi * 8u * 32u) / WIDTH;

        __syncthreads();

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_ITERS; ++i) *(int4*) &output_threadblock[lane_offset + (row + 16u * i) * WIDTH] = *(int4*) &act_shmem[lane_offset + (row + 16u * i) * (WIDTH + SKEW)];
    }

    template <std::uint32_t WIDTH, std::uint32_t N_ITERS, typename OUT_T, network::detail::Activation ACTIVATION, bool INFERENCE>
    __global__ void kernel_mlp_fused(const network::detail::Activation output_activation, const __half* __restrict__ input, const __half* __restrict__ weights, OUT_T* __restrict__ out_intermediate, OUT_T* __restrict__ out, const std::uint32_t output_stride, const std::uint32_t batch_size, const std::uint32_t in_width, const std::uint32_t out_width, const std::uint32_t n_hidden_matmuls, const nvcuda::wmma::layout_t input_layout, const nvcuda::wmma::layout_t output_layout) {
        extern __shared__ __half shmem[];
        __half* act_shmem = shmem;

        const std::uint32_t elem_idx = 16u * blockIdx.x * N_ITERS;

        if (input_layout == nvcuda::wmma::mem_col_major || in_width != WIDTH) {
            if (input_layout == nvcuda::wmma::mem_row_major)
                threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::row_major>(ACTIVATION, act_shmem, input + elem_idx * in_width, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
            else
                threadblock_input_layer_forward_dynamic<WIDTH, N_ITERS, OUT_T, nvcuda::wmma::col_major>(ACTIVATION, act_shmem, input + elem_idx, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr, in_width, batch_size);
        } else {
            threadblock_load_input_static<WIDTH, N_ITERS>(act_shmem, input + elem_idx * WIDTH);
            threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights, !INFERENCE ? (out_intermediate + elem_idx * WIDTH) : nullptr);
        }

        const std::uint32_t first_weights_stride = WIDTH * in_width;
        const std::uint32_t weights_stride       = WIDTH * WIDTH;
        const std::uint32_t layer_stride         = WIDTH * batch_size;

        for (std::uint32_t k = 0u; k < n_hidden_matmuls; ++k) threadblock_layer<WIDTH, N_ITERS, OUT_T>(ACTIVATION, act_shmem, weights + first_weights_stride + weights_stride * k, !INFERENCE ? (out_intermediate + layer_stride * (k + 1u) + elem_idx * WIDTH) : nullptr);

        if (out_width > 16u) {
            if (INFERENCE) threadblock_write_output_static<WIDTH, N_ITERS>(act_shmem, out_intermediate + elem_idx * WIDTH);
        } else if (out != nullptr) {
            if (output_layout == nvcuda::wmma::mem_row_major)
                threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx * output_stride, output_stride, output_layout);
            else
                threadblock_last_layer_forward<WIDTH, N_ITERS, OUT_T>(output_activation, act_shmem, weights + first_weights_stride + weights_stride * n_hidden_matmuls, out + elem_idx, output_stride, output_layout);
        }
    }

    template <std::uint32_t WIDTH, typename T, network::detail::Activation ACTIVATION, bool INFERENCE>
    void mlp_fused_forward(cudaStream_t stream, network::detail::Activation output_activation, const legacy::GPUMatrix<T, legacy::RM>& weights, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>& output_intermediate, legacy::GPUMatrixDynamic<T>* output, const std::uint32_t n_hidden_layers) {
        static_assert(std::is_same_v<T, __half>, "The fully fused forward pass only supports __half precision.");
        const std::uint32_t batch_size = input.cols();
        const std::uint32_t in_width   = input.rows();

        constexpr std::uint32_t SKEW         = WIDTH % 16u == 0u ? 8u : 0u;
        constexpr std::uint32_t INPUT_SKEW   = 8u;
        constexpr std::uint32_t N_BLOCK_ROWS = WIDTH / 16u;

        static_assert(WIDTH % 16u == 0u, "Width must be a multiply of 16.");

        legacy::check_or_throw(in_width % 16u == 0u);
        legacy::check_or_throw(weights.rows() == WIDTH);
        legacy::check_or_throw(weights.cols() % 16u == 0u);
        legacy::check_or_throw(output_intermediate.cols() == batch_size);
        legacy::check_or_throw(!output || output->cols() == batch_size);
        legacy::check_or_throw(input.layout() == legacy::RM || input.stride() == input.m());

        const std::uint32_t N_ITERS = WIDTH >= 256u ? 2u : 8u;
        if (batch_size % (16u * N_ITERS) != 0u) {
            std::ostringstream stream_message;
            stream_message << "Batch size must be a multiple of " << (16u * N_ITERS) << '.';
            throw std::runtime_error{stream_message.str()};
        }

        const dim3 threads                    = {32u, N_BLOCK_ROWS, 1u};
        const std::uint32_t n_elems_per_block = 16u * N_ITERS;
        const std::uint32_t n_blocks          = (batch_size + n_elems_per_block - 1u) / n_elems_per_block;

        std::size_t shmem_size = sizeof(__half) * (16u + 16u * N_ITERS) * (WIDTH + SKEW);
        if (in_width != WIDTH || input.layout() == legacy::RM) shmem_size = std::max(shmem_size, sizeof(__half) * (WIDTH + 16u) * (in_width + INPUT_SKEW));

        const dim3 blocks = {n_blocks, 1u, 1u};

        check_shmem_error(cudaFuncSetAttribute(kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) shmem_size));
        kernel_mlp_fused<WIDTH, N_ITERS, __half, ACTIVATION, INFERENCE>
            <<<blocks, threads, shmem_size, stream>>>(output_activation, input.data(), weights.data(), output_intermediate.data(), output ? output->data() : nullptr, output ? output->stride() : 0u, batch_size, in_width, output ? output->rows() : 0u, n_hidden_layers, input.layout() == legacy::RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major, output && output->layout() == legacy::RM ? nvcuda::wmma::mem_col_major : nvcuda::wmma::mem_row_major);
    }

    template <typename T, std::uint32_t WIDTH>
    class FullyFusedMLP {
    public:
        FullyFusedMLP(std::uint32_t input_width, std::uint32_t output_width, std::uint32_t n_hidden_layers, network::detail::Activation activation, network::detail::Activation output_activation);

        struct Scratch {
            std::vector<legacy::GPUMatrixDynamic<T>> forward_hidden;
            std::vector<legacy::GPUMatrixDynamic<T>> backward_hidden;
            legacy::GPUMatrixDynamic<T> backward_output;
            legacy::GpuAllocation forward_alloc;
            legacy::GpuAllocation backward_alloc;
        };

        void set_params(T* params, T* gradients);

        T* params() const {
            return m_params;
        }

        T* gradients() const {
            return m_gradients;
        }

        void inference(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>& output);
        void prepare_scratch(cudaStream_t stream, std::uint32_t batch_size, legacy::MatrixLayout output_layout, Scratch& scratch);
        void forward(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>* output, Scratch& scratch);
        void backward(cudaStream_t stream, Scratch& scratch, const legacy::GPUMatrixDynamic<T>& input, const legacy::GPUMatrixDynamic<T>& output, const legacy::GPUMatrixDynamic<T>& dL_doutput, legacy::GPUMatrixDynamic<T>* dL_dinput = nullptr, network::detail::GradientMode param_gradients_mode = network::detail::GradientMode::Overwrite);

        void initialize_params(legacy::math::pcg32& rnd, float* params_full_precision, float scale = 1.0f);

        legacy::GPUMatrix<T, legacy::RM>& input_weight_matrix() {
            return m_weight_matrices.front();
        }

        legacy::GPUMatrix<T, legacy::RM>& weight_matrix_at(const std::uint32_t idx) {
            return m_weight_matrices.at(1u + idx);
        }

        legacy::GPUMatrix<T, legacy::RM>& output_weight_matrix() {
            return m_weight_matrices.back();
        }

        legacy::GPUMatrix<T, legacy::RM>& input_gradient_matrix() {
            return m_gradient_matrices.front();
        }

        legacy::GPUMatrix<T, legacy::RM>& gradient_matrix_at(const std::uint32_t idx) {
            return m_gradient_matrices.at(1u + idx);
        }

        legacy::GPUMatrix<T, legacy::RM>& output_gradient_matrix() {
            return m_gradient_matrices.back();
        }

        std::size_t n_params() const {
            return m_total_n_params;
        }

        std::uint32_t input_width() const {
            return m_input_width;
        }

        std::uint32_t padded_output_width() const {
            return m_padded_output_width;
        }

        std::uint32_t output_width() const {
            return m_output_width;
        }

    private:
        std::uint32_t m_n_hidden_layers                 = 0u;
        std::uint32_t m_n_hidden_matmuls                = 0u;
        std::uint32_t m_input_width                     = 0u;
        std::uint32_t m_network_width                   = WIDTH;
        std::uint32_t m_output_width                    = 0u;
        std::uint32_t m_padded_output_width             = 0u;
        network::detail::Activation m_activation        = network::detail::Activation::None;
        network::detail::Activation m_output_activation = network::detail::Activation::None;
        std::vector<legacy::GPUMatrix<T, legacy::RM>> m_weight_matrices;
        std::size_t m_total_n_params = 0u;
        std::vector<legacy::GPUMatrix<T, legacy::RM>> m_gradient_matrices;
        T* m_params    = nullptr;
        T* m_gradients = nullptr;
    };

    template <typename T, std::uint32_t WIDTH>
    FullyFusedMLP<T, WIDTH>::FullyFusedMLP(const std::uint32_t input_width, const std::uint32_t output_width, const std::uint32_t n_hidden_layers, const network::detail::Activation activation, const network::detail::Activation output_activation) : m_n_hidden_layers{n_hidden_layers}, m_input_width{input_width}, m_output_width{output_width}, m_activation{activation}, m_output_activation{output_activation} {
        if (m_n_hidden_layers <= 0u) throw std::runtime_error{"FullyFusedMLP requires at least 1 hidden layer (3 layers in total)."};

        m_n_hidden_matmuls    = n_hidden_layers - 1u;
        m_padded_output_width = legacy::next_multiple(m_output_width, 16u);

        m_weight_matrices.emplace_back(nullptr, m_network_width, m_input_width);
        m_gradient_matrices.emplace_back(nullptr, m_network_width, m_input_width);

        for (std::uint32_t i = 0u; i < m_n_hidden_matmuls; ++i) {
            m_weight_matrices.emplace_back(nullptr, m_network_width, m_network_width);
            m_gradient_matrices.emplace_back(nullptr, m_network_width, m_network_width);
        }

        m_weight_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);
        m_gradient_matrices.emplace_back(nullptr, m_padded_output_width, m_network_width);

        m_total_n_params = 0u;
        for (const auto& matrix : m_weight_matrices) m_total_n_params += matrix.n_elements();
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::inference(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>& output) {
        legacy::check_or_throw(input.m() == input_width());
        legacy::check_or_throw(output.m() == padded_output_width());
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(params() != nullptr);

        const std::uint32_t batch_size            = input.n();
        legacy::GPUMatrixDynamic<T> inference_tmp = m_output_width > 16u ? ngp::legacy::GPUMatrixDynamic<T>{m_network_width, batch_size, stream, legacy::CM} : ngp::legacy::GPUMatrixDynamic<T>{nullptr, m_network_width, batch_size, legacy::CM};

        switch (m_activation) {
        case network::detail::Activation::None: mlp_fused_forward<WIDTH, T, network::detail::Activation::None, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Exponential: mlp_fused_forward<WIDTH, T, network::detail::Activation::Exponential, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Sigmoid: mlp_fused_forward<WIDTH, T, network::detail::Activation::Sigmoid, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::ReLU: mlp_fused_forward<WIDTH, T, network::detail::Activation::ReLU, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::LeakyReLU: mlp_fused_forward<WIDTH, T, network::detail::Activation::LeakyReLU, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Squareplus: mlp_fused_forward<WIDTH, T, network::detail::Activation::Squareplus, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Softplus: mlp_fused_forward<WIDTH, T, network::detail::Activation::Softplus, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Tanh: mlp_fused_forward<WIDTH, T, network::detail::Activation::Tanh, true>(stream, m_output_activation, input_weight_matrix(), input, inference_tmp, &output, m_n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        if (m_output_width > 16u) fc_multiply<LastLayer>(stream, output_weight_matrix(), inference_tmp, output, m_output_activation);
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::forward(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>* output, Scratch& scratch) {
        legacy::check_or_throw(input.m() == input_width());
        legacy::check_or_throw(!output || output->m() == padded_output_width());
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(!output || input.n() == output->n());
        legacy::check_or_throw(params() != nullptr);

        switch (m_activation) {
        case network::detail::Activation::None: mlp_fused_forward<WIDTH, T, network::detail::Activation::None, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Exponential: mlp_fused_forward<WIDTH, T, network::detail::Activation::Exponential, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Sigmoid: mlp_fused_forward<WIDTH, T, network::detail::Activation::Sigmoid, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::ReLU: mlp_fused_forward<WIDTH, T, network::detail::Activation::ReLU, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::LeakyReLU: mlp_fused_forward<WIDTH, T, network::detail::Activation::LeakyReLU, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Squareplus: mlp_fused_forward<WIDTH, T, network::detail::Activation::Squareplus, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Softplus: mlp_fused_forward<WIDTH, T, network::detail::Activation::Softplus, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        case network::detail::Activation::Tanh: mlp_fused_forward<WIDTH, T, network::detail::Activation::Tanh, false>(stream, m_output_activation, input_weight_matrix(), input, scratch.forward_hidden.at(0), output, m_n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        if (output && m_output_width > 16u) fc_multiply<LastLayer>(stream, output_weight_matrix(), scratch.forward_hidden.back(), *output, m_output_activation);
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::backward(cudaStream_t stream, Scratch& scratch, const legacy::GPUMatrixDynamic<T>& input, const legacy::GPUMatrixDynamic<T>& output, const legacy::GPUMatrixDynamic<T>& dL_doutput, legacy::GPUMatrixDynamic<T>* dL_dinput, const network::detail::GradientMode param_gradients_mode) {
        legacy::check_or_throw(input.m() == input_width());
        legacy::check_or_throw(output.m() == padded_output_width());
        legacy::check_or_throw(dL_doutput.m() == padded_output_width());
        legacy::check_or_throw(!dL_dinput || dL_dinput->m() == input_width());
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(input.n() == dL_doutput.n());
        legacy::check_or_throw(!dL_dinput || input.n() == dL_dinput->n());
        legacy::check_or_throw(params() != nullptr);
        if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(gradients() != nullptr);

        const std::uint32_t batch_size = dL_doutput.n();
        if (m_output_activation != network::detail::Activation::None) activation_backward_output_gpu(stream, dL_doutput.n_elements(), m_output_activation, output.data(), dL_doutput.data(), scratch.backward_output.data());

        const float param_gradient_beta = param_gradients_mode == network::detail::GradientMode::Accumulate ? 1.0f : 0.0f;
        std::vector<network::detail::SyncedStreamReservation> multi_streams;
        const std::uint32_t split_k_factor                = batch_size / std::min(1u << 12u, batch_size);
        const legacy::GPUMatrixDynamic<T>& tmp_dL_doutput = m_output_activation == network::detail::Activation::None ? dL_doutput : scratch.backward_output;

        auto dynamic_view = [](auto& matrix) { return ngp::legacy::GPUMatrixDynamic<typename std::remove_reference_t<decltype(matrix)>::Type>{matrix.data(), matrix.m(), matrix.n(), matrix.layout(), matrix.stride()}; };

        std::uint32_t tmp_idx          = m_n_hidden_matmuls;
        std::uint32_t backward_tmp_idx = 0u;

        if (param_gradients_mode != network::detail::GradientMode::Ignore) {
            multi_streams.emplace_back(stream, 2u);
            auto output_gradient = dynamic_view(output_gradient_matrix());
            fc_multiply_split_k<LastLayerK>(multi_streams.back().aux_stream, tmp_dL_doutput, scratch.forward_hidden.at(tmp_idx).transposed(), output_gradient, split_k_factor, param_gradient_beta);
        }

        if (m_output_width > 16u) fc_multiply<FullLayer>(stream, output_weight_matrix().transposed(), tmp_dL_doutput, scratch.forward_hidden.at(tmp_idx), scratch.backward_hidden.at(backward_tmp_idx), m_activation, true);

        legacy::GPUMatrixDynamic<T>* dL_dinput_fused = input.m() == scratch.forward_hidden.at(0).m() && input.layout() == legacy::CM ? dL_dinput : nullptr;

        switch (m_activation) {
        case network::detail::Activation::None: mlp_fused_backward<WIDTH, T, network::detail::Activation::None>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::Exponential: mlp_fused_backward<WIDTH, T, network::detail::Activation::Exponential>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::Sigmoid: mlp_fused_backward<WIDTH, T, network::detail::Activation::Sigmoid>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::ReLU: mlp_fused_backward<WIDTH, T, network::detail::Activation::ReLU>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::LeakyReLU: mlp_fused_backward<WIDTH, T, network::detail::Activation::LeakyReLU>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::Squareplus: mlp_fused_backward<WIDTH, T, network::detail::Activation::Squareplus>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::Softplus: mlp_fused_backward<WIDTH, T, network::detail::Activation::Softplus>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        case network::detail::Activation::Tanh: mlp_fused_backward<WIDTH, T, network::detail::Activation::Tanh>(stream, input_weight_matrix(), weight_matrix_at(0), tmp_dL_doutput, scratch.backward_hidden.at(backward_tmp_idx), scratch.forward_hidden.at(0), dL_dinput_fused, m_n_hidden_matmuls); break;
        default: throw std::runtime_error{"Unsupported activation."};
        }

        tmp_idx -= 1u;
        ++backward_tmp_idx;

        for (std::uint32_t i = 0u; i < m_n_hidden_matmuls; ++i) {
            const std::uint32_t matrix_idx = m_n_hidden_matmuls - i - 1u;

            if (param_gradients_mode != network::detail::GradientMode::Ignore) {
                multi_streams.emplace_back(stream, 2u);
                auto gradient_matrix = dynamic_view(gradient_matrix_at(matrix_idx));
                fc_multiply_split_k<FullLayerK>(multi_streams.back().aux_stream, scratch.backward_hidden.at(backward_tmp_idx - 1u), scratch.forward_hidden.at(tmp_idx).transposed(), gradient_matrix, split_k_factor, param_gradient_beta);
            }

            tmp_idx -= 1u;
            ++backward_tmp_idx;
        }

        if (param_gradients_mode != network::detail::GradientMode::Ignore) {
            multi_streams.emplace_back(stream, 2u);
            auto input_gradient = dynamic_view(input_gradient_matrix());
            fc_multiply_split_k<FullLayerK>(multi_streams.back().aux_stream, scratch.backward_hidden.at(backward_tmp_idx - 1u), input.transposed(), input_gradient, split_k_factor, param_gradient_beta);
        }

        if (dL_dinput && !dL_dinput_fused) fc_multiply<FullLayer>(stream, input_weight_matrix().transposed(), scratch.backward_hidden.at(backward_tmp_idx - 1u), *dL_dinput);
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::prepare_scratch(cudaStream_t stream, const std::uint32_t batch_size, const legacy::MatrixLayout output_layout, Scratch& scratch) {
        scratch.forward_hidden.resize(m_n_hidden_layers);
        scratch.backward_hidden.resize(m_n_hidden_layers);

        std::size_t shared_forward_bytes  = 0u;
        std::size_t shared_backward_bytes = 0u;
        for (std::uint32_t i = 0u; i < m_n_hidden_layers; ++i) {
            scratch.forward_hidden[i].set_size_unsafe(m_network_width, batch_size);
            scratch.backward_hidden[i].set_size_unsafe(m_network_width, batch_size);
            shared_forward_bytes += scratch.forward_hidden[i].n_bytes();
            shared_backward_bytes += scratch.backward_hidden[i].n_bytes();
        }

        scratch.forward_alloc  = network::detail::allocate_workspace(stream, shared_forward_bytes);
        scratch.backward_alloc = network::detail::allocate_workspace(stream, shared_backward_bytes);

        void* forward_base          = scratch.forward_alloc.data();
        void* backward_base         = scratch.backward_alloc.data();
        std::size_t forward_offset  = 0u;
        std::size_t backward_offset = 0u;
        for (std::uint32_t i = 0u; i < m_n_hidden_layers; ++i) {
            scratch.forward_hidden[i].set_data_unsafe(static_cast<char*>(forward_base) + forward_offset);
            scratch.backward_hidden[i].set_data_unsafe(static_cast<char*>(backward_base) + backward_offset);
            forward_offset += scratch.forward_hidden[i].n_bytes();
            backward_offset += scratch.backward_hidden[i].n_bytes();
        }

        if (m_output_activation != network::detail::Activation::None)
            scratch.backward_output = {m_padded_output_width, batch_size, stream, output_layout};
        else
            scratch.backward_output = {};
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::set_params(T* params, T* gradients) {
        m_params    = params;
        m_gradients = gradients;

        std::size_t current_pos = 0u;
        for (std::size_t i = 0u; i < m_weight_matrices.size(); ++i) {
            m_weight_matrices[i].set_data_unsafe(params + current_pos);
            m_gradient_matrices[i].set_data_unsafe(gradients + current_pos);
            current_pos += m_weight_matrices[i].n_elements();
        }
    }

    template <typename T, std::uint32_t WIDTH>
    void FullyFusedMLP<T, WIDTH>::initialize_params(legacy::math::pcg32& rnd, float* params_full_precision, const float scale) {
        std::vector<legacy::GPUMatrix<float, legacy::RM>> weight_matrices_full_precision;
        weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_input_width);
        params_full_precision += weight_matrices_full_precision.back().n_elements();

        for (std::uint32_t i = 0u; i < m_n_hidden_matmuls; ++i) {
            weight_matrices_full_precision.emplace_back(params_full_precision, m_network_width, m_network_width);
            params_full_precision += weight_matrices_full_precision.back().n_elements();
        }

        weight_matrices_full_precision.emplace_back(params_full_precision, m_padded_output_width, m_network_width);
        for (auto& matrix : weight_matrices_full_precision) matrix.initialize_xavier_uniform(rnd, scale);
    }

} // namespace ngp::mlp

namespace ngp::optimizer {

    template <typename T>
    __global__ void adam_step(const std::uint32_t n_elements, const std::uint32_t n_matrix_weights, const float loss_scale, const float learning_rate, const float beta1, const float beta2, const float epsilon, const float l2_reg, float* __restrict__ weights_full_precision, T* __restrict__ weights, const T* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        float gradient = static_cast<float>(gradients[i]) / loss_scale;
        if (i >= n_matrix_weights && gradient == 0.0f) return;

        const float weight_fp = weights_full_precision[i];
        if (i < n_matrix_weights) gradient += l2_reg * weight_fp;

        const float gradient_sq  = gradient * gradient;
        const float first_moment = first_moments[i] = beta1 * first_moments[i] + (1.0f - beta1) * gradient;
        const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1.0f - beta2) * gradient_sq;
        const std::uint32_t current_step              = ++param_steps[i];
        const float corrected_learning_rate           = learning_rate * sqrtf(1.0f - powf(beta2, static_cast<float>(current_step))) / (1.0f - powf(beta1, static_cast<float>(current_step)));
        const float new_weight                        = weight_fp - corrected_learning_rate * first_moment / (sqrtf(second_moment) + epsilon);

        weights_full_precision[i] = new_weight;
        weights[i]                = static_cast<T>(new_weight);
    }

    template <typename T>
    class AdamOptimizer final {
    public:
        AdamOptimizer() = default;

        template <typename Config>
        explicit AdamOptimizer(const Config& params) {
            update_hyperparams(params);
        }

        template <typename Config>
        void update_hyperparams(const Config& params) {
            m_beta1              = params.beta1;
            m_beta2              = params.beta2;
            m_epsilon            = params.epsilon;
            m_base_learning_rate = params.learning_rate;
            m_l2_reg             = params.l2_reg;
        }

        void allocate(const std::uint32_t n_weights, const std::uint32_t n_matrix_weights) {
            m_n_weights = n_weights;
            if (m_n_weights > m_first_moments.size()) {
                m_first_moments.resize(m_n_weights);
                m_first_moments.memset(0);

                m_second_moments.resize(m_n_weights);
                m_second_moments.memset(0);

                m_param_steps.resize(m_n_weights);
                m_param_steps.memset(0);
            }

            m_n_matrix_weights = n_matrix_weights;
        }

        void step(cudaStream_t stream, const float loss_scale, float* weights_full_precision, T* weights, const T* gradients) {
            if (m_n_weights == 0) return;

            const std::uint32_t blocks = (m_n_weights + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            adam_step<T><<<blocks, network::detail::n_threads_linear, 0, stream>>>(m_n_weights, m_n_matrix_weights, loss_scale, m_base_learning_rate, m_beta1, m_beta2, m_epsilon, m_l2_reg, weights_full_precision, weights, gradients, m_first_moments.data(), m_second_moments.data(), m_param_steps.data());
        }

    private:
        std::uint32_t m_n_weights        = 0;
        std::uint32_t m_n_matrix_weights = 0;

        legacy::GpuBuffer<float> m_first_moments       = {};
        legacy::GpuBuffer<float> m_second_moments      = {};
        legacy::GpuBuffer<std::uint32_t> m_param_steps = {};

        float m_base_learning_rate = 1e-3f;
        float m_beta1              = 0.9f;
        float m_beta2              = 0.999f;
        float m_epsilon            = 1e-8f;
        float m_l2_reg             = 1e-8f;
    };

} // namespace ngp::optimizer

namespace ngp::network {

    template <typename T>
    __global__ void extract_density(const std::uint32_t n_elements, const std::uint32_t density_stride, const std::uint32_t rgbd_stride, const T* __restrict__ density, T* __restrict__ rgbd) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;
        rgbd[i * rgbd_stride] = density[i * density_stride];
    }

    template <typename T>
    __global__ void extract_rgb(const std::uint32_t n_elements, const std::uint32_t rgb_stride, const std::uint32_t output_stride, const T* __restrict__ rgbd, T* __restrict__ rgb) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t elem_idx         = i / 3u;
        const std::uint32_t dim_idx          = i - elem_idx * 3u;
        rgb[elem_idx * rgb_stride + dim_idx] = rgbd[elem_idx * output_stride + dim_idx];
    }

    template <typename T>
    __global__ void add_density_gradient(const std::uint32_t n_elements, const std::uint32_t rgbd_stride, const T* __restrict__ rgbd, const std::uint32_t density_stride, T* __restrict__ density) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;
        density[i * density_stride] += rgbd[i * rgbd_stride + 3u];
    }

} // namespace ngp::network

#include "stb/stb_image_write.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>

namespace ngp::network::detail {

    struct AuxStreamSlot final {
        AuxStreamSlot() {
            ngp::legacy::cuda_check(cudaStreamCreate(&stream));
            ngp::legacy::cuda_check(cudaEventCreate(&event));
        }

        ~AuxStreamSlot() {
            if (event) cudaEventDestroy(event);
            if (stream) cudaStreamDestroy(stream);
        }

        AuxStreamSlot& operator=(const AuxStreamSlot&) = delete;
        AuxStreamSlot(const AuxStreamSlot&)            = delete;

        cudaStream_t stream = {};
        cudaEvent_t event   = {};
    };

    std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>>>& aux_stream_pools() {
        static auto* pools = new std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>>>{};
        return *pools;
    }

    void free_aux_stream_pool(const cudaStream_t parent_stream) {
        ngp::legacy::check_or_throw(parent_stream != nullptr);
        aux_stream_pools().erase(parent_stream);
    }

    SyncedStreamReservation::SyncedStreamReservation(const cudaStream_t stream, const std::size_t n_streams) : main_stream{stream} {
        if (n_streams == 0u) throw std::runtime_error{"SyncedStreamReservation: must request at least one stream"};
        if (n_streams == 1u) return;
        if (n_streams != 2u) throw std::runtime_error{"SyncedStreamReservation: this repository only supports a single auxiliary stream"};

        ngp::legacy::check_or_throw(main_stream != nullptr);
        auto& pool = aux_stream_pools()[main_stream];
        if (pool.empty()) pool.push(std::unique_ptr<AuxStreamSlot, void (*)(AuxStreamSlot*)>{new AuxStreamSlot{}, +[](AuxStreamSlot* aux_stream) { delete aux_stream; }});
        aux_stream_slot = std::move(pool.top());
        pool.pop();
        aux_stream      = aux_stream_slot->stream;
        ngp::legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, main_stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(aux_stream_slot->stream, aux_stream_slot->event, 0));
    }

    SyncedStreamReservation::~SyncedStreamReservation() {
        if (!aux_stream_slot) return;

        ngp::legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, aux_stream_slot->stream));
        ngp::legacy::cuda_check(cudaStreamWaitEvent(main_stream, aux_stream_slot->event, 0));
        auto& pools = aux_stream_pools();
        if (!pools.contains(main_stream)) std::terminate();
        pools[main_stream].push(std::move(aux_stream_slot));
        aux_stream = nullptr;
    }

    SyncedStreamReservation& SyncedStreamReservation::operator=(SyncedStreamReservation&& other) {
        std::swap(aux_stream_slot, other.aux_stream_slot);
        std::swap(aux_stream, other.aux_stream);
        std::swap(main_stream, other.main_stream);
        return *this;
    }

    SyncedStreamReservation::SyncedStreamReservation(SyncedStreamReservation&& other) {
        *this = std::move(other);
    }

} // namespace ngp::network::detail

namespace ngp::mlp {

    template class FullyFusedMLP<__half, ngp::density_network_width>;
#if NGP_RGB_NETWORK_WIDTH != NGP_DENSITY_NETWORK_WIDTH
    template class FullyFusedMLP<__half, ngp::rgb_network_width>;
#endif

} // namespace ngp::mlp

namespace ngp {

    inline constexpr std::uint32_t NERF_STEPS                  = 1024u;
    inline constexpr float MIN_CONE_STEPSIZE                   = 1.73205080757f / static_cast<float>(NERF_STEPS);
    inline constexpr std::uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY = 16u;
    inline constexpr float NERF_MIN_OPTICAL_THICKNESS          = 0.01f;

    struct TrainingStepWorkspace final {
        legacy::GpuAllocation alloc                             = {};
        std::uint32_t* ray_indices                              = nullptr;
        void* rays_unnormalized                                 = nullptr;
        std::uint32_t* numsteps                                 = nullptr;
        float* coords                                           = nullptr;
        __half* mlp_out                                         = nullptr;
        __half* dloss_dmlp_out                                  = nullptr;
        float* coords_compacted                                 = nullptr;
        std::uint32_t* ray_counter                              = nullptr;
        std::uint32_t max_samples                               = 0u;
        std::uint32_t max_inference                             = 0u;
        std::uint32_t floats_per_coord                          = 0u;
        std::uint32_t padded_output_width                       = 0u;
        std::uint32_t n_rays_total                              = 0u;
        legacy::GPUMatrixDynamic<float> coords_matrix           = {};
        legacy::GPUMatrixDynamic<__half> rgbsigma_matrix        = {};
        legacy::GPUMatrixDynamic<float> compacted_coords_matrix = {};
        legacy::GPUMatrixDynamic<__half> gradient_matrix        = {};
        legacy::GPUMatrixDynamic<__half> compacted_output       = {};
    };

    inline __device__ instant_ngp_detail::Ray uv_to_ray(const std::uint32_t spp, const legacy::math::vec2& uv, const legacy::math::ivec2& resolution, const float focal_length, const legacy::math::mat4x3& camera_matrix, const float near_distance = 0.0f) {
        static_cast<void>(spp);
        legacy::math::vec3 dir    = {(uv.x - 0.5f) * static_cast<float>(resolution.x) / focal_length, (uv.y - 0.5f) * static_cast<float>(resolution.y) / focal_length, 1.0f};
        dir                       = legacy::math::mat3(camera_matrix) * dir;
        legacy::math::vec3 origin = camera_matrix[3];
        origin += dir * near_distance;
        return {origin, dir};
    }

    inline __device__ float network_to_density(const float val) {
        return expf(val);
    }

    struct DensityGridReduceOp final {
        std::uint32_t base_grid_elements = 0u;

        __device__ float operator()(const float val) const {
            return fmaxf(val, 0.0f) / static_cast<float>(base_grid_elements);
        }
    };

    struct SumIdentityOp final {
        __device__ float operator()(const float val) const {
            return val;
        }
    };

    inline __device__ legacy::math::vec3 warp_position(const legacy::math::vec3& pos, const legacy::BoundingBox& aabb) {
        return aabb.relative_pos(pos);
    }

    inline __device__ legacy::math::vec3 warp_direction(const legacy::math::vec3& dir) {
        return (dir + 1.0f) * 0.5f;
    }

    inline __device__ float warp_dt(const float dt) {
        static_cast<void>(dt);
        return 0.0f;
    }

    inline __device__ std::uint32_t density_grid_idx_at(const legacy::math::vec3& pos) {
        const legacy::math::ivec3 i = pos * static_cast<float>(legacy::NERF_GRIDSIZE);
        if (i.x < 0 || i.x >= static_cast<int>(legacy::NERF_GRIDSIZE) || i.y < 0 || i.y >= static_cast<int>(legacy::NERF_GRIDSIZE) || i.z < 0 || i.z >= static_cast<int>(legacy::NERF_GRIDSIZE)) return 0xFFFFFFFFu;
        return network::detail::morton3D(i.x, i.y, i.z);
    }

    inline __device__ bool density_grid_occupied_at(const legacy::math::vec3& pos, const std::uint8_t* density_grid_bitfield) {
        const std::uint32_t idx = density_grid_idx_at(pos);
        if (idx == 0xFFFFFFFFu) return false;
        return density_grid_bitfield[idx / 8u] & (1u << (idx % 8u));
    }

    inline __device__ float advance_n_steps(const float t, const float n) {
        return t + n * MIN_CONE_STEPSIZE;
    }

    inline __device__ float calc_dt() {
        return MIN_CONE_STEPSIZE;
    }

    inline __device__ float advance_to_next_voxel(const float t, const legacy::math::vec3& pos, const legacy::math::vec3& dir, const legacy::math::vec3& idir) {
        const legacy::math::vec3 p = static_cast<float>(legacy::NERF_GRIDSIZE) * (pos - 0.5f);
        const float tx             = (floorf(p.x + 0.5f + 0.5f * legacy::math::sign(dir.x)) - p.x) * idir.x;
        const float ty             = (floorf(p.y + 0.5f + 0.5f * legacy::math::sign(dir.y)) - p.y) * idir.y;
        const float tz             = (floorf(p.z + 0.5f + 0.5f * legacy::math::sign(dir.z)) - p.z) * idir.z;
        const float t_target       = t + fmaxf(fminf(fminf(tx, ty), tz) / static_cast<float>(legacy::NERF_GRIDSIZE), 0.0f);
        return t + ceilf(fmaxf((t_target - t) / MIN_CONE_STEPSIZE, 0.5f)) * MIN_CONE_STEPSIZE;
    }

    inline __device__ legacy::math::vec2 nerf_random_image_pos_training(legacy::math::pcg32& rng, const legacy::math::ivec2& resolution, const bool snap_to_pixel_centers) {
        legacy::math::vec2 uv = {rng.next_float(), rng.next_float()};
        if (snap_to_pixel_centers) uv = (legacy::math::vec2(ngp::legacy::math::clamp(legacy::math::ivec2(uv * legacy::math::vec2(resolution)), 0, resolution - 1)) + 0.5f) / legacy::math::vec2(resolution);
        return uv;
    }

    inline __device__ std::uint32_t image_idx(const std::uint32_t base_idx, const std::uint32_t n_rays, const std::uint32_t n_rays_total, const std::uint32_t n_training_images) {
        static_cast<void>(n_rays_total);
        return ((base_idx * n_training_images) / n_rays) % n_training_images;
    }

    inline __host__ __device__ legacy::math::vec3 srgb_to_linear(const legacy::math::vec3& x) {
        return {
            x.x <= 0.04045f ? x.x / 12.92f : powf((x.x + 0.055f) / 1.055f, 2.4f),
            x.y <= 0.04045f ? x.y / 12.92f : powf((x.y + 0.055f) / 1.055f, 2.4f),
            x.z <= 0.04045f ? x.z / 12.92f : powf((x.z + 0.055f) / 1.055f, 2.4f),
        };
    }

    inline __host__ __device__ legacy::math::vec3 linear_to_srgb(const legacy::math::vec3& x) {
        return {
            x.x < 0.0031308f ? 12.92f * x.x : 1.055f * powf(x.x, 0.41666f) - 0.055f,
            x.y < 0.0031308f ? 12.92f * x.y : 1.055f * powf(x.y, 0.41666f) - 0.055f,
            x.z < 0.0031308f ? 12.92f * x.z : 1.055f * powf(x.z, 0.41666f) - 0.055f,
        };
    }

    inline __host__ __device__ legacy::math::vec4 read_rgba(const legacy::math::ivec2& px, const legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        const std::uint32_t rgba32 = reinterpret_cast<const std::uint32_t*>(pixels)[px.x + px.y * resolution.x + img * static_cast<std::uint64_t>(resolution.x) * resolution.y];
        legacy::math::vec4 result  = {
            static_cast<float>((rgba32 & 0x000000FFu) >> 0u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0x0000FF00u) >> 8u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0x00FF0000u) >> 16u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0xFF000000u) >> 24u) * (1.0f / 255.0f),
        };
        result.rgb() = srgb_to_linear(result.rgb()) * result.a;
        return result;
    }

    inline __device__ float network_to_rgb_derivative(const float val) {
        const float rgb = network::detail::logistic(val);
        return rgb * (1.0f - rgb);
    }

    template <typename T>
    __device__ legacy::math::vec3 network_to_rgb_vec(const T& val) {
        return {
            network::detail::logistic(static_cast<float>(val[0])),
            network::detail::logistic(static_cast<float>(val[1])),
            network::detail::logistic(static_cast<float>(val[2])),
        };
    }

    inline __device__ float unwarp_dt(const float dt) {
        static_cast<void>(dt);
        return MIN_CONE_STEPSIZE;
    }

    inline __host__ __device__ legacy::math::vec3 clamp_rgb01(const legacy::math::vec3& value) {
        return {
            cuda::std::clamp(value.x, 0.0f, 1.0f),
            cuda::std::clamp(value.y, 0.0f, 1.0f),
            cuda::std::clamp(value.z, 0.0f, 1.0f),
        };
    }

    template <typename T>
    __global__ void fill_rollover(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i                = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;
        inout[i] = inout[i % (n_input_elements * stride)];
    }

    template <typename T>
    __global__ void fill_rollover_and_rescale(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i                = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;

        T result = inout[i % (n_input_elements * stride)];
        result   = static_cast<T>(static_cast<float>(result) * static_cast<float>(n_input_elements) / static_cast<float>(n_elements));
        inout[i] = result;
    }

    template <typename T>
    __device__ T warp_reduce(T val) {
        TCNN_PRAGMA_UNROLL
        for (int offset = warpSize / 2; offset > 0; offset /= 2) val += __shfl_xor_sync(0xffffffff, val, offset);
        return val;
    }

    template <typename T, typename T_OUT, typename F>
    __global__ void block_reduce(const std::uint32_t n_elements, const F fun, const T* __restrict__ input, T_OUT* __restrict__ output, const std::uint32_t n_blocks) {
        const std::uint32_t sum_idx        = blockIdx.x / n_blocks;
        const std::uint32_t sub_blocks_idx = blockIdx.x % n_blocks;
        const std::uint32_t i              = threadIdx.x + sub_blocks_idx * blockDim.x;
        const std::uint32_t block_offset   = sum_idx * n_elements;

        static __shared__ T_OUT sdata[32];

        const std::uint32_t lane = threadIdx.x % warpSize;
        const std::uint32_t wid  = threadIdx.x / warpSize;

        T_OUT val = {};
        if constexpr (std::is_same_v<std::decay_t<T>, __half> || std::is_same_v<std::decay_t<T>, half>) {
            if (i < n_elements) {
                half vals[8];
                *reinterpret_cast<int4*>(&vals[0]) = *(reinterpret_cast<const int4*>(input) + i + block_offset);
                val                                = fun(static_cast<T>(vals[0])) + fun(static_cast<T>(vals[1])) + fun(static_cast<T>(vals[2])) + fun(static_cast<T>(vals[3])) + fun(static_cast<T>(vals[4])) + fun(static_cast<T>(vals[5])) + fun(static_cast<T>(vals[6])) + fun(static_cast<T>(vals[7]));
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, float>) {
            if (i < n_elements) {
                const float4 vals = *(reinterpret_cast<const float4*>(input) + i + block_offset);
                val               = fun(static_cast<T>(vals.x)) + fun(static_cast<T>(vals.y)) + fun(static_cast<T>(vals.z)) + fun(static_cast<T>(vals.w));
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
            if (i < n_elements) {
                const double2 vals = *(reinterpret_cast<const double2*>(input) + i + block_offset);
                val                = fun(static_cast<T>(vals.x)) + fun(static_cast<T>(vals.y));
            }
        } else {
            assert(false);
            return;
        }

        val = warp_reduce(val);

        if (lane == 0) sdata[wid] = val;
        __syncthreads();

        if (wid == 0) {
            val = threadIdx.x < blockDim.x / warpSize ? sdata[lane] : static_cast<T_OUT>(0);
            val = warp_reduce(val);
            if (lane == 0) atomicAdd(&output[sum_idx], val);
        }
    }

    template <typename T, typename T_OUT, typename F>
    void reduce_sum(T* device_pointer, F fun, T_OUT* workspace, std::uint32_t n_elements, cudaStream_t stream, const std::uint32_t n_sums = 1u) {
        constexpr std::uint32_t threads      = 1024u;
        const std::uint32_t n_elems_per_load = 16u / sizeof(T);

        if (n_elements % n_elems_per_load != 0u) throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
        if ((reinterpret_cast<std::size_t>(device_pointer) % 16u) != 0u) throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};

        n_elements /= n_elems_per_load;
        const std::uint32_t blocks = (n_elements + threads - 1u) / threads;
        block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
    }

    inline std::uint32_t reduce_sum_workspace_size(const std::uint32_t n_elements) {
        return (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
    }

    __global__ void mark_untrained_density_grid(const std::uint32_t n_elements, float* __restrict__ grid_out, const std::uint32_t n_training_images, const instant_ngp_detail::GpuFrame* __restrict__ frames) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t x = network::detail::morton3D_invert(i >> 0u);
        const std::uint32_t y = network::detail::morton3D_invert(i >> 1u);
        const std::uint32_t z = network::detail::morton3D_invert(i >> 2u);

        constexpr float voxel_size   = 1.0f / static_cast<float>(legacy::NERF_GRIDSIZE);
        const legacy::math::vec3 pos = legacy::math::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} / static_cast<float>(legacy::NERF_GRIDSIZE);

        legacy::math::vec3 corners[8] = {
            pos + legacy::math::vec3{0.0f, 0.0f, 0.0f},
            pos + legacy::math::vec3{voxel_size, 0.0f, 0.0f},
            pos + legacy::math::vec3{0.0f, voxel_size, 0.0f},
            pos + legacy::math::vec3{voxel_size, voxel_size, 0.0f},
            pos + legacy::math::vec3{0.0f, 0.0f, voxel_size},
            pos + legacy::math::vec3{voxel_size, 0.0f, voxel_size},
            pos + legacy::math::vec3{0.0f, voxel_size, voxel_size},
            pos + legacy::math::vec3{voxel_size, voxel_size, voxel_size},
        };

        constexpr std::uint32_t min_count = 1u;
        std::uint32_t count               = 0u;

        for (std::uint32_t j = 0u; j < n_training_images && count < min_count; ++j) {
            const auto& frame = frames[j];
            const auto& xform = frame.camera;

            for (const legacy::math::vec3& corner : corners) {
                const legacy::math::vec3 dir = ngp::legacy::math::normalize(corner - xform[3]);
                if (ngp::legacy::math::dot(dir, xform[2]) < 1e-4f) continue;

                const legacy::math::vec3 offset = corner - xform[3];
                legacy::math::vec3 camera_dir   = {
                    ngp::legacy::math::dot(xform[0], offset),
                    ngp::legacy::math::dot(xform[1], offset),
                    ngp::legacy::math::dot(xform[2], offset),
                };
                camera_dir /= camera_dir.z;
                const legacy::math::vec2 uv = camera_dir.xy() * frame.focal_length / legacy::math::vec2(frame.resolution) + legacy::math::vec2(0.5f);
                const instant_ngp_detail::Ray ray = uv_to_ray(0u, uv, frame.resolution, frame.focal_length, xform);
                if (ngp::legacy::math::distance(ngp::legacy::math::normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
                    ++count;
                    break;
                }
            }
        }

        grid_out[i] = count >= min_count ? 0.0f : -1.0f;
    }

    __global__ void generate_grid_samples_nerf_nonuniform(const std::uint32_t n_elements, legacy::math::pcg32 rng, const std::uint32_t step, legacy::BoundingBox aabb, const float* __restrict__ grid_in, legacy::NerfPosition* __restrict__ out, std::uint32_t* __restrict__ indices, const float thresh) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        rng.advance(i * 4u);
        std::uint32_t idx = 0u;
        for (std::uint32_t j = 0u; j < 10u; ++j) {
            idx = ((i + step * n_elements) * 56924617u + j * 19349663u + 96925573u) % legacy::NERF_GRID_N_CELLS;
            if (grid_in[idx] > thresh) break;
        }

        const std::uint32_t x        = network::detail::morton3D_invert(idx >> 0u);
        const std::uint32_t y        = network::detail::morton3D_invert(idx >> 1u);
        const std::uint32_t z        = network::detail::morton3D_invert(idx >> 2u);
        const legacy::math::vec3 pos = (legacy::math::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} + network::detail::random_val_3d(rng)) / static_cast<float>(legacy::NERF_GRIDSIZE);

        out[i]     = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE)};
        indices[i] = idx;
    }

    __global__ void splat_grid_samples_nerf_max_nearest_neighbor(const std::uint32_t n_elements, const std::uint32_t* __restrict__ indices, const __half* __restrict__ network_output, float* __restrict__ grid_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t idx = indices[i];
        const float mlp         = network_to_density(network_output[i]);
        const float thickness   = mlp * MIN_CONE_STEPSIZE;
        atomicMax(reinterpret_cast<std::uint32_t*>(&grid_out[idx]), __float_as_uint(thickness));
    }

    __global__ void ema_grid_samples_nerf(const std::uint32_t n_elements, const float decay, const std::uint32_t count, float* __restrict__ grid_out, const float* __restrict__ grid_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        static_cast<void>(count);
        const float importance = grid_in[i];
        const float prev_val   = grid_out[i];
        const float val        = prev_val < 0.0f ? prev_val : fmaxf(prev_val * decay, importance);
        grid_out[i]            = val;
    }

    __global__ void grid_to_bitfield(const std::uint32_t n_elements, const float* __restrict__ grid, std::uint8_t* __restrict__ grid_bitfield, const float* __restrict__ mean_density_ptr) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        std::uint8_t bits  = 0u;
        const float thresh = fminf(NERF_MIN_OPTICAL_THICKNESS, *mean_density_ptr);

        TCNN_PRAGMA_UNROLL
        for (std::uint8_t j = 0u; j < 8u; ++j) bits |= grid[i * 8u + j] > thresh ? (static_cast<std::uint8_t>(1u) << j) : 0u;

        grid_bitfield[i] = bits;
    }

    __global__ void generate_training_samples_nerf(const std::uint32_t n_rays, const legacy::BoundingBox aabb, const std::uint32_t max_samples, const std::uint32_t n_rays_total, legacy::math::pcg32 rng, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ numsteps_counter, std::uint32_t* __restrict__ ray_indices_out, instant_ngp_detail::Ray* __restrict__ rays_out_unnormalized, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out,
        const std::uint32_t n_training_images, const instant_ngp_detail::GpuFrame* __restrict__ frames, const std::uint8_t* __restrict__ density_grid, const bool snap_to_pixel_centers) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;

        const std::uint32_t img              = image_idx(i, n_rays, n_rays_total, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY);
        const legacy::math::vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);

        const float focal_length         = frame.focal_length;
        const legacy::math::mat4x3 xform = frame.camera;

        instant_ngp_detail::Ray ray_unnormalized = uv_to_ray(0u, uv, resolution, focal_length, xform);
        if (!ray_unnormalized.is_valid()) ray_unnormalized = {xform[3], xform[2]};

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);

        const float startt            = advance_n_steps(tminmax.x, rng.next_float());
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t j = 0u;
        float t         = startt;
        legacy::math::vec3 pos;

        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
        if (j == 0u) return;

        const std::uint32_t numsteps = j;
        const std::uint32_t base     = atomicAdd(numsteps_counter, numsteps);
        if (base + numsteps > max_samples) return;

        coords_out += base;

        const std::uint32_t ray_idx     = atomicAdd(ray_counter, 1u);
        ray_indices_out[ray_idx]        = i;
        rays_out_unnormalized[ray_idx]  = ray_unnormalized;
        numsteps_out[ray_idx * 2u + 0u] = numsteps;
        numsteps_out[ray_idx * 2u + 1u] = base;

        const legacy::math::vec3 warped_dir = warp_direction(ray_d_normalized);
        t                                   = startt;
        j                                   = 0u;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, warp_dt(dt));
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void compute_loss_kernel_train_nerf(const std::uint32_t n_rays, const legacy::BoundingBox aabb, const std::uint32_t n_rays_total, legacy::math::pcg32 rng, const std::uint32_t max_samples_compacted, const std::uint32_t* __restrict__ rays_counter, float loss_scale, const int padded_output_width, const std::uint32_t n_training_images, const instant_ngp_detail::GpuFrame* __restrict__ frames, const __half* network_output, std::uint32_t* __restrict__ numsteps_counter,
        const std::uint32_t* __restrict__ ray_indices_in, const instant_ngp_detail::Ray* __restrict__ rays_in_unnormalized, std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const legacy::NerfCoordinate> coords_in, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out, __half* dloss_doutput, float* __restrict__ loss_output, const bool snap_to_pixel_centers, const float* __restrict__ mean_density_ptr, const float near_distance) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= *rays_counter) return;

        std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
        std::uint32_t base     = numsteps_in[i * 2u + 1u];

        coords_in += base;
        network_output += base * padded_output_width;

        float T                 = 1.0f;
        constexpr float epsilon = 1e-4f;

        auto rgb_ray                     = legacy::math::vec3(0.0f);
        std::uint32_t compacted_numsteps = 0u;
        const legacy::math::vec3 ray_o   = rays_in_unnormalized[i].o;
        for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
            if (T < epsilon) break;

            const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const float dt                                            = unwarp_dt(coords_in.ptr->dt);
            const float density                                       = network_to_density(local_network_output[3]);

            const float alpha  = 1.0f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            T *= (1.0f - alpha);

            network_output += padded_output_width;
            coords_in += 1u;
        }

        const std::uint32_t ray_idx = ray_indices_in[i];
        rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY);

        const std::uint32_t img              = image_idx(ray_idx, n_rays, n_rays_total, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        const legacy::math::vec2 uv               = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);
        const legacy::math::vec3 background_color = network::detail::random_val_3d(rng);
        const legacy::math::ivec2 texel           = ngp::legacy::math::clamp(legacy::math::ivec2(uv * legacy::math::vec2(resolution)), 0, resolution - 1);
        const legacy::math::vec4 texsamp          = read_rgba(texel, resolution, frame.pixels);
        const legacy::math::vec3 rgbtarget        = linear_to_srgb(texsamp.rgb() + (1.0f - texsamp.a) * srgb_to_linear(background_color));

        if (compacted_numsteps == numsteps) rgb_ray += T * background_color;

        network_output -= padded_output_width * compacted_numsteps;
        coords_in -= compacted_numsteps;

        std::uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps);
        compacted_numsteps           = std::min(max_samples_compacted - std::min(max_samples_compacted, compacted_base), compacted_numsteps);
        numsteps_in[i * 2u + 0u]     = compacted_numsteps;
        numsteps_in[i * 2u + 1u]     = compacted_base;
        if (compacted_numsteps == 0u) return;

        coords_out += compacted_base;
        dloss_doutput += compacted_base * padded_output_width;

        const legacy::math::vec3 difference = rgb_ray - rgbtarget;
        const legacy::math::vec3 loss       = difference * difference;
        const legacy::math::vec3 gradient   = 2.0f * difference;
        const float mean_loss               = ngp::legacy::math::mean(loss);
        if (loss_output) loss_output[i] = mean_loss / static_cast<float>(n_rays);

        loss_scale /= static_cast<float>(n_rays);

        const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS ? 1e-4f : 0.0f;

        legacy::math::vec3 rgb_ray2 = {0.0f, 0.0f, 0.0f};
        T                           = 1.0f;
        for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
            legacy::NerfCoordinate* coord_out      = coords_out(j);
            const legacy::NerfCoordinate* coord_in = coords_in(j);
            *coord_out                             = *coord_in;

            const legacy::math::vec3 pos                              = aabb.min + coord_in->pos.p * aabb.diag();
            const float depth                                         = ngp::legacy::math::distance(pos, ray_o);
            const float dt                                            = unwarp_dt(coord_in->dt);
            const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const float density                                       = network_to_density(local_network_output[3]);
            const float alpha                                         = 1.0f - __expf(-density * dt);
            const float weight                                        = alpha * T;
            rgb_ray2 += weight * rgb;
            T *= (1.0f - alpha);

            const legacy::math::vec3 suffix        = rgb_ray - rgb_ray2;
            const legacy::math::vec3 dloss_by_drgb = weight * gradient;

            legacy::math::tvec<__half, 4u> local_dL_doutput{};
            local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0]));
            local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1]));
            local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2]));

            const float density_derivative = expf(cuda::std::clamp(static_cast<float>(local_network_output[3]), -15.0f, 15.0f));
            const float dloss_by_dmlp      = density_derivative * (dt * ngp::legacy::math::dot(gradient, T * rgb - suffix));
            local_dL_doutput[3]            = loss_scale * dloss_by_dmlp + (static_cast<float>(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) + (static_cast<float>(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);

            *reinterpret_cast<legacy::math::tvec<__half, 4u>*>(dloss_doutput) = local_dL_doutput;

            dloss_doutput += padded_output_width;
            network_output += padded_output_width;
        }
    }

    __global__ void generate_validation_samples_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, legacy::BoundingBox aabb, const std::uint32_t max_samples, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ overflow_counter, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out, instant_ngp_detail::GpuFrame frame, const std::uint8_t* __restrict__ density_grid) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        numsteps_out[i * 2u + 0u] = 0u;
        numsteps_out[i * 2u + 1u] = 0u;

        const std::uint32_t global_pixel     = pixel_offset + i;
        const legacy::math::ivec2 resolution = frame.resolution;
        const legacy::math::ivec2 px         = {static_cast<int>(global_pixel % static_cast<std::uint32_t>(resolution.x)), static_cast<int>(global_pixel / static_cast<std::uint32_t>(resolution.x))};
        const legacy::math::vec2 uv          = (legacy::math::vec2{static_cast<float>(px.x) + 0.5f, static_cast<float>(px.y) + 0.5f}) / legacy::math::vec2(resolution);
        const legacy::math::mat4x3 xform     = frame.camera;

        instant_ngp_detail::Ray ray_unnormalized = uv_to_ray(0u, uv, resolution, frame.focal_length, xform);
        if (!ray_unnormalized.is_valid()) return;

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);
        if (tminmax.y <= tminmax.x) return;

        float t                       = advance_n_steps(tminmax.x, 0.5f);
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t numsteps = 0u;
        legacy::math::vec3 pos;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && numsteps < NERF_STEPS) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                ++numsteps;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }

        if (numsteps == 0u) return;

        const std::uint32_t base = atomicAdd(sample_counter, numsteps);
        if (base + numsteps > max_samples) {
            atomicAdd(overflow_counter, 1u);
            return;
        }

        numsteps_out[i * 2u + 0u] = numsteps;
        numsteps_out[i * 2u + 1u] = base;

        coords_out += base;
        const legacy::math::vec3 warped_dir = warp_direction(ray_d_normalized);
        t                                   = advance_n_steps(tminmax.x, 0.5f);

        std::uint32_t j = 0u;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, warp_dt(dt));
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void composite_validation_kernel_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, const std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const legacy::NerfCoordinate> coords_in, const __half* __restrict__ network_output, const std::uint32_t padded_output_width, const legacy::math::vec3 background_color, legacy::math::vec3* __restrict__ image_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        const std::uint32_t global_pixel = pixel_offset + i;
        const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
        const std::uint32_t base         = numsteps_in[i * 2u + 1u];
        float T                          = 1.0f;
        auto rgb_ray                     = legacy::math::vec3(0.0f);

        if (numsteps > 0u) {
            coords_in += base;
            network_output += base * padded_output_width;

            for (std::uint32_t j = 0u; j < numsteps; ++j) {
                const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
                const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
                const float dt                                            = unwarp_dt(coords_in.ptr->dt);
                const float density                                       = network_to_density(local_network_output[3]);
                const float alpha                                         = 1.0f - __expf(-density * dt);
                const float weight                                        = alpha * T;

                rgb_ray += weight * rgb;
                T *= (1.0f - alpha);

                coords_in += 1;
                network_output += padded_output_width;
                if (T < 1e-4f) break;
            }
        }

        image_out[global_pixel] = clamp_rgb01(rgb_ray + T * background_color);
    }

    struct InstantNGP::ModelState final {
        struct Layout final {
            std::uint32_t pos_input_width      = 0u;
            std::uint32_t pos_output_width     = 0u;
            legacy::MatrixLayout pos_layout    = legacy::CM;
            std::size_t pos_param_count        = 0u;
            std::uint32_t dir_input_width      = 0u;
            std::uint32_t dir_output_width     = 0u;
            legacy::MatrixLayout dir_layout    = legacy::CM;
            std::uint32_t density_output_width = 0u;
            std::uint32_t rgb_output_width     = 0u;
            std::uint32_t padded_output_width  = 0u;
        };

        struct ParamLayout final {
            std::size_t density_network = 0u;
            std::size_t rgb_network     = 0u;
            std::size_t pos_encoding    = 0u;
            std::size_t total           = 0u;
        };

        explicit ModelState(const InstantNGP::NetworkConfig& config, const InstantNGP::TrainPlan& plan);

        std::variant<ngp::encoding::GridEncodingTemplated<__half, 3u, 1u>, ngp::encoding::GridEncodingTemplated<__half, 3u, 2u>, ngp::encoding::GridEncodingTemplated<__half, 3u, 4u>, ngp::encoding::GridEncodingTemplated<__half, 3u, 8u>> pos_encoding;
        ngp::encoding::SphericalHarmonicsEncoding<__half> dir_encoding;
        ngp::mlp::FullyFusedMLP<__half, density_network_width> density_network;
        ngp::mlp::FullyFusedMLP<__half, rgb_network_width> rgb_network;
        Layout layout = {};
        std::uint32_t rgb_network_input_width = 0u;
        std::uint32_t dir_offset              = 0u;
        ParamLayout params                    = {};
    };

    struct InstantNGP::ModelScratch final {
        legacy::GPUMatrixDynamic<__half> density_network_input  = {};
        legacy::GPUMatrixDynamic<__half> density_network_output = {};
        legacy::GPUMatrixDynamic<__half> rgb_network_input      = {};
        legacy::GPUMatrixDynamic<__half> rgb_network_output     = {};
        legacy::GPUMatrixDynamic<__half> dL_drgb                = {};
        legacy::GPUMatrixDynamic<__half> dL_drgb_input          = {};
        legacy::GPUMatrixDynamic<__half> dL_ddensity_input      = {};
        ngp::mlp::FullyFusedMLP<__half, density_network_width>::Scratch density_network = {};
        ngp::mlp::FullyFusedMLP<__half, rgb_network_width>::Scratch rgb_network         = {};
    };

    struct InstantNGP::OptimizerState final {
        ngp::optimizer::AdamOptimizer<__half> impl = {};
    };

    InstantNGP::ModelState::ModelState(const InstantNGP::NetworkConfig& config, const InstantNGP::TrainPlan& plan)
        : pos_encoding{ngp::encoding::create_position_encoding<__half>(plan.network.n_pos_dims, config.encoding, plan.network.density_alignment)},
          dir_encoding{ngp::encoding::create_direction_encoding<__half>(plan.network.n_dir_dims, config.direction_encoding, plan.network.rgb_alignment)},
          density_network{plan.network.density_input_dims, plan.network.density_output_dims, config.density_network.n_hidden_layers, ngp::network::detail::activation_from_config(config.density_network.activation), ngp::network::detail::activation_from_config(config.density_network.output_activation)},
          rgb_network{plan.network.rgb_input_dims, plan.network.rgb_output_dims, config.rgb_network.n_hidden_layers, ngp::network::detail::activation_from_config(config.rgb_network.activation), ngp::network::detail::activation_from_config(config.rgb_network.output_activation)},
          rgb_network_input_width{plan.network.rgb_input_dims},
          dir_offset{plan.network.dir_offset} {
        layout.pos_input_width      = ngp::network::detail::visit_module(pos_encoding, [](const auto& impl) { return impl.input_width(); });
        layout.pos_output_width     = ngp::network::detail::visit_module(pos_encoding, [](const auto& impl) { return impl.padded_output_width(); });
        layout.pos_layout           = ngp::network::detail::visit_module(pos_encoding, [](const auto& impl) { return impl.preferred_output_layout(); });
        layout.pos_param_count      = ngp::network::detail::visit_module(pos_encoding, [](const auto& impl) { return impl.n_params(); });
        layout.dir_input_width      = dir_encoding.input_width();
        layout.dir_output_width     = dir_encoding.padded_output_width();
        layout.dir_layout           = dir_encoding.preferred_output_layout();
        layout.density_output_width = density_network.padded_output_width();
        layout.rgb_output_width     = rgb_network.padded_output_width();
        layout.padded_output_width  = std::max(layout.rgb_output_width, 4u);

        params.density_network = 0u;
        params.rgb_network     = params.density_network + density_network.n_params();
        params.pos_encoding    = params.rgb_network + rgb_network.n_params();
        params.total           = params.pos_encoding + ngp::network::detail::visit_module(pos_encoding, [](const auto& impl) { return impl.n_params(); });
    }

    InstantNGP::DeviceState::~DeviceState() noexcept {
        if (!stream) return;

        network::detail::free_aux_stream_pool(stream);
        cudaStreamDestroy(stream);
    }

    InstantNGP::DeviceState::DeviceState(DeviceState&& other) noexcept {
        stream = std::exchange(other.stream, nullptr);
    }

    auto InstantNGP::DeviceState::operator=(DeviceState&& other) noexcept -> DeviceState& {
        if (this == &other) return *this;
        if (stream) {
            network::detail::free_aux_stream_pool(stream);
            cudaStreamDestroy(stream);
        }

        stream = std::exchange(other.stream, nullptr);
        return *this;
    }

    InstantNGP::NetworkState::~NetworkState() noexcept {
        try {
            if (graph_capture.graph) {
                ngp::legacy::cuda_check(cudaGraphDestroy(graph_capture.graph));
                graph_capture.graph = nullptr;
            }
            if (graph_capture.graph_instance) {
                ngp::legacy::cuda_check(cudaGraphExecDestroy(graph_capture.graph_instance));
                graph_capture.graph_instance = nullptr;
            }
        } catch (const std::runtime_error& error) {
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos) std::fprintf(stderr, "Could not destroy cuda graph: %s\n", error.what());
        }
    }

    InstantNGP::NetworkState::NetworkState(NetworkState&& other) noexcept
        : definition{std::move(other.definition)}, scratch{std::move(other.scratch)}, optimizer{std::move(other.optimizer)}, parameters{std::move(other.parameters)}, graph_capture{other.graph_capture} {
        other.graph_capture = {};
    }

    auto InstantNGP::NetworkState::operator=(NetworkState&& other) noexcept -> NetworkState& {
        if (this == &other) return *this;
        try {
            if (graph_capture.graph) {
                ngp::legacy::cuda_check(cudaGraphDestroy(graph_capture.graph));
                graph_capture.graph = nullptr;
            }
            if (graph_capture.graph_instance) {
                ngp::legacy::cuda_check(cudaGraphExecDestroy(graph_capture.graph_instance));
                graph_capture.graph_instance = nullptr;
            }
        } catch (const std::runtime_error& error) {
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos) std::fprintf(stderr, "Could not destroy cuda graph: %s\n", error.what());
        }

        definition    = std::move(other.definition);
        scratch       = std::move(other.scratch);
        optimizer     = std::move(other.optimizer);
        parameters    = std::move(other.parameters);
        graph_capture = other.graph_capture;
        other.graph_capture = {};
        return *this;
    }

    void InstantNGP::initialize_network_state() {
        network.definition = std::unique_ptr<ModelState, void (*)(ModelState*)>{new ModelState{config.network, config.plan}, +[](ModelState* model) { delete model; }};
        network.scratch = std::unique_ptr<ModelScratch, void (*)(ModelScratch*)>{new ModelScratch{}, +[](ModelScratch* scratch) { delete scratch; }};
        network.optimizer = std::unique_ptr<OptimizerState, void (*)(OptimizerState*)>{new OptimizerState{}, +[](OptimizerState* optimizer) { delete optimizer; }};
        network.optimizer->impl.update_hyperparams(config.network.optimizer);
        const std::size_t param_count = total_parameter_count();
        network.optimizer->impl.allocate(static_cast<std::uint32_t>(param_count), matrix_parameter_count());
        network.parameters.buffer.resize(sizeof(float) * param_count + sizeof(__half) * param_count * 2u);
        network.parameters.buffer.memset(0);
        network.parameters.full_precision = reinterpret_cast<float*>(network.parameters.buffer.data());
        network.parameters.values = reinterpret_cast<__half*>(network.parameters.buffer.data() + sizeof(float) * param_count);
        network.parameters.gradients = reinterpret_cast<__half*>(network.parameters.buffer.data() + sizeof(float) * param_count + sizeof(__half) * param_count);
        set_model_params(network.parameters.values, network.parameters.gradients);

        legacy::math::pcg32 init_rng{config.seed};
        initialize_model_params(init_rng, network.parameters.full_precision);
        if (param_count > 0u) {
            const std::uint32_t blocks = (static_cast<std::uint32_t>(param_count) + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ngp::network::detail::cast<__half><<<blocks, ngp::network::detail::n_threads_linear, 0, nullptr>>>(static_cast<std::uint32_t>(param_count), network.parameters.full_precision, network.parameters.values);
        }

        legacy::cuda_check(cudaDeviceSynchronize());
        prepare_model_scratch(config.plan.training.batch_size, legacy::CM);
    }

    void InstantNGP::set_model_params(__half* params_ptr, __half* gradients_ptr) {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before binding parameters."};

        auto& model = *network.definition;
        model.density_network.set_params(params_ptr + model.params.density_network, gradients_ptr + model.params.density_network);
        model.rgb_network.set_params(params_ptr + model.params.rgb_network, gradients_ptr + model.params.rgb_network);
        ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.set_params(params_ptr + model.params.pos_encoding, gradients_ptr + model.params.pos_encoding); });
    }

    void InstantNGP::initialize_model_params(legacy::math::pcg32& rng, float* params_full_precision, const float scale) {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before parameter initialization."};

        auto& model = *network.definition;
        model.density_network.initialize_params(rng, params_full_precision + model.params.density_network, scale);
        model.rgb_network.initialize_params(rng, params_full_precision + model.params.rgb_network, scale);
        ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.initialize_params(rng, params_full_precision + model.params.pos_encoding, scale); });
    }

    auto InstantNGP::total_parameter_count() const -> std::size_t {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before querying parameter count."};
        return network.definition->params.total;
    }

    auto InstantNGP::matrix_parameter_count() const -> std::uint32_t {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before querying matrix parameter count."};
        return static_cast<std::uint32_t>(network.definition->params.pos_encoding);
    }

    auto InstantNGP::model_input_width() const -> std::uint32_t {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before querying input width."};
        return network.definition->dir_offset + network.definition->layout.dir_input_width;
    }

    auto InstantNGP::model_padded_output_width() const -> std::uint32_t {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before querying output width."};
        return network.definition->layout.padded_output_width;
    }

    void InstantNGP::prepare_model_scratch(const std::uint32_t batch_size, const legacy::MatrixLayout output_layout) {
        if (!network.definition || !network.scratch) throw std::runtime_error{"Network model must be initialized before preparing scratch."};

        auto& model = *network.definition;
        auto& scratch = *network.scratch;
        scratch.density_network_input  = ngp::legacy::GPUMatrixDynamic<__half>{model.layout.pos_output_width, batch_size, device.stream, model.layout.pos_layout};
        scratch.rgb_network_input      = ngp::legacy::GPUMatrixDynamic<__half>{model.rgb_network_input_width, batch_size, device.stream, model.layout.dir_layout};
        scratch.density_network_output = scratch.rgb_network_input.slice_rows(0u, model.layout.density_output_width);
        scratch.dL_drgb                = ngp::legacy::GPUMatrixDynamic<__half>{model.layout.rgb_output_width, batch_size, device.stream, output_layout};
        scratch.dL_drgb_input          = ngp::legacy::GPUMatrixDynamic<__half>{model.rgb_network_input_width, batch_size, device.stream, model.layout.dir_layout};
        if (model.layout.pos_param_count > 0u)
            scratch.dL_ddensity_input = ngp::legacy::GPUMatrixDynamic<__half>{model.layout.pos_output_width, batch_size, device.stream, model.layout.pos_layout};
        else
            scratch.dL_ddensity_input = {};

        model.density_network.prepare_scratch(device.stream, batch_size, output_layout, scratch.density_network);
        model.rgb_network.prepare_scratch(device.stream, batch_size, output_layout, scratch.rgb_network);
    }

    void InstantNGP::density(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>& output) const {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before density inference."};
        if (input.layout() != legacy::CM) throw std::runtime_error{"model density input must be in column major format."};

        auto& model                    = *network.definition;
        const std::uint32_t batch_size = output.n();
        legacy::GPUMatrixDynamic<__half> density_input{model.layout.pos_output_width, batch_size, stream, model.layout.pos_layout};
        ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.encode(stream, input.slice_rows(0u, model.layout.pos_input_width), density_input); });
        model.density_network.inference(stream, density_input, output);
    }

    void InstantNGP::inference(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>& output) const {
        if (!network.definition) throw std::runtime_error{"Network model must be initialized before inference."};

        auto& model = *network.definition;
        legacy::check_or_throw(input.m() == model_input_width());
        legacy::check_or_throw(output.m() == model_padded_output_width());
        legacy::check_or_throw(input.n() % ngp::network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());

        const std::uint32_t batch_size = input.n();
        legacy::GPUMatrixDynamic<__half> density_input{model.layout.pos_output_width, batch_size, stream, model.layout.pos_layout};
        legacy::GPUMatrixDynamic<__half> rgb_input{model.rgb_network_input_width, batch_size, stream, model.layout.dir_layout};
        legacy::GPUMatrixDynamic<__half> density_output = rgb_input.slice_rows(0u, model.layout.density_output_width);
        legacy::GPUMatrixDynamic<__half> rgb_output{output.data(), model.layout.rgb_output_width, batch_size, output.layout()};

        ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.encode(stream, input.slice_rows(0u, model.layout.pos_input_width), density_input); });
        model.density_network.inference(stream, density_input, density_output);

        auto dir_output = rgb_input.slice_rows(model.layout.density_output_width, model.layout.dir_output_width);
        model.dir_encoding.encode(stream, input.slice_rows(model.dir_offset, model.layout.dir_input_width), dir_output);
        model.rgb_network.inference(stream, rgb_input, rgb_output);

        if (batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ngp::network::extract_density<__half><<<blocks, ngp::network::detail::n_threads_linear, 0, stream>>>(batch_size, density_output.layout() == legacy::AoS ? density_output.stride() : 1u, output.layout() == legacy::AoS ? model_padded_output_width() : 1u, density_output.data(), output.data() + 3u * (output.layout() == legacy::AoS ? 1u : batch_size));
        }
    }

    void InstantNGP::forward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>* output) {
        if (!network.definition || !network.scratch) throw std::runtime_error{"Network model must be initialized before forward."};

        auto& model = *network.definition;
        auto& scratch = *network.scratch;
        legacy::check_or_throw(input.m() == model_input_width());
        legacy::check_or_throw(!output || output->m() == model_padded_output_width());
        legacy::check_or_throw(input.n() % ngp::network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(!output || input.n() == output->n());

        const std::uint32_t batch_size = input.n();
        ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.encode(stream, input.slice_rows(0u, model.layout.pos_input_width), scratch.density_network_input); });
        model.density_network.forward(stream, scratch.density_network_input, &scratch.density_network_output, scratch.density_network);

        auto dir_output = scratch.rgb_network_input.slice_rows(model.layout.density_output_width, model.layout.dir_output_width);
        model.dir_encoding.encode(stream, input.slice_rows(model.dir_offset, model.layout.dir_input_width), dir_output);

        if (output) scratch.rgb_network_output = ngp::legacy::GPUMatrixDynamic<__half>{output->data(), model.layout.rgb_output_width, batch_size, output->layout()};
        model.rgb_network.forward(stream, scratch.rgb_network_input, output ? &scratch.rgb_network_output : nullptr, scratch.rgb_network);

        if (output && batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ngp::network::extract_density<__half><<<blocks, ngp::network::detail::n_threads_linear, 0, stream>>>(batch_size, model.layout.dir_layout == legacy::AoS ? scratch.density_network_output.stride() : 1u, model_padded_output_width(), scratch.density_network_output.data(), output->data() + 3u);
        }
    }

    void InstantNGP::backward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, const legacy::GPUMatrixDynamic<__half>& output, const legacy::GPUMatrixDynamic<__half>& dL_doutput, const GradientMode gradient_mode) {
        if (!network.definition || !network.scratch) throw std::runtime_error{"Network model must be initialized before backward."};

        auto& model = *network.definition;
        auto& scratch = *network.scratch;
        const ngp::network::detail::GradientMode model_gradient_mode = gradient_mode == GradientMode::Ignore ? ngp::network::detail::GradientMode::Ignore : (gradient_mode == GradientMode::Overwrite ? ngp::network::detail::GradientMode::Overwrite : ngp::network::detail::GradientMode::Accumulate);
        legacy::check_or_throw(input.m() == model_input_width());
        legacy::check_or_throw(output.m() == model_padded_output_width());
        legacy::check_or_throw(dL_doutput.m() == model_padded_output_width());
        legacy::check_or_throw(input.n() % ngp::network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(input.n() == dL_doutput.n());

        const std::uint32_t batch_size = input.n();
        legacy::cuda_check(cudaMemsetAsync(scratch.dL_drgb.data(), 0, scratch.dL_drgb.n_bytes(), stream));

        if (batch_size > 0u) {
            const std::uint32_t rgb_elements = batch_size * 3u;
            const std::uint32_t blocks       = (rgb_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ngp::network::extract_rgb<__half><<<blocks, ngp::network::detail::n_threads_linear, 0, stream>>>(rgb_elements, scratch.dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), scratch.dL_drgb.data());
        }

        const legacy::GPUMatrixDynamic<__half> rgb_output{(__half*) output.data(), model.layout.rgb_output_width, batch_size, output.layout()};
        model.rgb_network.backward(stream, scratch.rgb_network, scratch.rgb_network_input, rgb_output, scratch.dL_drgb, &scratch.dL_drgb_input, model_gradient_mode);

        auto dL_ddensity_output = scratch.dL_drgb_input.slice_rows(0u, model.layout.density_output_width);
        if (batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ngp::network::add_density_gradient<__half><<<blocks, ngp::network::detail::n_threads_linear, 0, stream>>>(batch_size, dL_doutput.m(), dL_doutput.data(), dL_ddensity_output.layout() == legacy::RM ? 1u : dL_ddensity_output.stride(), dL_ddensity_output.data());
        }

        if (model.layout.pos_param_count > 0u) {
            model.density_network.backward(stream, scratch.density_network, scratch.density_network_input, scratch.density_network_output, dL_ddensity_output, &scratch.dL_ddensity_input, model_gradient_mode);
            ngp::network::detail::visit_module(model.pos_encoding, [&](auto& impl) { impl.backward(stream, input.slice_rows(0u, model.layout.pos_input_width), scratch.dL_ddensity_input, model_gradient_mode); });
        } else {
            model.density_network.backward(stream, scratch.density_network, scratch.density_network_input, scratch.density_network_output, dL_ddensity_output, (legacy::GPUMatrixDynamic<__half>*) nullptr, model_gradient_mode);
        }
    }

    InstantNGP::InstantNGP(const NetworkConfig& network_config) {
        config.network = network_config;

        config.plan                             = {};
        config.plan.training.batch_size         = 1u << 18;
        config.plan.training.floats_per_coord   = sizeof(legacy::NerfCoordinate) / sizeof(float);
        config.plan.validation.floats_per_coord = config.plan.training.floats_per_coord;
        config.plan.validation.max_samples      = config.plan.validation.tile_rays * config.plan.validation.max_samples_per_ray;

        constexpr std::uint32_t n_pos_dims    = sizeof(legacy::NerfPosition) / sizeof(float);
        config.plan.network.n_pos_dims          = n_pos_dims;
        config.plan.network.n_dir_dims          = 3u;
        config.plan.network.dir_offset          = n_pos_dims + 1u;
        config.plan.network.density_alignment   = 16u;
        config.plan.network.density_output_dims = 16u;
        config.plan.network.rgb_alignment       = 16u;
        config.plan.network.rgb_output_dims     = 3u;

        const std::uint32_t encoding_output_dims   = config.network.encoding.n_levels * config.network.encoding.n_features_per_level;
        config.plan.network.density_input_dims       = legacy::next_multiple(encoding_output_dims, std::lcm(config.plan.network.density_alignment, config.network.encoding.n_features_per_level));
        const std::uint32_t dir_output_dims        = config.network.direction_encoding.sh_degree * config.network.direction_encoding.sh_degree;
        config.plan.network.dir_encoding_output_dims = legacy::next_multiple(dir_output_dims, config.plan.network.rgb_alignment);
        config.plan.network.rgb_input_dims           = legacy::next_multiple(config.plan.network.density_output_dims + config.plan.network.dir_encoding_output_dims, config.plan.network.rgb_alignment);

        config.plan.training.padded_output_width     = std::max(legacy::next_multiple(config.plan.network.rgb_output_dims, 16u), 4u);
        config.plan.training.max_samples             = config.plan.training.batch_size * 16u;
        config.plan.prep.uniform_samples_warmup      = legacy::NERF_GRID_N_CELLS;
        config.plan.prep.uniform_samples_steady      = legacy::NERF_GRID_N_CELLS / 4u;
        config.plan.prep.nonuniform_samples_steady   = legacy::NERF_GRID_N_CELLS / 4u;
        config.plan.density_grid.padded_output_width = config.plan.network.density_output_dims;
        config.plan.density_grid.query_batch_size    = legacy::NERF_GRID_N_CELLS * 2u;
        config.plan.density_grid.n_elements          = legacy::NERF_GRID_N_CELLS;
        config.plan.validation.padded_output_width   = config.plan.training.padded_output_width;

        cudaStream_t created_stream = {};
        legacy::cuda_check(cudaStreamCreate(&created_stream));
        device.stream = created_stream;

        training.rng         = legacy::math::pcg32{config.seed};
        sampling.density_rng = legacy::math::pcg32{training.rng.next_uint()};
        initialize_network_state();
    }

    auto InstantNGP::train(const std::int32_t iters) -> TrainResult {
        if (!network.definition || network.parameters.values == nullptr || dataset.device.frames.size() == 0u) throw std::runtime_error{"training data must be loaded before train()."};

        for (std::int32_t i = 0; i < iters; ++i) {
            std::uint32_t n_prep_to_skip = training.step / config.plan.prep.skip_growth_interval;
            if (n_prep_to_skip < 1u) n_prep_to_skip = 1u;
            if (n_prep_to_skip > config.plan.prep.max_skip) n_prep_to_skip = config.plan.prep.max_skip;
            if (training.step % n_prep_to_skip == 0u) {
                const auto prep_start = std::chrono::steady_clock::now();
                legacy::ScopeGuard prep_timing_guard{[&] { training.last_prep_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prep_start).count() / static_cast<float>(n_prep_to_skip); }};

                const std::uint32_t n_uniform_density_grid_samples    = training.step < config.plan.prep.warmup_steps ? config.plan.prep.uniform_samples_warmup : config.plan.prep.uniform_samples_steady;
                const std::uint32_t n_nonuniform_density_grid_samples = training.step < config.plan.prep.warmup_steps ? 0u : config.plan.prep.nonuniform_samples_steady;
                const std::uint32_t n_elements                        = config.plan.density_grid.n_elements;

                sampling.density.values.resize(n_elements);
                const std::uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;
                const std::uint32_t padded_output_width    = config.plan.density_grid.padded_output_width;

                const std::size_t positions_bytes        = legacy::align_to_cacheline(n_density_grid_samples * sizeof(legacy::NerfPosition));
                const std::size_t indices_bytes          = legacy::align_to_cacheline(n_elements * sizeof(std::uint32_t));
                const std::size_t density_tmp_bytes      = legacy::align_to_cacheline(n_elements * sizeof(float));
                const std::size_t mlp_out_bytes          = legacy::align_to_cacheline(n_density_grid_samples * padded_output_width * sizeof(__half));
                auto& update_workspace = sampling.update;
                update_workspace.arena.enlarge(positions_bytes + indices_bytes + density_tmp_bytes + mlp_out_bytes, device.stream);

                std::uint8_t* density_grid_base = reinterpret_cast<std::uint8_t*>(update_workspace.arena.data());
                std::size_t density_grid_offset = 0u;
                update_workspace.positions      = reinterpret_cast<legacy::NerfPosition*>(density_grid_base + density_grid_offset);
                density_grid_offset += positions_bytes;
                update_workspace.indices = reinterpret_cast<std::uint32_t*>(density_grid_base + density_grid_offset);
                density_grid_offset += indices_bytes;
                update_workspace.density_scratch = reinterpret_cast<float*>(density_grid_base + density_grid_offset);
                density_grid_offset += density_tmp_bytes;
                update_workspace.mlp_out = reinterpret_cast<__half*>(density_grid_base + density_grid_offset);

                if (training.step == 0u) {
                    sampling.density.ema_step = 0u;
                    if (n_elements > 0u) {
                        const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                        mark_untrained_density_grid<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_elements, sampling.density.values.data(), static_cast<std::uint32_t>(dataset.device.frames.size()), dataset.device.frames.data());
                    }
                }

                legacy::cuda_check(cudaMemsetAsync(update_workspace.density_scratch, 0, sizeof(float) * n_elements, device.stream));

                if (n_uniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_uniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_uniform_density_grid_samples, sampling.density_rng, sampling.density.ema_step, sampling.aabb, sampling.density.values.data(), update_workspace.positions, update_workspace.indices, -0.01f);
                }
                sampling.density_rng.advance();

                if (n_nonuniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_nonuniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_nonuniform_density_grid_samples, sampling.density_rng, sampling.density.ema_step, sampling.aabb, sampling.density.values.data(), update_workspace.positions + n_uniform_density_grid_samples, update_workspace.indices + n_uniform_density_grid_samples, NERF_MIN_OPTICAL_THICKNESS);
                }
                sampling.density_rng.advance();

                const std::size_t density_batch_size = config.plan.density_grid.query_batch_size;
                for (std::size_t density_batch_offset = 0u; density_batch_offset < n_density_grid_samples; density_batch_offset += density_batch_size) {
                    const std::size_t density_query_size = std::min(density_batch_size, static_cast<std::size_t>(n_density_grid_samples) - density_batch_offset);
                    legacy::GPUMatrixDynamic<__half> density_matrix(update_workspace.mlp_out + density_batch_offset, padded_output_width, density_query_size, legacy::RM);
                    legacy::GPUMatrixDynamic<float> density_position_matrix(reinterpret_cast<float*>(update_workspace.positions + density_batch_offset), sizeof(legacy::NerfPosition) / sizeof(float), density_query_size, legacy::CM);
                    density(device.stream, density_position_matrix, density_matrix);
                }

                if (n_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    splat_grid_samples_nerf_max_nearest_neighbor<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_density_grid_samples, update_workspace.indices, update_workspace.mlp_out, update_workspace.density_scratch);
                }

                if (n_elements > 0u) {
                    const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    ema_grid_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_elements, sampling.density.ema_decay, sampling.density.ema_step, sampling.density.values.data(), update_workspace.density_scratch);
                }
                ++sampling.density.ema_step;

                constexpr std::uint32_t base_grid_elements = legacy::NERF_GRID_N_CELLS;
                sampling.density.occupancy.enlarge(base_grid_elements / 8u);
                sampling.density.reduction.enlarge(reduce_sum_workspace_size(base_grid_elements));

                legacy::cuda_check(cudaMemsetAsync(sampling.density.reduction.data(), 0, sizeof(float), device.stream));
                reduce_sum(sampling.density.values.data(), DensityGridReduceOp{base_grid_elements}, sampling.density.reduction.data(), base_grid_elements, device.stream);

                if (base_grid_elements / 8u > 0u) {
                    constexpr std::uint32_t blocks = ((base_grid_elements / 8u) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    grid_to_bitfield<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(base_grid_elements / 8u, sampling.density.values.data(), sampling.density.occupancy.data(), sampling.density.reduction.data());
                }

                legacy::cuda_check(cudaStreamSynchronize(device.stream));
            }

            network.optimizer->impl.update_hyperparams(config.network.optimizer);
            const bool get_loss_scalar = training.step % 16u == 0u;
            const auto train_start     = std::chrono::steady_clock::now();
            legacy::ScopeGuard train_timing_guard{[&] { training.last_train_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(); }};

            auto& counters                 = training.counters;
            auto& workspace                = training.workspace;
            const std::uint32_t batch_size = config.plan.training.batch_size;

            counters.numsteps_counter.enlarge(1u);
            counters.numsteps_counter_compacted.enlarge(1u);
            counters.loss.enlarge(counters.rays_per_batch);
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter_compacted.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(counters.loss.data(), 0, sizeof(float) * counters.rays_per_batch, device.stream));

            workspace.padded_output_width = config.plan.training.padded_output_width;
            workspace.floats_per_coord    = config.plan.training.floats_per_coord;
            workspace.max_samples         = config.plan.training.max_samples;

            const std::size_t ray_indices_bytes       = legacy::align_to_cacheline(counters.rays_per_batch * sizeof(std::uint32_t));
            const std::size_t rays_unnormalized_bytes = legacy::align_to_cacheline(counters.rays_per_batch * sizeof(instant_ngp_detail::Ray));
            const std::size_t numsteps_bytes          = legacy::align_to_cacheline(counters.rays_per_batch * 2u * sizeof(std::uint32_t));
            const std::size_t coords_bytes            = legacy::align_to_cacheline(workspace.max_samples * workspace.floats_per_coord * sizeof(float));
            const std::size_t mlp_out_bytes           = legacy::align_to_cacheline(std::max(batch_size, workspace.max_samples) * workspace.padded_output_width * sizeof(__half));
            const std::size_t dloss_bytes             = legacy::align_to_cacheline(batch_size * workspace.padded_output_width * sizeof(__half));
            const std::size_t compacted_coords_bytes  = legacy::align_to_cacheline(batch_size * workspace.floats_per_coord * sizeof(float));
            const std::size_t ray_counter_bytes       = legacy::align_to_cacheline(sizeof(std::uint32_t));
            const std::size_t total_bytes             = ray_indices_bytes + rays_unnormalized_bytes + numsteps_bytes + coords_bytes + mlp_out_bytes + dloss_bytes + compacted_coords_bytes + ray_counter_bytes;

            workspace.arena.enlarge(total_bytes, device.stream);
            std::uint8_t* workspace_base = reinterpret_cast<std::uint8_t*>(workspace.arena.data());
            std::size_t workspace_offset = 0u;

            workspace.ray_indices = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);
            workspace_offset += ray_indices_bytes;
            workspace.rays_unnormalized = workspace_base + workspace_offset;
            workspace_offset += rays_unnormalized_bytes;
            workspace.numsteps = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);
            workspace_offset += numsteps_bytes;
            workspace.coords = reinterpret_cast<float*>(workspace_base + workspace_offset);
            workspace_offset += coords_bytes;
            workspace.mlp_out = reinterpret_cast<__half*>(workspace_base + workspace_offset);
            workspace_offset += mlp_out_bytes;
            workspace.dloss_dmlp_out = reinterpret_cast<__half*>(workspace_base + workspace_offset);
            workspace_offset += dloss_bytes;
            workspace.coords_compacted = reinterpret_cast<float*>(workspace_base + workspace_offset);
            workspace_offset += compacted_coords_bytes;
            workspace.ray_counter = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);

            workspace.max_inference = counters.measured_batch_size_before_compaction == 0u ? workspace.max_samples : legacy::next_multiple(std::min(counters.measured_batch_size_before_compaction, workspace.max_samples), network::detail::batch_size_granularity);
            if (counters.measured_batch_size_before_compaction == 0u) counters.measured_batch_size_before_compaction = workspace.max_inference;

            workspace.coords_matrix           = legacy::GPUMatrixDynamic<float>{workspace.coords, workspace.floats_per_coord, workspace.max_inference, legacy::CM};
            workspace.rgbsigma_matrix         = legacy::GPUMatrixDynamic<__half>{workspace.mlp_out, workspace.padded_output_width, workspace.max_inference, legacy::CM};
            workspace.compacted_coords_matrix = legacy::GPUMatrixDynamic<float>{workspace.coords_compacted, workspace.floats_per_coord, batch_size, legacy::CM};
            workspace.gradient_matrix         = legacy::GPUMatrixDynamic<__half>{workspace.dloss_dmlp_out, workspace.padded_output_width, batch_size, legacy::CM};
            workspace.compacted_output        = legacy::GPUMatrixDynamic<__half>{config.plan.training.padded_output_width, batch_size, device.stream, legacy::CM};

            if (training.step == 0u) counters.n_rays_total = 0u;
            workspace.n_rays_total = counters.n_rays_total;
            counters.n_rays_total += counters.rays_per_batch;

            legacy::cuda_check(cudaMemsetAsync(workspace.ray_counter, 0, sizeof(std::uint32_t), device.stream));
            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_training_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(counters.rays_per_batch, sampling.aabb, workspace.max_inference, workspace.n_rays_total, training.rng, workspace.ray_counter, counters.numsteps_counter.data(), workspace.ray_indices, static_cast<instant_ngp_detail::Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords), 1u},
                    static_cast<std::uint32_t>(dataset.device.pixels.size()), dataset.device.frames.data(), sampling.density.occupancy.data(), sampling.snap_to_pixel_centers);
            }
            inference(device.stream, workspace.coords_matrix, workspace.rgbsigma_matrix);

            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                compute_loss_kernel_train_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(counters.rays_per_batch, sampling.aabb, workspace.n_rays_total, training.rng, batch_size, workspace.ray_counter, network::detail::default_loss_scale<__half>, static_cast<int>(workspace.padded_output_width), static_cast<std::uint32_t>(dataset.device.pixels.size()), dataset.device.frames.data(), workspace.mlp_out, counters.numsteps_counter_compacted.data(),
                    workspace.ray_indices, static_cast<const instant_ngp_detail::Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<const legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords), 1u}, legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords_compacted), 1u}, workspace.dloss_dmlp_out, counters.loss.data(), sampling.snap_to_pixel_centers, sampling.density.reduction.data(),
                    sampling.near_distance);
            }

            const std::uint32_t dloss_elements  = batch_size * workspace.padded_output_width;
            const std::uint32_t coords_elements = batch_size * workspace.floats_per_coord;
            fill_rollover_and_rescale<__half><<<((dloss_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(batch_size, workspace.padded_output_width, counters.numsteps_counter_compacted.data(), workspace.dloss_dmlp_out);
            fill_rollover<float><<<((coords_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(batch_size, workspace.floats_per_coord, counters.numsteps_counter_compacted.data(), workspace.coords_compacted);

            auto& graph_capture = network.graph_capture;
            if (device.stream == nullptr || device.stream == cudaStreamLegacy) {
                forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                backward(device.stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, GradientMode::Overwrite);
            } else {
                cudaStreamCaptureStatus capture_status;
                legacy::cuda_check(cudaStreamIsCapturing(device.stream, &capture_status));
                if (capture_status != cudaStreamCaptureStatusNone) {
                    forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                    backward(device.stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, GradientMode::Overwrite);
                } else {
                    cudaError_t capture_result = cudaStreamIsCapturing(cudaStreamLegacy, &capture_status);
                    if (capture_result == cudaErrorStreamCaptureImplicit) {
                        forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                        backward(device.stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, GradientMode::Overwrite);
                    } else {
                        legacy::cuda_check(capture_result);
                        if (capture_status != cudaStreamCaptureStatusNone) {
                            forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                            backward(device.stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, GradientMode::Overwrite);
                        } else {
                            if (graph_capture.graph) {
                                legacy::cuda_check(cudaGraphDestroy(graph_capture.graph));
                                graph_capture.graph = nullptr;
                            }

                            legacy::cuda_check(cudaStreamBeginCapture(device.stream, cudaStreamCaptureModeRelaxed));
                            legacy::current_graph_captures().push_back(&graph_capture);
                            try {
                                forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                                backward(device.stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, GradientMode::Overwrite);
                                legacy::cuda_check(cudaStreamEndCapture(device.stream, &graph_capture.graph));
                                if (legacy::current_graph_captures().back() != &graph_capture) throw std::runtime_error{"Graph capture must end in reverse order of creation."};
                                legacy::current_graph_captures().pop_back();
                            } catch (...) {
                                if (!legacy::current_graph_captures().empty() && legacy::current_graph_captures().back() == &graph_capture) legacy::current_graph_captures().pop_back();

                                cudaGraph_t aborted_graph      = nullptr;
                                cudaError_t end_capture_result = cudaStreamEndCapture(device.stream, &aborted_graph);
                                if (end_capture_result == cudaSuccess && aborted_graph)
                                    cudaGraphDestroy(aborted_graph);
                                else
                                    cudaGetLastError();

                                graph_capture.graph = nullptr;
                                throw;
                            }

                            if (graph_capture.synchronize_when_capture_done) {
                                legacy::cuda_check(cudaDeviceSynchronize());
                                graph_capture.synchronize_when_capture_done = false;
                            }

                            if (!graph_capture.graph) {
                                if (graph_capture.graph_instance) legacy::cuda_check(cudaGraphExecDestroy(graph_capture.graph_instance));
                                graph_capture.graph          = nullptr;
                                graph_capture.graph_instance = nullptr;
                            } else {
                                if (graph_capture.graph_instance) {
                                    cudaGraphExecUpdateResultInfo update_result;
                                    legacy::cuda_check(cudaGraphExecUpdate(graph_capture.graph_instance, graph_capture.graph, &update_result));
                                    if (update_result.result != cudaGraphExecUpdateSuccess) {
                                        legacy::cuda_check(cudaGraphExecDestroy(graph_capture.graph_instance));
                                        graph_capture.graph_instance = nullptr;
                                    }
                                }

                                if (!graph_capture.graph_instance) legacy::cuda_check(cudaGraphInstantiate(&graph_capture.graph_instance, graph_capture.graph, nullptr, nullptr, 0));
                                legacy::cuda_check(cudaGraphLaunch(graph_capture.graph_instance, device.stream));
                            }
                        }
                    }
                }
            }
            network.optimizer->impl.step(device.stream, network::detail::default_loss_scale<__half>, network.parameters.full_precision, network.parameters.values, network.parameters.gradients);
            ++training.step;

            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            std::uint32_t measured_batch_size_before_compaction = 0u;
            std::uint32_t measured_batch_size                   = 0u;
            counters.numsteps_counter.copy_to_host(&measured_batch_size_before_compaction, 1u);
            counters.numsteps_counter_compacted.copy_to_host(&measured_batch_size, 1u);
            counters.measured_batch_size_before_compaction = 0u;
            counters.measured_batch_size                   = 0u;

            float last_loss = 0.0f;
            if (measured_batch_size_before_compaction != 0u && measured_batch_size != 0u) {
                counters.measured_batch_size_before_compaction = measured_batch_size_before_compaction;
                counters.measured_batch_size                   = measured_batch_size;

                if (get_loss_scalar) {
                    training.loss_reduction.enlarge(reduce_sum_workspace_size(counters.rays_per_batch), device.stream);
                    legacy::cuda_check(cudaMemsetAsync(training.loss_reduction.data(), 0, sizeof(float), device.stream));
                    reduce_sum(counters.loss.data(), SumIdentityOp{}, training.loss_reduction.data(), counters.rays_per_batch, device.stream);
                    legacy::cuda_check(cudaMemcpyAsync(&last_loss, training.loss_reduction.data(), sizeof(float), cudaMemcpyDeviceToHost, device.stream));
                    legacy::cuda_check(cudaStreamSynchronize(device.stream));
                    last_loss *= static_cast<float>(counters.measured_batch_size) / static_cast<float>(batch_size);
                }

                counters.rays_per_batch = static_cast<std::uint32_t>(static_cast<float>(counters.rays_per_batch) * static_cast<float>(batch_size) / static_cast<float>(counters.measured_batch_size));
                counters.rays_per_batch = std::min(legacy::next_multiple(counters.rays_per_batch, network::detail::batch_size_granularity), 1u << 18);
            }

            if (get_loss_scalar) training.last_loss = last_loss;

            if (counters.measured_batch_size == 0u) {
                training.last_loss = 0.0f;
                throw std::runtime_error{"Training stopped unexpectedly."};
            }

            training.rng.advance();
        }

        TrainResult result{};
        result.loss                                  = training.last_loss;
        result.train_ms                              = training.last_train_ms;
        result.prep_ms                               = training.last_prep_ms;
        result.step                                  = training.step;
        result.batch_size                            = config.plan.training.batch_size;
        result.rays_per_batch                        = training.counters.rays_per_batch;
        result.measured_batch_size                   = training.counters.measured_batch_size;
        result.measured_batch_size_before_compaction = training.counters.measured_batch_size_before_compaction;
        return result;
    }

    auto InstantNGP::validate(const std::filesystem::path& report_path) -> ValidateResult {
        if (report_path.empty()) throw std::invalid_argument{"validation benchmark report path must not be empty."};
        if (!network.definition || network.parameters.values == nullptr) throw std::runtime_error{"Validation benchmark requires an initialized network."};

        const std::size_t validation_count = dataset.host.validation.size();
        if (validation_count == 0u) throw std::runtime_error{"No validation images are available in the current dataset."};

        const std::filesystem::path parent_dir = report_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create validation report directory '" + parent_dir.string() + "'."};

        std::ofstream report_stream{report_path, std::ios::binary};
        if (!report_stream.is_open()) throw std::runtime_error{"Failed to open validation benchmark report '" + report_path.string() + "'."};
        report_stream << std::setprecision(9);
        report_stream << "image_index,width,height,mse,psnr\n";

        ValidateResult result{};
        result.image_count = static_cast<std::uint32_t>(validation_count);
        result.min_psnr    = std::numeric_limits<float>::infinity();
        result.max_psnr    = -std::numeric_limits<float>::infinity();

        double total_squared_error = 0.0;
        double total_psnr          = 0.0;
        const auto benchmark_start = std::chrono::steady_clock::now();

        for (std::size_t validation_image_index = 0u; validation_image_index < validation_count; ++validation_image_index) {
            const DatasetState::HostData::Frame& source = dataset.host.validation[validation_image_index];
            if (source.width == 0u || source.height == 0u) throw std::runtime_error{"Validation frame has zero resolution."};
            if (!std::isfinite(source.focal_length_x) || source.focal_length_x <= 0.0f) throw std::runtime_error{"Validation frame has an invalid focal_length_x."};
            if (!std::isfinite(source.focal_length_y) || source.focal_length_y <= 0.0f) throw std::runtime_error{"Validation frame has an invalid focal_length_y."};

            const float focal_length_difference = std::fabs(source.focal_length_x - source.focal_length_y);
            const float focal_length_scale      = std::max(1.0f, std::max(std::fabs(source.focal_length_x), std::fabs(source.focal_length_y)));
            if (focal_length_difference > 1e-6f * focal_length_scale) throw std::runtime_error{"Validation benchmark currently requires focal_length_x and focal_length_y to match."};

            auto& frame = render.frame;
            frame.resolution   = legacy::math::ivec2{static_cast<int>(source.width), static_cast<int>(source.height)};
            frame.focal_length = source.focal_length_x;
            for (std::size_t row = 0u; row < 3u; ++row) {
                for (std::size_t column = 0u; column < 4u; ++column) frame.camera[column][row] = source.transform_matrix_4x4[row * 4u + column];
            }
            frame.camera[1] *= -1.0f;
            frame.camera[2] *= -1.0f;
            frame.camera[3]                      = frame.camera[3] * 0.33f + legacy::math::vec3(0.5f);
            const legacy::math::vec4 camera_row0 = ngp::legacy::math::row(frame.camera, 0);
            frame.camera                         = ngp::legacy::math::row(frame.camera, 0, ngp::legacy::math::row(frame.camera, 1));
            frame.camera                         = ngp::legacy::math::row(frame.camera, 1, ngp::legacy::math::row(frame.camera, 2));
            frame.camera                         = ngp::legacy::math::row(frame.camera, 2, camera_row0);

            const legacy::math::ivec2 resolution = frame.resolution;
            auto& workspace = render.workspace;
            workspace.total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
            workspace.padded_output_width = config.plan.validation.padded_output_width;
            workspace.floats_per_coord    = config.plan.validation.floats_per_coord;
            workspace.max_samples         = config.plan.validation.max_samples;
            workspace.rendered.resize(workspace.total_pixels);
            workspace.tile_numsteps.resize(config.plan.validation.tile_rays * 2u);
            workspace.tile_coords.resize(workspace.max_samples * workspace.floats_per_coord);
            workspace.tile_mlp_out.resize(workspace.max_samples * workspace.padded_output_width);
            workspace.sample_counter.resize(1u);
            workspace.overflow_counter.resize(1u);

            for (std::uint32_t pixel_offset = 0u; pixel_offset < workspace.total_pixels; pixel_offset += config.plan.validation.tile_rays) {
                const std::uint32_t tile_pixels = std::min(config.plan.validation.tile_rays, workspace.total_pixels - pixel_offset);
                legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), device.stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), device.stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), device.stream));

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, sampling.aabb, workspace.max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
                }

                render.used_samples    = 0u;
                render.overflowed_rays = 0u;
                legacy::cuda_check(cudaMemcpyAsync(&render.used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
                legacy::cuda_check(cudaMemcpyAsync(&render.overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
                legacy::cuda_check(cudaStreamSynchronize(device.stream));

                if (render.overflowed_rays != 0u) {
                    std::ostringstream message;
                    message << "Validation benchmark sample budget overflowed for " << render.overflowed_rays << " rays at image " << validation_image_index << ". Reduce tile size or increase the sample budget.";
                    throw std::runtime_error{message.str()};
                }

                if (render.used_samples > 0u) {
                    const std::uint32_t padded_used_samples = legacy::next_multiple(render.used_samples, network::detail::batch_size_granularity);
                    const std::uint32_t coord_elements      = padded_used_samples * workspace.floats_per_coord;
                    fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(padded_used_samples, workspace.floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                    legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), workspace.floats_per_coord, padded_used_samples, legacy::CM);
                    legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), workspace.padded_output_width, padded_used_samples, legacy::CM);
                    inference(device.stream, coords_matrix, rgbsigma_matrix);
                }

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const legacy::NerfCoordinate>{reinterpret_cast<const legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), workspace.padded_output_width, render.background, workspace.rendered.data());
                }
            }

            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            render.rendered_host.resize(workspace.total_pixels);
            workspace.rendered.copy_to_host(render.rendered_host);

            double image_squared_error = 0.0;
            for (int y = 0; y < resolution.y; ++y) {
                for (int x = 0; x < resolution.x; ++x) {
                    const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                    const legacy::math::vec3 prediction = clamp_rgb01(render.rendered_host[pixel_index]);
                    const legacy::math::vec4 gt         = read_rgba(legacy::math::ivec2{x, y}, resolution, source.rgba.data());
                    const legacy::math::vec3 target     = clamp_rgb01(linear_to_srgb(gt.rgb() + (1.0f - gt.a) * render.background));
                    const legacy::math::vec3 diff       = prediction - target;
                    image_squared_error += static_cast<double>(ngp::legacy::math::mean(diff * diff));
                }
            }

            const auto mse   = static_cast<float>(image_squared_error / static_cast<double>(workspace.total_pixels));
            const float psnr = mse > 0.0f ? -10.0f * std::log10(mse) : std::numeric_limits<float>::infinity();

            report_stream << validation_image_index << ',' << resolution.x << ',' << resolution.y << ',' << mse << ',' << psnr << '\n';
            if (!report_stream.good()) throw std::runtime_error{"Failed while writing validation benchmark report '" + report_path.string() + "'."};

            total_squared_error += image_squared_error;
            total_psnr += static_cast<double>(psnr);
            result.total_pixels += static_cast<std::uint64_t>(workspace.total_pixels);
            result.min_psnr = std::min(result.min_psnr, psnr);
            result.max_psnr = std::max(result.max_psnr, psnr);
        }

        result.benchmark_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - benchmark_start).count();
        result.mean_mse     = result.total_pixels > 0u ? static_cast<float>(total_squared_error / static_cast<double>(result.total_pixels)) : 0.0f;
        result.mean_psnr    = result.image_count > 0u ? static_cast<float>(total_psnr / static_cast<double>(result.image_count)) : 0.0f;
        result.split_psnr   = result.mean_mse > 0.0f ? -10.0f * std::log10(result.mean_mse) : std::numeric_limits<float>::infinity();
        return result;
    }

    auto InstantNGP::inference(const std::filesystem::path& output_path, const InferenceCamera& camera) const -> InferenceResult {
        if (output_path.empty()) throw std::invalid_argument{"inference output path must not be empty."};
        if (!network.definition || network.parameters.values == nullptr) throw std::runtime_error{"Inference requires an initialized network."};
        if (camera.resolution.x <= 0 || camera.resolution.y <= 0) throw std::runtime_error{"Inference camera has an invalid resolution."};
        if (!std::isfinite(camera.focal_length) || camera.focal_length <= 0.0f) throw std::runtime_error{"Inference camera has an invalid focal length."};
        for (std::size_t column = 0u; column < 4u; ++column) {
            for (std::size_t row = 0u; row < 3u; ++row) {
                if (!std::isfinite(camera.camera[column][row])) throw std::runtime_error{"Inference camera has a non-finite transform element."};
            }
        }

        const auto render_start = std::chrono::steady_clock::now();
        instant_ngp_detail::GpuFrame frame{};
        frame.resolution   = camera.resolution;
        frame.focal_length = camera.focal_length;
        frame.camera       = camera.camera;

        const legacy::math::ivec2 resolution = frame.resolution;
        const auto background                = legacy::math::vec3(1.0f);

        auto& workspace = render.workspace;
        workspace.total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
        workspace.padded_output_width = config.plan.validation.padded_output_width;
        workspace.floats_per_coord    = config.plan.validation.floats_per_coord;
        workspace.max_samples         = config.plan.validation.max_samples;
        workspace.rendered.resize(workspace.total_pixels);
        workspace.tile_numsteps.resize(config.plan.validation.tile_rays * 2u);
        workspace.tile_coords.resize(workspace.max_samples * workspace.floats_per_coord);
        workspace.tile_mlp_out.resize(workspace.max_samples * workspace.padded_output_width);
        workspace.sample_counter.resize(1u);
        workspace.overflow_counter.resize(1u);

        for (std::uint32_t pixel_offset = 0u; pixel_offset < workspace.total_pixels; pixel_offset += config.plan.validation.tile_rays) {
            const std::uint32_t tile_pixels = std::min(config.plan.validation.tile_rays, workspace.total_pixels - pixel_offset);
            legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), device.stream));

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, sampling.aabb, workspace.max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
            }

            std::uint32_t used_samples    = 0u;
            std::uint32_t overflowed_rays = 0u;
            legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
            legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            if (overflowed_rays != 0u) {
                std::ostringstream message;
                message << "Inference sample budget overflowed for " << overflowed_rays << " rays. Reduce tile size or increase the sample budget.";
                throw std::runtime_error{message.str()};
            }

            if (used_samples > 0u) {
                const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                const std::uint32_t coord_elements      = padded_used_samples * workspace.floats_per_coord;
                fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(padded_used_samples, workspace.floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), workspace.floats_per_coord, padded_used_samples, legacy::CM);
                legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), workspace.padded_output_width, padded_used_samples, legacy::CM);
                inference(device.stream, coords_matrix, rgbsigma_matrix);
            }

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const legacy::NerfCoordinate>{reinterpret_cast<const legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), workspace.padded_output_width, background, workspace.rendered.data());
            }
        }

        legacy::cuda_check(cudaStreamSynchronize(device.stream));

        std::vector<legacy::math::vec3> rendered_host(workspace.total_pixels);
        workspace.rendered.copy_to_host(rendered_host);

        std::vector<std::uint8_t> png_rgb(static_cast<std::size_t>(workspace.total_pixels) * 3u);
        for (int y = 0; y < resolution.y; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                png_rgb[pixel_index * 3u + 0u]      = static_cast<std::uint8_t>(lrintf(prediction.x * 255.0f));
                png_rgb[pixel_index * 3u + 1u]      = static_cast<std::uint8_t>(lrintf(prediction.y * 255.0f));
                png_rgb[pixel_index * 3u + 2u]      = static_cast<std::uint8_t>(lrintf(prediction.z * 255.0f));
            }
        }

        const std::filesystem::path parent_dir = output_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create image directory '" + parent_dir.string() + "'."};
        if (stbi_write_png(output_path.string().c_str(), resolution.x, resolution.y, 3, png_rgb.data(), resolution.x * 3) == 0) throw std::runtime_error{"Failed to write PNG image '" + output_path.string() + "'."};

        InferenceResult result{};
        result.width     = static_cast<std::uint32_t>(resolution.x);
        result.height    = static_cast<std::uint32_t>(resolution.y);
        result.render_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - render_start).count();
        return result;
    }

    auto InstantNGP::test(const std::filesystem::path& report_path) -> TestResult {
        if (report_path.empty()) throw std::invalid_argument{"test benchmark report path must not be empty."};
        if (!network.definition || network.parameters.values == nullptr) throw std::runtime_error{"Test benchmark requires an initialized network."};

        const std::size_t test_count = dataset.host.test.size();
        if (test_count == 0u) throw std::runtime_error{"No test images are available in the current dataset."};

        const std::filesystem::path parent_dir = report_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create test report directory '" + parent_dir.string() + "'."};

        std::ofstream report_stream{report_path, std::ios::binary};
        if (!report_stream.is_open()) throw std::runtime_error{"Failed to open test benchmark report '" + report_path.string() + "'."};
        report_stream << std::setprecision(9);
        report_stream << "image_index,width,height,mse,psnr\n";

        TestResult result{};
        result.image_count = static_cast<std::uint32_t>(test_count);
        result.min_psnr    = std::numeric_limits<float>::infinity();
        result.max_psnr    = -std::numeric_limits<float>::infinity();

        double total_squared_error = 0.0;
        double total_psnr          = 0.0;
        const auto benchmark_start = std::chrono::steady_clock::now();

        for (std::size_t test_image_index = 0u; test_image_index < test_count; ++test_image_index) {
            const DatasetState::HostData::Frame& source = dataset.host.test[test_image_index];
            if (source.width == 0u || source.height == 0u) throw std::runtime_error{"Test frame has zero resolution."};
            if (!std::isfinite(source.focal_length_x) || source.focal_length_x <= 0.0f) throw std::runtime_error{"Test frame has an invalid focal_length_x."};
            if (!std::isfinite(source.focal_length_y) || source.focal_length_y <= 0.0f) throw std::runtime_error{"Test frame has an invalid focal_length_y."};

            const float focal_length_difference = std::fabs(source.focal_length_x - source.focal_length_y);
            const float focal_length_scale      = std::max(1.0f, std::max(std::fabs(source.focal_length_x), std::fabs(source.focal_length_y)));
            if (focal_length_difference > 1e-6f * focal_length_scale) throw std::runtime_error{"Test benchmark currently requires focal_length_x and focal_length_y to match."};

            instant_ngp_detail::GpuFrame frame{};
            frame.resolution   = legacy::math::ivec2{static_cast<int>(source.width), static_cast<int>(source.height)};
            frame.focal_length = source.focal_length_x;
            for (std::size_t row = 0u; row < 3u; ++row) {
                for (std::size_t column = 0u; column < 4u; ++column) frame.camera[column][row] = source.transform_matrix_4x4[row * 4u + column];
            }
            frame.camera[1] *= -1.0f;
            frame.camera[2] *= -1.0f;
            frame.camera[3]                      = frame.camera[3] * 0.33f + legacy::math::vec3(0.5f);
            const legacy::math::vec4 camera_row0 = ngp::legacy::math::row(frame.camera, 0);
            frame.camera                         = ngp::legacy::math::row(frame.camera, 0, ngp::legacy::math::row(frame.camera, 1));
            frame.camera                         = ngp::legacy::math::row(frame.camera, 1, ngp::legacy::math::row(frame.camera, 2));
            frame.camera                         = ngp::legacy::math::row(frame.camera, 2, camera_row0);

            const legacy::math::ivec2 resolution = frame.resolution;
            const auto background                = legacy::math::vec3(1.0f);

            auto& workspace = render.workspace;
            workspace.total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
            workspace.padded_output_width = config.plan.validation.padded_output_width;
            workspace.floats_per_coord    = config.plan.validation.floats_per_coord;
            workspace.max_samples         = config.plan.validation.max_samples;
            workspace.rendered.resize(workspace.total_pixels);
            workspace.tile_numsteps.resize(config.plan.validation.tile_rays * 2u);
            workspace.tile_coords.resize(workspace.max_samples * workspace.floats_per_coord);
            workspace.tile_mlp_out.resize(workspace.max_samples * workspace.padded_output_width);
            workspace.sample_counter.resize(1u);
            workspace.overflow_counter.resize(1u);

            for (std::uint32_t pixel_offset = 0u; pixel_offset < workspace.total_pixels; pixel_offset += config.plan.validation.tile_rays) {
                const std::uint32_t tile_pixels = std::min(config.plan.validation.tile_rays, workspace.total_pixels - pixel_offset);
                legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), device.stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), device.stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), device.stream));

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, sampling.aabb, workspace.max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
                }

                std::uint32_t used_samples    = 0u;
                std::uint32_t overflowed_rays = 0u;
                legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
                legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
                legacy::cuda_check(cudaStreamSynchronize(device.stream));

                if (overflowed_rays != 0u) {
                    std::ostringstream message;
                    message << "Test benchmark sample budget overflowed for " << overflowed_rays << " rays at image " << test_image_index << ". Reduce tile size or increase the sample budget.";
                    throw std::runtime_error{message.str()};
                }

                if (used_samples > 0u) {
                    const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                    const std::uint32_t coord_elements      = padded_used_samples * workspace.floats_per_coord;
                    fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(padded_used_samples, workspace.floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                    legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), workspace.floats_per_coord, padded_used_samples, legacy::CM);
                    legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), workspace.padded_output_width, padded_used_samples, legacy::CM);
                    inference(device.stream, coords_matrix, rgbsigma_matrix);
                }

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const legacy::NerfCoordinate>{reinterpret_cast<const legacy::NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), workspace.padded_output_width, background, workspace.rendered.data());
                }
            }

            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            std::vector<legacy::math::vec3> rendered_host(workspace.total_pixels);
            workspace.rendered.copy_to_host(rendered_host);

            double image_squared_error = 0.0;
            for (int y = 0; y < resolution.y; ++y) {
                for (int x = 0; x < resolution.x; ++x) {
                    const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                    const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                    const legacy::math::vec4 gt         = read_rgba(legacy::math::ivec2{x, y}, resolution, source.rgba.data());
                    const legacy::math::vec3 target     = clamp_rgb01(linear_to_srgb(gt.rgb() + (1.0f - gt.a) * background));
                    const legacy::math::vec3 diff       = prediction - target;
                    image_squared_error += static_cast<double>(ngp::legacy::math::mean(diff * diff));
                }
            }

            const auto mse   = static_cast<float>(image_squared_error / static_cast<double>(workspace.total_pixels));
            const float psnr = mse > 0.0f ? -10.0f * std::log10(mse) : std::numeric_limits<float>::infinity();

            report_stream << test_image_index << ',' << resolution.x << ',' << resolution.y << ',' << mse << ',' << psnr << '\n';
            if (!report_stream.good()) throw std::runtime_error{"Failed while writing test benchmark report '" + report_path.string() + "'."};

            total_squared_error += image_squared_error;
            total_psnr += static_cast<double>(psnr);
            result.total_pixels += static_cast<std::uint64_t>(workspace.total_pixels);
            result.min_psnr = std::min(result.min_psnr, psnr);
            result.max_psnr = std::max(result.max_psnr, psnr);
        }

        result.benchmark_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - benchmark_start).count();
        result.mean_mse     = result.total_pixels > 0u ? static_cast<float>(total_squared_error / static_cast<double>(result.total_pixels)) : 0.0f;
        result.mean_psnr    = result.image_count > 0u ? static_cast<float>(total_psnr / static_cast<double>(result.image_count)) : 0.0f;
        result.split_psnr   = result.mean_mse > 0.0f ? -10.0f * std::log10(result.mean_mse) : std::numeric_limits<float>::infinity();
        return result;
    }

} // namespace ngp
