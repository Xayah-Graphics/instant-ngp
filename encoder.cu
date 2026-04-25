#include "encoder.cuh"
#include <algorithm>
#include <limits>
#include <numeric>
#include <sstream>
#include <type_traits>

namespace ngp::encoding {

#if defined(TCNN_PARAMS_UNALIGNED)
    inline constexpr bool params_aligned = false;
#else
    inline constexpr bool params_aligned = true;
#endif

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

    inline std::uint32_t powi(const std::uint32_t base, const std::uint32_t exponent) {
        std::uint32_t result = 1u;
        for (std::uint32_t i = 0u; i < exponent; ++i) result *= base;
        return result;
    }

    template <std::uint32_t N_DIMS, std::uint32_t N_PRIMES>
    __device__ std::uint32_t lcg_hash(const legacy::math::uvec<N_DIMS>& pos_grid, const std::uint32_t primes[N_PRIMES]) {
        static_assert(N_DIMS <= N_PRIMES, "lcg_hash can only hash up to N_PRIMES dimensions.");

        std::uint32_t result = 0u;
        TCNN_PRAGMA_UNROLL
        for (std::uint32_t i = 0u; i < N_DIMS; ++i) result ^= pos_grid[i] * primes[i];
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
        static_assert(N_DIMS <= std::size(max_bases), "grid_index can only be used for N_DIMS <= 10");

        if (grid_resolution_value <= max_bases[N_DIMS]) {
            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0u; dim < N_DIMS; ++dim) {
                index += pos_grid[dim] * stride;
                stride *= grid_resolution_value;
            }
        } else {
            stride = 0xFFFFFFFFu;
        }

        if (grid_type == GridType::Hash && hashmap_size < stride) index = coherent_prime_hash<N_DIMS>(pos_grid);
        return index % hashmap_size;
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
        for (std::size_t j = 0u; j < N_TO_GENERATE; ++j) {
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
        const std::size_t blocks    = (n_threads + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
        generate_random_uniform_kernel<T, RNG, n_to_generate><<<blocks, network::detail::n_threads_linear>>>(n_elements, rng, out, lower, upper);
        rng.advance(n_elements);
    }

    __device__ inline float random_val(const std::uint32_t seed, const std::uint32_t idx) {
        legacy::math::pcg32 rng{seed};
        rng.advance(idx);
        return rng.next_float();
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

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    __global__ void kernel_grid(const std::uint32_t num_elements, const std::uint32_t num_grid_features, const ParamsOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, float max_level, const GridType grid_type, const T* __restrict__ grid, legacy::MatrixView<const float> positions_in, T* __restrict__ encoded_positions) {
        const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i >= num_elements) return;

        const std::uint32_t level = blockIdx.y;

        max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

        if (level >= max_level + 1e-3f) {
            if (encoded_positions) {
                TCNN_PRAGMA_UNROLL
                for (std::uint32_t f = 0u; f < N_FEATURES_PER_LEVEL; ++f) encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = static_cast<T>(0.0f);
            }
            return;
        }

        grid += offset_table.data[level] * N_FEATURES_PER_LEVEL;
        const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
        const float scale                = exp2f(static_cast<float>(level) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
        const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;

        float pos[N_POS_DIMS];
        legacy::math::uvec<N_POS_DIMS> pos_grid = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t dim = 0u; dim < N_POS_DIMS; ++dim) pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale);

        auto grid_val = [&](const legacy::math::uvec<N_POS_DIMS>& local_pos) {
            const std::uint32_t index = grid_index<N_POS_DIMS>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL;
            return *reinterpret_cast<const legacy::math::tvec<T, N_FEATURES_PER_LEVEL, params_aligned ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)>*>(&grid[index]);
        };

        if (encoded_positions) {
            legacy::math::tvec<T, N_FEATURES_PER_LEVEL, params_aligned ? sizeof(T) * N_FEATURES_PER_LEVEL : sizeof(T)> result = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t idx = 0u; idx < (1u << N_POS_DIMS); ++idx) {
                float weight                                  = 1.0f;
                legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

                TCNN_PRAGMA_UNROLL
                for (std::uint32_t dim = 0u; dim < N_POS_DIMS; ++dim) {
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
                for (std::uint32_t feature_idx = 0u; feature_idx < N_FEATURES_PER_LEVEL; ++feature_idx) {
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
            for (std::uint32_t f = 0u; f < N_FEATURES_PER_LEVEL; ++f) encoded_positions[i + (level * N_FEATURES_PER_LEVEL + f) * num_elements] = result[f];
        }
    }

    template <typename T, typename GradT, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL, std::uint32_t N_FEATURES_PER_THREAD>
    __global__ void kernel_grid_backward(const std::uint32_t num_elements, const std::uint32_t num_grid_features, const ParamsOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, float max_level, const bool stochastic_interpolation, const GridType grid_type, GradT* __restrict__ grid_gradient, legacy::MatrixView<const float> positions_in, const T* __restrict__ dL_dy) {
        const std::uint32_t i = ((blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD) / N_FEATURES_PER_LEVEL;
        if (i >= num_elements) return;

        const std::uint32_t level   = blockIdx.y;
        const std::uint32_t feature = (blockIdx.x * blockDim.x + threadIdx.x) * N_FEATURES_PER_THREAD - i * N_FEATURES_PER_LEVEL;

        max_level = (max_level * num_grid_features) / N_FEATURES_PER_LEVEL;

        if (level > max_level + 1e-3f) return;

        grid_gradient += offset_table.data[level] * N_FEATURES_PER_LEVEL;
        const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
        const float scale                = exp2f(static_cast<float>(level) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
        const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;

        auto add_grid_gradient = [&](const legacy::math::uvec<N_POS_DIMS>& local_pos, const legacy::math::tvec<GradT, N_FEATURES_PER_THREAD>& grad, const float weight) {
            const std::uint32_t index = grid_index<N_POS_DIMS>(grid_type, hashmap_size, resolution, local_pos) * N_FEATURES_PER_LEVEL + feature;
            legacy::math::atomic_add_gmem(grid_gradient + index, static_cast<GradT>(weight) * grad);
        };

        float pos[N_POS_DIMS];
        legacy::math::uvec<N_POS_DIMS> pos_grid = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t dim = 0u; dim < N_POS_DIMS; ++dim) pos_fract(positions_in(dim, i), &pos[dim], &pos_grid[dim], scale);

        legacy::math::tvec<T, N_FEATURES_PER_THREAD> grad = {};

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t f = 0u; f < N_FEATURES_PER_THREAD; ++f) grad[f] = dL_dy[i + (level * N_FEATURES_PER_LEVEL + feature + f) * num_elements];

        if (stochastic_interpolation) {
            const float sample                            = random_val(1337u, i + level * num_elements);
            legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0u; dim < N_POS_DIMS; ++dim) {
                if (sample >= pos[dim])
                    pos_grid_local[dim] = pos_grid[dim];
                else
                    pos_grid_local[dim] = pos_grid[dim] + 1u;
            }

            add_grid_gradient(pos_grid_local, grad, 1.0f);
            return;
        }

        TCNN_PRAGMA_UNROLL
        for (std::uint32_t idx = 0u; idx < (1u << N_POS_DIMS); ++idx) {
            float weight                                  = 1.0f;
            legacy::math::uvec<N_POS_DIMS> pos_grid_local = {};

            TCNN_PRAGMA_UNROLL
            for (std::uint32_t dim = 0u; dim < N_POS_DIMS; ++dim) {
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

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::GridEncodingTemplated(const std::uint32_t n_features, const std::uint32_t log2_hashmap_size, const std::uint32_t base_resolution, const float per_level_scale, const bool stochastic_interpolation, const GridType grid_type) : n_features{n_features}, base_resolution{base_resolution}, per_level_scale{per_level_scale}, stochastic_interpolation{stochastic_interpolation}, grid_type{grid_type} {
        n_levels             = (this->n_features + N_FEATURES_PER_LEVEL - 1u) / N_FEATURES_PER_LEVEL;
        std::uint32_t offset = 0u;

        if (n_levels > max_n_levels) {
            std::ostringstream stream;
            stream << "GridEncoding: n_levels=" << n_levels << " must be at most MAX_N_LEVELS=" << max_n_levels;
            throw std::runtime_error{stream.str()};
        }

        for (std::uint32_t i = 0u; i < n_levels; ++i) {
            const float scale                  = exp2f(static_cast<float>(i) * std::log2(per_level_scale)) * static_cast<float>(base_resolution) - 1.0f;
            const std::uint32_t resolution     = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
            constexpr std::uint32_t max_params = std::numeric_limits<std::uint32_t>::max() / 2u;
            std::uint32_t params_in_level      = std::pow(static_cast<float>(resolution), N_POS_DIMS) > static_cast<float>(max_params) ? max_params : powi(resolution, N_POS_DIMS);

            params_in_level = legacy::next_multiple(params_in_level, 8u);

            if (grid_type == GridType::Dense) {
            } else if (grid_type == GridType::Tiled) {
                params_in_level = std::min(params_in_level, powi(base_resolution, N_POS_DIMS));
            } else if (grid_type == GridType::Hash) {
                params_in_level = std::min(params_in_level, 1u << log2_hashmap_size);
            } else {
                throw std::runtime_error{"GridEncoding: invalid grid type."};
            }

            offset_table.data[i] = offset;
            offset += params_in_level;
        }

        offset_table.data[n_levels] = offset;
        n_params                    = static_cast<std::size_t>(offset_table.data[n_levels]) * N_FEATURES_PER_LEVEL;
        output_width                = this->n_features;
        padded_output_width         = output_width;

        if (this->n_features % N_FEATURES_PER_LEVEL != 0u) {
            std::ostringstream stream;
            stream << "GridEncoding: n_features=" << this->n_features << " must be a multiple of N_FEATURES_PER_LEVEL=" << N_FEATURES_PER_LEVEL;
            throw std::runtime_error{stream.str()};
        }
    }

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    void GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(output.m() == padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        if (n_params > 0u) legacy::check_or_throw(params != nullptr);

        const std::uint32_t num_elements = input.n();
        if (padded_output_width == 0u || num_elements == 0u) return;
        const std::uint32_t n_to_pad = padded_output_width - output_width;

        network::detail::SyncedStreamReservation synced_streams{stream, n_to_pad > 0u ? 2u : 1u};
        const cudaStream_t main_stream = synced_streams.main_stream;
        const cudaStream_t aux_stream  = synced_streams.aux_stream ? synced_streams.aux_stream : synced_streams.main_stream;

        if (n_to_pad > 0u) {
            if (output.layout() == legacy::AoS) {
                const dim3 threads         = {n_to_pad, (network::detail::n_threads_linear + n_to_pad - 1u) / n_to_pad, 1u};
                const std::uint32_t blocks = (num_elements + threads.y - 1u) / threads.y;
                zero_padded_output_aos<T><<<blocks, threads, 0, aux_stream>>>(num_elements, output_width, n_to_pad, output.pitched_ptr());
            } else {
                legacy::cuda_check(cudaMemsetAsync(output.data() + num_elements * output_width, 0, sizeof(T) * num_elements * n_to_pad, aux_stream));
            }
        }

        static constexpr std::uint32_t n_threads_hashgrid = 512u;
        const dim3 blocks_hashgrid                        = {(num_elements + n_threads_hashgrid - 1u) / n_threads_hashgrid, n_levels, 1u};

        T* encoded_positions_soa        = output.data();
        legacy::GpuAllocation workspace = {};
        if (output.layout() == legacy::AoS) {
            workspace             = legacy::GpuAllocation{static_cast<std::size_t>(num_elements) * n_features * sizeof(T), main_stream};
            encoded_positions_soa = reinterpret_cast<T*>(workspace.data());
        }

        kernel_grid<T, N_POS_DIMS, N_FEATURES_PER_LEVEL><<<blocks_hashgrid, n_threads_hashgrid, 0, main_stream>>>(num_elements, n_features, offset_table, base_resolution, std::log2(per_level_scale), max_level, grid_type, params, input.view(), encoded_positions_soa);

        if (output.layout() == legacy::AoS) {
            const dim3 threads_transpose         = {n_levels * N_FEATURES_PER_LEVEL, 8u, 1u};
            const std::uint32_t blocks_transpose = (num_elements + threads_transpose.y - 1u) / threads_transpose.y;
            transpose_encoded_position<T><<<blocks_transpose, threads_transpose, 0, main_stream>>>(num_elements, encoded_positions_soa, output.pitched_ptr());
        }
    }

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    void GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::backward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, const legacy::GPUMatrixDynamic<T>& dL_doutput, const network::detail::GradientMode param_gradients_mode) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(dL_doutput.m() == padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == dL_doutput.n());
        if (n_params > 0u) {
            legacy::check_or_throw(params != nullptr);
            if (param_gradients_mode != network::detail::GradientMode::Ignore) legacy::check_or_throw(gradients != nullptr);
        }

        const std::uint32_t num_elements = input.n();
        if (param_gradients_mode == network::detail::GradientMode::Ignore || num_elements == 0u) return;

        const T* dL_dy_rm               = dL_doutput.data();
        legacy::GpuAllocation workspace = {};
        if (dL_doutput.layout() == legacy::CM) {
            workspace = legacy::GpuAllocation{static_cast<std::size_t>(num_elements) * n_features * sizeof(T), stream};

            const dim3 threads_transpose         = {n_levels * N_FEATURES_PER_LEVEL, 8u, 1u};
            const std::uint32_t blocks_transpose = (num_elements + threads_transpose.y - 1u) / threads_transpose.y;
            transpose_gradients<T><<<blocks_transpose, threads_transpose, 0, stream>>>(num_elements, reinterpret_cast<T*>(workspace.data()), dL_doutput.pitched_ptr());

            dL_dy_rm = reinterpret_cast<const T*>(workspace.data());
        }

        std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>* grid_gradient = nullptr;
        legacy::GpuAllocation grid_gradient_tmp                                 = {};

        if constexpr (!std::is_same_v<std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>, T>) {
            grid_gradient_tmp = legacy::GpuAllocation{n_params * sizeof(std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>), stream};
            grid_gradient     = reinterpret_cast<std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>*>(grid_gradient_tmp.data());
        } else {
            grid_gradient = reinterpret_cast<std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>*>(gradients);
        }

        if (param_gradients_mode == network::detail::GradientMode::Overwrite) legacy::cuda_check(cudaMemsetAsync(grid_gradient, 0, n_params * sizeof(std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>), stream));

        static constexpr std::uint32_t n_threads_hashgrid    = 256u;
        static constexpr std::uint32_t n_features_per_thread = std::min(2u, N_FEATURES_PER_LEVEL);

        const dim3 blocks_hashgrid = {((num_elements * N_FEATURES_PER_LEVEL / n_features_per_thread) + n_threads_hashgrid - 1u) / n_threads_hashgrid, n_levels, 1u};
        kernel_grid_backward<T, std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>, N_POS_DIMS, N_FEATURES_PER_LEVEL, n_features_per_thread><<<blocks_hashgrid, n_threads_hashgrid, 0, stream>>>(num_elements, n_features, offset_table, base_resolution, std::log2(per_level_scale), max_level, stochastic_interpolation, grid_type, grid_gradient, input.view(), dL_dy_rm);

        if constexpr (!std::is_same_v<std::conditional_t<N_FEATURES_PER_LEVEL == 1u, float, T>, T>) {
            if (n_params > 0u) {
                const std::uint32_t blocks = (static_cast<std::uint32_t>(n_params) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                network::detail::cast<T><<<blocks, network::detail::n_threads_linear, 0, stream>>>(static_cast<std::uint32_t>(n_params), reinterpret_cast<const float*>(grid_gradient), gradients);
            }
        }
    }

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    void GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL>::initialize_params(legacy::math::pcg32& rng, float* params_full_precision, const float scale) {
        generate_random_uniform<float>(rng, n_params, params_full_precision, -1e-4f * scale, 1e-4f * scale);
    }

    template <typename T, std::uint32_t N_POS_DIMS, std::uint32_t N_FEATURES_PER_LEVEL>
    GridEncodingTemplated<T, N_POS_DIMS, N_FEATURES_PER_LEVEL> make_hash_grid_encoding(const InstantNGP::NetworkConfig::HashGridConfig& config) {
        const GridType grid_type       = config.storage == InstantNGP::GridStorage::Hash ? GridType::Hash : (config.storage == InstantNGP::GridStorage::Dense ? GridType::Dense : (config.storage == InstantNGP::GridStorage::Tiled ? GridType::Tiled : throw std::runtime_error{"Unsupported grid storage mode."}));
        const std::uint32_t n_features = N_FEATURES_PER_LEVEL * config.n_levels;
        const float per_level_scale    = config.per_level_scale.has_value() ? *config.per_level_scale : (grid_type == GridType::Dense && config.n_levels > 1u && config.base_resolution > 0u ? std::exp(std::log(256.0f / static_cast<float>(config.base_resolution)) / (config.n_levels - 1u)) : 2.0f);
        return {n_features, config.log2_hashmap_size, config.base_resolution, per_level_scale, config.stochastic_interpolation, grid_type};
    }

    template <typename T>
    std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>> create_position_encoding(const std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::HashGridConfig& config, const std::uint32_t alignment) {
        if (config.n_levels == 0u) throw std::runtime_error{"HashGrid encoding requires at least one level."};
        if (config.base_resolution == 0u) throw std::runtime_error{"HashGrid encoding base_resolution must be greater than zero."};
        if (n_dims_to_encode != 3u) throw std::runtime_error{"HashGrid encoding in this repository only supports 3D positions."};

        switch (config.n_features_per_level) {
        case 1u:
            {
                auto encoding = make_hash_grid_encoding<T, 3u, 1u>(config);
                if (alignment > 0u) encoding.padded_output_width = legacy::next_multiple(encoding.output_width, std::lcm(alignment, encoding.required_output_alignment));
                return std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{std::move(encoding)};
            }
        case 2u:
            {
                auto encoding = make_hash_grid_encoding<T, 3u, 2u>(config);
                if (alignment > 0u) encoding.padded_output_width = legacy::next_multiple(encoding.output_width, std::lcm(alignment, encoding.required_output_alignment));
                return std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{std::move(encoding)};
            }
        case 4u:
            {
                auto encoding = make_hash_grid_encoding<T, 3u, 4u>(config);
                if (alignment > 0u) encoding.padded_output_width = legacy::next_multiple(encoding.output_width, std::lcm(alignment, encoding.required_output_alignment));
                return std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{std::move(encoding)};
            }
        case 8u:
            {
                auto encoding = make_hash_grid_encoding<T, 3u, 8u>(config);
                if (alignment > 0u) encoding.padded_output_width = legacy::next_multiple(encoding.output_width, std::lcm(alignment, encoding.required_output_alignment));
                return std::variant<GridEncodingTemplated<T, 3u, 1u>, GridEncodingTemplated<T, 3u, 2u>, GridEncodingTemplated<T, 3u, 4u>, GridEncodingTemplated<T, 3u, 8u>>{std::move(encoding)};
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
        for (std::uint32_t j = 0u; j < num_to_pad; ++j) data_out(j) = static_cast<T>(1.0f);

        data_out.advance_rows(num_to_pad);
        sh_enc<T, legacy::MatrixView<T>>(degree, data_in(0u, i) * 2.0f - 1.0f, data_in(1u, i) * 2.0f - 1.0f, data_in(2u, i) * 2.0f - 1.0f, data_out);
    }

    template <typename T>
    SphericalHarmonicsEncoding<T>::SphericalHarmonicsEncoding(const std::uint32_t degree, const std::uint32_t n_dims_to_encode) : degree{degree} {
        output_width        = degree * degree;
        padded_output_width = output_width;

        if (n_dims_to_encode != 3u) throw std::runtime_error{"Can only encode 3D directions in spherical harmonics."};
        if (this->degree <= 0u) throw std::runtime_error{"Spherical harmonics must have positive degree."};
        if (this->degree > 8u) throw std::runtime_error{"Spherical harmonics are only implemented up to degree 8."};
    }

    template <typename T>
    void SphericalHarmonicsEncoding<T>::encode(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<T>& output) {
        legacy::check_or_throw(input.m() == input_width);
        legacy::check_or_throw(output.m() == padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        if (padded_output_width == 0u) return;

        const std::uint32_t num_elements = input.n();
        if (num_elements > 0u) {
            const std::uint32_t blocks = (num_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            kernel_sh<T><<<blocks, network::detail::n_threads_linear, 0, stream>>>(num_elements, degree, padded_output_width - output_width, input.view(), output.view());
        }
    }

    template <typename T>
    SphericalHarmonicsEncoding<T> create_direction_encoding(const std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::DirectionEncodingConfig& config, const std::uint32_t alignment) {
        auto result = SphericalHarmonicsEncoding<T>{config.sh_degree, n_dims_to_encode};
        if (alignment > 0u) result.padded_output_width = legacy::next_multiple(result.output_width, std::lcm(alignment, result.required_output_alignment));
        return result;
    }

    template struct GridEncodingTemplated<__half, 3u, 1u>;
    template struct GridEncodingTemplated<__half, 3u, 2u>;
    template struct GridEncodingTemplated<__half, 3u, 4u>;
    template struct GridEncodingTemplated<__half, 3u, 8u>;
    template std::variant<GridEncodingTemplated<__half, 3u, 1u>, GridEncodingTemplated<__half, 3u, 2u>, GridEncodingTemplated<__half, 3u, 4u>, GridEncodingTemplated<__half, 3u, 8u>> create_position_encoding<__half>(std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::HashGridConfig& config, std::uint32_t alignment);
    template struct SphericalHarmonicsEncoding<__half>;
    template SphericalHarmonicsEncoding<__half> create_direction_encoding<__half>(std::uint32_t n_dims_to_encode, const InstantNGP::NetworkConfig::DirectionEncodingConfig& config, std::uint32_t alignment);

} // namespace ngp::encoding
