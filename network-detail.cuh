#pragma once

#include "instant-ngp.h"
#include <memory>
#include <stack>
#include <unordered_map>

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

    inline constexpr std::uint32_t batch_size_granularity = 256u;
    inline constexpr std::uint32_t n_threads_linear       = 128u;

    template <typename T>
    __global__ void cast(const std::uint32_t num_elements, const float* __restrict__ full_precision, T* __restrict__ target) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;
        target[i] = static_cast<T>(full_precision[i]);
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

    template <typename RNG>
    __device__ legacy::math::vec3 random_val_3d(RNG& rng) {
        return {rng.next_float(), rng.next_float(), rng.next_float()};
    }

    inline Activation activation_from_config(const InstantNGP::ActivationMode activation) {
        switch (activation) {
        case InstantNGP::ActivationMode::None: return Activation::None;
        case InstantNGP::ActivationMode::ReLU: return Activation::ReLU;
        case InstantNGP::ActivationMode::Exponential: return Activation::Exponential;
        case InstantNGP::ActivationMode::Sigmoid: return Activation::Sigmoid;
        case InstantNGP::ActivationMode::Squareplus: return Activation::Squareplus;
        case InstantNGP::ActivationMode::Softplus: return Activation::Softplus;
        case InstantNGP::ActivationMode::Tanh: return Activation::Tanh;
        case InstantNGP::ActivationMode::LeakyReLU: return Activation::LeakyReLU;
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
    std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot>>>& aux_stream_pools();
    void free_aux_stream_pool(cudaStream_t parent_stream);

    struct SyncedStreamReservation final {
        SyncedStreamReservation() = default;
        SyncedStreamReservation(cudaStream_t stream, std::size_t n_streams);
        ~SyncedStreamReservation();
        SyncedStreamReservation& operator=(const SyncedStreamReservation&) = delete;
        SyncedStreamReservation(const SyncedStreamReservation&)            = delete;
        SyncedStreamReservation& operator=(SyncedStreamReservation&& other) noexcept;
        SyncedStreamReservation(SyncedStreamReservation&& other) noexcept;

        AuxStreamSlot* aux_stream_slot = nullptr;
        cudaStream_t aux_stream        = nullptr;
        cudaStream_t main_stream       = nullptr;
    };

} // namespace ngp::network::detail

