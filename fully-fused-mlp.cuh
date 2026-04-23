#pragma once

#include "network-detail.cuh"

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

namespace ngp::mlp {

    template <typename T, std::uint32_t WIDTH>
    struct FullyFusedMLP final {
        FullyFusedMLP(std::uint32_t input_width, std::uint32_t output_width, std::uint32_t n_hidden_layers, network::detail::Activation activation, network::detail::Activation output_activation);

        struct Scratch {
            std::vector<legacy::GPUMatrixDynamic<T>> forward_hidden  = {};
            std::vector<legacy::GPUMatrixDynamic<T>> backward_hidden = {};
            legacy::GPUMatrixDynamic<T> backward_output              = {};
            legacy::GpuAllocation forward_alloc                      = {};
            legacy::GpuAllocation backward_alloc                     = {};
        };

        void inference(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>& output);
        void prepare_scratch(cudaStream_t stream, std::uint32_t batch_size, legacy::MatrixLayout output_layout, Scratch& scratch);
        void forward(cudaStream_t stream, const legacy::GPUMatrixDynamic<T>& input, legacy::GPUMatrixDynamic<T>* output, Scratch& scratch);
        void backward(cudaStream_t stream, Scratch& scratch, const legacy::GPUMatrixDynamic<T>& input, const legacy::GPUMatrixDynamic<T>& output, const legacy::GPUMatrixDynamic<T>& dL_doutput, legacy::GPUMatrixDynamic<T>* dL_dinput = nullptr, network::detail::GradientMode param_gradients_mode = network::detail::GradientMode::Overwrite);
        void initialize_params(legacy::math::pcg32& rnd, float* params_full_precision, float scale = 1.0f);

        std::uint32_t n_hidden_layers                                   = 0u;
        std::uint32_t n_hidden_matmuls                                  = 0u;
        std::uint32_t input_width                                       = 0u;
        std::uint32_t network_width                                     = WIDTH;
        std::uint32_t output_width                                      = 0u;
        std::uint32_t padded_output_width                               = 0u;
        network::detail::Activation activation                          = network::detail::Activation::None;
        network::detail::Activation output_activation                   = network::detail::Activation::None;
        std::vector<legacy::GPUMatrix<T, legacy::RM>> weight_matrices   = {};
        std::vector<legacy::GPUMatrix<T, legacy::RM>> gradient_matrices = {};
        std::size_t n_params                                            = 0u;
        T* params                                                       = nullptr;
        T* gradients                                                    = nullptr;
    };

} // namespace ngp::mlp
