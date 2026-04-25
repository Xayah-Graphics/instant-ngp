#pragma once

#include "common.cuh"

namespace ngp {

    constexpr bool supported_fully_fused_width(const std::uint32_t width) {
        return width == 16u || width == 32u || width == 64u || width == 128u;
    }

    inline constexpr std::uint32_t density_network_width = 64u;
    inline constexpr std::uint32_t rgb_network_width     = 64u;

    static_assert(supported_fully_fused_width(density_network_width), "density_network_width must be one of 16, 32, 64, or 128.");
    static_assert(supported_fully_fused_width(rgb_network_width), "rgb_network_width must be one of 16, 32, 64, or 128.");

} // namespace ngp

namespace ngp::mlp {

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

    template <typename T, std::uint32_t WIDTH>
    struct FullyFusedMLP final {
        FullyFusedMLP(std::uint32_t input_width, std::uint32_t output_width, std::uint32_t n_hidden_layers, Activation activation, Activation output_activation);

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
        Activation activation                                           = Activation::None;
        Activation output_activation                                    = Activation::None;
        std::vector<legacy::GPUMatrix<T, legacy::RM>> weight_matrices   = {};
        std::vector<legacy::GPUMatrix<T, legacy::RM>> gradient_matrices = {};
        std::size_t n_params                                            = 0u;
        T* params                                                       = nullptr;
        T* gradients                                                    = nullptr;
    };

} // namespace ngp::mlp
