#pragma once

#include "common.cuh"

namespace ngp {

    inline constexpr std::uint32_t density_network_width = 64u;
    inline constexpr std::uint32_t rgb_network_width     = 64u;

    static_assert(density_network_width == 16u || density_network_width == 32u || density_network_width == 64u || density_network_width == 128u, "density_network_width must be one of 16, 32, 64, or 128.");
    static_assert(rgb_network_width == 16u || rgb_network_width == 32u || rgb_network_width == 64u || rgb_network_width == 128u, "rgb_network_width must be one of 16, 32, 64, or 128.");

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
            std::vector<legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>> forward_hidden  = {};
            std::vector<legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>> backward_hidden = {};
            legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic> backward_output              = {};
            legacy::GpuAllocation forward_alloc                      = {};
            legacy::GpuAllocation backward_alloc                     = {};
        };

        void inference(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& output);
        void prepare_scratch(cudaStream_t stream, std::uint32_t batch_size, legacy::MatrixLayout output_layout, Scratch& scratch);
        void forward(cudaStream_t stream, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* output, Scratch& scratch);
        void backward(cudaStream_t stream, const cudaStream_t* aux_streams, const cudaEvent_t* aux_events, std::uint32_t n_aux_streams, Scratch& scratch, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& input, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& output, const legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>& dL_doutput, legacy::GPUMatrix<T, legacy::MatrixLayout::Dynamic>* dL_dinput, network::GradientMode param_gradients_mode);
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
