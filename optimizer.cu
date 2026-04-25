#include "optimizer.cuh"

namespace ngp {

    template <typename T>
    __global__ void adam_step(const std::uint32_t n_elements, const std::uint32_t n_matrix_weights, const float loss_scale, const float learning_rate, const float beta1, const float beta2, const float epsilon, const float l2_reg, float* __restrict__ weights_full_precision, T* __restrict__ weights, const T* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        float gradient = static_cast<float>(gradients[i]) / loss_scale;
        if (i >= n_matrix_weights && gradient == 0.0f) return;

        const float weight_fp = weights_full_precision[i];
        if (i < n_matrix_weights) gradient += l2_reg * weight_fp;

        const float gradient_sq               = gradient * gradient;
        const float first_moment              = first_moments[i] = beta1 * first_moments[i] + (1.0f - beta1) * gradient;
        const float second_moment             = second_moments[i] = beta2 * second_moments[i] + (1.0f - beta2) * gradient_sq;
        const std::uint32_t current_step      = ++param_steps[i];
        const float corrected_learning_rate   = learning_rate * sqrtf(1.0f - powf(beta2, static_cast<float>(current_step))) / (1.0f - powf(beta1, static_cast<float>(current_step)));
        const float new_weight                = weight_fp - corrected_learning_rate * first_moment / (sqrtf(second_moment) + epsilon);

        weights_full_precision[i] = new_weight;
        weights[i]                = static_cast<T>(new_weight);
    }

    void InstantNGP::Optimizer::allocate(const std::uint32_t n_weights, const std::uint32_t n_matrix_weights) {
        this->n_weights = n_weights;
        if (this->n_weights > first_moments.size()) {
            first_moments.resize(this->n_weights);
            first_moments.memset(0);

            second_moments.resize(this->n_weights);
            second_moments.memset(0);

            param_steps.resize(this->n_weights);
            param_steps.memset(0);
        }

        this->n_matrix_weights = n_matrix_weights;
    }

    void InstantNGP::Optimizer::step(cudaStream_t stream, const float loss_scale, float* weights_full_precision, __half* weights, const __half* gradients) {
        if (n_weights == 0u) return;

        const std::uint32_t blocks = (n_weights + network::n_threads_linear - 1u) / network::n_threads_linear;
        adam_step<__half><<<blocks, network::n_threads_linear, 0, stream>>>(n_weights, n_matrix_weights, loss_scale, base_learning_rate, beta1, beta2, epsilon, l2_reg, weights_full_precision, weights, gradients, first_moments.data(), second_moments.data(), param_steps.data());
    }

} // namespace ngp
