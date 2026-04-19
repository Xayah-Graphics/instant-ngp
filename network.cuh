#ifndef NETWORK_CUH
#define NETWORK_CUH

#include "common.cuh"


namespace ngp::mlp {}

namespace ngp::optimizer {

    // template <typename T>
    // __global__ void adam_step(const uint32_t n_elements, const uint32_t n_matrix_weights, const float loss_scale, const float learning_rate, const float beta1, const float beta2, const float epsilon, const float l2_reg, float* __restrict__ weights_full_precision, T* __restrict__ weights, const T* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, uint32_t* __restrict__ param_steps) {
    //     const uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
    //     if (i >= n_elements) {
    //         return;
    //     }
    //
    //     float gradient = (float) gradients[i] / loss_scale;
    //     if (i >= n_matrix_weights && gradient == 0.0f) {
    //         return;
    //     }
    //
    //     const float weight_fp = weights_full_precision[i];
    //     if (i < n_matrix_weights) {
    //         gradient += l2_reg * weight_fp;
    //     }
    //
    //     const float gradient_sq = gradient * gradient;
    //     float first_moment = first_moments[i] = beta1 * first_moments[i] + (1 - beta1) * gradient;
    //     const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1 - beta2) * gradient_sq;
    //
    //     const uint32_t current_step         = ++param_steps[i];
    //     const float corrected_learning_rate = learning_rate * sqrtf(1 - powf(beta2, (float) current_step)) / (1 - powf(beta1, (float) current_step));
    //     const float new_weight              = weight_fp - corrected_learning_rate * first_moment / (sqrtf(second_moment) + epsilon);
    //
    //     weights_full_precision[i] = new_weight;
    //     weights[i]                = (T) new_weight;
    // }
    //
    // template <typename T>
    // class AdamOptimizer {
    // public:
    //     struct Config {
    //         float learning_rate = 1e-2f;
    //         float beta1         = 0.9f;
    //         float beta2         = 0.99f;
    //         float epsilon       = 1e-15f;
    //         float l2_reg        = 1e-6f;
    //     };
    //
    //     explicit AdamOptimizer(const Config& params) {
    //         update_hyperparams(params);
    //     }
    //
    //     void update_hyperparams(const Config& params) {
    //         m_beta1              = params.beta1;
    //         m_beta2              = params.beta2;
    //         m_epsilon            = params.epsilon;
    //         m_base_learning_rate = params.learning_rate;
    //         m_l2_reg             = params.l2_reg;
    //     }
    //
    //     void allocate(uint32_t n_weights, uint32_t n_matrix_weights) {
    //         m_n_weights = n_weights;
    //         if (m_n_weights > m_first_moments.size()) {
    //             m_first_moments.resize(m_n_weights);
    //             m_first_moments.memset(0);
    //
    //             m_second_moments.resize(m_n_weights);
    //             m_second_moments.memset(0);
    //
    //             m_param_steps.resize(m_n_weights);
    //             m_param_steps.memset(0);
    //         }
    //         m_n_matrix_weights = n_matrix_weights;
    //     }
    //
    //     void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, T* weights, const T* gradients) {
    //         if (m_n_weights > 0) {
    //             const uint32_t blocks = (m_n_weights + N_THREADS_LINEAR - 1) / N_THREADS_LINEAR;
    //             adam_step<T><<<blocks, N_THREADS_LINEAR, 0, stream>>>(m_n_weights, m_n_matrix_weights, loss_scale, m_base_learning_rate, m_beta1, m_beta2, m_epsilon, m_l2_reg, weights_full_precision, weights, gradients, m_first_moments.data(), m_second_moments.data(), m_param_steps.data());
    //         }
    //     }
    //
    // private:
    //     uint32_t m_n_weights        = 0;
    //     uint32_t m_n_matrix_weights = 0;
    //
    //     legacy::GpuBuffer<float> m_first_moments;
    //     legacy::GpuBuffer<float> m_second_moments;
    //     legacy::GpuBuffer<uint32_t> m_param_steps;
    //
    //     float m_base_learning_rate = 1e-3f;
    //     float m_beta1              = 0.9f;
    //     float m_beta2              = 0.999f;
    //     float m_epsilon            = 1e-8f;
    //     float m_l2_reg             = 1e-8f;
    // };
} // namespace ngp::optimizer

#endif
