#pragma once

#include "instant-ngp.h"

namespace ngp {

    struct InstantNGP::Optimizer final {
        void allocate(std::uint32_t n_weights, std::uint32_t n_matrix_weights);
        void step(cudaStream_t stream, float loss_scale, float* weights_full_precision, __half* weights, const __half* gradients);

        std::uint32_t n_weights                      = 0u;
        std::uint32_t n_matrix_weights               = 0u;
        legacy::GpuBuffer<float> first_moments       = {};
        legacy::GpuBuffer<float> second_moments      = {};
        legacy::GpuBuffer<std::uint32_t> param_steps = {};
        float base_learning_rate                     = 1e-3f;
        float beta1                                  = 0.9f;
        float beta2                                  = 0.999f;
        float epsilon                                = 1e-8f;
        float l2_reg                                 = 1e-8f;
    };

} // namespace ngp
