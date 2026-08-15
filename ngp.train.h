#ifndef NGP_TRAIN_H
#define NGP_TRAIN_H

#include "ngp.train.config.h"
#include <cstdint>
#include <type_traits>

namespace ngp::cuda {
    // Device memory.
    void free_device_buffers(void** pointers, std::size_t count) noexcept;
    void destroy_cublaslt(void*& handle) noexcept;

    template <typename... Pointers>
        requires ((std::is_pointer_v<Pointers> && ...))
    void free_device_buffers(Pointers&... pointers) noexcept {
        if constexpr (sizeof...(Pointers) > 0u) {
            void* raw[] = {const_cast<void*>(static_cast<const void*>(pointers))...};
            free_device_buffers(raw, sizeof...(Pointers));
            ((pointers = nullptr), ...);
        }
    }

    // Dataset.
    void upload_dataset(const std::uint8_t* pixels, std::size_t pixels_bytes, const float* camera, std::size_t camera_count, const std::uint8_t*& out_pixels, const float*& out_camera);

    // Sampler.
    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy);
    void set_occupancy_grid_full(std::uint8_t* occupancy, std::uint32_t* occupancy_grid_occupied_count);
    void sample_training_batch(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint64_t seed, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);
    void allocate_density_grid_buffers(float*& out_density_grid_values, float*& out_density_grid_scratch, std::uint32_t*& out_density_grid_indices, float*& out_density_grid_mean, std::uint32_t*& out_occupancy_grid_occupied_count);
    bool update_density_grid_values(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint64_t seed, std::uint32_t current_step, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* density_grid_output, float* density_grid_values, float* density_grid_scratch, std::uint32_t* density_grid_indices, std::uint32_t& density_grid_ema_step, bool reset_density_grid);
    void update_occupancy_grid_from_density_grid(const float* density_grid_values, float* density_grid_mean, std::uint32_t* occupancy_grid_occupied_count, std::uint8_t* occupancy);

    // Loss and compaction.
    void allocate_training_loss_buffers(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values);
    void compute_training_loss_and_compact_samples(std::uint32_t rays_per_batch, std::uint64_t seed, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, const std::uint16_t* network_output, std::uint32_t* compacted_sample_counter, const std::uint32_t* ray_indices, const float* rays, std::uint32_t* numsteps, const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, float* loss_values);
    void pad_compacted_training_batch(const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);

    // Evaluation.
    void allocate_evaluation_buffers(std::uint32_t*& out_evaluation_numsteps, std::uint32_t*& out_evaluation_sample_counter, std::uint32_t*& out_evaluation_overflow_counter, double*& out_evaluation_loss_sum);
    void allocate_evaluation_comparison_buffer(std::uint32_t width, std::uint32_t height, std::uint8_t*& out_comparison_pixels);
    void run_evaluation(const std::uint8_t* evaluation_pixels, const float* evaluation_camera, std::uint32_t evaluation_frame_count, std::uint32_t evaluation_image_begin, std::uint32_t evaluation_image_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, const std::uint8_t* occupancy, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, std::uint32_t* evaluation_numsteps, std::uint32_t* evaluation_sample_counter, std::uint32_t* evaluation_overflow_counter, double* evaluation_loss_sum, std::uint8_t* comparison_pixels, std::uint8_t* host_comparison_pixels, double& out_loss_sum);

    // Network buffers and parameters.
    void allocate_network_buffers(std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden, std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace);
    void allocate_trainable_parameter_buffers(float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients);
    void initialize_mlp_parameters(std::uint64_t seed, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
    void initialize_grid_parameters(std::uint64_t seed, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
    void download_trainable_parameters(const float* params_full_precision, float* out_params_full_precision);
    void upload_trainable_parameters(const float* params_full_precision, float* out_params_full_precision, std::uint16_t* out_params, std::uint16_t* out_param_gradients, float* optimizer_first_moments, float* optimizer_second_moments, std::uint32_t* optimizer_param_steps);

    // Network execution.
    void evaluate_network(std::uint32_t sample_count, const float* sample_coords, const std::uint16_t* params, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output);
    void forward_network(const float* sample_coords, const std::uint16_t* params, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* density_forward_hidden, std::uint16_t* rgb_forward_hidden, std::uint16_t* network_output);
    void backward_network(const float* sample_coords, const std::uint16_t* params, std::uint16_t* gradients, const std::uint16_t* density_input, const std::uint16_t* rgb_input, const std::uint16_t* density_forward_hidden, const std::uint16_t* rgb_forward_hidden, const std::uint16_t* network_output, const std::uint16_t* network_output_gradients, std::uint16_t* rgb_output_gradients, std::uint16_t* rgb_input_gradients, std::uint16_t* density_input_gradients, std::uint16_t* density_backward_hidden, std::uint16_t* rgb_backward_hidden, void* cublaslt_handle, std::uint8_t* cublaslt_workspace);

    // Optimizer.
    void allocate_adam_state(float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps);
    void step_optimizer(float* params_full_precision, std::uint16_t* params, const std::uint16_t* gradients, float* first_moments, float* second_moments, std::uint32_t* param_steps);

    // Host readback.
    void read_counter(const std::uint32_t* counter, std::uint32_t& out_value);
    void read_loss_sum(const float* loss_values, std::uint32_t loss_count, float& out_loss_sum);
} // namespace ngp::cuda

#endif // NGP_TRAIN_H
