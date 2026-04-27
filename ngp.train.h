#ifndef NGP_TRAIN_H
#define NGP_TRAIN_H

#include <cstdint>
#include <string>
#include <type_traits>

namespace ngp::cuda::config {
    // Grid encoding.
    inline constexpr std::uint32_t GRID_N_LEVELS           = 8u;
    inline constexpr std::uint32_t GRID_FEATURES_PER_LEVEL = 4u;
    inline constexpr std::uint32_t GRID_OUTPUT_WIDTH       = GRID_N_LEVELS * GRID_FEATURES_PER_LEVEL;
    inline constexpr std::uint32_t GRID_BASE_RESOLUTION    = 16u;
    inline constexpr std::uint32_t GRID_LOG2_HASHMAP_SIZE  = 19u;
    inline constexpr std::uint32_t GRID_OFFSET_COUNT       = GRID_N_LEVELS + 1u;
    inline constexpr float GRID_PER_LEVEL_SCALE            = 2.0f;

    // Fully fused MLP shape.
    inline constexpr std::uint32_t MLP_WIDTH              = 64u;
    inline constexpr std::uint32_t DENSITY_HIDDEN_LAYERS  = 1u;
    inline constexpr std::uint32_t RGB_HIDDEN_LAYERS      = 2u;
    inline constexpr std::uint32_t DENSITY_OUTPUT_WIDTH   = 16u;
    inline constexpr std::uint32_t DIRECTION_OUTPUT_WIDTH = 16u;
    inline constexpr std::uint32_t RGB_INPUT_WIDTH        = DENSITY_OUTPUT_WIDTH + DIRECTION_OUTPUT_WIDTH;
    inline constexpr std::uint32_t NETWORK_OUTPUT_WIDTH   = 16u;
    inline constexpr std::uint32_t MLP_INPUT_WIDTH        = GRID_OUTPUT_WIDTH;
    inline constexpr std::uint32_t MLP_OUTPUT_WIDTH       = NETWORK_OUTPUT_WIDTH;

    // Training batch shape.
    inline constexpr std::uint32_t NETWORK_BATCH_SIZE        = 1u << 18u;
    inline constexpr std::uint32_t NETWORK_BATCH_GRANULARITY = 16u * 8u;

    static_assert(GRID_OUTPUT_WIDTH % GRID_FEATURES_PER_LEVEL == 0u);
    static_assert(GRID_OUTPUT_WIDTH == MLP_INPUT_WIDTH);
    static_assert(RGB_INPUT_WIDTH == MLP_INPUT_WIDTH);
    static_assert(NETWORK_OUTPUT_WIDTH == MLP_OUTPUT_WIDTH);

    constexpr std::uint32_t round_up(const std::uint32_t value, const std::uint32_t granularity) {
        return ((value + granularity - 1u) / granularity) * granularity;
    }
} // namespace ngp::cuda::config

namespace ngp::cuda {
    // Device memory.
    void free_device_data(void** pointers, std::size_t count) noexcept;

    template <typename... Pointers>
        requires ((std::is_pointer_v<Pointers> && ...))
    void free_device_data(Pointers&... pointers) noexcept {
        if constexpr (sizeof...(Pointers) > 0u) {
            void* raw[] = {const_cast<void*>(static_cast<const void*>(pointers))...};
            free_device_data(raw, sizeof...(Pointers));
            ((pointers = nullptr), ...);
        }
    }

    // Dataset.
    std::string copy_dataset_to_device_once(const std::uint8_t* pixels, std::size_t pixels_bytes, const float* camera, std::size_t camera_count, const std::uint8_t*& out_pixels, const float*& out_camera);

    // Sampler.
    std::string allocate_sampler_once(std::uint32_t rays_per_batch, std::uint32_t max_samples, float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy);
    std::string sample_training_batch(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_length, std::uint32_t current_step, std::uint64_t seed, std::uint32_t rays_per_batch, std::uint32_t max_samples, bool snap_to_pixel_centers, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);

    // Loss and compaction.
    std::string allocate_training_loss_once(std::uint32_t batch_size, std::uint32_t rays_per_batch, std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values);
    std::string compute_loss_and_compact_once(std::uint32_t rays_per_batch, std::uint32_t batch_size, std::uint64_t seed, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, bool snap_to_pixel_centers, const std::uint16_t* network_output, std::uint32_t* compacted_sample_counter, const std::uint32_t* ray_indices, const float* rays, std::uint32_t* numsteps,
        const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, float* loss_values);
    std::string fill_rollover_once(std::uint32_t batch_size, const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);

    // Network buffers and parameters.
    std::string allocate_network_once(std::uint32_t batch_size, std::uint32_t max_samples, std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden,
        std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, std::uint8_t*& out_cutlass_workspace);
    std::string allocate_trainable_params_once(std::uint32_t param_count, float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients);
    std::string initialize_mlp_params_once(std::uint64_t seed, std::uint32_t density_input_width, std::uint32_t density_output_width, std::uint32_t density_hidden_layers, std::uint32_t density_param_offset, std::uint32_t rgb_input_width, std::uint32_t rgb_output_width, std::uint32_t rgb_hidden_layers, std::uint32_t rgb_param_offset, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
    std::string initialize_grid_params_once(std::uint32_t param_count, std::uint64_t seed, std::uint64_t rng_offset, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);

    // Encoding and network execution.
    std::string encode_grid_forward(std::uint32_t sample_count, const float* sample_coords, const std::uint32_t* grid_offsets, std::uint32_t grid_n_levels, std::uint32_t grid_features_per_level, std::uint32_t grid_base_resolution, float grid_per_level_scale, const std::uint16_t* grid_params, std::uint16_t* encoded_positions);
    std::string encode_grid_backward(std::uint32_t sample_count, const float* sample_coords, const std::uint32_t* grid_offsets, std::uint32_t grid_n_levels, std::uint32_t grid_features_per_level, std::uint32_t grid_base_resolution, float grid_per_level_scale, const std::uint16_t* encoded_position_gradients, std::uint16_t* grid_param_gradients);
    std::string network_inference_once(
        std::uint32_t sample_count, std::uint32_t batch_size, const float* sample_coords, const std::uint32_t* grid_offsets, std::uint32_t grid_n_levels, std::uint32_t grid_features_per_level, std::uint32_t grid_base_resolution, float grid_per_level_scale, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output);
    std::string network_forward_once(std::uint32_t batch_size, const float* sample_coords, const std::uint32_t* grid_offsets, std::uint32_t grid_n_levels, std::uint32_t grid_features_per_level, std::uint32_t grid_base_resolution, float grid_per_level_scale, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, std::uint16_t* density_input, std::uint16_t* rgb_input,
        std::uint16_t* density_forward_hidden, std::uint16_t* rgb_forward_hidden, std::uint16_t* network_output);
    std::string network_backward_once(std::uint32_t batch_size, const float* sample_coords, const std::uint32_t* grid_offsets, std::uint32_t grid_n_levels, std::uint32_t grid_features_per_level, std::uint32_t grid_base_resolution, float grid_per_level_scale, const std::uint16_t* params, std::uint16_t* gradients, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, const std::uint16_t* density_input,
        const std::uint16_t* rgb_input, const std::uint16_t* density_forward_hidden, const std::uint16_t* rgb_forward_hidden, const std::uint16_t* network_output, const std::uint16_t* network_output_gradients, std::uint16_t* rgb_output_gradients, std::uint16_t* rgb_input_gradients, std::uint16_t* density_input_gradients, std::uint16_t* density_backward_hidden, std::uint16_t* rgb_backward_hidden, std::uint8_t* cutlass_workspace);

    // Optimizer.
    std::string allocate_adam_state_once(std::uint32_t param_count, float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps);
    std::string optimize(std::uint32_t param_count, std::uint32_t mlp_param_count, float loss_scale, float learning_rate, float beta1, float beta2, float epsilon, float l2_reg, float* params_full_precision, std::uint16_t* params, const std::uint16_t* gradients, float* first_moments, float* second_moments, std::uint32_t* param_steps);

    // Host readback.
    std::string read_counter_once(const std::uint32_t* counter, std::uint32_t& out_value);
    std::string read_loss_sum_once(const float* loss_values, std::uint32_t loss_count, float& out_loss_sum);
} // namespace ngp::cuda

#endif // NGP_TRAIN_H
