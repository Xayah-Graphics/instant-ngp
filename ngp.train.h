#ifndef NGP_TRAIN_H
#define NGP_TRAIN_H

#include <cstdint>
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
    inline constexpr float GRID_LOG2_PER_LEVEL_SCALE       = 1.0f;

    // Fully fused MLP shape.
    inline constexpr std::uint32_t MLP_WIDTH              = 64u;
    inline constexpr std::uint32_t DENSITY_HIDDEN_LAYERS  = 1u;
    inline constexpr std::uint32_t RGB_HIDDEN_LAYERS      = 2u;
    inline constexpr std::uint32_t DENSITY_OUTPUT_WIDTH   = 16u;
    inline constexpr std::uint32_t DIRECTION_OUTPUT_WIDTH = 16u;
    inline constexpr std::uint32_t RGB_INPUT_WIDTH        = DENSITY_OUTPUT_WIDTH + DIRECTION_OUTPUT_WIDTH;
    inline constexpr std::uint32_t NETWORK_OUTPUT_WIDTH   = 16u;

    // Training batch shape.
    inline constexpr std::uint32_t NETWORK_BATCH_SIZE               = 1u << 18u;
    inline constexpr std::uint32_t NETWORK_BATCH_GRANULARITY        = 16u * 8u;
    inline constexpr std::uint32_t INITIAL_RAYS_PER_BATCH           = 1u << 12u;
    inline constexpr std::uint32_t MAX_SAMPLES_PER_BATCH_MULTIPLIER = 16u;
    inline constexpr std::uint32_t MAX_SAMPLES                      = NETWORK_BATCH_SIZE * MAX_SAMPLES_PER_BATCH_MULTIPLIER;

    static_assert(GRID_OUTPUT_WIDTH % GRID_FEATURES_PER_LEVEL == 0u);
    static_assert(GRID_OUTPUT_WIDTH == RGB_INPUT_WIDTH);
    static_assert(GRID_PER_LEVEL_SCALE == 2.0f);
    static_assert(DENSITY_HIDDEN_LAYERS == 1u, "The handwritten density MLP path currently supports exactly one hidden layer.");
    static_assert(RGB_HIDDEN_LAYERS == 2u, "The handwritten RGB MLP path currently supports exactly two hidden layers.");
} // namespace ngp::cuda::config

namespace ngp::cuda {
    // Device memory.
    void free_device_data(void** pointers, std::size_t count) noexcept;
    void destroy_cublaslt_once(void*& handle) noexcept;

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
    void copy_dataset_to_device_once(const std::uint8_t* pixels, std::size_t pixels_bytes, const float* camera, std::size_t camera_count, const std::uint8_t*& out_pixels, const float*& out_camera);

    // Sampler.
    void allocate_sampler_once(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy);
    void sample_training_batch(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_length, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);
    void allocate_density_grid_once(float*& out_density_grid_values, float*& out_density_grid_scratch, std::uint32_t*& out_density_grid_indices, float*& out_density_grid_mean, std::uint32_t*& out_density_grid_occupied_count);
    void update_density_grid_once(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_length, std::uint32_t current_step, const std::uint32_t* grid_offsets, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t grid_param_offset, float* sample_coords, std::uint16_t* density_input, std::uint16_t* network_output, float* density_grid_values, float* density_grid_scratch, std::uint32_t* density_grid_indices, float* density_grid_mean, std::uint32_t* density_grid_occupied_count, std::uint8_t* occupancy, std::uint32_t& density_grid_ema_step, float& out_elapsed_ms);

    // Loss and compaction.
    void allocate_training_loss_once(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values);
    void compute_loss_and_compact_once(std::uint32_t rays_per_batch, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, const std::uint16_t* network_output, std::uint32_t* compacted_sample_counter, const std::uint32_t* ray_indices, const float* rays, std::uint32_t* numsteps, const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, float* loss_values);
    void fill_rollover_once(const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);

    // Network buffers and parameters.
    void allocate_network_once(std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden, std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace);
    void allocate_trainable_params_once(std::uint32_t param_count, float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients);
    void initialize_mlp_params_once(std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
    void initialize_grid_params_once(std::uint32_t param_count, std::uint64_t rng_offset, float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);

    // Network execution.
    void network_inference_once(std::uint32_t sample_count, const float* sample_coords, const std::uint32_t* grid_offsets, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output);
    void network_forward_once(const float* sample_coords, const std::uint32_t* grid_offsets, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* density_forward_hidden, std::uint16_t* rgb_forward_hidden, std::uint16_t* network_output);
    void network_backward_once(const float* sample_coords, const std::uint32_t* grid_offsets, const std::uint16_t* params, std::uint16_t* gradients, std::uint32_t density_param_offset, std::uint32_t rgb_param_offset, std::uint32_t grid_param_offset, const std::uint16_t* density_input, const std::uint16_t* rgb_input, const std::uint16_t* density_forward_hidden, const std::uint16_t* rgb_forward_hidden, const std::uint16_t* network_output, const std::uint16_t* network_output_gradients, std::uint16_t* rgb_output_gradients, std::uint16_t* rgb_input_gradients, std::uint16_t* density_input_gradients, std::uint16_t* density_backward_hidden, std::uint16_t* rgb_backward_hidden, void* cublaslt_handle, std::uint8_t* cublaslt_workspace);

    // Optimizer.
    void allocate_adam_state_once(std::uint32_t param_count, float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps);
    void optimize(std::uint32_t param_count, std::uint32_t mlp_param_count, float* params_full_precision, std::uint16_t* params, const std::uint16_t* gradients, float* first_moments, float* second_moments, std::uint32_t* param_steps);

    // Host readback.
    void read_counter_once(const std::uint32_t* counter, std::uint32_t& out_value);
    void read_loss_sum_once(const float* loss_values, std::uint32_t loss_count, float& out_loss_sum);
} // namespace ngp::cuda

#endif // NGP_TRAIN_H
