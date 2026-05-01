#ifndef NGP_TRAIN_H
#define NGP_TRAIN_H

#include <array>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace ngp::cuda::config {
    // Grid encoding.
    inline constexpr std::uint32_t GRID_N_LEVELS           = 8u;
    inline constexpr std::uint32_t GRID_FEATURES_PER_LEVEL = 4u;
    inline constexpr std::uint32_t GRID_OUTPUT_WIDTH       = GRID_N_LEVELS * GRID_FEATURES_PER_LEVEL;
    inline constexpr std::uint32_t GRID_BASE_RESOLUTION    = 16u;
    inline constexpr std::uint32_t GRID_LOG2_HASHMAP_SIZE  = 19u;

    // Fully fused MLP shape.
    inline constexpr std::uint32_t MLP_WIDTH              = 64u;
    inline constexpr std::uint32_t DENSITY_HIDDEN_LAYERS  = 1u;
    inline constexpr std::uint32_t RGB_HIDDEN_LAYERS      = 2u;
    inline constexpr std::uint32_t DENSITY_OUTPUT_WIDTH   = 16u;
    inline constexpr std::uint32_t DIRECTION_OUTPUT_WIDTH = 16u;
    inline constexpr std::uint32_t RGB_INPUT_WIDTH        = DENSITY_OUTPUT_WIDTH + DIRECTION_OUTPUT_WIDTH;
    inline constexpr std::uint32_t NETWORK_OUTPUT_WIDTH   = 16u;

    struct NetworkParameterLayout final {
        std::array<std::uint32_t, GRID_N_LEVELS + 1u> grid_offsets = {};
        std::uint32_t density_param_offset                         = 0u;
        std::uint32_t density_input_weight_offset                  = 0u;
        std::uint32_t density_output_weight_offset                 = 0u;
        std::uint32_t density_param_count                          = 0u;
        std::uint32_t rgb_param_offset                             = 0u;
        std::uint32_t rgb_input_weight_offset                      = 0u;
        std::uint32_t rgb_hidden_weight_offset                     = 0u;
        std::uint32_t rgb_output_weight_offset                     = 0u;
        std::uint32_t rgb_param_count                              = 0u;
        std::uint32_t mlp_param_count                              = 0u;
        std::uint32_t grid_param_offset                            = 0u;
        std::uint32_t grid_param_count                             = 0u;
        std::uint32_t total_param_count                            = 0u;
    };

    constexpr NetworkParameterLayout make_network_parameter_layout() {
        NetworkParameterLayout layout              = {};
        std::uint32_t grid_cursor                  = 0u;
        constexpr std::uint32_t grid_hashmap_size  = 1u << GRID_LOG2_HASHMAP_SIZE;
        constexpr std::uint64_t grid_max_positions = std::numeric_limits<std::uint32_t>::max() / 2ull;

        for (std::uint32_t level = 0u; level < GRID_N_LEVELS; ++level) {
            const std::uint32_t resolution = GRID_BASE_RESOLUTION << level;
            const std::uint64_t dense      = static_cast<std::uint64_t>(resolution) * resolution * resolution;
            std::uint64_t positions        = dense > grid_max_positions ? grid_max_positions : dense;
            positions                      = ((positions + 7u) / 8u) * 8u;
            if (positions > grid_hashmap_size) positions = grid_hashmap_size;
            layout.grid_offsets[level] = grid_cursor;
            grid_cursor += static_cast<std::uint32_t>(positions);
        }

        layout.grid_offsets[GRID_N_LEVELS] = grid_cursor;

        std::uint32_t param_cursor         = 0u;
        layout.density_param_offset        = param_cursor;
        layout.density_input_weight_offset = param_cursor;
        param_cursor += MLP_WIDTH * GRID_OUTPUT_WIDTH;
        layout.density_output_weight_offset = param_cursor;
        param_cursor += DENSITY_OUTPUT_WIDTH * MLP_WIDTH;
        layout.density_param_count = param_cursor - layout.density_param_offset;

        layout.rgb_param_offset        = param_cursor;
        layout.rgb_input_weight_offset = param_cursor;
        param_cursor += MLP_WIDTH * RGB_INPUT_WIDTH;
        layout.rgb_hidden_weight_offset = param_cursor;
        param_cursor += MLP_WIDTH * MLP_WIDTH;
        layout.rgb_output_weight_offset = param_cursor;
        param_cursor += NETWORK_OUTPUT_WIDTH * MLP_WIDTH;
        layout.rgb_param_count = param_cursor - layout.rgb_param_offset;

        layout.mlp_param_count   = param_cursor;
        layout.grid_param_offset = param_cursor;
        layout.grid_param_count  = layout.grid_offsets[GRID_N_LEVELS] * GRID_FEATURES_PER_LEVEL;
        param_cursor += layout.grid_param_count;
        layout.total_param_count = param_cursor;
        return layout;
    }

    inline constexpr NetworkParameterLayout NETWORK_PARAMETER_LAYOUT = make_network_parameter_layout();

    // Training batch shape.
    inline constexpr std::uint32_t NETWORK_BATCH_SIZE        = 1u << 18u;
    inline constexpr std::uint32_t NETWORK_BATCH_GRANULARITY = 16u * 8u;
    inline constexpr std::uint32_t INITIAL_RAYS_PER_BATCH    = 1u << 12u;
    inline constexpr std::uint32_t MAX_SAMPLES               = NETWORK_BATCH_SIZE * 16u;

    static_assert(GRID_OUTPUT_WIDTH % GRID_FEATURES_PER_LEVEL == 0u);
    static_assert(GRID_OUTPUT_WIDTH == RGB_INPUT_WIDTH);
    static_assert(DENSITY_HIDDEN_LAYERS == 1u, "The handwritten density MLP path currently supports exactly one hidden layer.");
    static_assert(RGB_HIDDEN_LAYERS == 2u, "The handwritten RGB MLP path currently supports exactly two hidden layers.");
    static_assert(NETWORK_PARAMETER_LAYOUT.grid_offsets[0u] == 0u);
    static_assert(NETWORK_PARAMETER_LAYOUT.density_param_offset == 0u);
    static_assert(NETWORK_PARAMETER_LAYOUT.rgb_param_offset == NETWORK_PARAMETER_LAYOUT.density_param_offset + NETWORK_PARAMETER_LAYOUT.density_param_count);
    static_assert(NETWORK_PARAMETER_LAYOUT.mlp_param_count == NETWORK_PARAMETER_LAYOUT.rgb_param_offset + NETWORK_PARAMETER_LAYOUT.rgb_param_count);
    static_assert(NETWORK_PARAMETER_LAYOUT.grid_param_offset == NETWORK_PARAMETER_LAYOUT.mlp_param_count);
    static_assert(NETWORK_PARAMETER_LAYOUT.grid_param_count == NETWORK_PARAMETER_LAYOUT.grid_offsets[GRID_N_LEVELS] * GRID_FEATURES_PER_LEVEL);
    static_assert(NETWORK_PARAMETER_LAYOUT.total_param_count == NETWORK_PARAMETER_LAYOUT.grid_param_offset + NETWORK_PARAMETER_LAYOUT.grid_param_count);
} // namespace ngp::cuda::config

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
    void sample_training_batch(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint32_t current_step, std::uint32_t rays_per_batch, std::uint32_t sample_limit, const std::uint8_t* occupancy, float* sample_coords, float* rays, std::uint32_t* ray_indices, std::uint32_t* numsteps, std::uint32_t* ray_counter, std::uint32_t* sample_counter);
    void allocate_density_grid_buffers(float*& out_density_grid_values, float*& out_density_grid_scratch, std::uint32_t*& out_density_grid_indices, float*& out_density_grid_mean, std::uint32_t*& out_density_grid_occupied_count);
    void update_density_grid(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, std::uint32_t current_step, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* density_grid_output, float* density_grid_values, float* density_grid_scratch, std::uint32_t* density_grid_indices, float* density_grid_mean, std::uint32_t* density_grid_occupied_count, std::uint8_t* occupancy, std::uint32_t& density_grid_ema_step, float& out_elapsed_ms);

    // Loss and compaction.
    void allocate_training_loss_buffers(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values);
    void compute_training_loss_and_compact_samples(std::uint32_t rays_per_batch, std::uint32_t current_step, const std::uint32_t* ray_counter, const std::uint8_t* pixels, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, const std::uint16_t* network_output, std::uint32_t* compacted_sample_counter, const std::uint32_t* ray_indices, const float* rays, std::uint32_t* numsteps, const float* sample_coords, float* compacted_sample_coords, std::uint16_t* network_output_gradients, float* loss_values);
    void pad_compacted_training_batch(const std::uint32_t* compacted_sample_counter, float* compacted_sample_coords, std::uint16_t* network_output_gradients);

    // Evaluation.
    void allocate_evaluation_buffers(std::uint32_t*& out_evaluation_numsteps, std::uint32_t*& out_evaluation_sample_counter, std::uint32_t*& out_evaluation_overflow_counter, double*& out_evaluation_loss_sum);
    void allocate_test_comparison_buffer(std::uint32_t width, std::uint32_t height, std::uint8_t*& out_test_comparison_pixels);
    void run_evaluation(const std::uint8_t* evaluation_pixels, const float* evaluation_camera, std::uint32_t evaluation_frame_count, std::uint32_t evaluation_image_begin, std::uint32_t evaluation_image_count, std::uint32_t width, std::uint32_t height, float focal_x, float focal_y, float principal_x, float principal_y, const std::uint8_t* occupancy, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, std::uint32_t* evaluation_numsteps, std::uint32_t* evaluation_sample_counter, std::uint32_t* evaluation_overflow_counter, double* evaluation_loss_sum, std::uint8_t* test_comparison_pixels, std::uint8_t* host_test_comparison_pixels, double& out_loss_sum);

    // Network buffers and parameters.
    void allocate_network_buffers(std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden, std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace);
    void allocate_trainable_parameter_buffers(float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients);
    void initialize_mlp_parameters(float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
    void initialize_grid_parameters(float* params_full_precision, std::uint16_t* params, std::uint16_t* param_gradients);
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
