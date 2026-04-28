#ifndef NGP_TRAIN_H
#define NGP_TRAIN_H

#include <cstdint>
#include <type_traits>

#if defined(__CUDACC__)
#define NGP_CUDA_HOST_DEVICE __host__ __device__
#else
#define NGP_CUDA_HOST_DEVICE
#endif

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

    // Training behavior.
    inline constexpr std::uint64_t TRAIN_SEED = 1337u;

    static_assert(GRID_OUTPUT_WIDTH % GRID_FEATURES_PER_LEVEL == 0u);
    static_assert(GRID_OUTPUT_WIDTH == RGB_INPUT_WIDTH);
    static_assert(GRID_PER_LEVEL_SCALE == 2.0f);
    static_assert(DENSITY_HIDDEN_LAYERS == 1u, "The handwritten density MLP path currently supports exactly one hidden layer.");
    static_assert(RGB_HIDDEN_LAYERS == 2u, "The handwritten RGB MLP path currently supports exactly two hidden layers.");
} // namespace ngp::cuda::config

namespace ngp::cuda {
    inline constexpr std::uint64_t PCG32_DEFAULT_STATE  = 0x853c49e6748fea9bULL;
    inline constexpr std::uint64_t PCG32_DEFAULT_STREAM = 0xda3e39cb94b95bdbULL;
    inline constexpr std::uint64_t PCG32_MULT           = 0x5851f42d4c957f2dULL;

    struct Pcg32 final {
        std::uint64_t state = PCG32_DEFAULT_STATE;
        std::uint64_t inc   = PCG32_DEFAULT_STREAM;

        Pcg32() = default;

        NGP_CUDA_HOST_DEVICE explicit Pcg32(const std::uint64_t initstate, const std::uint64_t initseq = 1u) {
            this->seed(initstate, initseq);
        }

        NGP_CUDA_HOST_DEVICE void seed(const std::uint64_t initstate, const std::uint64_t initseq) {
            this->state = 0u;
            this->inc   = (initseq << 1u) | 1u;
            this->next_uint();
            this->state += initstate;
            this->next_uint();
        }

        NGP_CUDA_HOST_DEVICE std::uint32_t next_uint() {
            const std::uint64_t oldstate = this->state;
            this->state                  = oldstate * PCG32_MULT + this->inc;
            const auto xorshifted        = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
            const auto rot               = static_cast<std::uint32_t>(oldstate >> 59u);
            return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
        }

        NGP_CUDA_HOST_DEVICE float next_float() {
            union {
                std::uint32_t bits;
                float value;
            } result    = {};
            result.bits = (this->next_uint() >> 9u) | 0x3f800000u;
            return result.value - 1.0f;
        }

        NGP_CUDA_HOST_DEVICE void advance(std::uint64_t delta) {
            std::uint64_t cur_mult = PCG32_MULT;
            std::uint64_t cur_plus = this->inc;
            std::uint64_t acc_mult = 1u;
            std::uint64_t acc_plus = 0u;

            while (delta > 0u) {
                if ((delta & 1u) != 0u) {
                    acc_mult *= cur_mult;
                    acc_plus = acc_plus * cur_mult + cur_plus;
                }

                cur_plus = (cur_mult + 1u) * cur_plus;
                cur_mult *= cur_mult;
                delta >>= 1u;
            }

            this->state = acc_mult * this->state + acc_plus;
        }
    };

#undef NGP_CUDA_HOST_DEVICE

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
    void update_density_grid_once(const float* camera, std::uint32_t frame_count, std::uint32_t width, std::uint32_t height, float focal_length, std::uint32_t current_step, const std::uint32_t* grid_offsets, const std::uint16_t* params, std::uint32_t density_param_offset, std::uint32_t grid_param_offset, float* sample_coords, std::uint16_t* density_input, std::uint16_t* density_grid_output, float* density_grid_values, float* density_grid_scratch, std::uint32_t* density_grid_indices, float* density_grid_mean, std::uint32_t* density_grid_occupied_count, std::uint8_t* occupancy, std::uint32_t& density_grid_ema_step, Pcg32& density_grid_rng, float& out_elapsed_ms);

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
