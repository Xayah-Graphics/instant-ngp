module;
#include "ngp.train.h"
export module ngp.train;
import std;

namespace ngp::train {
    namespace config = cuda::config;

    template <typename T>
    concept ByteBufferLike = requires(T& buffer) {
        { buffer.data() };
        { buffer.size() } -> std::convertible_to<std::size_t>;
        requires std::same_as<std::remove_cv_t<std::remove_pointer_t<decltype(buffer.data())>>, std::uint8_t>;
    };

    template <typename T>
    concept CameraLike = requires(T& camera) {
        { camera[0] } -> std::convertible_to<float>;
        { camera[11] } -> std::convertible_to<float>;
    };

    template <typename T>
    concept FrameLike = requires(T& frame) {
        requires ByteBufferLike<std::remove_cvref_t<decltype(frame.rgba)>>;
        requires CameraLike<std::remove_cvref_t<decltype(frame.camera)>>;
        { frame.width } -> std::convertible_to<std::uint32_t>;
        { frame.height } -> std::convertible_to<std::uint32_t>;
        { frame.focal_length } -> std::convertible_to<float>;
    };

    template <typename T>
    concept FrameRangeLike = std::ranges::range<T> && FrameLike<std::ranges::range_value_t<T>>;

    template <typename T>
    concept NerfDatasetLike = requires(T& dataset) {
        requires FrameRangeLike<std::remove_cvref_t<decltype(dataset.train)>>;
        requires FrameRangeLike<std::remove_cvref_t<decltype(dataset.validation)>>;
        requires FrameRangeLike<std::remove_cvref_t<decltype(dataset.test)>>;
    };

    export struct TrainStats final {
        std::uint32_t step                                    = 0u;
        std::uint32_t rays_per_batch                          = 0u;
        std::uint32_t measured_sample_count_before_compaction = 0u;
        std::uint32_t measured_sample_count                   = 0u;
        std::uint32_t density_grid_occupied_cells             = 0u;
        float loss                                            = 0.0f;
        float elapsed_ms                                      = 0.0f;
        float density_grid_update_ms                          = 0.0f;
        float density_grid_occupancy_ratio                    = 0.0f;
    };

    export struct ValidationStats final {
        std::uint32_t step        = 0u;
        std::uint32_t image_count = 0u;
        std::uint64_t pixel_count = 0u;
        float mse                 = 0.0f;
        float psnr                = 0.0f;
        float elapsed_ms          = 0.0f;
    };

    export class InstantNGP final {
    public:
        explicit InstantNGP(const NerfDatasetLike auto& dataset) {
            try {
                std::vector<std::uint8_t> host_pixels;
                std::vector<float> host_camera;

                const std::size_t frame_count = std::ranges::size(dataset.train);
                if (frame_count == 0) throw std::runtime_error{"pixels is empty."};
                if (frame_count > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"too many training frames."};

                const auto& first_frame = *std::ranges::begin(dataset.train);
                host_pixels.reserve(std::ranges::fold_left(dataset.train | std::views::transform([](const auto& frame) { return frame.rgba.size(); }), 0uz, std::plus{}));
                host_camera.reserve(frame_count * 12uz);

                std::ranges::for_each(dataset.train, [&](const auto& frame) {
                    if (static_cast<std::uint32_t>(frame.width) != static_cast<std::uint32_t>(first_frame.width)) throw std::runtime_error{"training frame width mismatch."};
                    if (static_cast<std::uint32_t>(frame.height) != static_cast<std::uint32_t>(first_frame.height)) throw std::runtime_error{"training frame height mismatch."};
                    if (static_cast<float>(frame.focal_length) != static_cast<float>(first_frame.focal_length)) throw std::runtime_error{"training frame focal length mismatch."};

                    host_pixels.append_range(frame.rgba);
                    host_camera.append_range(std::views::iota(0uz, 12uz) | std::views::transform([&](const std::size_t i) { return static_cast<float>(frame.camera[i]); }));
                });

                cuda::copy_dataset_to_device_once(host_pixels.data(), host_pixels.size(), host_camera.data(), host_camera.size(), this->device.pixels, this->device.camera);

                this->host.frame_count  = static_cast<std::uint32_t>(frame_count);
                this->host.width        = static_cast<std::uint32_t>(first_frame.width);
                this->host.height       = static_cast<std::uint32_t>(first_frame.height);
                this->host.focal_length = static_cast<float>(first_frame.focal_length);

                const std::size_t validation_frame_count = std::ranges::size(dataset.validation);
                if (validation_frame_count > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"too many validation frames."};
                if (validation_frame_count != 0uz) {
                    const auto& first_validation_frame = *std::ranges::begin(dataset.validation);
                    const auto validation_width        = static_cast<std::uint32_t>(first_validation_frame.width);
                    const auto validation_height       = static_cast<std::uint32_t>(first_validation_frame.height);
                    const auto validation_focal_length = static_cast<float>(first_validation_frame.focal_length);
                    if (validation_width == 0u || validation_height == 0u || validation_focal_length <= 0.0f) throw std::runtime_error{"invalid validation frame metadata."};

                    std::vector<std::uint8_t> validation_pixels;
                    std::vector<float> validation_camera;
                    validation_pixels.reserve(validation_frame_count * first_validation_frame.rgba.size());
                    validation_camera.reserve(validation_frame_count * 12uz);

                    for (const auto& frame : dataset.validation) {
                        if (static_cast<std::uint32_t>(frame.width) != validation_width) throw std::runtime_error{"validation frame width mismatch."};
                        if (static_cast<std::uint32_t>(frame.height) != validation_height) throw std::runtime_error{"validation frame height mismatch."};
                        if (static_cast<float>(frame.focal_length) != validation_focal_length) throw std::runtime_error{"validation frame focal length mismatch."};
                        if (frame.rgba.size() != static_cast<std::size_t>(validation_width) * validation_height * 4uz) throw std::runtime_error{"validation frame RGBA size mismatch."};

                        validation_pixels.append_range(frame.rgba);
                        for (std::size_t i = 0uz; i < 12uz; ++i) validation_camera.push_back(static_cast<float>(frame.camera[i]));
                    }

                    cuda::copy_dataset_to_device_once(validation_pixels.data(), validation_pixels.size(), validation_camera.data(), validation_camera.size(), this->device.validation_pixels, this->device.validation_camera);
                    this->host.validation_frame_count  = static_cast<std::uint32_t>(validation_frame_count);
                    this->host.validation_width        = validation_width;
                    this->host.validation_height       = validation_height;
                    this->host.validation_focal_length = validation_focal_length;
                }

                this->host.density_param_offset = 0u;
                this->host.density_param_count  = config::MLP_WIDTH * config::GRID_OUTPUT_WIDTH + (config::DENSITY_HIDDEN_LAYERS - 1u) * config::MLP_WIDTH * config::MLP_WIDTH + config::DENSITY_OUTPUT_WIDTH * config::MLP_WIDTH;
                this->host.rgb_param_offset     = this->host.density_param_offset + this->host.density_param_count;
                this->host.rgb_param_count      = config::MLP_WIDTH * config::RGB_INPUT_WIDTH + (config::RGB_HIDDEN_LAYERS - 1u) * config::MLP_WIDTH * config::MLP_WIDTH + config::NETWORK_OUTPUT_WIDTH * config::MLP_WIDTH;
                this->host.mlp_param_count      = this->host.rgb_param_offset + this->host.rgb_param_count;
                this->host.grid_param_offset    = this->host.mlp_param_count;

                std::uint32_t grid_offset = 0u;
                for (std::uint32_t level = 0u; level < config::GRID_N_LEVELS; ++level) {
                    const float scale                         = std::exp2(static_cast<float>(level) * config::GRID_LOG2_PER_LEVEL_SCALE) * static_cast<float>(config::GRID_BASE_RESOLUTION) - 1.0f;
                    const std::uint32_t resolution            = static_cast<std::uint32_t>(std::ceil(scale)) + 1u;
                    constexpr std::uint32_t MAX_PARAMS        = std::numeric_limits<std::uint32_t>::max() / 2u;
                    const std::uint64_t dense_params_in_level = static_cast<std::uint64_t>(resolution) * resolution * resolution;
                    std::uint32_t params_in_level             = dense_params_in_level > MAX_PARAMS ? MAX_PARAMS : static_cast<std::uint32_t>(dense_params_in_level);

                    params_in_level                = ((params_in_level + 7u) / 8u) * 8u;
                    params_in_level                = std::min(params_in_level, 1u << config::GRID_LOG2_HASHMAP_SIZE);
                    this->host.grid_offsets[level] = grid_offset;
                    grid_offset += params_in_level;
                }
                this->host.grid_offsets[config::GRID_N_LEVELS] = grid_offset;
                this->host.grid_param_count                    = grid_offset * config::GRID_FEATURES_PER_LEVEL;
                this->host.total_param_count                   = this->host.mlp_param_count + this->host.grid_param_count;
                cuda::Pcg32 training_rng{config::TRAIN_SEED};
                this->host.density_grid_rng = cuda::Pcg32{training_rng.next_uint()};

                cuda::allocate_sampler_once(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy);
                cuda::allocate_network_once(this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::allocate_density_grid_once(this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count);
                cuda::allocate_training_loss_once(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values);
                if (this->host.validation_frame_count != 0u) cuda::allocate_validation_once(this->device.validation_numsteps, this->device.validation_sample_counter, this->device.validation_overflow_counter, this->device.validation_loss_sum);
                cuda::allocate_trainable_params_once(this->host.total_param_count, this->device.params_full_precision, this->device.params, this->device.param_gradients);
                cuda::allocate_adam_state_once(this->host.total_param_count, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                cuda::initialize_mlp_params_once(this->host.density_param_offset, this->host.rgb_param_offset, this->device.params_full_precision, this->device.params, this->device.param_gradients);
                cuda::initialize_grid_params_once(this->host.grid_param_count, this->host.mlp_param_count, this->device.params_full_precision + this->host.grid_param_offset, this->device.params + this->host.grid_param_offset, this->device.param_gradients + this->host.grid_param_offset);
            } catch (...) {
                cuda::destroy_cublaslt_once(this->device.cublaslt_handle);
                cuda::free_device_data(this->device.pixels, this->device.camera, this->device.validation_pixels, this->device.validation_camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.validation_numsteps, this->device.validation_sample_counter, this->device.validation_overflow_counter, this->device.validation_loss_sum, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden,
                    this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                throw;
            }
        }

        ~InstantNGP() noexcept {
            cuda::destroy_cublaslt_once(this->device.cublaslt_handle);
            cuda::free_device_data(this->device.pixels, this->device.camera, this->device.validation_pixels, this->device.validation_camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.validation_numsteps, this->device.validation_sample_counter, this->device.validation_overflow_counter, this->device.validation_loss_sum, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden,
                this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
        }

        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = delete;
        InstantNGP& operator=(InstantNGP&&) noexcept = delete;

        std::expected<TrainStats, std::string> train(const std::int32_t iters) {
            try {
                const auto train_start            = std::chrono::steady_clock::now();
                std::uint32_t loss_rays_per_batch = this->host.rays_per_batch;
                this->host.density_grid_update_ms = 0.0f;
                for (std::int32_t i = 0; i < iters; ++i) {
                    loss_rays_per_batch          = this->host.rays_per_batch;
                    float density_grid_update_ms = 0.0f;
                    cuda::update_density_grid_once(this->device.camera, this->host.frame_count, this->host.width, this->host.height, this->host.focal_length, this->host.current_step, this->host.grid_offsets.data(), this->device.params, this->host.density_param_offset, this->host.grid_param_offset, this->device.sample_coords, this->device.density_input, this->device.network_output, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.occupancy, this->host.density_grid_ema_step, this->host.density_grid_rng, density_grid_update_ms);
                    this->host.density_grid_update_ms += density_grid_update_ms;
                    cuda::sample_training_batch(this->device.camera, this->host.frame_count, this->host.width, this->host.height, this->host.focal_length, this->host.current_step, this->host.rays_per_batch, this->host.inference_sample_count, this->device.occupancy, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter);
                    cuda::network_inference_once(this->host.inference_sample_count, this->device.sample_coords, this->host.grid_offsets.data(), this->device.params, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.network_output);
                    cuda::compute_loss_and_compact_once(this->host.rays_per_batch, this->host.current_step, this->device.ray_counter, this->device.pixels, this->host.frame_count, this->host.width, this->host.height, this->device.network_output, this->device.compacted_sample_counter, this->device.ray_indices, this->device.rays, this->device.numsteps, this->device.sample_coords, this->device.compacted_sample_coords, this->device.network_output_gradients, this->device.loss_values);
                    cuda::fill_rollover_once(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.network_output_gradients);
                    cuda::network_forward_once(this->device.compacted_sample_coords, this->host.grid_offsets.data(), this->device.params, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output);
                    cuda::network_backward_once(this->device.compacted_sample_coords, this->host.grid_offsets.data(), this->device.params, this->device.param_gradients, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                    cuda::optimize(this->host.total_param_count, this->host.mlp_param_count, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                    cuda::read_counter_once(this->device.sample_counter, this->host.measured_sample_count_before_compaction);
                    cuda::read_counter_once(this->device.compacted_sample_counter, this->host.measured_sample_count);
                    if (this->host.measured_sample_count == 0u) {
                        cuda::read_counter_once(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
                        throw std::runtime_error{std::format("Training stopped unexpectedly. density_grid_occupied_cells={}", this->host.density_grid_occupied_cells)};
                    }

                    this->host.inference_sample_count = ((std::min(this->host.measured_sample_count_before_compaction, config::MAX_SAMPLES) + config::NETWORK_BATCH_GRANULARITY - 1u) / config::NETWORK_BATCH_GRANULARITY) * config::NETWORK_BATCH_GRANULARITY;
                    this->host.rays_per_batch         = std::min(std::max(((static_cast<std::uint32_t>(std::min((static_cast<std::uint64_t>(this->host.rays_per_batch) * config::NETWORK_BATCH_SIZE) / this->host.measured_sample_count, static_cast<std::uint64_t>(config::NETWORK_BATCH_SIZE))) + config::NETWORK_BATCH_GRANULARITY - 1u) / config::NETWORK_BATCH_GRANULARITY) * config::NETWORK_BATCH_GRANULARITY, config::NETWORK_BATCH_GRANULARITY), config::NETWORK_BATCH_SIZE);

                    ++this->host.current_step;
                }

                float loss_sum = 0.0f;
                cuda::read_loss_sum_once(this->device.loss_values, loss_rays_per_batch, loss_sum);
                cuda::read_counter_once(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
                return TrainStats{
                    .step                                    = this->host.current_step,
                    .rays_per_batch                          = this->host.rays_per_batch,
                    .measured_sample_count_before_compaction = this->host.measured_sample_count_before_compaction,
                    .measured_sample_count                   = this->host.measured_sample_count,
                    .density_grid_occupied_cells             = this->host.density_grid_occupied_cells,
                    .loss                                    = loss_sum * static_cast<float>(this->host.measured_sample_count) / static_cast<float>(config::NETWORK_BATCH_SIZE),
                    .elapsed_ms                              = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(),
                    .density_grid_update_ms                  = this->host.density_grid_update_ms,
                    .density_grid_occupancy_ratio            = static_cast<float>(this->host.density_grid_occupied_cells) / (128.0f * 128.0f * 128.0f),
                };
            } catch (const std::exception& error) {
                return std::unexpected{std::string{error.what()}};
            }
        }

        std::expected<ValidationStats, std::string> validate() {
            try {
                if (this->host.validation_frame_count == 0u) throw std::runtime_error{"No validation images are available in the current dataset."};

                const auto validation_start          = std::chrono::steady_clock::now();
                const std::uint64_t pixels_per_image = static_cast<std::uint64_t>(this->host.validation_width) * this->host.validation_height;
                const std::uint64_t pixel_count      = pixels_per_image * this->host.validation_frame_count;
                double validation_loss_sum           = 0.0;

                cuda::validate_once(this->device.validation_pixels, this->device.validation_camera, this->host.validation_frame_count, this->host.validation_width, this->host.validation_height, this->host.validation_focal_length, this->device.occupancy, this->host.grid_offsets.data(), this->device.params, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.sample_coords, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.validation_numsteps, this->device.validation_sample_counter, this->device.validation_overflow_counter, this->device.validation_loss_sum, validation_loss_sum);

                const double mse = validation_loss_sum / (static_cast<double>(pixel_count) * 3.0);
                if (!std::isfinite(mse)) throw std::runtime_error{"validation produced non-finite MSE."};

                return ValidationStats{
                    .step        = this->host.current_step,
                    .image_count = this->host.validation_frame_count,
                    .pixel_count = pixel_count,
                    .mse         = static_cast<float>(mse),
                    .psnr        = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity(),
                    .elapsed_ms  = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - validation_start).count(),
                };
            } catch (const std::exception& error) {
                return std::unexpected{std::string{error.what()}};
            }
        }

    private:
        struct HostData {
            // Stable after construction: dataset metadata.
            std::uint32_t frame_count            = 0u;
            std::uint32_t width                  = 0u;
            std::uint32_t height                 = 0u;
            float focal_length                   = 0.0f;
            std::uint32_t validation_frame_count = 0u;
            std::uint32_t validation_width       = 0u;
            std::uint32_t validation_height      = 0u;
            float validation_focal_length        = 0.0f;

            // Stable after construction: network parameter layout.
            std::array<std::uint32_t, config::GRID_OFFSET_COUNT> grid_offsets = {};
            std::uint32_t density_param_offset                                = 0u;
            std::uint32_t density_param_count                                 = 0u;
            std::uint32_t rgb_param_offset                                    = 0u;
            std::uint32_t rgb_param_count                                     = 0u;
            std::uint32_t mlp_param_count                                     = 0u;
            std::uint32_t grid_param_offset                                   = 0u;
            std::uint32_t grid_param_count                                    = 0u;
            std::uint32_t total_param_count                                   = 0u;

            // Mutated by train(): step, adaptive batch shape, and latest counters.
            std::uint32_t current_step                            = 0u;
            std::uint32_t rays_per_batch                          = config::INITIAL_RAYS_PER_BATCH;
            std::uint32_t inference_sample_count                  = config::MAX_SAMPLES;
            std::uint32_t measured_sample_count_before_compaction = 0u;
            std::uint32_t measured_sample_count                   = 0u;
            std::uint32_t density_grid_ema_step                   = 0u;
            std::uint32_t density_grid_occupied_cells             = 0u;
            cuda::Pcg32 density_grid_rng                          = {};
            float density_grid_update_ms                          = 0.0f;
        } host;

        struct DeviceData {
            // Dataset.
            const std::uint8_t* pixels            = nullptr;
            const float* camera                   = nullptr;
            const std::uint8_t* validation_pixels = nullptr;
            const float* validation_camera        = nullptr;

            // Sampler.
            std::uint8_t* occupancy                    = nullptr;
            float* sample_coords                       = nullptr;
            float* rays                                = nullptr;
            std::uint32_t* ray_indices                 = nullptr;
            std::uint32_t* numsteps                    = nullptr;
            std::uint32_t* ray_counter                 = nullptr;
            std::uint32_t* sample_counter              = nullptr;
            float* density_grid_values                 = nullptr;
            float* density_grid_scratch                = nullptr;
            std::uint32_t* density_grid_indices        = nullptr;
            float* density_grid_mean                   = nullptr;
            std::uint32_t* density_grid_occupied_count = nullptr;

            // Loss and compaction.
            std::uint32_t* compacted_sample_counter    = nullptr;
            float* compacted_sample_coords             = nullptr;
            float* loss_values                         = nullptr;
            std::uint16_t* network_output_gradients    = nullptr;
            std::uint32_t* validation_numsteps         = nullptr;
            std::uint32_t* validation_sample_counter   = nullptr;
            std::uint32_t* validation_overflow_counter = nullptr;
            double* validation_loss_sum                = nullptr;

            // Network.
            std::uint16_t* density_input           = nullptr;
            std::uint16_t* rgb_input               = nullptr;
            std::uint16_t* network_output          = nullptr;
            std::uint16_t* rgb_output_gradients    = nullptr;
            std::uint16_t* rgb_input_gradients     = nullptr;
            std::uint16_t* density_input_gradients = nullptr;
            std::uint16_t* density_forward_hidden  = nullptr;
            std::uint16_t* rgb_forward_hidden      = nullptr;
            std::uint16_t* density_backward_hidden = nullptr;
            std::uint16_t* rgb_backward_hidden     = nullptr;
            void* cublaslt_handle                  = nullptr;
            std::uint8_t* cublaslt_workspace       = nullptr;

            // Trainable parameters.
            float* params_full_precision   = nullptr;
            std::uint16_t* params          = nullptr;
            std::uint16_t* param_gradients = nullptr;

            // Optimizer.
            float* optimizer_first_moments       = nullptr;
            float* optimizer_second_moments      = nullptr;
            std::uint32_t* optimizer_param_steps = nullptr;
        } device;

        static constexpr std::array<float, 6> AABB = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    };
} // namespace ngp::train
