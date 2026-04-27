module;
#include "ngp.train.h"
#include <exception>
#include <stdexcept>
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
        float loss                                            = 0.0f;
        float elapsed_ms                                      = 0.0f;
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

                if (const std::string error = cuda::copy_dataset_to_device_once(host_pixels.data(), host_pixels.size(), host_camera.data(), host_camera.size(), this->device.pixels, this->device.camera); !error.empty()) throw std::runtime_error{error};

                this->host.frame_count  = static_cast<std::uint32_t>(frame_count);
                this->host.width        = static_cast<std::uint32_t>(first_frame.width);
                this->host.height       = static_cast<std::uint32_t>(first_frame.height);
                this->host.focal_length = static_cast<float>(first_frame.focal_length);

                this->host.density_param_offset = 0u;
                this->host.density_param_count  = config::mlp_width * config::grid_output_width + (config::density_hidden_layers - 1u) * config::mlp_width * config::mlp_width + config::density_output_width * config::mlp_width;
                this->host.rgb_param_offset     = this->host.density_param_offset + this->host.density_param_count;
                this->host.rgb_param_count      = config::mlp_width * config::rgb_input_width + (config::rgb_hidden_layers - 1u) * config::mlp_width * config::mlp_width + config::network_output_width * config::mlp_width;
                this->host.mlp_param_count      = this->host.rgb_param_offset + this->host.rgb_param_count;
                this->host.grid_param_offset    = this->host.mlp_param_count;

                std::uint32_t grid_offset = 0u;
                for (std::uint32_t level = 0u; level < config::grid_n_levels; ++level) {
                    const float scale                         = std::exp2(static_cast<float>(level) * std::log2(config::grid_per_level_scale)) * static_cast<float>(config::grid_base_resolution) - 1.0f;
                    const std::uint32_t resolution            = static_cast<std::uint32_t>(std::ceil(scale)) + 1u;
                    constexpr std::uint32_t max_params        = std::numeric_limits<std::uint32_t>::max() / 2u;
                    const std::uint64_t dense_params_in_level = static_cast<std::uint64_t>(resolution) * resolution * resolution;
                    std::uint32_t params_in_level             = dense_params_in_level > max_params ? max_params : static_cast<std::uint32_t>(dense_params_in_level);

                    params_in_level                = ((params_in_level + 7u) / 8u) * 8u;
                    params_in_level                = std::min(params_in_level, 1u << config::grid_log2_hashmap_size);
                    this->host.grid_offsets[level] = grid_offset;
                    grid_offset += params_in_level;
                }
                this->host.grid_offsets[config::grid_n_levels] = grid_offset;
                this->host.grid_param_count                    = grid_offset * config::grid_features_per_level;
                this->host.total_param_count                   = this->host.mlp_param_count + this->host.grid_param_count;

                if (const std::string error = cuda::allocate_sampler_once(this->host.max_rays_per_batch, this->host.max_samples, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy); !error.empty()) throw std::runtime_error{error};
                if (const std::string error = cuda::allocate_network_once(
                        config::network_batch_size, this->host.max_samples, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cutlass_workspace);
                    !error.empty())
                    throw std::runtime_error{error};
                if (const std::string error = cuda::allocate_training_loss_once(config::network_batch_size, this->host.max_rays_per_batch, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values); !error.empty()) throw std::runtime_error{error};
                if (const std::string error = cuda::allocate_trainable_params_once(this->host.total_param_count, this->device.params_full_precision, this->device.params, this->device.param_gradients); !error.empty()) throw std::runtime_error{error};
                if (const std::string error = cuda::allocate_adam_state_once(this->host.total_param_count, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps); !error.empty()) throw std::runtime_error{error};
                if (const std::string error = cuda::initialize_mlp_params_once(this->host.seed, config::grid_output_width, config::density_output_width, config::density_hidden_layers, this->host.density_param_offset, config::rgb_input_width, config::network_output_width, config::rgb_hidden_layers, this->host.rgb_param_offset, this->device.params_full_precision, this->device.params, this->device.param_gradients); !error.empty()) throw std::runtime_error{error};
                if (const std::string error = cuda::initialize_grid_params_once(this->host.grid_param_count, this->host.seed, this->host.mlp_param_count, this->device.params_full_precision + this->host.grid_param_offset, this->device.params + this->host.grid_param_offset, this->device.param_gradients + this->host.grid_param_offset); !error.empty()) throw std::runtime_error{error};
            } catch (...) {
                cuda::free_device_data(this->device.pixels, this->device.camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients,
                    this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cutlass_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments,
                    this->device.optimizer_param_steps);
                throw;
            }
        }

        ~InstantNGP() noexcept {
            cuda::free_device_data(this->device.pixels, this->device.camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients,
                this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cutlass_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments,
                this->device.optimizer_param_steps);
        }

        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = delete;
        InstantNGP& operator=(InstantNGP&&) noexcept = delete;

        std::expected<TrainStats, std::string> train(const std::int32_t iters) {
            const auto train_start            = std::chrono::steady_clock::now();
            std::uint32_t loss_rays_per_batch = this->host.rays_per_batch;
            for (std::int32_t i = 0; i < iters; ++i) {
                loss_rays_per_batch = this->host.rays_per_batch;
                if (const std::string error =
                        cuda::sample_training_batch(this->device.camera, this->host.frame_count, this->host.width, this->host.height, this->host.focal_length, this->host.current_step, this->host.seed, this->host.rays_per_batch, this->host.inference_sample_count, this->host.snap_to_pixel_centers, this->device.occupancy, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter);
                    !error.empty())
                    return std::unexpected{error};

                if (const std::string error = cuda::network_inference_once(this->host.inference_sample_count, config::network_batch_size, this->device.sample_coords, this->host.grid_offsets.data(), config::grid_n_levels, config::grid_features_per_level, config::grid_base_resolution, config::grid_per_level_scale, this->device.params, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.network_output);
                    !error.empty())
                    return std::unexpected{error};
                if (const std::string error = cuda::compute_loss_and_compact_once(this->host.rays_per_batch, config::network_batch_size, this->host.seed, this->host.current_step, this->device.ray_counter, this->device.pixels, this->host.frame_count, this->host.width, this->host.height, this->host.snap_to_pixel_centers, this->device.network_output, this->device.compacted_sample_counter, this->device.ray_indices, this->device.rays, this->device.numsteps, this->device.sample_coords,
                        this->device.compacted_sample_coords, this->device.network_output_gradients, this->device.loss_values);
                    !error.empty())
                    return std::unexpected{error};
                if (const std::string error = cuda::fill_rollover_once(config::network_batch_size, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.network_output_gradients); !error.empty()) return std::unexpected{error};
                if (const std::string error = cuda::network_forward_once(
                        config::network_batch_size, this->device.compacted_sample_coords, this->host.grid_offsets.data(), config::grid_n_levels, config::grid_features_per_level, config::grid_base_resolution, config::grid_per_level_scale, this->device.params, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output);
                    !error.empty())
                    return std::unexpected{error};
                if (const std::string error = cuda::network_backward_once(config::network_batch_size, this->device.compacted_sample_coords, this->host.grid_offsets.data(), config::grid_n_levels, config::grid_features_per_level, config::grid_base_resolution, config::grid_per_level_scale, this->device.params, this->device.param_gradients, this->host.density_param_offset, this->host.rgb_param_offset, this->host.grid_param_offset, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden,
                        this->device.rgb_forward_hidden, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cutlass_workspace);
                    !error.empty())
                    return std::unexpected{error};
                if (const std::string error = cuda::optimize(
                        this->host.total_param_count, this->host.mlp_param_count, this->host.optimizer_loss_scale, this->host.optimizer_learning_rate, this->host.optimizer_beta1, this->host.optimizer_beta2, this->host.optimizer_epsilon, this->host.optimizer_l2_reg, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                    !error.empty())
                    return std::unexpected{error};
                if (const std::string error = cuda::read_counter_once(this->device.sample_counter, this->host.measured_sample_count_before_compaction); !error.empty()) return std::unexpected{error};
                if (const std::string error = cuda::read_counter_once(this->device.compacted_sample_counter, this->host.measured_sample_count); !error.empty()) return std::unexpected{error};
                if (this->host.measured_sample_count == 0u) return std::unexpected{std::string{"Training stopped unexpectedly."}};

                this->host.inference_sample_count = config::round_up(std::min(this->host.measured_sample_count_before_compaction, this->host.max_samples), config::network_batch_granularity);
                this->host.rays_per_batch         = std::min(std::max(config::round_up(static_cast<std::uint32_t>(std::min((static_cast<std::uint64_t>(this->host.rays_per_batch) * config::network_batch_size) / this->host.measured_sample_count, static_cast<std::uint64_t>(this->host.max_rays_per_batch))), config::network_batch_granularity), config::network_batch_granularity), this->host.max_rays_per_batch);

                ++this->host.current_step;
            }

            float loss_sum = 0.0f;
            if (const std::string error = cuda::read_loss_sum_once(this->device.loss_values, loss_rays_per_batch, loss_sum); !error.empty()) return std::unexpected{error};

            return TrainStats{
                .step                                    = this->host.current_step,
                .rays_per_batch                          = this->host.rays_per_batch,
                .measured_sample_count_before_compaction = this->host.measured_sample_count_before_compaction,
                .measured_sample_count                   = this->host.measured_sample_count,
                .loss                                    = loss_sum * static_cast<float>(this->host.measured_sample_count) / static_cast<float>(config::network_batch_size),
                .elapsed_ms                              = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(),
            };
        }

    private:
        struct HostData {
            // Dataset.
            std::uint32_t frame_count = 0u;
            std::uint32_t width       = 0u;
            std::uint32_t height      = 0u;
            float focal_length        = 0.0f;

            // Training state.
            std::uint32_t current_step = 0u;
            std::uint64_t seed         = 1337u;
            bool snap_to_pixel_centers = true;

            // Sampler and loss.
            std::uint32_t rays_per_batch                          = 1u << 12u;
            std::uint32_t max_rays_per_batch                      = config::network_batch_size;
            std::uint32_t max_samples                             = config::network_batch_size * 16u;
            std::uint32_t inference_sample_count                  = max_samples;
            std::uint32_t measured_sample_count_before_compaction = 0u;
            std::uint32_t measured_sample_count                   = 0u;

            // Network parameter layout.
            std::array<std::uint32_t, config::grid_offset_count> grid_offsets = {};
            std::uint32_t density_param_offset                               = 0u;
            std::uint32_t density_param_count                                = 0u;
            std::uint32_t rgb_param_offset                                   = 0u;
            std::uint32_t rgb_param_count                                    = 0u;
            std::uint32_t mlp_param_count                                    = 0u;
            std::uint32_t grid_param_offset                                  = 0u;
            std::uint32_t grid_param_count                                   = 0u;
            std::uint32_t total_param_count                                  = 0u;

            // Optimizer.
            float optimizer_learning_rate = 1e-2f;
            float optimizer_beta1         = 0.9f;
            float optimizer_beta2         = 0.99f;
            float optimizer_epsilon       = 1e-15f;
            float optimizer_l2_reg        = 1e-6f;
            float optimizer_loss_scale    = 128.0f;
        } host;

        struct DeviceData {
            // Dataset.
            const std::uint8_t* pixels = nullptr;
            const float* camera        = nullptr;

            // Sampler.
            std::uint8_t* occupancy       = nullptr;
            float* sample_coords          = nullptr;
            float* rays                   = nullptr;
            std::uint32_t* ray_indices    = nullptr;
            std::uint32_t* numsteps       = nullptr;
            std::uint32_t* ray_counter    = nullptr;
            std::uint32_t* sample_counter = nullptr;

            // Loss and compaction.
            std::uint32_t* compacted_sample_counter = nullptr;
            float* compacted_sample_coords          = nullptr;
            float* loss_values                      = nullptr;
            std::uint16_t* network_output_gradients = nullptr;

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
            std::uint8_t* cutlass_workspace        = nullptr;

            // Trainable parameters.
            float* params_full_precision   = nullptr;
            std::uint16_t* params          = nullptr;
            std::uint16_t* param_gradients = nullptr;

            // Optimizer.
            float* optimizer_first_moments       = nullptr;
            float* optimizer_second_moments      = nullptr;
            std::uint32_t* optimizer_param_steps = nullptr;
        } device;

        static constexpr std::array<float, 6> aabb = {0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    };
} // namespace ngp::train
