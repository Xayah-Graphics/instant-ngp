module;
#include "ngp.train.h"

#include "json/json.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
module ngp.train;
import std;

namespace ngp::train {
    void InstantNGP::initialize(const std::span<const FrameSetView> frame_sets, const float scene_scale) {
        try {
            this->host.scene_scale = scene_scale;
            if (!std::isfinite(scene_scale) || scene_scale <= 0.0f) throw std::runtime_error{"scene scale must be finite and positive."};
            if (frame_sets.empty()) throw std::runtime_error{"dataset must contain at least one loaded frame set."};

            // ====================================================================================================
            // 1. UPLOAD FRAME SETS TO GPU
            // ====================================================================================================
            {
                this->host.frame_sets.reserve(frame_sets.size());
                this->device.frame_sets.reserve(frame_sets.size());
                for (const FrameSetView& frame_set : frame_sets) {
                    if (frame_set.name.empty()) throw std::runtime_error{"frame set name must not be empty."};
                    if (frame_set.frames.empty()) throw std::runtime_error{std::format("frame set '{}' contains no frames.", frame_set.name)};
                    for (const HostFrameSet& existing_frame_set : this->host.frame_sets)
                        if (existing_frame_set.name == frame_set.name) throw std::runtime_error{std::format("frame set '{}' was loaded more than once.", frame_set.name)};

                    std::vector<std::uint8_t> pixels;
                    std::vector<float> camera;
                    pixels.reserve(std::ranges::fold_left(frame_set.frames | std::views::transform([](const auto& frame) { return frame.rgba.size(); }), 0uz, std::plus{}));
                    camera.reserve(frame_set.frames.size() * 12uz);
                    for (const FrameView& frame : frame_set.frames) {
                        if (frame.width == 0u || frame.height == 0u) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid dimensions.", frame_set.name)};
                        if (!std::isfinite(frame.focal_x) || !std::isfinite(frame.focal_y) || frame.focal_x <= 0.0f || frame.focal_y <= 0.0f) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid focal length.", frame_set.name)};
                        if (!std::isfinite(frame.principal_x) || !std::isfinite(frame.principal_y) || frame.principal_x < 0.0f || frame.principal_y < 0.0f || frame.principal_x >= static_cast<float>(frame.width) || frame.principal_y >= static_cast<float>(frame.height)) throw std::runtime_error{std::format("frame set '{}' contains a frame with invalid principal point.", frame_set.name)};
                        if (frame.camera.size() != 12uz) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} camera values; expected 12.", frame_set.name, frame.camera.size())};
                        if (frame.height > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(frame.width) / 4ull) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                        const std::uint64_t rgba_byte_count = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 4ull;
                        if (rgba_byte_count > std::numeric_limits<std::size_t>::max()) throw std::runtime_error{std::format("frame set '{}' contains an image that is too large.", frame_set.name)};
                        if (frame.rgba.size() != static_cast<std::size_t>(rgba_byte_count)) throw std::runtime_error{std::format("frame set '{}' contains a frame with {} RGBA bytes; expected {}.", frame_set.name, frame.rgba.size(), rgba_byte_count)};
                        pixels.append_range(frame.rgba);
                        camera.append_range(frame.camera);
                    }

                    this->device.frame_sets.push_back({});
                    DeviceFrameSet& device_frame_set = this->device.frame_sets.back();
                    cuda::upload_dataset(pixels.data(), pixels.size(), camera.data(), camera.size(), device_frame_set.pixels, device_frame_set.camera);

                    const FrameView& first_frame = frame_set.frames.front();
                    this->host.frame_sets.push_back(HostFrameSet{
                        .name         = std::string{frame_set.name},
                        .frame_count  = static_cast<std::uint32_t>(frame_set.frames.size()),
                        .width        = first_frame.width,
                        .height       = first_frame.height,
                        .focal_x      = first_frame.focal_x,
                        .focal_y      = first_frame.focal_y,
                        .principal_x  = first_frame.principal_x,
                        .principal_y  = first_frame.principal_y,
                    });
                    this->host.comparison_width  = std::max(this->host.comparison_width, first_frame.width);
                    this->host.comparison_height = std::max(this->host.comparison_height, first_frame.height);
                }
            }

            // ====================================================================================================
            // 2. ALLOCATE AND INITIALIZE GPU BUFFERS
            // ====================================================================================================
            {
                cuda::allocate_sampler_buffers(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy);
                cuda::allocate_network_buffers(this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::allocate_density_grid_buffers(this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count);
                cuda::allocate_training_loss_buffers(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values);
                cuda::allocate_evaluation_buffers(this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum);
                cuda::allocate_evaluation_comparison_buffer(this->host.comparison_width, this->host.comparison_height, this->device.comparison_pixels);
                cuda::allocate_trainable_parameter_buffers(this->device.params_full_precision, this->device.params, this->device.param_gradients);
                cuda::allocate_adam_state(this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                cuda::initialize_mlp_parameters(this->device.params_full_precision, this->device.params, this->device.param_gradients);
                cuda::initialize_grid_parameters(this->device.params_full_precision, this->device.params, this->device.param_gradients);
            }

            this->host.current_step = 0u;
            this->host.rays_per_batch = ngp::train::config::initial_rays_per_batch;
            this->host.inference_sample_count = ngp::train::config::max_samples;
            this->host.measured_sample_count_before_compaction = 0u;
            this->host.measured_sample_count = 0u;
            this->host.density_grid_ema_step = 0u;
            this->host.density_grid_occupied_cells = 0u;

        } catch (...) {
            for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera);
            cuda::destroy_cublaslt(this->device.cublaslt_handle);
            cuda::free_device_buffers(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.comparison_pixels, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients,
                this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            throw;
        }
    }

    InstantNGP::~InstantNGP() noexcept {
        for (DeviceFrameSet& frame_set : this->device.frame_sets) cuda::free_device_buffers(frame_set.pixels, frame_set.camera);
        cuda::destroy_cublaslt(this->device.cublaslt_handle);
        cuda::free_device_buffers(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.comparison_pixels, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients,
            this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
    }

    std::expected<OptimizationStats, std::string> InstantNGP::optimize(const OptimizationRequest request) {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"optimization frame set must not be empty."};
            if (request.iterations < 1) throw std::runtime_error{"optimization iterations must be positive."};
            const HostFrameSet* host_frame_set = nullptr;
            const DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                if (this->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set = std::addressof(this->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("optimization frame set '{}' is not loaded.", request.frame_set)};

            const auto train_start            = std::chrono::steady_clock::now();
            std::uint32_t loss_value_count    = this->host.rays_per_batch;
            for (std::int32_t i = 0; i < request.iterations; ++i) {
                loss_value_count = this->host.rays_per_batch;
                cuda::update_density_grid(device_frame_set->camera, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->host.current_step, this->device.params, this->device.sample_coords, this->device.density_input, this->device.network_output, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.occupancy, this->host.density_grid_ema_step);
                cuda::sample_training_batch(device_frame_set->camera, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->host.current_step, this->host.rays_per_batch, this->host.inference_sample_count, this->device.occupancy, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter);
                cuda::evaluate_network(this->host.inference_sample_count, this->device.sample_coords, this->device.params, this->device.density_input, this->device.rgb_input, this->device.network_output);
                cuda::compute_training_loss_and_compact_samples(this->host.rays_per_batch, this->host.current_step, this->device.ray_counter, device_frame_set->pixels, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, this->device.network_output, this->device.compacted_sample_counter, this->device.ray_indices, this->device.rays, this->device.numsteps, this->device.sample_coords, this->device.compacted_sample_coords, this->device.network_output_gradients, this->device.loss_values);
                cuda::pad_compacted_training_batch(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.network_output_gradients);
                cuda::forward_network(this->device.compacted_sample_coords, this->device.params, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output);
                cuda::backward_network(this->device.compacted_sample_coords, this->device.params, this->device.param_gradients, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::step_optimizer(this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                cuda::read_counter(this->device.sample_counter, this->host.measured_sample_count_before_compaction);
                cuda::read_counter(this->device.compacted_sample_counter, this->host.measured_sample_count);
                if (this->host.measured_sample_count == 0u) {
                    cuda::read_counter(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
                    throw std::runtime_error{std::format("Optimization stopped unexpectedly. density_grid_occupied_cells={}", this->host.density_grid_occupied_cells)};
                }

                this->host.inference_sample_count = ((std::min(this->host.measured_sample_count_before_compaction, ngp::train::config::max_samples) + ngp::train::config::network_batch_granularity - 1u) / ngp::train::config::network_batch_granularity) * ngp::train::config::network_batch_granularity;
                this->host.rays_per_batch         = std::min(std::max(((static_cast<std::uint32_t>(std::min((static_cast<std::uint64_t>(this->host.rays_per_batch) * ngp::train::config::network_batch_size) / this->host.measured_sample_count, static_cast<std::uint64_t>(ngp::train::config::network_batch_size))) + ngp::train::config::network_batch_granularity - 1u) / ngp::train::config::network_batch_granularity) * ngp::train::config::network_batch_granularity, ngp::train::config::network_batch_granularity), ngp::train::config::network_batch_size);

                ++this->host.current_step;
            }

            float loss_sum = 0.0f;
            cuda::read_loss_sum(this->device.loss_values, loss_value_count, loss_sum);
            cuda::read_counter(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
            return OptimizationStats{
                .step                                    = this->host.current_step,
                .next_rays_per_batch                     = this->host.rays_per_batch,
                .measured_sample_count_before_compaction = this->host.measured_sample_count_before_compaction,
                .measured_sample_count                   = this->host.measured_sample_count,
                .density_grid_occupied_cells             = this->host.density_grid_occupied_cells,
                .loss                                    = loss_sum * static_cast<float>(this->host.measured_sample_count) / static_cast<float>(ngp::train::config::network_batch_size),
                .elapsed_ms                              = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(),
                .sample_efficiency_ratio                 = this->host.measured_sample_count_before_compaction == 0u ? 0.0f : static_cast<float>(this->host.measured_sample_count) / static_cast<float>(this->host.measured_sample_count_before_compaction),
                .density_grid_occupancy_ratio            = static_cast<float>(this->host.density_grid_occupied_cells) / static_cast<float>(ngp::train::config::nerf_grid_cells),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<EvaluationStats, std::string> InstantNGP::evaluate(const EvaluationRequest request) const {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"evaluation frame set must not be empty."};
            const HostFrameSet* host_frame_set = nullptr;
            const DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->host.frame_sets.size(); ++frame_set_index) {
                if (this->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set = std::addressof(this->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("evaluation frame set '{}' is not loaded.", request.frame_set)};
            if (request.refresh_acceleration) this->host.density_grid_ema_step = 0u;
            if (this->host.density_grid_ema_step == 0u) cuda::update_density_grid(device_frame_set->camera, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, 0u, this->device.params, this->device.sample_coords, this->device.density_input, this->device.network_output, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.occupancy, this->host.density_grid_ema_step);

            const bool writes_comparison = request.comparison_output_dir.has_value();
            if (writes_comparison) {
                if (request.comparison_output_dir->empty()) throw std::runtime_error{"comparison output directory must not be empty."};
                if (this->device.comparison_pixels == nullptr) throw std::runtime_error{"comparison image buffer is not initialized."};
                if (std::filesystem::exists(*request.comparison_output_dir) && !std::filesystem::is_directory(*request.comparison_output_dir)) throw std::runtime_error{std::format("comparison output path '{}' is not a directory.", request.comparison_output_dir->string())};
                std::filesystem::create_directories(*request.comparison_output_dir);
                if (!std::filesystem::is_directory(*request.comparison_output_dir)) throw std::runtime_error{std::format("failed to create comparison output directory '{}'.", request.comparison_output_dir->string())};
                if (host_frame_set->width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 2) || host_frame_set->height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) throw std::runtime_error{"comparison image dimensions exceed PNG writer limits."};
            }

            const auto evaluation_start          = std::chrono::steady_clock::now();
            const std::uint64_t pixels_per_image = static_cast<std::uint64_t>(host_frame_set->width) * host_frame_set->height;
            const std::uint64_t pixel_count      = pixels_per_image * host_frame_set->frame_count;
            double evaluation_loss_sum           = 0.0;
            std::uint32_t comparison_image_count = 0u;

            if (!writes_comparison) {
                cuda::run_evaluation(device_frame_set->pixels, device_frame_set->camera, host_frame_set->frame_count, 0u, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->device.occupancy, this->device.params, this->device.sample_coords, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, nullptr, nullptr, evaluation_loss_sum);
            } else {
                std::vector<std::uint8_t> comparison_image(static_cast<std::size_t>(host_frame_set->width) * host_frame_set->height * 2uz * 3uz);
                for (std::uint32_t image_index = 0u; image_index < host_frame_set->frame_count; ++image_index) {
                    double image_loss_sum = 0.0;
                    cuda::run_evaluation(device_frame_set->pixels, device_frame_set->camera, host_frame_set->frame_count, image_index, 1u, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->device.occupancy, this->device.params, this->device.sample_coords, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.comparison_pixels, comparison_image.data(), image_loss_sum);
                    evaluation_loss_sum += image_loss_sum;

                    const std::filesystem::path output_path = *request.comparison_output_dir / std::format("{}_{:04}.png", host_frame_set->name, image_index);
                    const std::string output_path_text      = output_path.string();
                    const int output_width                  = static_cast<int>(host_frame_set->width * 2u);
                    const int output_height                 = static_cast<int>(host_frame_set->height);
                    if (stbi_write_png(output_path_text.c_str(), output_width, output_height, 3, comparison_image.data(), output_width * 3) == 0) throw std::runtime_error{std::format("failed to write comparison image '{}'.", output_path_text)};
                    ++comparison_image_count;
                }
            }

            const double mse = evaluation_loss_sum / (static_cast<double>(pixel_count) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"evaluation produced non-finite MSE."};

            return EvaluationStats{
                .frame_set              = host_frame_set->name,
                .step                   = this->host.current_step,
                .image_count            = host_frame_set->frame_count,
                .comparison_image_count = comparison_image_count,
                .pixel_count            = pixel_count,
                .mse                    = static_cast<float>(mse),
                .psnr                   = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity(),
                .elapsed_ms             = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - evaluation_start).count(),
                .output_dir             = request.comparison_output_dir.value_or(std::filesystem::path{}),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> InstantNGP::export_weights(const std::filesystem::path& path) const {
        try {
            static_assert(std::endian::native == std::endian::little);
            if (path.empty()) throw std::runtime_error{"weights export path must not be empty."};
            if (!path.parent_path().empty() && !std::filesystem::is_directory(path.parent_path())) throw std::runtime_error{std::format("weights export parent directory '{}' does not exist.", path.parent_path().string())};
            if (this->device.params_full_precision == nullptr) throw std::runtime_error{"trainable parameters are not initialized."};

            struct SafetensorsTensor final {
                std::string_view name;
                std::uint32_t param_offset;
                std::uint64_t rows;
                std::uint64_t cols;
            };

            constexpr std::array tensors = std::to_array<SafetensorsTensor>({
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = ngp::train::config::network_parameter_layout.density_input_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::grid_output_width},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = ngp::train::config::network_parameter_layout.density_output_weight_offset, .rows = ngp::train::config::density_output_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "rgb_mlp.input.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_input_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::rgb_input_width},
                SafetensorsTensor{.name = "rgb_mlp.hidden.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_hidden_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "rgb_mlp.output.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_output_weight_offset, .rows = ngp::train::config::network_output_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "hash_grid.params", .param_offset = ngp::train::config::network_parameter_layout.grid_param_offset, .rows = ngp::train::config::network_parameter_layout.grid_offsets[ngp::train::config::grid_n_levels], .cols = ngp::train::config::grid_features_per_level},
            });

            std::vector<float> host_params(ngp::train::config::network_parameter_layout.total_param_count);
            cuda::download_trainable_parameters(this->device.params_full_precision, host_params.data());

            std::string grid_offsets_text;
            for (std::uint32_t i = 0u; i < ngp::train::config::grid_n_levels + 1u; ++i) {
                if (!grid_offsets_text.empty()) grid_offsets_text += ",";
                grid_offsets_text += std::format("{}", ngp::train::config::network_parameter_layout.grid_offsets[i]);
            }

            nlohmann::json metadata             = nlohmann::json::object();
            metadata["format"]                  = "instant-ngp-new.weights.v2";
            metadata["grid_n_levels"]           = std::format("{}", ngp::train::config::grid_n_levels);
            metadata["grid_features_per_level"] = std::format("{}", ngp::train::config::grid_features_per_level);
            metadata["grid_base_resolution"]    = std::format("{}", ngp::train::config::grid_base_resolution);
            metadata["grid_log2_hashmap_size"]  = std::format("{}", ngp::train::config::grid_log2_hashmap_size);
            metadata["mlp_width"]               = std::format("{}", ngp::train::config::mlp_width);
            metadata["density_hidden_layers"]   = std::format("{}", ngp::train::config::density_hidden_layers);
            metadata["rgb_hidden_layers"]       = std::format("{}", ngp::train::config::rgb_hidden_layers);
            metadata["density_output_width"]    = std::format("{}", ngp::train::config::density_output_width);
            metadata["direction_output_width"]  = std::format("{}", ngp::train::config::direction_output_width);
            metadata["rgb_input_width"]         = std::format("{}", ngp::train::config::rgb_input_width);
            metadata["network_output_width"]    = std::format("{}", ngp::train::config::network_output_width);
            metadata["grid_offsets"]            = grid_offsets_text;
            metadata["scene_scale"]             = std::format("{:.9g}", this->host.scene_scale);

            nlohmann::json header  = nlohmann::json::object();
            header["__metadata__"] = metadata;

            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::uint64_t byte_count   = tensor.rows * tensor.cols * sizeof(float);
                header[std::string{tensor.name}] = nlohmann::json{
                    {"dtype", "F32"},
                    {"shape", nlohmann::json::array({tensor.rows, tensor.cols})},
                    {"data_offsets", nlohmann::json::array({data_offset, data_offset + byte_count})},
                };
                data_offset += byte_count;
            }

            const std::string header_text   = header.dump();
            const std::uint64_t header_size = header_text.size();
            std::ofstream output{path, std::ios::binary | std::ios::trunc};
            if (!output) throw std::runtime_error{std::format("failed to open weights export path '{}'.", path.string())};

            output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
            output.write(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            for (const SafetensorsTensor& tensor : tensors) output.write(reinterpret_cast<const char*>(host_params.data() + tensor.param_offset), static_cast<std::streamsize>(tensor.rows * tensor.cols * sizeof(float)));
            if (!output) throw std::runtime_error{std::format("failed to write weights file '{}'.", path.string())};

            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<void, std::string> InstantNGP::load_weights(const std::filesystem::path& path) {
        try {
            static_assert(std::endian::native == std::endian::little);
            if (this->host.current_step != 0u) throw std::runtime_error{"weights can only be loaded before training starts."};
            if (path.empty()) throw std::runtime_error{"weights load path must not be empty."};
            if (!std::filesystem::is_regular_file(path)) throw std::runtime_error{std::format("weights file '{}' does not exist.", path.string())};
            if (this->device.params_full_precision == nullptr) throw std::runtime_error{"trainable parameters are not initialized."};

            struct SafetensorsTensor final {
                std::string_view name;
                std::uint32_t param_offset;
                std::uint64_t rows;
                std::uint64_t cols;
            };

            constexpr std::array tensors = std::to_array<SafetensorsTensor>({
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = ngp::train::config::network_parameter_layout.density_input_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::grid_output_width},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = ngp::train::config::network_parameter_layout.density_output_weight_offset, .rows = ngp::train::config::density_output_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "rgb_mlp.input.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_input_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::rgb_input_width},
                SafetensorsTensor{.name = "rgb_mlp.hidden.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_hidden_weight_offset, .rows = ngp::train::config::mlp_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "rgb_mlp.output.weight", .param_offset = ngp::train::config::network_parameter_layout.rgb_output_weight_offset, .rows = ngp::train::config::network_output_width, .cols = ngp::train::config::mlp_width},
                SafetensorsTensor{.name = "hash_grid.params", .param_offset = ngp::train::config::network_parameter_layout.grid_param_offset, .rows = ngp::train::config::network_parameter_layout.grid_offsets[ngp::train::config::grid_n_levels], .cols = ngp::train::config::grid_features_per_level},
            });

            std::string grid_offsets_text;
            for (std::uint32_t i = 0u; i < ngp::train::config::grid_n_levels + 1u; ++i) {
                if (!grid_offsets_text.empty()) grid_offsets_text += ",";
                grid_offsets_text += std::format("{}", ngp::train::config::network_parameter_layout.grid_offsets[i]);
            }

            nlohmann::json expected_metadata             = nlohmann::json::object();
            expected_metadata["format"]                  = "instant-ngp-new.weights.v2";
            expected_metadata["grid_n_levels"]           = std::format("{}", ngp::train::config::grid_n_levels);
            expected_metadata["grid_features_per_level"] = std::format("{}", ngp::train::config::grid_features_per_level);
            expected_metadata["grid_base_resolution"]    = std::format("{}", ngp::train::config::grid_base_resolution);
            expected_metadata["grid_log2_hashmap_size"]  = std::format("{}", ngp::train::config::grid_log2_hashmap_size);
            expected_metadata["mlp_width"]               = std::format("{}", ngp::train::config::mlp_width);
            expected_metadata["density_hidden_layers"]   = std::format("{}", ngp::train::config::density_hidden_layers);
            expected_metadata["rgb_hidden_layers"]       = std::format("{}", ngp::train::config::rgb_hidden_layers);
            expected_metadata["density_output_width"]    = std::format("{}", ngp::train::config::density_output_width);
            expected_metadata["direction_output_width"]  = std::format("{}", ngp::train::config::direction_output_width);
            expected_metadata["rgb_input_width"]         = std::format("{}", ngp::train::config::rgb_input_width);
            expected_metadata["network_output_width"]    = std::format("{}", ngp::train::config::network_output_width);
            expected_metadata["grid_offsets"]            = grid_offsets_text;
            expected_metadata["scene_scale"]             = std::format("{:.9g}", this->host.scene_scale);

            const std::uintmax_t file_size = std::filesystem::file_size(path);
            if (file_size < sizeof(std::uint64_t)) throw std::runtime_error{"weights file is too small for a safetensors header."};

            std::ifstream input{path, std::ios::binary};
            if (!input) throw std::runtime_error{std::format("failed to open weights file '{}'.", path.string())};

            std::uint64_t header_size = 0u;
            input.read(reinterpret_cast<char*>(&header_size), sizeof(header_size));
            if (!input) throw std::runtime_error{"failed to read safetensors header length."};
            if (header_size == 0u || header_size > 100ull * 1024ull * 1024ull) throw std::runtime_error{"invalid safetensors header length."};
            if (sizeof(std::uint64_t) + header_size > file_size) throw std::runtime_error{"safetensors header length exceeds file size."};

            std::string header_text(header_size, '\0');
            input.read(header_text.data(), static_cast<std::streamsize>(header_text.size()));
            if (!input) throw std::runtime_error{"failed to read safetensors header."};
            if (header_text.empty() || header_text.front() != '{') throw std::runtime_error{"safetensors header must begin with '{'."};

            const nlohmann::json header = nlohmann::json::parse(header_text);
            if (!header.is_object()) throw std::runtime_error{"safetensors header must be a JSON object."};
            if (header.size() != tensors.size() + 1uz) throw std::runtime_error{"safetensors header contains unexpected tensors."};
            if (!header.contains("__metadata__") || !header.at("__metadata__").is_object()) throw std::runtime_error{"safetensors metadata is missing."};
            if (header.at("__metadata__") != expected_metadata) throw std::runtime_error{"safetensors metadata does not match the current InstantNGP configuration."};

            std::uint64_t expected_data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::string tensor_name{tensor.name};
                if (!header.contains(tensor_name)) throw std::runtime_error{std::format("safetensors tensor '{}' is missing.", tensor_name)};
                const nlohmann::json& tensor_header = header.at(tensor_name);
                if (!tensor_header.is_object() || tensor_header.size() != 3uz) throw std::runtime_error{std::format("safetensors tensor '{}' has an invalid header.", tensor_name)};
                if (!tensor_header.contains("dtype") || !tensor_header.at("dtype").is_string() || tensor_header.at("dtype").get<std::string>() != "F32") throw std::runtime_error{std::format("safetensors tensor '{}' must use dtype F32.", tensor_name)};
                if (!tensor_header.contains("shape") || !tensor_header.at("shape").is_array() || tensor_header.at("shape").size() != 2uz) throw std::runtime_error{std::format("safetensors tensor '{}' has an invalid shape.", tensor_name)};
                if ((!tensor_header.at("shape").at(0uz).is_number_integer() && !tensor_header.at("shape").at(0uz).is_number_unsigned()) || (!tensor_header.at("shape").at(1uz).is_number_integer() && !tensor_header.at("shape").at(1uz).is_number_unsigned())) throw std::runtime_error{std::format("safetensors tensor '{}' shape must contain integer dimensions.", tensor_name)};
                const std::int64_t actual_rows = tensor_header.at("shape").at(0uz).get<std::int64_t>();
                const std::int64_t actual_cols = tensor_header.at("shape").at(1uz).get<std::int64_t>();
                if (actual_rows < 0 || actual_cols < 0 || static_cast<std::uint64_t>(actual_rows) != tensor.rows || static_cast<std::uint64_t>(actual_cols) != tensor.cols) throw std::runtime_error{std::format("safetensors tensor '{}' shape mismatch.", tensor_name)};
                if (!tensor_header.contains("data_offsets") || !tensor_header.at("data_offsets").is_array() || tensor_header.at("data_offsets").size() != 2uz) throw std::runtime_error{std::format("safetensors tensor '{}' has invalid data_offsets.", tensor_name)};
                if ((!tensor_header.at("data_offsets").at(0uz).is_number_integer() && !tensor_header.at("data_offsets").at(0uz).is_number_unsigned()) || (!tensor_header.at("data_offsets").at(1uz).is_number_integer() && !tensor_header.at("data_offsets").at(1uz).is_number_unsigned())) throw std::runtime_error{std::format("safetensors tensor '{}' offsets must be integers.", tensor_name)};
                const std::int64_t actual_begin_signed = tensor_header.at("data_offsets").at(0uz).get<std::int64_t>();
                const std::int64_t actual_end_signed   = tensor_header.at("data_offsets").at(1uz).get<std::int64_t>();
                if (actual_begin_signed < 0 || actual_end_signed < 0) throw std::runtime_error{std::format("safetensors tensor '{}' offsets must be non-negative.", tensor_name)};
                const std::uint64_t actual_begin = static_cast<std::uint64_t>(actual_begin_signed);
                const std::uint64_t actual_end   = static_cast<std::uint64_t>(actual_end_signed);
                const std::uint64_t byte_count   = tensor.rows * tensor.cols * sizeof(float);
                if (actual_begin != expected_data_offset || actual_end != actual_begin + byte_count) throw std::runtime_error{std::format("safetensors tensor '{}' data_offsets mismatch.", tensor_name)};
                expected_data_offset += byte_count;
            }

            const std::uint64_t file_data_size = file_size - sizeof(std::uint64_t) - header_size;
            if (expected_data_offset != file_data_size) throw std::runtime_error{"safetensors data buffer size does not match tensor offsets."};

            std::vector<char> data(file_data_size);
            if (!data.empty()) input.read(data.data(), static_cast<std::streamsize>(data.size()));
            if (!input) throw std::runtime_error{"failed to read safetensors tensor data."};

            std::vector host_params(ngp::train::config::network_parameter_layout.total_param_count, 0.0f);
            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::uint64_t byte_count = tensor.rows * tensor.cols * sizeof(float);
                std::memcpy(host_params.data() + tensor.param_offset, data.data() + data_offset, byte_count);
                data_offset += byte_count;
            }

            for (const float value : host_params)
                if (!std::isfinite(value)) throw std::runtime_error{"weights file contains non-finite values."};

            cuda::upload_trainable_parameters(host_params.data(), this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            this->host.density_grid_ema_step = 0u;
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace ngp::train
