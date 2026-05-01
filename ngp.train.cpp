module;
#include "ngp.train.h"

#include "json/json.hpp"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
module ngp.train;
import std;
import ngp.dataset;

namespace ngp::train {
    InstantNGP::InstantNGP(const dataset::NGPDataset& dataset) {
        try {
            {
                if (this->host.frame_count = std::ranges::size(dataset.train); assert(this->host.frame_count > 0uz), this->host.frame_count > 0uz) {
                    std::vector<std::uint8_t> train_pixels;
                    std::vector<float> train_camera;
                    train_pixels.reserve(std::ranges::fold_left(dataset.train | std::views::transform([](const auto& frame) { return frame.rgba.size(); }), 0uz, std::plus{}));
                    train_camera.reserve(std::ranges::size(dataset.train) * 12uz);
                    for (const auto& frame : dataset.train) {
                        train_pixels.append_range(frame.rgba);
                        train_camera.append_range(frame.camera);
                    }
                    cuda::upload_dataset(train_pixels.data(), train_pixels.size(), train_camera.data(), train_camera.size(), this->device.pixels, this->device.camera);
                    this->host.frame_count = std::ranges::size(dataset.train);
                    this->host.width       = std::ranges::begin(dataset.train)->width;
                    this->host.height      = std::ranges::begin(dataset.train)->height;
                    this->host.focal_x     = std::ranges::begin(dataset.train)->focal_x;
                    this->host.focal_y     = std::ranges::begin(dataset.train)->focal_y;
                    this->host.principal_x = std::ranges::begin(dataset.train)->principal_x;
                    this->host.principal_y = std::ranges::begin(dataset.train)->principal_y;
                    this->host.scene_scale = dataset.scene_scale;
                }

                if (this->host.validation_frame_count = std::ranges::size(dataset.validation); this->host.validation_frame_count > 0) {
                    std::vector<std::uint8_t> validation_pixels;
                    std::vector<float> validation_camera;
                    validation_pixels.reserve(std::ranges::fold_left(dataset.validation | std::views::transform([](const auto& frame) { return frame.rgba.size(); }), 0uz, std::plus{}));
                    validation_camera.reserve(std::ranges::size(dataset.validation) * 12uz);
                    for (const auto& frame : dataset.validation) {
                        validation_pixels.append_range(frame.rgba);
                        validation_camera.append_range(frame.camera);
                    }
                    cuda::upload_dataset(validation_pixels.data(), validation_pixels.size(), validation_camera.data(), validation_camera.size(), this->device.validation_pixels, this->device.validation_camera);
                    this->host.validation_frame_count = std::ranges::size(dataset.validation);
                    this->host.validation_width       = std::ranges::begin(dataset.validation)->width;
                    this->host.validation_height      = std::ranges::begin(dataset.validation)->height;
                    this->host.validation_focal_x     = std::ranges::begin(dataset.validation)->focal_x;
                    this->host.validation_focal_y     = std::ranges::begin(dataset.validation)->focal_y;
                    this->host.validation_principal_x = std::ranges::begin(dataset.validation)->principal_x;
                    this->host.validation_principal_y = std::ranges::begin(dataset.validation)->principal_y;
                }

                if (this->host.test_frame_count = std::ranges::size(dataset.test); this->host.test_frame_count > 0) {
                    std::vector<std::uint8_t> test_pixels;
                    std::vector<float> test_camera;
                    test_pixels.reserve(std::ranges::fold_left(dataset.test | std::views::transform([](const auto& frame) { return frame.rgba.size(); }), 0uz, std::plus{}));
                    test_camera.reserve(std::ranges::size(dataset.test) * 12uz);
                    for (const auto& frame : dataset.test) {
                        test_pixels.append_range(frame.rgba);
                        test_camera.append_range(frame.camera);
                    }
                    cuda::upload_dataset(test_pixels.data(), test_pixels.size(), test_camera.data(), test_camera.size(), this->device.test_pixels, this->device.test_camera);

                    this->host.test_frame_count = std::ranges::size(dataset.test);
                    this->host.test_width       = std::ranges::begin(dataset.test)->width;
                    this->host.test_height      = std::ranges::begin(dataset.test)->height;
                    this->host.test_focal_x     = std::ranges::begin(dataset.test)->focal_x;
                    this->host.test_focal_y     = std::ranges::begin(dataset.test)->focal_y;
                    this->host.test_principal_x = std::ranges::begin(dataset.test)->principal_x;
                    this->host.test_principal_y = std::ranges::begin(dataset.test)->principal_y;
                }
            }

            cuda::Pcg32 training_rng{cuda::config::TRAIN_SEED};
            this->host.density_grid_rng = cuda::Pcg32{training_rng.next_uint()};

            cuda::allocate_sampler_buffers(this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy);
            cuda::allocate_network_buffers(this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
            cuda::allocate_density_grid_buffers(this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count);
            cuda::allocate_training_loss_buffers(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values);
            if (this->host.validation_frame_count != 0u || this->host.test_frame_count != 0u) cuda::allocate_evaluation_buffers(this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum);
            if (this->host.test_frame_count != 0u) cuda::allocate_test_comparison_buffer(this->host.test_width, this->host.test_height, this->device.test_comparison_pixels);
            cuda::allocate_trainable_parameter_buffers(this->device.params_full_precision, this->device.params, this->device.param_gradients);
            cuda::allocate_adam_state(this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            cuda::initialize_mlp_parameters(this->device.params_full_precision, this->device.params, this->device.param_gradients);
            cuda::initialize_grid_parameters(this->device.params_full_precision, this->device.params, this->device.param_gradients);
        } catch (...) {
            cuda::destroy_cublaslt(this->device.cublaslt_handle);
            cuda::free_device_buffers(this->device.pixels, this->device.camera, this->device.validation_pixels, this->device.validation_camera, this->device.test_pixels, this->device.test_camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.test_comparison_pixels, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients,
                this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            throw;
        }
    }

    InstantNGP::~InstantNGP() noexcept {
        cuda::destroy_cublaslt(this->device.cublaslt_handle);
        cuda::free_device_buffers(this->device.pixels, this->device.camera, this->device.validation_pixels, this->device.validation_camera, this->device.test_pixels, this->device.test_camera, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter, this->device.occupancy, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.loss_values, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.test_comparison_pixels, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients,
            this->device.density_input_gradients, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_workspace, this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
    }

    std::expected<TrainStats, std::string> InstantNGP::train(const std::int32_t iters) {
        try {
            const auto train_start            = std::chrono::steady_clock::now();
            std::uint32_t loss_rays_per_batch = this->host.rays_per_batch;
            this->host.density_grid_update_ms = 0.0f;
            for (std::int32_t i = 0; i < iters; ++i) {
                loss_rays_per_batch          = this->host.rays_per_batch;
                float density_grid_update_ms = 0.0f;
                cuda::update_density_grid(this->device.camera, this->host.frame_count, this->host.width, this->host.height, this->host.focal_x, this->host.focal_y, this->host.principal_x, this->host.principal_y, this->host.current_step, this->device.params, this->device.sample_coords, this->device.density_input, this->device.network_output, this->device.density_grid_values, this->device.density_grid_scratch, this->device.density_grid_indices, this->device.density_grid_mean, this->device.density_grid_occupied_count, this->device.occupancy, this->host.density_grid_ema_step, this->host.density_grid_rng, density_grid_update_ms);
                this->host.density_grid_update_ms += density_grid_update_ms;
                cuda::sample_training_batch(this->device.camera, this->host.frame_count, this->host.width, this->host.height, this->host.focal_x, this->host.focal_y, this->host.principal_x, this->host.principal_y, this->host.current_step, this->host.rays_per_batch, this->host.inference_sample_count, this->device.occupancy, this->device.sample_coords, this->device.rays, this->device.ray_indices, this->device.numsteps, this->device.ray_counter, this->device.sample_counter);
                cuda::evaluate_network(this->host.inference_sample_count, this->device.sample_coords, this->device.params, this->device.density_input, this->device.rgb_input, this->device.network_output);
                cuda::compute_training_loss_and_compact_samples(this->host.rays_per_batch, this->host.current_step, this->device.ray_counter, this->device.pixels, this->host.frame_count, this->host.width, this->host.height, this->device.network_output, this->device.compacted_sample_counter, this->device.ray_indices, this->device.rays, this->device.numsteps, this->device.sample_coords, this->device.compacted_sample_coords, this->device.network_output_gradients, this->device.loss_values);
                cuda::pad_compacted_training_batch(this->device.compacted_sample_counter, this->device.compacted_sample_coords, this->device.network_output_gradients);
                cuda::forward_network(this->device.compacted_sample_coords, this->device.params, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output);
                cuda::backward_network(this->device.compacted_sample_coords, this->device.params, this->device.param_gradients, this->device.density_input, this->device.rgb_input, this->device.density_forward_hidden, this->device.rgb_forward_hidden, this->device.network_output, this->device.network_output_gradients, this->device.rgb_output_gradients, this->device.rgb_input_gradients, this->device.density_input_gradients, this->device.density_backward_hidden, this->device.rgb_backward_hidden, this->device.cublaslt_handle, this->device.cublaslt_workspace);
                cuda::step_optimizer(this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
                cuda::read_counter(this->device.sample_counter, this->host.measured_sample_count_before_compaction);
                cuda::read_counter(this->device.compacted_sample_counter, this->host.measured_sample_count);
                if (this->host.measured_sample_count == 0u) {
                    cuda::read_counter(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
                    throw std::runtime_error{std::format("Training stopped unexpectedly. density_grid_occupied_cells={}", this->host.density_grid_occupied_cells)};
                }

                this->host.inference_sample_count = ((std::min(this->host.measured_sample_count_before_compaction, cuda::config::MAX_SAMPLES) + cuda::config::NETWORK_BATCH_GRANULARITY - 1u) / cuda::config::NETWORK_BATCH_GRANULARITY) * cuda::config::NETWORK_BATCH_GRANULARITY;
                this->host.rays_per_batch         = std::min(std::max(((static_cast<std::uint32_t>(std::min((static_cast<std::uint64_t>(this->host.rays_per_batch) * cuda::config::NETWORK_BATCH_SIZE) / this->host.measured_sample_count, static_cast<std::uint64_t>(cuda::config::NETWORK_BATCH_SIZE))) + cuda::config::NETWORK_BATCH_GRANULARITY - 1u) / cuda::config::NETWORK_BATCH_GRANULARITY) * cuda::config::NETWORK_BATCH_GRANULARITY, cuda::config::NETWORK_BATCH_GRANULARITY), cuda::config::NETWORK_BATCH_SIZE);

                ++this->host.current_step;
            }

            float loss_sum = 0.0f;
            cuda::read_loss_sum(this->device.loss_values, loss_rays_per_batch, loss_sum);
            cuda::read_counter(this->device.density_grid_occupied_count, this->host.density_grid_occupied_cells);
            return TrainStats{
                .step                                    = this->host.current_step,
                .rays_per_batch                          = this->host.rays_per_batch,
                .measured_sample_count_before_compaction = this->host.measured_sample_count_before_compaction,
                .measured_sample_count                   = this->host.measured_sample_count,
                .density_grid_occupied_cells             = this->host.density_grid_occupied_cells,
                .loss                                    = loss_sum * static_cast<float>(this->host.measured_sample_count) / static_cast<float>(cuda::config::NETWORK_BATCH_SIZE),
                .elapsed_ms                              = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(),
                .density_grid_update_ms                  = this->host.density_grid_update_ms,
                .density_grid_occupancy_ratio            = static_cast<float>(this->host.density_grid_occupied_cells) / (128.0f * 128.0f * 128.0f),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<ValidationStats, std::string> InstantNGP::validate() const {
        try {
            if (this->host.validation_frame_count == 0u) throw std::runtime_error{"No validation images are available in the current dataset."};

            const auto validation_start          = std::chrono::steady_clock::now();
            const std::uint64_t pixels_per_image = static_cast<std::uint64_t>(this->host.validation_width) * this->host.validation_height;
            const std::uint64_t pixel_count      = pixels_per_image * this->host.validation_frame_count;
            double evaluation_loss_sum           = 0.0;

            cuda::run_evaluation(this->device.validation_pixels, this->device.validation_camera, this->host.validation_frame_count, 0u, this->host.validation_frame_count, this->host.validation_width, this->host.validation_height, this->host.validation_focal_x, this->host.validation_focal_y, this->host.validation_principal_x, this->host.validation_principal_y, this->device.occupancy, this->device.params, this->device.sample_coords, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, nullptr, nullptr, evaluation_loss_sum);

            const double mse = evaluation_loss_sum / (static_cast<double>(pixel_count) * 3.0);
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

    std::expected<TestStats, std::string> InstantNGP::test() const {
        try {
            if (this->host.test_frame_count == 0u) throw std::runtime_error{"No test images are available in the current dataset."};
            const std::filesystem::path output_dir = "test";
            if (output_dir.empty()) throw std::runtime_error{"test output directory must not be empty."};
            if (this->device.test_comparison_pixels == nullptr) throw std::runtime_error{"test comparison image buffer is not initialized."};
            if (std::filesystem::exists(output_dir) && !std::filesystem::is_directory(output_dir)) throw std::runtime_error{std::format("test output path '{}' is not a directory.", output_dir.string())};
            std::filesystem::create_directories(output_dir);
            if (!std::filesystem::is_directory(output_dir)) throw std::runtime_error{std::format("failed to create test output directory '{}'.", output_dir.string())};
            if (this->host.test_width > static_cast<std::uint32_t>(std::numeric_limits<int>::max() / 2) || this->host.test_height > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) throw std::runtime_error{"test image dimensions exceed PNG writer limits."};

            const auto test_start                = std::chrono::steady_clock::now();
            const std::uint64_t pixels_per_image = static_cast<std::uint64_t>(this->host.test_width) * this->host.test_height;
            const std::uint64_t pixel_count      = pixels_per_image * this->host.test_frame_count;
            double evaluation_loss_sum           = 0.0;
            std::vector<std::uint8_t> comparison_image(static_cast<std::size_t>(this->host.test_width) * this->host.test_height * 2uz * 3uz);

            for (std::uint32_t image_index = 0u; image_index < this->host.test_frame_count; ++image_index) {
                double image_loss_sum = 0.0;
                cuda::run_evaluation(this->device.test_pixels, this->device.test_camera, this->host.test_frame_count, image_index, 1u, this->host.test_width, this->host.test_height, this->host.test_focal_x, this->host.test_focal_y, this->host.test_principal_x, this->host.test_principal_y, this->device.occupancy, this->device.params, this->device.sample_coords, this->device.density_input, this->device.rgb_input, this->device.network_output, this->device.evaluation_numsteps, this->device.evaluation_sample_counter, this->device.evaluation_overflow_counter, this->device.evaluation_loss_sum, this->device.test_comparison_pixels, comparison_image.data(), image_loss_sum);
                evaluation_loss_sum += image_loss_sum;

                const std::filesystem::path output_path = output_dir / std::format("test_{:04}.png", image_index);
                const std::string output_path_text      = output_path.string();
                const int output_width                  = static_cast<int>(this->host.test_width * 2u);
                const int output_height                 = static_cast<int>(this->host.test_height);
                if (stbi_write_png(output_path_text.c_str(), output_width, output_height, 3, comparison_image.data(), output_width * 3) == 0) throw std::runtime_error{std::format("failed to write test comparison image '{}'.", output_path_text)};
            }

            const double mse = evaluation_loss_sum / (static_cast<double>(pixel_count) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"test produced non-finite MSE."};

            return TestStats{
                .step                   = this->host.current_step,
                .image_count            = this->host.test_frame_count,
                .comparison_image_count = this->host.test_frame_count,
                .pixel_count            = pixel_count,
                .mse                    = static_cast<float>(mse),
                .psnr                   = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity(),
                .elapsed_ms             = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - test_start).count(),
                .output_dir             = output_dir,
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
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.density_input_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::GRID_OUTPUT_WIDTH},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.density_output_weight_offset, .rows = cuda::config::DENSITY_OUTPUT_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.input.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_input_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::RGB_INPUT_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.hidden.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_hidden_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.output.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_output_weight_offset, .rows = cuda::config::NETWORK_OUTPUT_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "hash_grid.params", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.grid_param_offset, .rows = cuda::config::NETWORK_PARAMETER_LAYOUT.grid_offsets[cuda::config::GRID_N_LEVELS], .cols = cuda::config::GRID_FEATURES_PER_LEVEL},
            });

            std::vector<float> host_params(cuda::config::NETWORK_PARAMETER_LAYOUT.total_param_count);
            cuda::download_trainable_parameters(this->device.params_full_precision, host_params.data());

            std::string grid_offsets_text;
            for (std::uint32_t i = 0u; i < cuda::config::GRID_OFFSET_COUNT; ++i) {
                if (!grid_offsets_text.empty()) grid_offsets_text += ",";
                grid_offsets_text += std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.grid_offsets[i]);
            }

            nlohmann::json metadata               = nlohmann::json::object();
            metadata["format"]                    = "instant-ngp-new.weights.v1";
            metadata["grid_n_levels"]             = std::format("{}", cuda::config::GRID_N_LEVELS);
            metadata["grid_features_per_level"]   = std::format("{}", cuda::config::GRID_FEATURES_PER_LEVEL);
            metadata["grid_base_resolution"]      = std::format("{}", cuda::config::GRID_BASE_RESOLUTION);
            metadata["grid_log2_hashmap_size"]    = std::format("{}", cuda::config::GRID_LOG2_HASHMAP_SIZE);
            metadata["grid_per_level_scale"]      = std::format("{}", cuda::config::GRID_PER_LEVEL_SCALE);
            metadata["grid_log2_per_level_scale"] = std::format("{}", cuda::config::GRID_LOG2_PER_LEVEL_SCALE);
            metadata["mlp_width"]                 = std::format("{}", cuda::config::MLP_WIDTH);
            metadata["density_hidden_layers"]     = std::format("{}", cuda::config::DENSITY_HIDDEN_LAYERS);
            metadata["rgb_hidden_layers"]         = std::format("{}", cuda::config::RGB_HIDDEN_LAYERS);
            metadata["density_output_width"]      = std::format("{}", cuda::config::DENSITY_OUTPUT_WIDTH);
            metadata["direction_output_width"]    = std::format("{}", cuda::config::DIRECTION_OUTPUT_WIDTH);
            metadata["rgb_input_width"]           = std::format("{}", cuda::config::RGB_INPUT_WIDTH);
            metadata["network_output_width"]      = std::format("{}", cuda::config::NETWORK_OUTPUT_WIDTH);
            metadata["grid_offsets"]              = grid_offsets_text;
            metadata["density_param_count"]       = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.density_param_count);
            metadata["rgb_param_count"]           = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_param_count);
            metadata["mlp_param_count"]           = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.mlp_param_count);
            metadata["grid_param_count"]          = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.grid_param_count);
            metadata["total_param_count"]         = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.total_param_count);
            metadata["scene_scale"]               = std::format("{:.9g}", this->host.scene_scale);

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
                SafetensorsTensor{.name = "density_mlp.input.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.density_input_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::GRID_OUTPUT_WIDTH},
                SafetensorsTensor{.name = "density_mlp.output.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.density_output_weight_offset, .rows = cuda::config::DENSITY_OUTPUT_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.input.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_input_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::RGB_INPUT_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.hidden.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_hidden_weight_offset, .rows = cuda::config::MLP_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "rgb_mlp.output.weight", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_output_weight_offset, .rows = cuda::config::NETWORK_OUTPUT_WIDTH, .cols = cuda::config::MLP_WIDTH},
                SafetensorsTensor{.name = "hash_grid.params", .param_offset = cuda::config::NETWORK_PARAMETER_LAYOUT.grid_param_offset, .rows = cuda::config::NETWORK_PARAMETER_LAYOUT.grid_offsets[cuda::config::GRID_N_LEVELS], .cols = cuda::config::GRID_FEATURES_PER_LEVEL},
            });

            std::string grid_offsets_text;
            for (std::uint32_t i = 0u; i < cuda::config::GRID_OFFSET_COUNT; ++i) {
                if (!grid_offsets_text.empty()) grid_offsets_text += ",";
                grid_offsets_text += std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.grid_offsets[i]);
            }

            nlohmann::json expected_metadata               = nlohmann::json::object();
            expected_metadata["format"]                    = "instant-ngp-new.weights.v1";
            expected_metadata["grid_n_levels"]             = std::format("{}", cuda::config::GRID_N_LEVELS);
            expected_metadata["grid_features_per_level"]   = std::format("{}", cuda::config::GRID_FEATURES_PER_LEVEL);
            expected_metadata["grid_base_resolution"]      = std::format("{}", cuda::config::GRID_BASE_RESOLUTION);
            expected_metadata["grid_log2_hashmap_size"]    = std::format("{}", cuda::config::GRID_LOG2_HASHMAP_SIZE);
            expected_metadata["grid_per_level_scale"]      = std::format("{}", cuda::config::GRID_PER_LEVEL_SCALE);
            expected_metadata["grid_log2_per_level_scale"] = std::format("{}", cuda::config::GRID_LOG2_PER_LEVEL_SCALE);
            expected_metadata["mlp_width"]                 = std::format("{}", cuda::config::MLP_WIDTH);
            expected_metadata["density_hidden_layers"]     = std::format("{}", cuda::config::DENSITY_HIDDEN_LAYERS);
            expected_metadata["rgb_hidden_layers"]         = std::format("{}", cuda::config::RGB_HIDDEN_LAYERS);
            expected_metadata["density_output_width"]      = std::format("{}", cuda::config::DENSITY_OUTPUT_WIDTH);
            expected_metadata["direction_output_width"]    = std::format("{}", cuda::config::DIRECTION_OUTPUT_WIDTH);
            expected_metadata["rgb_input_width"]           = std::format("{}", cuda::config::RGB_INPUT_WIDTH);
            expected_metadata["network_output_width"]      = std::format("{}", cuda::config::NETWORK_OUTPUT_WIDTH);
            expected_metadata["grid_offsets"]              = grid_offsets_text;
            expected_metadata["density_param_count"]       = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.density_param_count);
            expected_metadata["rgb_param_count"]           = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.rgb_param_count);
            expected_metadata["mlp_param_count"]           = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.mlp_param_count);
            expected_metadata["grid_param_count"]          = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.grid_param_count);
            expected_metadata["total_param_count"]         = std::format("{}", cuda::config::NETWORK_PARAMETER_LAYOUT.total_param_count);
            expected_metadata["scene_scale"]               = std::format("{:.9g}", this->host.scene_scale);

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

            std::vector host_params(cuda::config::NETWORK_PARAMETER_LAYOUT.total_param_count, 0.0f);
            std::uint64_t data_offset = 0u;
            for (const SafetensorsTensor& tensor : tensors) {
                const std::uint64_t byte_count = tensor.rows * tensor.cols * sizeof(float);
                std::memcpy(host_params.data() + tensor.param_offset, data.data() + data_offset, byte_count);
                data_offset += byte_count;
            }

            for (const float value : host_params)
                if (!std::isfinite(value)) throw std::runtime_error{"weights file contains non-finite values."};

            cuda::upload_trainable_parameters(host_params.data(), this->device.params_full_precision, this->device.params, this->device.param_gradients, this->device.optimizer_first_moments, this->device.optimizer_second_moments, this->device.optimizer_param_steps);
            return {};
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace ngp::train
