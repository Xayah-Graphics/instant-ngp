module;
#include "ngp.train.h"
#include <cstdint>

namespace ngp::inspector::kernels {
    void sample_color_grid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z, float reference_x, float reference_y, float reference_z, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, float* output_rgb);
}

module ngp.inspector;
import std;
import ngp.train;

namespace ngp::inspector {
    Inspector::Inspector(const train::InstantNGP& trainer) : trainer{std::addressof(trainer)} {}

    std::expected<ColorGridSampleStats, std::string> Inspector::sample_color_grid(ColorGridSampleRequest request) const {
        try {
            if (request.dimensions != std::array<std::uint32_t, 3u>{train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size}) throw std::runtime_error{"color grid sample dimensions must match the direct density grid dimensions."};
            if (request.encoding != ColorGridEncoding::MortonFloat32x3) throw std::runtime_error{"color grid sample only supports MortonFloat32x3 encoding."};
            if (request.output_rgb == nullptr) throw std::runtime_error{"color grid sample output pointer must not be null."};
            const std::uint64_t expected_byte_size = static_cast<std::uint64_t>(train::config::nerf_grid_cells) * 3u * sizeof(float);
            if (request.byte_size < expected_byte_size) throw std::runtime_error{"color grid sample output buffer is too small."};
            const float direction_length = std::sqrt(request.reference_direction[0] * request.reference_direction[0] + request.reference_direction[1] * request.reference_direction[1] + request.reference_direction[2] * request.reference_direction[2]);
            if (!std::isfinite(direction_length) || direction_length <= 0.0f) throw std::runtime_error{"color grid sample reference direction must be finite and non-zero."};
            const float reference_x = request.reference_direction[0] / direction_length;
            const float reference_y = request.reference_direction[1] / direction_length;
            const float reference_z = request.reference_direction[2] / direction_length;
            kernels::sample_color_grid(request.dimensions[0], request.dimensions[1], request.dimensions[2], reference_x, reference_y, reference_z, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.rgb_input, this->trainer->device.network_output, request.output_rgb);
            return ColorGridSampleStats{
                .dimensions = request.dimensions,
                .byte_size = expected_byte_size,
                .revision = this->trainer->host.current_step + 1u,
                .encoding = ColorGridEncoding::MortonFloat32x3,
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    DensityGridDeviceView Inspector::density_grid_device_view() const {
        return DensityGridDeviceView{
            .dimensions = {train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size},
            .cell_count = train::config::nerf_grid_cells,
            .values = this->trainer->device.density_grid_values,
            .byte_size = static_cast<std::uint64_t>(train::config::nerf_grid_cells) * sizeof(float),
            .revision = this->trainer->host.density_grid_ema_step,
            .optical_thickness_step = train::config::min_cone_stepsize,
            .encoding = DensityGridEncoding::MortonFloat32,
            .initialized = this->trainer->host.density_grid_ema_step > 0u,
        };
    }

    OccupancyGridDeviceView Inspector::occupancy_grid_device_view() const {
        return OccupancyGridDeviceView{
            .dimensions = {train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size},
            .cell_count = train::config::nerf_grid_cells,
            .bitfield = this->trainer->device.occupancy,
            .bitfield_bytes = train::config::nerf_grid_cells / 8u,
            .occupied_cells = this->trainer->host.density_grid_occupied_cells,
            .revision = this->trainer->host.density_grid_ema_step,
            .encoding = OccupancyGridEncoding::MortonBitfield,
            .initialized = this->trainer->host.density_grid_ema_step > 0u,
        };
    }

    std::expected<EvaluationPreviewResult, std::string> Inspector::evaluate_preview(const EvaluationPreviewRequest request) const {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"evaluation preview frame set must not be empty."};
            const train::InstantNGP::HostFrameSet* host_frame_set = nullptr;
            const train::InstantNGP::DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->trainer->host.frame_sets.size(); ++frame_set_index) {
                if (this->trainer->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set = std::addressof(this->trainer->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->trainer->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("evaluation preview frame set '{}' is not loaded.", request.frame_set)};
            if (request.image_index >= host_frame_set->frame_count) throw std::runtime_error{std::format("evaluation preview image_index {} is out of range for frame set '{}' with {} frames.", request.image_index, request.frame_set, host_frame_set->frame_count)};
            if (request.refresh_acceleration) this->trainer->host.density_grid_ema_step = 0u;
            if (this->trainer->host.density_grid_ema_step == 0u) cuda::update_density_grid(device_frame_set->camera, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, 0u, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.network_output, this->trainer->device.density_grid_values, this->trainer->device.density_grid_scratch, this->trainer->device.density_grid_indices, this->trainer->device.density_grid_mean, this->trainer->device.density_grid_occupied_count, this->trainer->device.occupancy, this->trainer->host.density_grid_ema_step, true);
            if (this->trainer->device.comparison_pixels == nullptr) throw std::runtime_error{"evaluation preview comparison image buffer is not initialized."};

            const auto evaluation_start = std::chrono::steady_clock::now();
            const std::uint64_t pixel_count_64 = static_cast<std::uint64_t>(host_frame_set->width) * host_frame_set->height;
            if (pixel_count_64 > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max() / 4u)) throw std::runtime_error{"evaluation preview image is too large."};
            const std::size_t pixel_count = static_cast<std::size_t>(pixel_count_64);
            std::vector<std::uint8_t> comparison_image(pixel_count * 2uz * 3uz);
            double image_loss_sum = 0.0;
            cuda::run_evaluation(device_frame_set->pixels, device_frame_set->camera, host_frame_set->frame_count, request.image_index, 1u, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->trainer->device.occupancy, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.rgb_input, this->trainer->device.network_output, this->trainer->device.evaluation_numsteps, this->trainer->device.evaluation_sample_counter, this->trainer->device.evaluation_overflow_counter, this->trainer->device.evaluation_loss_sum, this->trainer->device.comparison_pixels, comparison_image.data(), image_loss_sum);

            EvaluationPreviewResult result{
                .frame_set = std::string{host_frame_set->name},
                .image_index = request.image_index,
                .step = this->trainer->host.current_step,
                .width = host_frame_set->width,
                .height = host_frame_set->height,
                .elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - evaluation_start).count(),
                .ground_truth_rgba8 = std::vector<std::uint8_t>(pixel_count * 4uz),
                .prediction_rgba8 = std::vector<std::uint8_t>(pixel_count * 4uz),
                .error_rgba8 = std::vector<std::uint8_t>(pixel_count * 4uz),
            };

            const double mse = image_loss_sum / (static_cast<double>(pixel_count) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"evaluation preview produced non-finite MSE."};
            result.mse = static_cast<float>(mse);
            result.psnr = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity();

            for (std::uint32_t y = 0u; y < host_frame_set->height; ++y) {
                for (std::uint32_t x = 0u; x < host_frame_set->width; ++x) {
                    const std::size_t pixel_index = static_cast<std::size_t>(y) * host_frame_set->width + x;
                    const std::size_t source_row = static_cast<std::size_t>(y) * host_frame_set->width * 2uz * 3uz;
                    const std::size_t target_base = source_row + static_cast<std::size_t>(x) * 3uz;
                    const std::size_t prediction_base = source_row + (static_cast<std::size_t>(host_frame_set->width) + x) * 3uz;
                    const std::size_t destination_base = pixel_index * 4uz;
                    for (std::size_t channel = 0uz; channel < 3uz; ++channel) {
                        const std::uint8_t target_value = comparison_image[target_base + channel];
                        const std::uint8_t prediction_value = comparison_image[prediction_base + channel];
                        result.ground_truth_rgba8[destination_base + channel] = target_value;
                        result.prediction_rgba8[destination_base + channel] = prediction_value;
                        result.error_rgba8[destination_base + channel] = static_cast<std::uint8_t>(std::abs(static_cast<int>(prediction_value) - static_cast<int>(target_value)));
                    }
                    result.ground_truth_rgba8[destination_base + 3uz] = 255u;
                    result.prediction_rgba8[destination_base + 3uz] = 255u;
                    result.error_rgba8[destination_base + 3uz] = 255u;
                }
            }
            return result;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
}
