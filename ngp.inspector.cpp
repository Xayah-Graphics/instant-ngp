module;
#include "ngp.train.h"
#include "ngp.inspector.h"

module ngp.inspector;
import std;
import ngp.train;

namespace ngp::inspector {
    Inspector::Inspector(const train::InstantNGP& trainer) : trainer{std::addressof(trainer)} {}

    std::expected<ColorGridSampleStats, std::string> Inspector::sample_color_grid(ColorGridSampleRequest request) const {
        try {
            if (request.dimensions != std::array{train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size}) throw std::runtime_error{"color grid sample dimensions must match the direct density grid dimensions."};
            if (request.encoding != ColorGridEncoding::MortonFloat32x3) throw std::runtime_error{"color grid sample only supports MortonFloat32x3 encoding."};
            if (request.output_rgb == nullptr) throw std::runtime_error{"color grid sample output pointer must not be null."};
            constexpr std::uint64_t expected_byte_size = static_cast<std::uint64_t>(train::config::nerf_grid_cells) * 3u * sizeof(float);
            if (request.byte_size < expected_byte_size) throw std::runtime_error{"color grid sample output buffer is too small."};
            const float direction_length = std::sqrt(request.reference_direction[0] * request.reference_direction[0] + request.reference_direction[1] * request.reference_direction[1] + request.reference_direction[2] * request.reference_direction[2]);
            if (!std::isfinite(direction_length) || direction_length <= 0.0f) throw std::runtime_error{"color grid sample reference direction must be finite and non-zero."};
            const float reference_x = request.reference_direction[0] / direction_length;
            const float reference_y = request.reference_direction[1] / direction_length;
            const float reference_z = request.reference_direction[2] / direction_length;
            cuda::sample_color_grid(request.dimensions[0], request.dimensions[1], request.dimensions[2], reference_x, reference_y, reference_z, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.rgb_input, this->trainer->device.network_output, request.output_rgb);
            return ColorGridSampleStats{
                .dimensions = request.dimensions,
                .byte_size  = expected_byte_size,
                .revision   = this->trainer->host.current_step + 1u,
                .encoding   = ColorGridEncoding::MortonFloat32x3,
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    DensityGridDeviceView Inspector::density_grid_device_view() const {
        return DensityGridDeviceView{
            .dimensions             = {train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size},
            .cell_count             = train::config::nerf_grid_cells,
            .values                 = this->trainer->device.density_grid_values,
            .byte_size              = static_cast<std::uint64_t>(train::config::nerf_grid_cells) * sizeof(float),
            .revision               = this->trainer->host.density_grid_ema_step,
            .optical_thickness_step = train::config::min_cone_stepsize,
            .encoding               = DensityGridEncoding::MortonFloat32,
            .initialized            = this->trainer->host.density_grid_ema_step > 0u,
        };
    }

    OccupancyGridDeviceView Inspector::occupancy_grid_device_view() const {
        return OccupancyGridDeviceView{
            .dimensions     = {train::config::nerf_grid_size, train::config::nerf_grid_size, train::config::nerf_grid_size},
            .cell_count     = train::config::nerf_grid_cells,
            .bitfield       = this->trainer->device.occupancy,
            .bitfield_bytes = train::config::nerf_grid_cells / 8u,
            .occupied_cells = this->trainer->host.density_grid_occupied_cells,
            .revision       = this->trainer->host.density_grid_ema_step,
            .encoding       = OccupancyGridEncoding::MortonBitfield,
            .initialized    = this->trainer->host.density_grid_ema_step > 0u,
        };
    }

    SamplerBatchDeviceView Inspector::sampler_batch_device_view() const {
        std::uint32_t ray_count = 0u;
        if (this->trainer->host.current_step != 0u && this->trainer->host.measured_sample_count != 0u) {
            if (this->trainer->device.ray_counter == nullptr) throw std::runtime_error{"sampler batch ray counter buffer is null."};
            cuda::read_counter(this->trainer->device.ray_counter, ray_count);
        }
        const bool initialized = this->trainer->host.current_step != 0u && ray_count != 0u && this->trainer->host.measured_sample_count != 0u;
        return SamplerBatchDeviceView{
            .current_step           = this->trainer->host.current_step,
            .ray_count              = ray_count,
            .compacted_sample_count = this->trainer->host.measured_sample_count,
            .rays                   = this->trainer->device.rays,
            .numsteps               = this->trainer->device.numsteps,
            .compacted_sample_coords = this->trainer->device.compacted_sample_coords,
            .loss_values            = this->trainer->device.loss_values,
            .revision               = this->trainer->host.current_step,
            .initialized            = initialized,
        };
    }

    std::expected<SamplerVisualizationStats, std::string> Inspector::write_sampler_visualization(SamplerVisualizationRequest request) const {
        try {
            const SamplerBatchDeviceView view = this->sampler_batch_device_view();
            if (!view.initialized) throw std::runtime_error{"sampler batch has not been initialized."};
            if (view.rays == nullptr || view.numsteps == nullptr || view.compacted_sample_coords == nullptr || view.loss_values == nullptr) throw std::runtime_error{"sampler batch device view contains null buffers."};
            if (!std::isfinite(request.point_radius) || request.point_radius <= 0.0f) throw std::runtime_error{"sampler point radius must be finite and positive."};
            if (!std::isfinite(request.ray_width) || request.ray_width <= 0.0f) throw std::runtime_error{"sampler ray width must be finite and positive."};
            if (request.width_mode > 1u) throw std::runtime_error{"sampler ray width mode is invalid."};
            if (view.compacted_sample_count > std::numeric_limits<std::uint64_t>::max() / SamplerPointInstanceBytes) throw std::runtime_error{"sampler point byte size overflows uint64."};
            if (view.ray_count > std::numeric_limits<std::uint64_t>::max() / SamplerSegmentInstanceBytes) throw std::runtime_error{"sampler segment byte size overflows uint64."};
            const std::uint64_t expected_point_bytes = static_cast<std::uint64_t>(view.compacted_sample_count) * SamplerPointInstanceBytes;
            const std::uint64_t expected_segment_bytes = static_cast<std::uint64_t>(view.ray_count) * SamplerSegmentInstanceBytes;
            const bool writes_points = request.point_instances != nullptr || request.point_byte_size != 0u;
            const bool writes_segments = request.segment_instances != nullptr || request.segment_byte_size != 0u;
            if (!writes_points && !writes_segments) throw std::runtime_error{"sampler visualization request must include points or segments."};
            if (writes_points && request.point_instances == nullptr) throw std::runtime_error{"sampler point output pointer must not be null."};
            if (writes_points && request.point_byte_size < expected_point_bytes) throw std::runtime_error{"sampler point output buffer is too small."};
            if (writes_segments && request.segment_instances == nullptr) throw std::runtime_error{"sampler segment output pointer must not be null."};
            if (writes_segments && request.segment_byte_size < expected_segment_bytes) throw std::runtime_error{"sampler segment output buffer is too small."};
            cuda::fill_sampler_visualization(view.ray_count, view.compacted_sample_count, view.rays, view.numsteps, view.compacted_sample_coords, view.loss_values, request.point_radius, request.ray_width, request.width_mode, request.point_instances, request.point_byte_size, request.segment_instances, request.segment_byte_size);
            return SamplerVisualizationStats{
                .ray_count         = view.ray_count,
                .point_count       = view.compacted_sample_count,
                .point_byte_size   = writes_points ? expected_point_bytes : 0u,
                .segment_byte_size = writes_segments ? expected_segment_bytes : 0u,
                .revision          = view.revision,
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<EvaluationPreviewResult, std::string> Inspector::evaluate_preview(const EvaluationPreviewRequest request) const {
        try {
            if (request.frame_set.empty()) throw std::runtime_error{"evaluation preview frame set must not be empty."};
            const train::InstantNGP::HostFrameSet* host_frame_set     = nullptr;
            const train::InstantNGP::DeviceFrameSet* device_frame_set = nullptr;
            for (std::size_t frame_set_index = 0uz; frame_set_index < this->trainer->host.frame_sets.size(); ++frame_set_index) {
                if (this->trainer->host.frame_sets[frame_set_index].name == request.frame_set) {
                    host_frame_set   = std::addressof(this->trainer->host.frame_sets[frame_set_index]);
                    device_frame_set = std::addressof(this->trainer->device.frame_sets[frame_set_index]);
                    break;
                }
            }
            if (host_frame_set == nullptr || device_frame_set == nullptr) throw std::runtime_error{std::format("evaluation preview frame set '{}' is not loaded.", request.frame_set)};
            if (request.image_index >= host_frame_set->frame_count) throw std::runtime_error{std::format("evaluation preview image_index {} is out of range for frame set '{}' with {} frames.", request.image_index, request.frame_set, host_frame_set->frame_count)};
            if (this->trainer->host.density_grid_ema_step == 0u) cuda::update_density_grid(device_frame_set->camera, host_frame_set->frame_count, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, 0u, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.network_output, this->trainer->device.density_grid_values, this->trainer->device.density_grid_scratch, this->trainer->device.density_grid_indices, this->trainer->device.density_grid_mean, this->trainer->device.density_grid_occupied_count, this->trainer->device.occupancy, this->trainer->host.density_grid_ema_step, true);
            const auto evaluation_start        = std::chrono::steady_clock::now();
            const std::uint64_t pixel_count_64 = static_cast<std::uint64_t>(host_frame_set->width) * host_frame_set->height;
            double image_loss_sum              = 0.0;
            cuda::run_evaluation(device_frame_set->pixels, device_frame_set->camera, host_frame_set->frame_count, request.image_index, 1u, host_frame_set->width, host_frame_set->height, host_frame_set->focal_x, host_frame_set->focal_y, host_frame_set->principal_x, host_frame_set->principal_y, this->trainer->device.occupancy, this->trainer->device.params, this->trainer->device.sample_coords, this->trainer->device.density_input, this->trainer->device.rgb_input, this->trainer->device.network_output, this->trainer->device.evaluation_numsteps, this->trainer->device.evaluation_sample_counter, this->trainer->device.evaluation_overflow_counter, this->trainer->device.evaluation_loss_sum, nullptr, nullptr, image_loss_sum);

            EvaluationPreviewResult result{
                .frame_set   = std::string{host_frame_set->name},
                .image_index = request.image_index,
                .step        = this->trainer->host.current_step,
                .width       = host_frame_set->width,
                .height      = host_frame_set->height,
                .elapsed_ms  = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - evaluation_start).count(),
            };

            const double mse = image_loss_sum / (static_cast<double>(pixel_count_64) * 3.0);
            if (!std::isfinite(mse)) throw std::runtime_error{"evaluation preview produced non-finite MSE."};
            result.mse  = static_cast<float>(mse);
            result.psnr = mse > 0.0 ? static_cast<float>(-10.0 * std::log10(mse)) : std::numeric_limits<float>::infinity();
            return result;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace ngp::inspector
