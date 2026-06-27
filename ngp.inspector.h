#ifndef NGP_INSPECTOR_H
#define NGP_INSPECTOR_H

#include <cstddef>
#include <cstdint>

namespace ngp::cuda {
    void sample_color_grid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z, float reference_x, float reference_y, float reference_z, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, float* output_rgb);
    void fill_sampler_visualization(std::uint32_t ray_count, std::uint32_t compacted_sample_count, const float* rays, const std::uint32_t* numsteps, const float* compacted_sample_coords, const float* loss_values, float point_radius, float ray_width, std::uint32_t width_mode, std::byte* point_instances, std::uint64_t point_byte_size, std::byte* segment_instances, std::uint64_t segment_byte_size);
} // namespace ngp::cuda

#endif // NGP_INSPECTOR_H
