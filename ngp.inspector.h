#ifndef NGP_INSPECTOR_H
#define NGP_INSPECTOR_H

#include <cstdint>

namespace ngp::inspector::kernels {
    void sample_color_grid(std::uint32_t dim_x, std::uint32_t dim_y, std::uint32_t dim_z, float reference_x, float reference_y, float reference_z, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, float* output_rgb);
}

#endif // NGP_INSPECTOR_H
