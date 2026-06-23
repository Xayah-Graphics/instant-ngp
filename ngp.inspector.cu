#include "ngp.train.h"
#include <cstdint>
#include <cuda/std/algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace ngp::inspector::kernels {
    namespace {
        __device__ float sigmoid(const float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ std::uint32_t color_grid_part1by2(std::uint32_t value) {
            value &= 0x000003ffu;
            value = (value | (value << 16u)) & 0x030000ffu;
            value = (value | (value << 8u)) & 0x0300f00fu;
            value = (value | (value << 4u)) & 0x030c30c3u;
            value = (value | (value << 2u)) & 0x09249249u;
            return value;
        }

        __device__ std::uint32_t color_grid_morton3d(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
            return color_grid_part1by2(x) | (color_grid_part1by2(y) << 1u) | (color_grid_part1by2(z) << 2u);
        }

        __global__ void fill_color_grid_coords_kernel(const std::uint32_t sample_count, const std::uint32_t valid_count, const std::uint32_t offset, const std::uint32_t dim_x, const std::uint32_t dim_y, const std::uint32_t dim_z, const float reference_x, const float reference_y, const float reference_z, float* __restrict__ sample_coords) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const std::uint32_t cell = offset + i % valid_count;
            const std::uint32_t x    = cell % dim_x;
            const std::uint32_t y    = (cell / dim_x) % dim_y;
            const std::uint32_t z    = cell / (dim_x * dim_y);
            float* coord             = sample_coords + static_cast<std::uint64_t>(i) * train::config::sample_coord_floats;
            coord[0]                 = (static_cast<float>(x) + 0.5f) / static_cast<float>(dim_x);
            coord[1]                 = (static_cast<float>(y) + 0.5f) / static_cast<float>(dim_y);
            coord[2]                 = (static_cast<float>(z) + 0.5f) / static_cast<float>(dim_z);
            coord[3]                 = train::config::min_cone_stepsize;
            coord[4]                 = (reference_x + 1.0f) * 0.5f;
            coord[5]                 = (reference_y + 1.0f) * 0.5f;
            coord[6]                 = (reference_z + 1.0f) * 0.5f;
        }

        __global__ void copy_color_grid_output_kernel(const std::uint32_t sample_count, const std::uint32_t offset, const std::uint32_t dim_x, const std::uint32_t dim_y, const std::uint32_t dim_z, const __half* __restrict__ network_output, float* __restrict__ output_rgb) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const std::uint32_t cell                                       = offset + i;
            const std::uint32_t x                                          = cell % dim_x;
            const std::uint32_t y                                          = (cell / dim_x) % dim_y;
            const std::uint32_t z                                          = cell / (dim_x * dim_y);
            const std::uint32_t output_index                               = color_grid_morton3d(x, y, z);
            const __half* output                                           = network_output + static_cast<std::uint64_t>(i) * train::config::network_output_width;
            output_rgb[static_cast<std::uint64_t>(output_index) * 3u + 0u] = sigmoid(__half2float(output[0u]));
            output_rgb[static_cast<std::uint64_t>(output_index) * 3u + 1u] = sigmoid(__half2float(output[1u]));
            output_rgb[static_cast<std::uint64_t>(output_index) * 3u + 2u] = sigmoid(__half2float(output[2u]));
        }
    } // namespace

    void sample_color_grid(const std::uint32_t dim_x, const std::uint32_t dim_y, const std::uint32_t dim_z, const float reference_x, const float reference_y, const float reference_z, const std::uint16_t* params, float* sample_coords, std::uint16_t* density_input, std::uint16_t* rgb_input, std::uint16_t* network_output, float* output_rgb) {
        if (dim_x == 0u || dim_y == 0u || dim_z == 0u || dim_x > 1024u || dim_y > 1024u || dim_z > 1024u || params == nullptr || sample_coords == nullptr || density_input == nullptr || rgb_input == nullptr || network_output == nullptr || output_rgb == nullptr) throw std::runtime_error{"invalid color grid sample input."};
        const std::uint64_t cell_count_64 = static_cast<std::uint64_t>(dim_x) * static_cast<std::uint64_t>(dim_y) * static_cast<std::uint64_t>(dim_z);
        if (cell_count_64 > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"color grid sample cell count exceeds uint32 range."};
        const std::uint32_t cell_count      = static_cast<std::uint32_t>(cell_count_64);
        constexpr std::uint32_t granularity = 16u * train::config::mlp_forward_iters;
        for (std::uint32_t offset = 0u; offset < cell_count; offset += train::config::network_batch_size) {
            const std::uint32_t remaining    = cell_count - offset;
            const std::uint32_t used_count   = ::cuda::std::min(train::config::network_batch_size, remaining);
            const std::uint32_t padded_count = ((used_count + granularity - 1u) / granularity) * granularity;
            fill_color_grid_coords_kernel<<<(padded_count + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(padded_count, used_count, offset, dim_x, dim_y, dim_z, reference_x, reference_y, reference_z, sample_coords);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"fill_color_grid_coords_kernel failed: "} + cudaGetErrorString(status)};
            cuda::evaluate_network(padded_count, sample_coords, params, density_input, rgb_input, network_output);
            copy_color_grid_output_kernel<<<(used_count + train::config::threads_per_block - 1u) / train::config::threads_per_block, train::config::threads_per_block>>>(used_count, offset, dim_x, dim_y, dim_z, reinterpret_cast<const __half*>(network_output), output_rgb);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"copy_color_grid_output_kernel failed: "} + cudaGetErrorString(status)};
        }
    }
} // namespace ngp::inspector::kernels
