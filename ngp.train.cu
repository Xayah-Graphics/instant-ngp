#include "ngp.train.h"
#include <chrono>
#include <cmath>
#include <cublasLt.h>
#include <cuda/std/algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <format>
#include <limits>
#include <mma.h>
#include <stdexcept>
#include <string>
#include <vector>

namespace ngp::cuda {
    void free_device_buffers(void** const pointers, const std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] != nullptr) cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    void destroy_cublaslt(void*& handle) noexcept {
        if (handle != nullptr) cublasLtDestroy(static_cast<cublasLtHandle_t>(handle));
        handle = nullptr;
    }

    namespace {
        // Launch configuration.
        inline constexpr std::uint32_t THREADS_PER_BLOCK = 128u;

        // Sampler.
        inline constexpr std::uint32_t NERF_GRID_SIZE                         = 128u;
        inline constexpr std::uint32_t NERF_GRID_CELLS                        = NERF_GRID_SIZE * NERF_GRID_SIZE * NERF_GRID_SIZE;
        inline constexpr std::uint32_t NERF_STEPS                             = 1024u;
        inline constexpr std::uint32_t MAX_RANDOM_SAMPLES_PER_RAY             = 16u;
        inline constexpr std::uint32_t RANDOM_VALUES_PER_THREAD               = 4u;
        inline constexpr std::uint32_t SAMPLE_COORD_FLOATS                    = 7u;
        inline constexpr std::uint32_t RAY_FLOATS                             = 6u;
        inline constexpr float MIN_CONE_STEPSIZE                              = 1.73205080757f / static_cast<float>(NERF_STEPS);
        inline constexpr float NERF_MIN_OPTICAL_THICKNESS                     = 0.01f;
        inline constexpr std::uint32_t DENSITY_GRID_WARMUP_STEPS              = 256u;
        inline constexpr std::uint32_t DENSITY_GRID_SKIP_INTERVAL             = 16u;
        inline constexpr std::uint32_t DENSITY_GRID_MAX_SKIP                  = 16u;
        inline constexpr float DENSITY_GRID_DECAY                             = 0.95f;
        inline constexpr std::uint32_t DENSITY_GRID_WARMUP_SAMPLES            = NERF_GRID_CELLS;
        inline constexpr std::uint32_t DENSITY_GRID_STEADY_UNIFORM_SAMPLES    = NERF_GRID_CELLS / 4u;
        inline constexpr std::uint32_t DENSITY_GRID_STEADY_NONUNIFORM_SAMPLES = NERF_GRID_CELLS / 4u;
        inline constexpr std::uint32_t VALIDATION_TILE_RAYS                   = 4096u;
        inline constexpr std::uint32_t VALIDATION_MAX_SAMPLES                 = VALIDATION_TILE_RAYS * NERF_STEPS;
        static_assert(VALIDATION_MAX_SAMPLES <= config::MAX_SAMPLES);
        static_assert(VALIDATION_MAX_SAMPLES % config::NETWORK_BATCH_GRANULARITY == 0u);

        // Grid encoding.
        inline constexpr std::uint32_t GRID_FORWARD_THREADS   = 512u;
        inline constexpr std::uint32_t GRID_BACKWARD_THREADS  = 256u;
        inline constexpr std::uint32_t GRID_BACKWARD_FEATURES = 2u;
        static_assert(config::GRID_N_LEVELS == 8u);

        // Fully fused MLP.
        inline constexpr std::uint32_t MLP_FORWARD_ITERS       = 8u;
        inline constexpr std::uint32_t MLP_INPUT_WIDTH         = config::GRID_OUTPUT_WIDTH;
        inline constexpr std::uint32_t MLP_OUTPUT_WIDTH        = config::NETWORK_OUTPUT_WIDTH;
        inline constexpr std::uint32_t MLP_WIDTH_BLOCKS        = config::MLP_WIDTH / 16u;
        inline constexpr std::uint32_t MLP_SKEW                = 8u;
        inline constexpr std::uint32_t MLP_INPUT_SKEW          = 8u;
        inline constexpr std::uint32_t MLP_FIRST_LAYER_PARAMS  = config::MLP_WIDTH * MLP_INPUT_WIDTH;
        inline constexpr std::uint32_t MLP_HIDDEN_LAYER_PARAMS = config::MLP_WIDTH * config::MLP_WIDTH;
        inline constexpr std::uint32_t MLP_LAST_LAYER_PARAMS   = MLP_OUTPUT_WIDTH * config::MLP_WIDTH;
        inline constexpr std::uint32_t DENSITY_NETWORK_PARAMS  = MLP_FIRST_LAYER_PARAMS + (config::DENSITY_HIDDEN_LAYERS - 1u) * MLP_HIDDEN_LAYER_PARAMS + MLP_LAST_LAYER_PARAMS;
        inline constexpr std::uint32_t RGB_NETWORK_PARAMS      = MLP_FIRST_LAYER_PARAMS + (config::RGB_HIDDEN_LAYERS - 1u) * MLP_HIDDEN_LAYER_PARAMS + MLP_LAST_LAYER_PARAMS;
        inline constexpr std::size_t CUBLASLT_WORKSPACE_BYTES  = static_cast<std::size_t>(64u) * 1024u * 1024u;
        static_assert(config::RGB_INPUT_WIDTH == MLP_INPUT_WIDTH);

        // Training behavior.
        inline constexpr bool SNAP_TO_PIXEL_CENTERS             = true;
        inline constexpr float TRANSMITTANCE_EPSILON            = 1e-4f;
        inline constexpr float OPTIMIZER_LEARNING_RATE          = 1e-2f;
        inline constexpr float OPTIMIZER_BETA1                  = 0.9f;
        inline constexpr float OPTIMIZER_BETA2                  = 0.99f;
        inline constexpr float OPTIMIZER_EPSILON                = 1e-15f;
        inline constexpr float OPTIMIZER_L2_REG                 = 1e-6f;
        inline constexpr float OPTIMIZER_LOSS_SCALE             = 128.0f;
        inline constexpr float DENSITY_GRADIENT_CLAMP_MIN       = -15.0f;
        inline constexpr float DENSITY_GRADIENT_CLAMP_MAX       = 15.0f;
        inline constexpr float DENSITY_REGULARIZATION_THRESHOLD = -10.0f;
        inline constexpr float DENSITY_REGULARIZATION_MAX_DEPTH = 0.1f;
        inline constexpr float DENSITY_REGULARIZATION_STRENGTH  = 1e-4f;

        struct CublasLtMatmulResources final {
            cublasLtMatmulDesc_t operation_desc   = nullptr;
            cublasLtMatrixLayout_t a_desc         = nullptr;
            cublasLtMatrixLayout_t b_desc         = nullptr;
            cublasLtMatrixLayout_t d_desc         = nullptr;
            cublasLtMatmulPreference_t preference = nullptr;

            CublasLtMatmulResources()                                          = default;
            CublasLtMatmulResources(const CublasLtMatmulResources&)            = delete;
            CublasLtMatmulResources& operator=(const CublasLtMatmulResources&) = delete;

            ~CublasLtMatmulResources() noexcept {
                if (this->preference != nullptr) cublasLtMatmulPreferenceDestroy(this->preference);
                if (this->d_desc != nullptr) cublasLtMatrixLayoutDestroy(this->d_desc);
                if (this->b_desc != nullptr) cublasLtMatrixLayoutDestroy(this->b_desc);
                if (this->a_desc != nullptr) cublasLtMatrixLayoutDestroy(this->a_desc);
                if (this->operation_desc != nullptr) cublasLtMatmulDescDestroy(this->operation_desc);
            }
        };

        __device__ std::uint32_t coherent_prime_hash(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
            return x ^ (y * 2654435761u) ^ (z * 805459861u);
        }

        __device__ std::uint32_t grid_index(const std::uint32_t hashmap_size, const std::uint32_t resolution, const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
            std::uint32_t stride = 1u;
            std::uint32_t index  = 0u;

            if (resolution <= 0x659u) {
                index += x * stride;
                stride *= resolution;
                index += y * stride;
                stride *= resolution;
                index += z * stride;
                stride *= resolution;
            } else {
                stride = 0xFFFFFFFFu;
            }

            if (hashmap_size < stride) index = coherent_prime_hash(x, y, z);
            return index % hashmap_size;
        }

        __device__ void grid_position_fraction(const float input, float& pos, std::uint32_t& pos_grid, const float scale) {
            pos                 = fmaf(scale, input, 0.5f);
            const float floored = floorf(pos);
            pos_grid            = static_cast<std::uint32_t>(static_cast<int>(floored));
            pos -= floored;
        }

        __device__ bool unit_aabb_contains(const float3 pos) {
            return pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f && pos.z >= 0.0f && pos.z <= 1.0f;
        }

        __device__ bool intersect_unit_aabb(const float3 origin, const float3 direction, float& out_tmin) {
            const float3 inv_dir = {1.0f / direction.x, 1.0f / direction.y, 1.0f / direction.z};
            const float3 t0      = {-origin.x * inv_dir.x, -origin.y * inv_dir.y, -origin.z * inv_dir.z};
            const float3 t1      = {(1.0f - origin.x) * inv_dir.x, (1.0f - origin.y) * inv_dir.y, (1.0f - origin.z) * inv_dir.z};

            const float tx_min = fminf(t0.x, t1.x);
            const float tx_max = fmaxf(t0.x, t1.x);
            const float ty_min = fminf(t0.y, t1.y);
            const float ty_max = fmaxf(t0.y, t1.y);
            const float tz_min = fminf(t0.z, t1.z);
            const float tz_max = fmaxf(t0.z, t1.z);

            const float tmin = fmaxf(fmaxf(tx_min, ty_min), tz_min);
            const float tmax = fminf(fminf(tx_max, ty_max), tz_max);
            out_tmin         = fmaxf(tmin, 0.0f);
            return tmax >= out_tmin;
        }

        __device__ bool is_density_grid_occupied(const float3 pos, const std::uint8_t* occupancy) {
            const int x = static_cast<int>(pos.x * static_cast<float>(NERF_GRID_SIZE));
            const int y = static_cast<int>(pos.y * static_cast<float>(NERF_GRID_SIZE));
            const int z = static_cast<int>(pos.z * static_cast<float>(NERF_GRID_SIZE));
            if (x < 0 || x >= static_cast<int>(NERF_GRID_SIZE) || y < 0 || y >= static_cast<int>(NERF_GRID_SIZE) || z < 0 || z >= static_cast<int>(NERF_GRID_SIZE)) return false;
            auto morton_x             = static_cast<std::uint32_t>(x);
            auto morton_y             = static_cast<std::uint32_t>(y);
            auto morton_z             = static_cast<std::uint32_t>(z);
            morton_x                  = (morton_x * 0x00010001u) & 0xFF0000FFu;
            morton_x                  = (morton_x * 0x00000101u) & 0x0F00F00Fu;
            morton_x                  = (morton_x * 0x00000011u) & 0xC30C30C3u;
            morton_x                  = (morton_x * 0x00000005u) & 0x49249249u;
            morton_y                  = (morton_y * 0x00010001u) & 0xFF0000FFu;
            morton_y                  = (morton_y * 0x00000101u) & 0x0F00F00Fu;
            morton_y                  = (morton_y * 0x00000011u) & 0xC30C30C3u;
            morton_y                  = (morton_y * 0x00000005u) & 0x49249249u;
            morton_z                  = (morton_z * 0x00010001u) & 0xFF0000FFu;
            morton_z                  = (morton_z * 0x00000101u) & 0x0F00F00Fu;
            morton_z                  = (morton_z * 0x00000011u) & 0xC30C30C3u;
            morton_z                  = (morton_z * 0x00000005u) & 0x49249249u;
            const std::uint32_t index = morton_x | (morton_y << 1u) | (morton_z << 2u);
            return (occupancy[index / 8u] & (1u << (index % 8u))) != 0u;
        }

        __device__ float advance_to_next_density_voxel(const float t, const float3 pos, const float3 direction, const float3 inv_direction) {
            constexpr auto scale = static_cast<float>(NERF_GRID_SIZE);
            const float3 p       = {(pos.x - 0.5f) * scale, (pos.y - 0.5f) * scale, (pos.z - 0.5f) * scale};
            const float tx       = (floorf(p.x + 0.5f + 0.5f * copysignf(1.0f, direction.x)) - p.x) * inv_direction.x;
            const float ty       = (floorf(p.y + 0.5f + 0.5f * copysignf(1.0f, direction.y)) - p.y) * inv_direction.y;
            const float tz       = (floorf(p.z + 0.5f + 0.5f * copysignf(1.0f, direction.z)) - p.z) * inv_direction.z;
            const float t_target = t + fmaxf(fminf(fminf(tx, ty), tz) / scale, 0.0f);
            return t + ceilf(fmaxf((t_target - t) / MIN_CONE_STEPSIZE, 0.5f)) * MIN_CONE_STEPSIZE;
        }

        __device__ float sigmoid(const float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ float density_activation(const float value) {
            return expf(value);
        }

        __device__ float rgb_activation_derivative(const float value) {
            const float rgb = sigmoid(value);
            return rgb * (1.0f - rgb);
        }

        __device__ float3 srgb_to_linear(const float3 value) {
            return {
                value.x <= 0.04045f ? value.x / 12.92f : powf((value.x + 0.055f) / 1.055f, 2.4f),
                value.y <= 0.04045f ? value.y / 12.92f : powf((value.y + 0.055f) / 1.055f, 2.4f),
                value.z <= 0.04045f ? value.z / 12.92f : powf((value.z + 0.055f) / 1.055f, 2.4f),
            };
        }

        __device__ float3 linear_to_srgb(const float3 value) {
            return {
                value.x < 0.0031308f ? 12.92f * value.x : 1.055f * powf(value.x, 0.41666f) - 0.055f,
                value.y < 0.0031308f ? 12.92f * value.y : 1.055f * powf(value.y, 0.41666f) - 0.055f,
                value.z < 0.0031308f ? 12.92f * value.z : 1.055f * powf(value.z, 0.41666f) - 0.055f,
            };
        }

        __device__ float4 read_premultiplied_linear_rgba(const std::uint32_t pixel_x, const std::uint32_t pixel_y, const std::uint32_t image, const std::uint32_t width, const std::uint32_t height, const std::uint8_t* pixels) {
            const std::uint32_t rgba32 = reinterpret_cast<const std::uint32_t*>(pixels)[pixel_x + static_cast<std::uint64_t>(pixel_y) * width + static_cast<std::uint64_t>(image) * width * height];
            float4 result              = {
                static_cast<float>((rgba32 & 0x000000FFu) >> 0u) * (1.0f / 255.0f),
                static_cast<float>((rgba32 & 0x0000FF00u) >> 8u) * (1.0f / 255.0f),
                static_cast<float>((rgba32 & 0x00FF0000u) >> 16u) * (1.0f / 255.0f),
                static_cast<float>((rgba32 & 0xFF000000u) >> 24u) * (1.0f / 255.0f),
            };
            const float3 linear_rgb = srgb_to_linear({result.x, result.y, result.z});
            result.x                = linear_rgb.x * result.w;
            result.y                = linear_rgb.y * result.w;
            result.z                = linear_rgb.z * result.w;
            return result;
        }

        __device__ std::uint32_t training_image_index(const std::uint32_t ray_index, const std::uint32_t rays_per_batch, const std::uint32_t frame_count) {
            return static_cast<std::uint32_t>((static_cast<std::uint64_t>(ray_index) * frame_count) / rays_per_batch) % frame_count;
        }

        __device__ void sample_training_pixel(Pcg32& rng, const std::uint32_t width, const std::uint32_t height, float& out_u, float& out_v) {
            out_u = rng.next_float();
            out_v = rng.next_float();
            if (!SNAP_TO_PIXEL_CENTERS) return;

            const std::uint32_t pixel_x = ::cuda::std::min(static_cast<std::uint32_t>(out_u * static_cast<float>(width)), width - 1u);
            const std::uint32_t pixel_y = ::cuda::std::min(static_cast<std::uint32_t>(out_v * static_cast<float>(height)), height - 1u);
            out_u                       = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
            out_v                       = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
        }

        __global__ void mark_untrained_density_grid_kernel(float* __restrict__ density_grid_values, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const float* __restrict__ camera) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NERF_GRID_CELLS) return;

            std::uint32_t x = i >> 0u;
            x               = x & 0x49249249u;
            x               = (x | (x >> 2u)) & 0xC30C30C3u;
            x               = (x | (x >> 4u)) & 0x0F00F00Fu;
            x               = (x | (x >> 8u)) & 0xFF0000FFu;
            x               = (x | (x >> 16u)) & 0x0000FFFFu;
            std::uint32_t y = i >> 1u;
            y               = y & 0x49249249u;
            y               = (y | (y >> 2u)) & 0xC30C30C3u;
            y               = (y | (y >> 4u)) & 0x0F00F00Fu;
            y               = (y | (y >> 8u)) & 0xFF0000FFu;
            y               = (y | (y >> 16u)) & 0x0000FFFFu;
            std::uint32_t z = i >> 2u;
            z               = z & 0x49249249u;
            z               = (z | (z >> 2u)) & 0xC30C30C3u;
            z               = (z | (z >> 4u)) & 0x0F00F00Fu;
            z               = (z | (z >> 8u)) & 0xFF0000FFu;
            z               = (z | (z >> 16u)) & 0x0000FFFFu;

            constexpr float voxel_size = 1.0f / static_cast<float>(NERF_GRID_SIZE);
            const float3 cell_min      = {static_cast<float>(x) * voxel_size, static_cast<float>(y) * voxel_size, static_cast<float>(z) * voxel_size};
            bool trained               = false;

            for (std::uint32_t frame = 0u; frame < frame_count && !trained; ++frame) {
                const float* frame_camera = camera + static_cast<std::uint64_t>(frame) * 12u;
                const float3 camera_x     = {frame_camera[0], frame_camera[1], frame_camera[2]};
                const float3 camera_y     = {frame_camera[3], frame_camera[4], frame_camera[5]};
                const float3 camera_z     = {frame_camera[6], frame_camera[7], frame_camera[8]};
                const float3 origin       = {frame_camera[9], frame_camera[10], frame_camera[11]};

                for (std::uint32_t corner = 0u; corner < 8u && !trained; ++corner) {
                    const float3 pos = {
                        cell_min.x + ((corner & 1u) != 0u ? voxel_size : 0.0f),
                        cell_min.y + ((corner & 2u) != 0u ? voxel_size : 0.0f),
                        cell_min.z + ((corner & 4u) != 0u ? voxel_size : 0.0f),
                    };
                    const float3 relative = {pos.x - origin.x, pos.y - origin.y, pos.z - origin.z};
                    const float distance  = norm3df(relative.x, relative.y, relative.z);
                    if (distance <= 0.0f) continue;

                    const float local_x = relative.x * camera_x.x + relative.y * camera_x.y + relative.z * camera_x.z;
                    const float local_y = relative.x * camera_y.x + relative.y * camera_y.y + relative.z * camera_y.z;
                    const float local_z = relative.x * camera_z.x + relative.y * camera_z.y + relative.z * camera_z.z;
                    if (local_z / distance < 1e-4f) continue;

                    const float u = local_x * focal_length / (local_z * static_cast<float>(width)) + 0.5f;
                    const float v = local_y * focal_length / (local_z * static_cast<float>(height)) + 0.5f;
                    if (u <= 0.0f || v <= 0.0f || u >= 1.0f || v >= 1.0f) continue;

                    const float ray_x    = (u - 0.5f) * static_cast<float>(width) / focal_length;
                    const float ray_y    = (v - 0.5f) * static_cast<float>(height) / focal_length;
                    const float3 ray_dir = {
                        camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                        camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                        camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
                    };
                    const float ray_length = norm3df(ray_dir.x, ray_dir.y, ray_dir.z);
                    if (ray_length <= 0.0f) continue;

                    const float3 ray_normalized = {ray_dir.x / ray_length, ray_dir.y / ray_length, ray_dir.z / ray_length};
                    const float3 direction      = {relative.x / distance, relative.y / distance, relative.z / distance};
                    const float direction_delta = norm3df(ray_normalized.x - direction.x, ray_normalized.y - direction.y, ray_normalized.z - direction.z);
                    trained                     = direction_delta < 1e-3f;
                }
            }

            density_grid_values[i] = trained ? 0.0f : -1.0f;
        }

        __global__ void generate_density_grid_samples_kernel(const std::uint32_t sample_count, Pcg32 rng, const std::uint32_t density_grid_ema_step, const float threshold, const float* __restrict__ density_grid_values, float* __restrict__ sample_coords, std::uint32_t* __restrict__ density_grid_indices) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            rng.advance(static_cast<std::uint64_t>(i) * RANDOM_VALUES_PER_THREAD);

            std::uint32_t idx = 0u;
            for (std::uint32_t j = 0u; j < 10u; ++j) {
                idx = static_cast<std::uint32_t>(((static_cast<std::uint64_t>(i) + static_cast<std::uint64_t>(density_grid_ema_step) * sample_count) * 56924617ull + static_cast<std::uint64_t>(j) * 19349663ull + 96925573ull) % NERF_GRID_CELLS);
                if (density_grid_values[idx] > threshold) break;
            }

            std::uint32_t x = idx >> 0u;
            x               = x & 0x49249249u;
            x               = (x | (x >> 2u)) & 0xC30C30C3u;
            x               = (x | (x >> 4u)) & 0x0F00F00Fu;
            x               = (x | (x >> 8u)) & 0xFF0000FFu;
            x               = (x | (x >> 16u)) & 0x0000FFFFu;
            std::uint32_t y = idx >> 1u;
            y               = y & 0x49249249u;
            y               = (y | (y >> 2u)) & 0xC30C30C3u;
            y               = (y | (y >> 4u)) & 0x0F00F00Fu;
            y               = (y | (y >> 8u)) & 0xFF0000FFu;
            y               = (y | (y >> 16u)) & 0x0000FFFFu;
            std::uint32_t z = idx >> 2u;
            z               = z & 0x49249249u;
            z               = (z | (z >> 2u)) & 0xC30C30C3u;
            z               = (z | (z >> 4u)) & 0x0F00F00Fu;
            z               = (z | (z >> 8u)) & 0xFF0000FFu;
            z               = (z | (z >> 16u)) & 0x0000FFFFu;

            float* coord            = sample_coords + static_cast<std::uint64_t>(i) * SAMPLE_COORD_FLOATS;
            coord[0]                = (static_cast<float>(x) + rng.next_float()) / static_cast<float>(NERF_GRID_SIZE);
            coord[1]                = (static_cast<float>(y) + rng.next_float()) / static_cast<float>(NERF_GRID_SIZE);
            coord[2]                = (static_cast<float>(z) + rng.next_float()) / static_cast<float>(NERF_GRID_SIZE);
            coord[3]                = MIN_CONE_STEPSIZE;
            coord[4]                = 0.5f;
            coord[5]                = 0.5f;
            coord[6]                = 0.5f;
            density_grid_indices[i] = idx;
        }

        __global__ void splat_density_grid_samples_kernel(const std::uint32_t sample_count, const std::uint32_t* __restrict__ density_grid_indices, const __half* __restrict__ density_output, float* __restrict__ density_grid_scratch) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const float thickness = density_activation(__half2float(density_output[i])) * MIN_CONE_STEPSIZE;
            atomicMax(reinterpret_cast<unsigned int*>(density_grid_scratch + density_grid_indices[i]), __float_as_uint(thickness));
        }

        __global__ void update_density_grid_ema_kernel(const float* __restrict__ density_grid_scratch, float* __restrict__ density_grid_values) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NERF_GRID_CELLS) return;

            const float prev_val   = density_grid_values[i];
            const float importance = density_grid_scratch[i];
            density_grid_values[i] = prev_val < 0.0f ? prev_val : fmaxf(prev_val * DENSITY_GRID_DECAY, importance);
        }

        __global__ void reduce_density_grid_mean_kernel(const float* __restrict__ density_grid_values, float* __restrict__ density_grid_mean) {
            __shared__ float sums[1024];
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            float sum             = 0.0f;
            if (i < NERF_GRID_CELLS / 4u) {
                const float4 values = reinterpret_cast<const float4*>(density_grid_values)[i];
                sum                 = fmaxf(values.x, 0.0f) + fmaxf(values.y, 0.0f) + fmaxf(values.z, 0.0f) + fmaxf(values.w, 0.0f);
            }

            sums[threadIdx.x] = sum;
            __syncthreads();

            for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
                if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
                __syncthreads();
            }

            if (threadIdx.x == 0u) atomicAdd(density_grid_mean, sums[0] / static_cast<float>(NERF_GRID_CELLS));
        }

        __global__ void build_density_grid_bitfield_kernel(const float* __restrict__ density_grid_values, const float* __restrict__ density_grid_mean, std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ density_grid_occupied_count) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= NERF_GRID_CELLS / 8u) return;

            std::uint8_t bits     = 0u;
            const float threshold = fminf(NERF_MIN_OPTICAL_THICKNESS, *density_grid_mean);
            for (std::uint8_t j = 0u; j < 8u; ++j) bits |= density_grid_values[i * 8u + j] > threshold ? static_cast<std::uint8_t>(1u << j) : 0u;

            occupancy[i]                 = bits;
            const std::uint32_t occupied = __popc(static_cast<std::uint32_t>(bits));
            if (occupied != 0u) atomicAdd(density_grid_occupied_count, occupied);
        }

        __global__ void generate_training_samples_kernel(const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint32_t current_step, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const float* __restrict__ camera, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ ray_indices_out, float* __restrict__ rays_out, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= rays_per_batch) return;

            const std::uint32_t image = static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * frame_count) / rays_per_batch) % frame_count;
            const float* frame_camera = camera + static_cast<std::uint64_t>(image) * 12u;

            Pcg32 rng{config::TRAIN_SEED};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(i) * MAX_RANDOM_SAMPLES_PER_RAY);

            float u = rng.next_float();
            float v = rng.next_float();
            if (SNAP_TO_PIXEL_CENTERS) {
                const std::uint32_t pixel_x = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
                const std::uint32_t pixel_y = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
                u                           = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
                v                           = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
            }
            const float ray_x       = (u - 0.5f) * static_cast<float>(width) / focal_length;
            const float ray_y       = (v - 0.5f) * static_cast<float>(height) / focal_length;
            const float3 camera_x   = {frame_camera[0], frame_camera[1], frame_camera[2]};
            const float3 camera_y   = {frame_camera[3], frame_camera[4], frame_camera[5]};
            const float3 camera_z   = {frame_camera[6], frame_camera[7], frame_camera[8]};
            const float3 ray_origin = {frame_camera[9], frame_camera[10], frame_camera[11]};
            float3 ray_direction    = {
                camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
            };
            if (ray_direction.x == 0.0f && ray_direction.y == 0.0f && ray_direction.z == 0.0f) ray_direction = camera_z;

            const float direction_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
            if (direction_length == 0.0f) return;
            const float3 ray_direction_normalized = {ray_direction.x / direction_length, ray_direction.y / direction_length, ray_direction.z / direction_length};

            float tmin = 0.0f;
            if (!intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin)) return;

            constexpr float dt         = MIN_CONE_STEPSIZE;
            const float start_t        = tmin + rng.next_float() * dt;
            const float3 inv_direction = {1.0f / ray_direction_normalized.x, 1.0f / ray_direction_normalized.y, 1.0f / ray_direction_normalized.z};

            std::uint32_t numsteps = 0u;
            float t                = start_t;
            float3 pos             = {};

            while (numsteps < NERF_STEPS) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_density_grid_occupied(pos, occupancy)) {
                    ++numsteps;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base + numsteps > sample_limit) return;

            const std::uint32_t ray_index = atomicAdd(ray_counter, 1u);
            ray_indices_out[ray_index]    = i;

            float* ray_out = rays_out + static_cast<std::uint64_t>(ray_index) * RAY_FLOATS;
            ray_out[0]     = ray_origin.x;
            ray_out[1]     = ray_origin.y;
            ray_out[2]     = ray_origin.z;
            ray_out[3]     = ray_direction.x;
            ray_out[4]     = ray_direction.y;
            ray_out[5]     = ray_direction.z;

            numsteps_out[ray_index * 2u + 0u] = numsteps;
            numsteps_out[ray_index * 2u + 1u] = base;

            const float3 warped_direction = {(ray_direction_normalized.x + 1.0f) * 0.5f, (ray_direction_normalized.y + 1.0f) * 0.5f, (ray_direction_normalized.z + 1.0f) * 0.5f};
            t                             = start_t;
            std::uint32_t j               = 0u;

            while (j < numsteps) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_density_grid_occupied(pos, occupancy)) {
                    float* coord = coords_out + static_cast<std::uint64_t>(base + j) * SAMPLE_COORD_FLOATS;
                    coord[0]     = pos.x;
                    coord[1]     = pos.y;
                    coord[2]     = pos.z;
                    coord[3]     = dt;
                    coord[4]     = warped_direction.x;
                    coord[5]     = warped_direction.y;
                    coord[6]     = warped_direction.z;

                    ++j;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }
        }

        __global__ void generate_validation_samples_kernel(const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t max_samples, const std::uint32_t width, const std::uint32_t height, const float focal_length, const float* __restrict__ validation_camera, const std::uint32_t validation_image_index, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ overflow_counter, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= tile_pixels) return;

            numsteps_out[i * 2u + 0u] = 0u;
            numsteps_out[i * 2u + 1u] = 0u;

            const std::uint32_t global_pixel = pixel_offset + i;
            const std::uint32_t pixel_x      = global_pixel % width;
            const std::uint32_t pixel_y      = global_pixel / width;
            const float u                    = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
            const float v                    = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
            const float* frame_camera        = validation_camera + static_cast<std::uint64_t>(validation_image_index) * 12u;

            const float ray_x       = (u - 0.5f) * static_cast<float>(width) / focal_length;
            const float ray_y       = (v - 0.5f) * static_cast<float>(height) / focal_length;
            const float3 camera_x   = {frame_camera[0], frame_camera[1], frame_camera[2]};
            const float3 camera_y   = {frame_camera[3], frame_camera[4], frame_camera[5]};
            const float3 camera_z   = {frame_camera[6], frame_camera[7], frame_camera[8]};
            const float3 ray_origin = {frame_camera[9], frame_camera[10], frame_camera[11]};
            float3 ray_direction    = {
                camera_x.x * ray_x + camera_y.x * ray_y + camera_z.x,
                camera_x.y * ray_x + camera_y.y * ray_y + camera_z.y,
                camera_x.z * ray_x + camera_y.z * ray_y + camera_z.z,
            };
            if (ray_direction.x == 0.0f && ray_direction.y == 0.0f && ray_direction.z == 0.0f) ray_direction = camera_z;

            const float direction_length = norm3df(ray_direction.x, ray_direction.y, ray_direction.z);
            if (direction_length == 0.0f) return;
            const float3 ray_direction_normalized = {ray_direction.x / direction_length, ray_direction.y / direction_length, ray_direction.z / direction_length};

            float tmin = 0.0f;
            if (!intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin)) return;

            constexpr float dt         = MIN_CONE_STEPSIZE;
            const float start_t        = tmin + 0.5f * dt;
            const float3 inv_direction = {1.0f / ray_direction_normalized.x, 1.0f / ray_direction_normalized.y, 1.0f / ray_direction_normalized.z};

            std::uint32_t numsteps = 0u;
            float t                = start_t;
            float3 pos             = {};

            while (numsteps < NERF_STEPS) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_density_grid_occupied(pos, occupancy)) {
                    ++numsteps;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base + numsteps > max_samples) {
                atomicAdd(overflow_counter, 1u);
                return;
            }

            numsteps_out[i * 2u + 0u] = numsteps;
            numsteps_out[i * 2u + 1u] = base;

            const float3 warped_direction = {(ray_direction_normalized.x + 1.0f) * 0.5f, (ray_direction_normalized.y + 1.0f) * 0.5f, (ray_direction_normalized.z + 1.0f) * 0.5f};
            t                             = start_t;
            std::uint32_t j               = 0u;

            while (j < numsteps) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!unit_aabb_contains(pos)) break;

                if (is_density_grid_occupied(pos, occupancy)) {
                    float* coord = coords_out + static_cast<std::uint64_t>(base + j) * SAMPLE_COORD_FLOATS;
                    coord[0]     = pos.x;
                    coord[1]     = pos.y;
                    coord[2]     = pos.z;
                    coord[3]     = dt;
                    coord[4]     = warped_direction.x;
                    coord[5]     = warped_direction.y;
                    coord[6]     = warped_direction.z;

                    ++j;
                    t += dt;
                } else {
                    t = advance_to_next_density_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }
        }

        __global__ void pad_validation_rollover_coords_kernel(const std::uint32_t used_sample_count, const std::uint32_t padded_sample_count, float* __restrict__ inout) {
            const std::uint32_t i             = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t used_elements = used_sample_count * SAMPLE_COORD_FLOATS;
            if (used_sample_count == 0u || i < used_elements || i >= padded_sample_count * SAMPLE_COORD_FLOATS) return;
            inout[i] = inout[i % used_elements];
        }

        __global__ void accumulate_validation_loss_kernel(const std::uint32_t tile_pixels, const std::uint32_t pixel_offset, const std::uint32_t validation_image_index, const std::uint32_t width, const std::uint32_t height, const std::uint8_t* __restrict__ validation_pixels, const std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, const __half* __restrict__ network_output, double* __restrict__ validation_loss_sum) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            double squared_error  = 0.0;

            if (i < tile_pixels) {
                const std::uint32_t global_pixel = pixel_offset + i;
                const std::uint32_t pixel_x      = global_pixel % width;
                const std::uint32_t pixel_y      = global_pixel / width;
                const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
                const std::uint32_t base         = numsteps_in[i * 2u + 1u];
                float transmittance              = 1.0f;
                float3 rgb_ray                   = {};

                const float* coord   = coords_in + static_cast<std::uint64_t>(base) * SAMPLE_COORD_FLOATS;
                const __half* output = network_output + static_cast<std::uint64_t>(base) * MLP_OUTPUT_WIDTH;

                for (std::uint32_t j = 0u; j < numsteps; ++j) {
                    const float rgb_x   = sigmoid(__half2float(output[0u]));
                    const float rgb_y   = sigmoid(__half2float(output[1u]));
                    const float rgb_z   = sigmoid(__half2float(output[2u]));
                    const float density = density_activation(__half2float(output[3u]));
                    const float alpha   = 1.0f - __expf(-density * coord[3u]);
                    const float weight  = alpha * transmittance;
                    rgb_ray.x += weight * rgb_x;
                    rgb_ray.y += weight * rgb_y;
                    rgb_ray.z += weight * rgb_z;
                    transmittance *= 1.0f - alpha;
                    if (transmittance < TRANSMITTANCE_EPSILON) break;

                    coord += SAMPLE_COORD_FLOATS;
                    output += MLP_OUTPUT_WIDTH;
                }

                const float4 texel       = read_premultiplied_linear_rgba(pixel_x, pixel_y, validation_image_index, width, height, validation_pixels);
                const float3 rgb_target  = linear_to_srgb({texel.x, texel.y, texel.z});
                const float prediction_r = fminf(fmaxf(rgb_ray.x, 0.0f), 1.0f);
                const float prediction_g = fminf(fmaxf(rgb_ray.y, 0.0f), 1.0f);
                const float prediction_b = fminf(fmaxf(rgb_ray.z, 0.0f), 1.0f);
                const float target_r     = fminf(fmaxf(rgb_target.x, 0.0f), 1.0f);
                const float target_g     = fminf(fmaxf(rgb_target.y, 0.0f), 1.0f);
                const float target_b     = fminf(fmaxf(rgb_target.z, 0.0f), 1.0f);
                const double diff_r      = static_cast<double>(prediction_r) - static_cast<double>(target_r);
                const double diff_g      = static_cast<double>(prediction_g) - static_cast<double>(target_g);
                const double diff_b      = static_cast<double>(prediction_b) - static_cast<double>(target_b);
                squared_error            = diff_r * diff_r + diff_g * diff_g + diff_b * diff_b;
            }

            __shared__ double sums[THREADS_PER_BLOCK];
            sums[threadIdx.x] = squared_error;
            __syncthreads();

            for (std::uint32_t stride = blockDim.x / 2u; stride > 0u; stride >>= 1u) {
                if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
                __syncthreads();
            }

            if (threadIdx.x == 0u) atomicAdd(validation_loss_sum, sums[0]);
        }

        __global__ void compute_training_loss_and_compact_kernel(const std::uint32_t rays_per_batch, const std::uint32_t current_step, const std::uint32_t* __restrict__ ray_counter, const std::uint8_t* __restrict__ pixels, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const __half* __restrict__ network_output, std::uint32_t* __restrict__ compacted_sample_counter, const std::uint32_t* __restrict__ ray_indices_in, const float* __restrict__ rays_in, std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, float* __restrict__ coords_out, __half* __restrict__ dloss_doutput, float* __restrict__ loss_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= *ray_counter) return;

            std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
            std::uint32_t base     = numsteps_in[i * 2u + 1u];

            const float* coord_in = coords_in + static_cast<std::uint64_t>(base) * SAMPLE_COORD_FLOATS;
            const __half* output  = network_output + static_cast<std::uint64_t>(base) * MLP_OUTPUT_WIDTH;

            float transmittance              = 1.0f;
            float3 rgb_ray                   = {};
            std::uint32_t compacted_numsteps = 0u;
            const float* ray                 = rays_in + static_cast<std::uint64_t>(i) * RAY_FLOATS;
            const float3 ray_origin          = {ray[0], ray[1], ray[2]};

            for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
                if (transmittance < TRANSMITTANCE_EPSILON) break;

                const float rgb_x   = sigmoid(__half2float(output[0u]));
                const float rgb_y   = sigmoid(__half2float(output[1u]));
                const float rgb_z   = sigmoid(__half2float(output[2u]));
                const float density = density_activation(__half2float(output[3u]));
                const float alpha   = 1.0f - __expf(-density * coord_in[3u]);
                const float weight  = alpha * transmittance;
                rgb_ray.x += weight * rgb_x;
                rgb_ray.y += weight * rgb_y;
                rgb_ray.z += weight * rgb_z;
                transmittance *= 1.0f - alpha;

                output += MLP_OUTPUT_WIDTH;
                coord_in += SAMPLE_COORD_FLOATS;
            }

            const std::uint32_t ray_index = ray_indices_in[i];
            Pcg32 rng{config::TRAIN_SEED};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(ray_index) * MAX_RANDOM_SAMPLES_PER_RAY);

            const std::uint32_t image = training_image_index(ray_index, rays_per_batch, frame_count);
            float u                   = 0.0f;
            float v                   = 0.0f;
            sample_training_pixel(rng, width, height, u, v);

            const float3 background_color  = {rng.next_float(), rng.next_float(), rng.next_float()};
            const std::uint32_t pixel_x    = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
            const std::uint32_t pixel_y    = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
            const float4 texel             = read_premultiplied_linear_rgba(pixel_x, pixel_y, image, width, height, pixels);
            const float3 background_linear = srgb_to_linear(background_color);
            const float3 rgb_target        = linear_to_srgb({texel.x + (1.0f - texel.w) * background_linear.x, texel.y + (1.0f - texel.w) * background_linear.y, texel.z + (1.0f - texel.w) * background_linear.z});

            if (compacted_numsteps == numsteps) {
                rgb_ray.x += transmittance * background_color.x;
                rgb_ray.y += transmittance * background_color.y;
                rgb_ray.z += transmittance * background_color.z;
            }

            output -= static_cast<std::uint64_t>(compacted_numsteps) * MLP_OUTPUT_WIDTH;
            coord_in -= static_cast<std::uint64_t>(compacted_numsteps) * SAMPLE_COORD_FLOATS;

            std::uint32_t compacted_base        = atomicAdd(compacted_sample_counter, compacted_numsteps);
            const std::uint32_t remaining_slots = compacted_base < config::NETWORK_BATCH_SIZE ? config::NETWORK_BATCH_SIZE - compacted_base : 0u;
            compacted_numsteps                  = compacted_numsteps < remaining_slots ? compacted_numsteps : remaining_slots;
            numsteps_in[i * 2u + 0u]            = compacted_numsteps;
            numsteps_in[i * 2u + 1u]            = compacted_base;
            if (compacted_numsteps == 0u) return;

            coords_out += static_cast<std::uint64_t>(compacted_base) * SAMPLE_COORD_FLOATS;
            dloss_doutput += static_cast<std::uint64_t>(compacted_base) * MLP_OUTPUT_WIDTH;

            const float3 difference = {rgb_ray.x - rgb_target.x, rgb_ray.y - rgb_target.y, rgb_ray.z - rgb_target.z};
            const float3 gradient   = {2.0f * difference.x, 2.0f * difference.y, 2.0f * difference.z};
            if (loss_output != nullptr) loss_output[i] = (difference.x * difference.x + difference.y * difference.y + difference.z * difference.z) / (3.0f * static_cast<float>(rays_per_batch));

            const float scaled_loss = OPTIMIZER_LOSS_SCALE / static_cast<float>(rays_per_batch);
            float3 rgb_ray2         = {};
            transmittance           = 1.0f;

            for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
                float* coord_out   = coords_out + static_cast<std::uint64_t>(j) * SAMPLE_COORD_FLOATS;
                const float* coord = coord_in + static_cast<std::uint64_t>(j) * SAMPLE_COORD_FLOATS;
                for (std::uint32_t k = 0u; k < SAMPLE_COORD_FLOATS; ++k) coord_out[k] = coord[k];

                const float3 pos        = {coord[0], coord[1], coord[2]};
                const float depth       = norm3df(pos.x - ray_origin.x, pos.y - ray_origin.y, pos.z - ray_origin.z);
                const float dt          = coord[3u];
                const float mlp_rgb_x   = __half2float(output[0u]);
                const float mlp_rgb_y   = __half2float(output[1u]);
                const float mlp_rgb_z   = __half2float(output[2u]);
                const float mlp_density = __half2float(output[3u]);
                const float3 rgb        = {sigmoid(mlp_rgb_x), sigmoid(mlp_rgb_y), sigmoid(mlp_rgb_z)};
                const float density     = density_activation(mlp_density);
                const float alpha       = 1.0f - __expf(-density * dt);
                const float weight      = alpha * transmittance;
                rgb_ray2.x += weight * rgb.x;
                rgb_ray2.y += weight * rgb.y;
                rgb_ray2.z += weight * rgb.z;
                transmittance *= 1.0f - alpha;

                const float3 suffix        = {rgb_ray.x - rgb_ray2.x, rgb_ray.y - rgb_ray2.y, rgb_ray.z - rgb_ray2.z};
                const float3 dloss_by_drgb = {weight * gradient.x, weight * gradient.y, weight * gradient.z};

                dloss_doutput[0u] = __float2half(scaled_loss * (dloss_by_drgb.x * rgb_activation_derivative(mlp_rgb_x)));
                dloss_doutput[1u] = __float2half(scaled_loss * (dloss_by_drgb.y * rgb_activation_derivative(mlp_rgb_y)));
                dloss_doutput[2u] = __float2half(scaled_loss * (dloss_by_drgb.z * rgb_activation_derivative(mlp_rgb_z)));

                const float density_derivative = expf(::cuda::std::clamp(mlp_density, static_cast<float>(DENSITY_GRADIENT_CLAMP_MIN), static_cast<float>(DENSITY_GRADIENT_CLAMP_MAX)));
                const float dloss_by_dmlp      = density_derivative * (dt * (gradient.x * (transmittance * rgb.x - suffix.x) + gradient.y * (transmittance * rgb.y - suffix.y) + gradient.z * (transmittance * rgb.z - suffix.z)));
                dloss_doutput[3u]              = __float2half(scaled_loss * dloss_by_dmlp + (mlp_density > DENSITY_REGULARIZATION_THRESHOLD && depth < DENSITY_REGULARIZATION_MAX_DEPTH ? DENSITY_REGULARIZATION_STRENGTH : 0.0f));

                dloss_doutput += MLP_OUTPUT_WIDTH;
                output += MLP_OUTPUT_WIDTH;
            }
        }

        __global__ void pad_rollover_coords_kernel(const std::uint32_t* __restrict__ input_count, float* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * SAMPLE_COORD_FLOATS || i >= config::NETWORK_BATCH_SIZE * SAMPLE_COORD_FLOATS || n == 0u) return;
            inout[i] = inout[i % (n * SAMPLE_COORD_FLOATS)];
        }

        __global__ void pad_rollover_network_output_gradients_kernel(const std::uint32_t* __restrict__ input_count, __half* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * MLP_OUTPUT_WIDTH || i >= config::NETWORK_BATCH_SIZE * MLP_OUTPUT_WIDTH || n == 0u) return;
            inout[i] = __float2half(__half2float(inout[i % (n * MLP_OUTPUT_WIDTH)]) * static_cast<float>(n) / static_cast<float>(config::NETWORK_BATCH_SIZE));
        }

        __global__ void encode_grid_forward_kernel(const std::uint32_t sample_count, const std::uint32_t grid_offset_0, const std::uint32_t grid_offset_1, const std::uint32_t grid_offset_2, const std::uint32_t grid_offset_3, const std::uint32_t grid_offset_4, const std::uint32_t grid_offset_5, const std::uint32_t grid_offset_6, const std::uint32_t grid_offset_7, const std::uint32_t grid_offset_8, const float* __restrict__ sample_coords, const __half* __restrict__ grid, __half* __restrict__ encoded_positions) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= sample_count) return;

            const std::uint32_t level             = blockIdx.y;
            const std::uint32_t level_offset      = level == 0u ? grid_offset_0 : level == 1u ? grid_offset_1 : level == 2u ? grid_offset_2 : level == 3u ? grid_offset_3 : level == 4u ? grid_offset_4 : level == 5u ? grid_offset_5 : level == 6u ? grid_offset_6 : grid_offset_7;
            const std::uint32_t next_level_offset = level == 0u ? grid_offset_1 : level == 1u ? grid_offset_2 : level == 2u ? grid_offset_3 : level == 3u ? grid_offset_4 : level == 4u ? grid_offset_5 : level == 5u ? grid_offset_6 : level == 6u ? grid_offset_7 : grid_offset_8;
            grid += level_offset * config::GRID_FEATURES_PER_LEVEL;
            const std::uint32_t hashmap_size = next_level_offset - level_offset;
            const float scale                = exp2f(static_cast<float>(level) * config::GRID_LOG2_PER_LEVEL_SCALE) * static_cast<float>(config::GRID_BASE_RESOLUTION) - 1.0f;
            const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
            const float* sample              = sample_coords + static_cast<std::uint64_t>(i) * SAMPLE_COORD_FLOATS;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            grid_position_fraction(sample[0], pos_x, grid_x, scale);
            grid_position_fraction(sample[1], pos_y, grid_y, scale);
            grid_position_fraction(sample[2], pos_z, grid_z, scale);

            __half result0 = 0.0f;
            __half result1 = 0.0f;
            __half result2 = 0.0f;
            __half result3 = 0.0f;

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * config::GRID_FEATURES_PER_LEVEL;
                const __half weight_half  = weight;
                result0                   = __hfma(weight_half, grid[index + 0u], result0);
                result1                   = __hfma(weight_half, grid[index + 1u], result1);
                result2                   = __hfma(weight_half, grid[index + 2u], result2);
                result3                   = __hfma(weight_half, grid[index + 3u], result3);
            }

            encoded_positions[i + (level * config::GRID_FEATURES_PER_LEVEL + 0u) * sample_count] = result0;
            encoded_positions[i + (level * config::GRID_FEATURES_PER_LEVEL + 1u) * sample_count] = result1;
            encoded_positions[i + (level * config::GRID_FEATURES_PER_LEVEL + 2u) * sample_count] = result2;
            encoded_positions[i + (level * config::GRID_FEATURES_PER_LEVEL + 3u) * sample_count] = result3;
        }

        __global__ void encode_grid_backward_kernel(const std::uint32_t sample_count, const std::uint32_t grid_offset_0, const std::uint32_t grid_offset_1, const std::uint32_t grid_offset_2, const std::uint32_t grid_offset_3, const std::uint32_t grid_offset_4, const std::uint32_t grid_offset_5, const std::uint32_t grid_offset_6, const std::uint32_t grid_offset_7, const std::uint32_t grid_offset_8, const float* __restrict__ sample_coords, const __half* __restrict__ encoded_position_gradients, __half* __restrict__ grid_gradients) {
            const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t i      = (thread * GRID_BACKWARD_FEATURES) / config::GRID_FEATURES_PER_LEVEL;
            if (i >= sample_count) return;

            const std::uint32_t level             = blockIdx.y;
            const std::uint32_t feature           = thread * GRID_BACKWARD_FEATURES - i * config::GRID_FEATURES_PER_LEVEL;
            const std::uint32_t level_offset      = level == 0u ? grid_offset_0 : level == 1u ? grid_offset_1 : level == 2u ? grid_offset_2 : level == 3u ? grid_offset_3 : level == 4u ? grid_offset_4 : level == 5u ? grid_offset_5 : level == 6u ? grid_offset_6 : grid_offset_7;
            const std::uint32_t next_level_offset = level == 0u ? grid_offset_1 : level == 1u ? grid_offset_2 : level == 2u ? grid_offset_3 : level == 3u ? grid_offset_4 : level == 4u ? grid_offset_5 : level == 5u ? grid_offset_6 : level == 6u ? grid_offset_7 : grid_offset_8;
            grid_gradients += level_offset * config::GRID_FEATURES_PER_LEVEL;
            const std::uint32_t hashmap_size = next_level_offset - level_offset;
            const float scale                = exp2f(static_cast<float>(level) * config::GRID_LOG2_PER_LEVEL_SCALE) * static_cast<float>(config::GRID_BASE_RESOLUTION) - 1.0f;
            const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
            const float* sample              = sample_coords + static_cast<std::uint64_t>(i) * SAMPLE_COORD_FLOATS;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            grid_position_fraction(sample[0], pos_x, grid_x, scale);
            grid_position_fraction(sample[1], pos_y, grid_y, scale);
            grid_position_fraction(sample[2], pos_z, grid_z, scale);

            const __half grad0 = encoded_position_gradients[i + (level * config::GRID_FEATURES_PER_LEVEL + feature + 0u) * sample_count];
            const __half grad1 = encoded_position_gradients[i + (level * config::GRID_FEATURES_PER_LEVEL + feature + 1u) * sample_count];

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * config::GRID_FEATURES_PER_LEVEL + feature;
                const __half weight_half  = weight;
                atomicAdd(reinterpret_cast<__half2*>(grid_gradients + index), __halves2half2(__hmul(weight_half, grad0), __hmul(weight_half, grad1)));
            }
        }

        __global__ void encode_spherical_harmonics_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, __half* __restrict__ output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const float* coord = sample_coords + static_cast<std::uint64_t>(i) * SAMPLE_COORD_FLOATS;
            const float x      = coord[4] * 2.0f - 1.0f;
            const float y      = coord[5] * 2.0f - 1.0f;
            const float z      = coord[6] * 2.0f - 1.0f;
            const float xy     = x * y;
            const float xz     = x * z;
            const float yz     = y * z;
            const float x2     = x * x;
            const float y2     = y * y;
            const float z2     = z * z;

            output[i + 0u * sample_count]  = static_cast<__half>(0.28209479177387814f);
            output[i + 1u * sample_count]  = static_cast<__half>(-0.48860251190291987f * y);
            output[i + 2u * sample_count]  = static_cast<__half>(0.48860251190291987f * z);
            output[i + 3u * sample_count]  = static_cast<__half>(-0.48860251190291987f * x);
            output[i + 4u * sample_count]  = static_cast<__half>(1.0925484305920792f * xy);
            output[i + 5u * sample_count]  = static_cast<__half>(-1.0925484305920792f * yz);
            output[i + 6u * sample_count]  = static_cast<__half>(0.94617469575755997f * z2 - 0.31539156525251999f);
            output[i + 7u * sample_count]  = static_cast<__half>(-1.0925484305920792f * xz);
            output[i + 8u * sample_count]  = static_cast<__half>(0.54627421529603959f * x2 - 0.54627421529603959f * y2);
            output[i + 9u * sample_count]  = static_cast<__half>(0.59004358992664352f * y * (-3.0f * x2 + y2));
            output[i + 10u * sample_count] = static_cast<__half>(2.8906114426405538f * xy * z);
            output[i + 11u * sample_count] = static_cast<__half>(0.45704579946446572f * y * (1.0f - 5.0f * z2));
            output[i + 12u * sample_count] = static_cast<__half>(0.3731763325901154f * z * (5.0f * z2 - 3.0f));
            output[i + 13u * sample_count] = static_cast<__half>(0.45704579946446572f * x * (1.0f - 5.0f * z2));
            output[i + 14u * sample_count] = static_cast<__half>(1.4453057213202769f * z * (x2 - y2));
            output[i + 15u * sample_count] = static_cast<__half>(0.59004358992664352f * x * (-x2 + 3.0f * y2));
        }

        __global__ void cast_params_to_half_kernel(const std::uint32_t param_count, const float* __restrict__ params_full_precision, __half* __restrict__ params) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;
            params[i] = static_cast<__half>(params_full_precision[i]);
        }

        __global__ void initialize_grid_params_kernel(const std::uint32_t param_count, const std::uint64_t rng_offset, float* __restrict__ params_full_precision, __half* __restrict__ params, __half* __restrict__ param_gradients) {
            const std::uint32_t i         = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n_threads = blockDim.x * gridDim.x;
            Pcg32 rng{config::TRAIN_SEED};
            rng.advance(rng_offset + static_cast<std::uint64_t>(i) * RANDOM_VALUES_PER_THREAD);

            for (std::uint32_t j = 0u; j < RANDOM_VALUES_PER_THREAD; ++j) {
                const std::uint32_t idx = i + n_threads * j;
                if (idx >= param_count) return;

                const float value          = rng.next_float() * 2e-4f - 1e-4f;
                params_full_precision[idx] = value;
                params[idx]                = static_cast<__half>(value);
                param_gradients[idx]       = static_cast<__half>(0.0f);
            }
        }

        template <typename Fragment>
        __device__ void relu_fragment(Fragment& fragment) {
            for (int i = 0; i < static_cast<int>(fragment.num_elements); ++i) fragment.x[i] = __hmax(fragment.x[i], static_cast<__half>(0.0f));
        }

        template <typename Fragment, typename ForwardFragment>
        __device__ void relu_backward_fragment(Fragment& fragment, const ForwardFragment& forward_fragment) {
            for (int i = 0; i < static_cast<int>(fragment.num_elements); ++i) fragment.x[i] = fragment.x[i] * static_cast<__half>(forward_fragment.x[i] > static_cast<__half>(0.0f));
        }

        __device__ void mlp_input_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ input_threadblock, const __half* __restrict__ weights_this_layer, __half* __restrict__ hidden_threadblock, const std::uint32_t batch_size) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::col_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[MLP_FORWARD_ITERS];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % config::MLP_WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / config::MLP_WIDTH;
            const std::uint32_t weights_col = 16u * wi;

            __half* __restrict__ weights_shmem        = act_shmem + 16u * (MLP_INPUT_WIDTH + MLP_INPUT_SKEW);
            constexpr std::uint32_t n_elems_per_load  = MLP_WIDTH_BLOCKS * 32u * 8u;
            const std::uint32_t thread_elem_idx       = (li + wi * 32u) * 8u;
            constexpr std::uint32_t n_weight_elements = config::MLP_WIDTH * MLP_INPUT_WIDTH;

            for (std::uint32_t idx = thread_elem_idx; idx < n_weight_elements; idx += n_elems_per_load) {
                const std::uint32_t idx_skewed                       = idx + idx / MLP_INPUT_WIDTH * MLP_INPUT_SKEW;
                *reinterpret_cast<int4*>(&weights_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&weights_this_layer[idx]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < MLP_INPUT_WIDTH / 16u; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, input_threadblock + 16u * i * batch_size + 16u * l, batch_size);
                    nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (MLP_INPUT_WIDTH + MLP_INPUT_SKEW), MLP_INPUT_WIDTH + MLP_INPUT_SKEW);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (config::MLP_WIDTH + MLP_SKEW), result_frag[l], config::MLP_WIDTH + MLP_SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < MLP_FORWARD_ITERS; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * config::MLP_WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (config::MLP_WIDTH + MLP_SKEW)]);
        }

        __device__ void mlp_hidden_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ hidden_threadblock) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[MLP_WIDTH_BLOCKS];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[MLP_FORWARD_ITERS];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % config::MLP_WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / config::MLP_WIDTH;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * config::MLP_WIDTH, config::MLP_WIDTH);

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (config::MLP_WIDTH + MLP_SKEW), config::MLP_WIDTH + MLP_SKEW);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + l * 16u * (config::MLP_WIDTH + MLP_SKEW), result_frag[l], config::MLP_WIDTH + MLP_SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < MLP_FORWARD_ITERS; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * config::MLP_WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (config::MLP_WIDTH + MLP_SKEW)]);
        }

        __device__ void mlp_last_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out, const std::uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[MLP_WIDTH_BLOCKS];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag;

            const std::uint32_t li = threadIdx.x;
            const std::uint32_t wi = threadIdx.y;

            __half* __restrict__ weights_shmem = act_shmem + MLP_FORWARD_ITERS * 16u * (config::MLP_WIDTH + MLP_SKEW);
            const std::uint32_t weights_row    = (8u * li) % config::MLP_WIDTH;
            const std::uint32_t weights_col    = (8u * li + 8u * 32u * wi) / config::MLP_WIDTH;

            *reinterpret_cast<int4*>(&weights_shmem[weights_row + weights_col * (config::MLP_WIDTH + MLP_SKEW)]) = *reinterpret_cast<const int4*>(&weights_this_layer[weights_row + weights_col * config::MLP_WIDTH]);
            __syncthreads();

            for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, config::MLP_WIDTH + MLP_SKEW);

            for (std::uint32_t idx = wi; idx < MLP_FORWARD_ITERS; idx += MLP_WIDTH_BLOCKS) {
                nvcuda::wmma::fill_fragment(result_frag, 0.0f);
                for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * idx) * (config::MLP_WIDTH + MLP_SKEW), config::MLP_WIDTH + MLP_SKEW);
                    nvcuda::wmma::mma_sync(result_frag, act_frag, weights_frag[i], result_frag);
                }

                if (output_layout == nvcuda::wmma::mem_row_major)
                    nvcuda::wmma::store_matrix_sync(out + idx * 16u * output_stride, result_frag, output_stride, output_layout);
                else
                    nvcuda::wmma::store_matrix_sync(out + idx * 16u, result_frag, output_stride, output_layout);
            }
        }

        __global__ void mlp_forward_64_relu_kernel(const std::uint32_t batch_size, const __half* __restrict__ input, const __half* __restrict__ weights, __half* __restrict__ hidden, __half* __restrict__ output, const bool output_row_major, const std::uint32_t hidden_layers) {
            extern __shared__ __half shmem[];
            const std::uint32_t elem_idx = 16u * blockIdx.x * MLP_FORWARD_ITERS;

            mlp_input_layer_forward(shmem, input + elem_idx, weights, hidden == nullptr ? nullptr : hidden + elem_idx * config::MLP_WIDTH, batch_size);
            if (hidden_layers == 2u) mlp_hidden_layer_forward(shmem, weights + MLP_FIRST_LAYER_PARAMS, hidden == nullptr ? nullptr : hidden + static_cast<std::uint64_t>(config::MLP_WIDTH) * batch_size + elem_idx * config::MLP_WIDTH);

            const __half* last_weights = weights + MLP_FIRST_LAYER_PARAMS + (hidden_layers - 1u) * MLP_HIDDEN_LAYER_PARAMS;
            if (output_row_major)
                mlp_last_layer_forward(shmem, last_weights, output + elem_idx * MLP_OUTPUT_WIDTH, MLP_OUTPUT_WIDTH, nvcuda::wmma::mem_row_major);
            else
                mlp_last_layer_forward(shmem, last_weights, output + elem_idx, batch_size, nvcuda::wmma::mem_col_major);
        }

        __device__ void mlp_hidden_layer_backward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag[MLP_WIDTH_BLOCKS];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[MLP_FORWARD_ITERS];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % config::MLP_WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / config::MLP_WIDTH;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * config::MLP_WIDTH + weights_col, config::MLP_WIDTH);

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < MLP_WIDTH_BLOCKS; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (config::MLP_WIDTH + MLP_SKEW), config::MLP_WIDTH + MLP_SKEW);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_hidden + weights_col + l * 16u * config::MLP_WIDTH, config::MLP_WIDTH);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (config::MLP_WIDTH + MLP_SKEW), result_frag[l], config::MLP_WIDTH + MLP_SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            for (std::uint32_t i = 0u; i < MLP_FORWARD_ITERS; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + i * 16u) * config::MLP_WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (config::MLP_WIDTH + MLP_SKEW)]);
        }

        template <typename OutputLayout>
        __global__ void mlp_backward_hidden_64_relu_kernel(const std::uint32_t batch_size, const __half* __restrict__ dloss_doutput, const __half* __restrict__ weights, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden, const std::uint32_t output_stride, const std::uint32_t hidden_layers) {
            const std::uint32_t wi            = threadIdx.y;
            const std::uint32_t elem_idx_base = 16u * blockIdx.x * MLP_FORWARD_ITERS;

            extern __shared__ __half shmem[];
            __half* act_shmem = shmem;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, OutputLayout> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[MLP_FORWARD_ITERS];

            const std::uint32_t weights_col = 16u * wi;
            const __half* last_weights      = weights + MLP_FIRST_LAYER_PARAMS + (hidden_layers - 1u) * MLP_HIDDEN_LAYER_PARAMS;
            const __half* forward_last      = forward_hidden + static_cast<std::uint64_t>(hidden_layers - 1u) * config::MLP_WIDTH * batch_size;
            nvcuda::wmma::load_matrix_sync(weights_frag, last_weights + weights_col, config::MLP_WIDTH);

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

                if constexpr (std::is_same_v<OutputLayout, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + (elem_idx_base + 16u * l) * output_stride, output_stride);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + elem_idx_base + 16u * l, output_stride);

                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_last + weights_col + (elem_idx_base + l * 16u) * config::MLP_WIDTH, config::MLP_WIDTH);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < MLP_FORWARD_ITERS; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (config::MLP_WIDTH + MLP_SKEW), result_frag[l], config::MLP_WIDTH + MLP_SKEW, nvcuda::wmma::mem_row_major);

            __syncthreads();

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t lane_offset = (8u * li) % config::MLP_WIDTH;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / config::MLP_WIDTH;

            for (std::uint32_t i = 0u; i < MLP_FORWARD_ITERS; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + elem_idx_base + i * 16u) * config::MLP_WIDTH]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (config::MLP_WIDTH + MLP_SKEW)]);

            if (hidden_layers == 2u) mlp_hidden_layer_backward(act_shmem, weights + MLP_FIRST_LAYER_PARAMS, forward_hidden + elem_idx_base * config::MLP_WIDTH, backward_hidden + static_cast<std::uint64_t>(config::MLP_WIDTH) * batch_size + elem_idx_base * config::MLP_WIDTH);
        }

        __global__ void extract_density_kernel(const std::uint32_t batch_size, const __half* __restrict__ density_output, __half* __restrict__ network_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            network_output[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 3u] = density_output[i];
        }

        __global__ void extract_rgb_gradients_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ rgb_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;

            const __half zero = 0.0f;
            for (std::uint32_t j = 0u; j < MLP_OUTPUT_WIDTH; ++j) rgb_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + j] = zero;
            rgb_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 0u] = network_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 0u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 1u] = network_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 1u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 2u] = network_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 2u];
        }

        __global__ void add_density_gradient_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ density_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            density_output_gradients[i] = density_output_gradients[i] + network_output_gradients[static_cast<std::uint64_t>(i) * MLP_OUTPUT_WIDTH + 3u];
        }

        __global__ void adam_step_kernel(const std::uint32_t param_count, const std::uint32_t mlp_param_count, float* __restrict__ params_full_precision, __half* __restrict__ params, const __half* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;

            float gradient = static_cast<float>(gradients[i]) / OPTIMIZER_LOSS_SCALE;
            if (i >= mlp_param_count && gradient == 0.0f) return;

            const float param = params_full_precision[i];
            if (i < mlp_param_count) gradient += OPTIMIZER_L2_REG * param;

            const float gradient_sq  = gradient * gradient;
            const float first_moment = first_moments[i] = OPTIMIZER_BETA1 * first_moments[i] + (1.0f - OPTIMIZER_BETA1) * gradient;
            const float second_moment = second_moments[i] = OPTIMIZER_BETA2 * second_moments[i] + (1.0f - OPTIMIZER_BETA2) * gradient_sq;
            const std::uint32_t step                      = ++param_steps[i];
            const float corrected_lr                      = OPTIMIZER_LEARNING_RATE * sqrtf(1.0f - powf(OPTIMIZER_BETA2, static_cast<float>(step))) / (1.0f - powf(OPTIMIZER_BETA1, static_cast<float>(step)));
            const float updated_param                     = param - corrected_lr * first_moment / (sqrtf(second_moment) + OPTIMIZER_EPSILON);

            params_full_precision[i] = updated_param;
            params[i]                = static_cast<__half>(updated_param);
        }
    } // namespace

    void upload_dataset(const std::uint8_t* const pixels, const std::size_t pixels_bytes, const float* const camera, const std::size_t camera_count, const std::uint8_t*& out_pixels, const float*& out_camera) {
        out_pixels = nullptr;
        out_camera = nullptr;

        if (pixels == nullptr || pixels_bytes == 0 || camera == nullptr || camera_count == 0) throw std::runtime_error{"invalid dataset upload input."};

        void* uploaded_pixels = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_pixels, pixels_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc pixels failed: "} + cudaGetErrorString(status)};
        out_pixels = static_cast<std::uint8_t*>(uploaded_pixels);

        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint8_t*>(out_pixels), pixels, pixels_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy pixels failed: "} + cudaGetErrorString(status)};

        void* uploaded_camera = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_camera, camera_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc camera failed: "} + cudaGetErrorString(status)};
        out_camera = static_cast<float*>(uploaded_camera);

        if (const cudaError_t status = cudaMemcpy(const_cast<float*>(out_camera), camera, camera_count * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy camera failed: "} + cudaGetErrorString(status)};
    }

    void allocate_sampler_buffers(float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy) {
        out_sample_coords  = nullptr;
        out_rays           = nullptr;
        out_ray_indices    = nullptr;
        out_numsteps       = nullptr;
        out_ray_counter    = nullptr;
        out_sample_counter = nullptr;
        out_occupancy      = nullptr;

        static_assert(config::NETWORK_BATCH_SIZE != 0u);
        static_assert(config::MAX_SAMPLES != 0u);

        if (const cudaError_t status = cudaMalloc(&out_sample_coords, static_cast<std::size_t>(config::MAX_SAMPLES) * SAMPLE_COORD_FLOATS * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample coords failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rays, static_cast<std::size_t>(config::NETWORK_BATCH_SIZE) * RAY_FLOATS * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler rays failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_indices, static_cast<std::size_t>(config::NETWORK_BATCH_SIZE) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray indices failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_numsteps, static_cast<std::size_t>(config::NETWORK_BATCH_SIZE) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler numsteps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_ray_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_occupancy, NERF_GRID_CELLS / 8u); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc sampler occupancy failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_occupancy, 0xFF, NERF_GRID_CELLS / 8u); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler occupancy failed: "} + cudaGetErrorString(status)};
    }

    void allocate_density_grid_buffers(float*& out_density_grid_values, float*& out_density_grid_scratch, std::uint32_t*& out_density_grid_indices, float*& out_density_grid_mean, std::uint32_t*& out_density_grid_occupied_count) {
        out_density_grid_values         = nullptr;
        out_density_grid_scratch        = nullptr;
        out_density_grid_indices        = nullptr;
        out_density_grid_mean           = nullptr;
        out_density_grid_occupied_count = nullptr;

        if (const cudaError_t status = cudaMalloc(&out_density_grid_values, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density grid values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_grid_scratch, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density grid scratch failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_grid_indices, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density grid indices failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_grid_mean, sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density grid mean failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_grid_occupied_count, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density grid occupied count failed: "} + cudaGetErrorString(status)};

        if (const cudaError_t status = cudaMemset(out_density_grid_values, 0, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid values failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_density_grid_scratch, 0, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid scratch failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_density_grid_mean, 0, sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid mean failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_density_grid_occupied_count, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid occupied count failed: "} + cudaGetErrorString(status)};
    }

    void allocate_network_buffers(std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden, std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, void*& out_cublaslt_handle, std::uint8_t*& out_cublaslt_workspace) {
        out_density_input            = nullptr;
        out_rgb_input                = nullptr;
        out_network_output           = nullptr;
        out_network_output_gradients = nullptr;
        out_rgb_output_gradients     = nullptr;
        out_rgb_input_gradients      = nullptr;
        out_density_input_gradients  = nullptr;
        out_density_forward_hidden   = nullptr;
        out_rgb_forward_hidden       = nullptr;
        out_density_backward_hidden  = nullptr;
        out_rgb_backward_hidden      = nullptr;
        out_cublaslt_handle          = nullptr;
        out_cublaslt_workspace       = nullptr;

        static_assert(config::NETWORK_BATCH_SIZE != 0u);
        static_assert(config::MAX_SAMPLES != 0u);
        static_assert(config::NETWORK_BATCH_SIZE % (16u * MLP_FORWARD_ITERS) == 0u);
        static_assert(config::MAX_SAMPLES % (16u * MLP_FORWARD_ITERS) == 0u);

        cublasLtHandle_t handle = nullptr;
        if (const cublasStatus_t status = cublasLtCreate(&handle); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtCreate failed: "} + cublasGetStatusString(status)};
        out_cublaslt_handle = reinterpret_cast<void*>(handle);

        if (const cudaError_t status = cudaMalloc(&out_density_input, static_cast<std::size_t>(MLP_INPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density network input failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rgb_input, static_cast<std::size_t>(MLP_INPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc rgb network input failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_output, static_cast<std::size_t>(MLP_OUTPUT_WIDTH) * config::MAX_SAMPLES * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network output failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_network_output_gradients, static_cast<std::size_t>(MLP_OUTPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc network output gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rgb_output_gradients, static_cast<std::size_t>(MLP_OUTPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc rgb output gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rgb_input_gradients, static_cast<std::size_t>(MLP_INPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc rgb input gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_input_gradients, static_cast<std::size_t>(MLP_INPUT_WIDTH) * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density input gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_forward_hidden, static_cast<std::size_t>(config::DENSITY_HIDDEN_LAYERS) * config::MLP_WIDTH * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density forward hidden failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rgb_forward_hidden, static_cast<std::size_t>(config::RGB_HIDDEN_LAYERS) * config::MLP_WIDTH * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc rgb forward hidden failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_density_backward_hidden, static_cast<std::size_t>(config::DENSITY_HIDDEN_LAYERS) * config::MLP_WIDTH * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc density backward hidden failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_rgb_backward_hidden, static_cast<std::size_t>(config::RGB_HIDDEN_LAYERS) * config::MLP_WIDTH * config::NETWORK_BATCH_SIZE * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc rgb backward hidden failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc cublasLt workspace failed: "} + cudaGetErrorString(status)};
    }

    void allocate_training_loss_buffers(std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values) {
        out_compacted_sample_counter = nullptr;
        out_compacted_sample_coords  = nullptr;
        out_loss_values              = nullptr;

        static_assert(config::NETWORK_BATCH_SIZE != 0u);

        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc compacted sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_coords, static_cast<std::size_t>(config::NETWORK_BATCH_SIZE) * SAMPLE_COORD_FLOATS * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc compacted sample coords failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_loss_values, static_cast<std::size_t>(config::NETWORK_BATCH_SIZE) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc loss values failed: "} + cudaGetErrorString(status)};
    }

    void allocate_validation_buffers(std::uint32_t*& out_validation_numsteps, std::uint32_t*& out_validation_sample_counter, std::uint32_t*& out_validation_overflow_counter, double*& out_validation_loss_sum) {
        out_validation_numsteps         = nullptr;
        out_validation_sample_counter   = nullptr;
        out_validation_overflow_counter = nullptr;
        out_validation_loss_sum         = nullptr;

        if (const cudaError_t status = cudaMalloc(&out_validation_numsteps, static_cast<std::size_t>(VALIDATION_TILE_RAYS) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc validation numsteps failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_validation_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc validation sample counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_validation_overflow_counter, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc validation overflow counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_validation_loss_sum, sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc validation loss sum failed: "} + cudaGetErrorString(status)};
    }

    void allocate_trainable_parameter_buffers(const std::uint32_t param_count, float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients) {
        out_params_full_precision = nullptr;
        out_params                = nullptr;
        out_param_gradients       = nullptr;

        if (param_count == 0u) return;

        if (const cudaError_t status = cudaMalloc(&out_params_full_precision, static_cast<std::size_t>(param_count) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable params full precision failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_params, static_cast<std::size_t>(param_count) * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable params failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_param_gradients, static_cast<std::size_t>(param_count) * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc trainable param gradients failed: "} + cudaGetErrorString(status)};
    }

    void initialize_mlp_parameters(const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {
        if (params_full_precision == nullptr || params == nullptr || param_gradients == nullptr) throw std::runtime_error{"invalid mlp parameter initialization input."};

        const std::uint32_t mlp_param_count = ::cuda::std::max(density_param_offset + DENSITY_NETWORK_PARAMS, rgb_param_offset + RGB_NETWORK_PARAMS);
        std::vector host_params(mlp_param_count, 0.0f);
        Pcg32 rng{config::TRAIN_SEED};

        auto initialize_matrix = [&](const std::uint32_t offset, const std::uint32_t rows, const std::uint32_t cols) {
            const float scale = std::sqrt(6.0f / static_cast<float>(rows + cols));
            for (std::uint32_t i = 0u; i < rows * cols; ++i) host_params[offset + i] = rng.next_float() * 2.0f * scale - scale;
        };

        initialize_matrix(density_param_offset, config::MLP_WIDTH, MLP_INPUT_WIDTH);
        initialize_matrix(density_param_offset + MLP_FIRST_LAYER_PARAMS, MLP_OUTPUT_WIDTH, config::MLP_WIDTH);
        initialize_matrix(rgb_param_offset, config::MLP_WIDTH, MLP_INPUT_WIDTH);
        initialize_matrix(rgb_param_offset + MLP_FIRST_LAYER_PARAMS, config::MLP_WIDTH, config::MLP_WIDTH);
        initialize_matrix(rgb_param_offset + MLP_FIRST_LAYER_PARAMS + MLP_HIDDEN_LAYER_PARAMS, MLP_OUTPUT_WIDTH, config::MLP_WIDTH);

        if (const cudaError_t status = cudaMemcpy(params_full_precision, host_params.data(), host_params.size() * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy mlp full precision params failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(param_gradients, 0, host_params.size() * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset mlp gradients failed: "} + cudaGetErrorString(status)};

        const std::uint32_t blocks = (mlp_param_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        cast_params_to_half_kernel<<<blocks, THREADS_PER_BLOCK>>>(mlp_param_count, params_full_precision, reinterpret_cast<__half*>(params));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"cast_params_to_half_kernel failed: "} + cudaGetErrorString(status)};
    }

    void initialize_grid_parameters(const std::uint32_t param_count, const std::uint64_t rng_offset, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {
        if (param_count == 0u) return;
        if (params_full_precision == nullptr || params == nullptr || param_gradients == nullptr) throw std::runtime_error{"grid parameter buffers are null."};

        const std::uint32_t n_threads = (param_count + RANDOM_VALUES_PER_THREAD - 1u) / RANDOM_VALUES_PER_THREAD;
        const std::uint32_t blocks    = (n_threads + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        initialize_grid_params_kernel<<<blocks, THREADS_PER_BLOCK>>>(param_count, rng_offset, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<__half*>(param_gradients));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"initialize_grid_params_kernel failed: "} + cudaGetErrorString(status)};
    }

    void download_trainable_parameters(const std::uint32_t param_count, const float* const params_full_precision, float* const out_params_full_precision) {
        if (param_count == 0u) return;
        if (params_full_precision == nullptr || out_params_full_precision == nullptr) throw std::runtime_error{"invalid trainable parameter download input."};
        if (const cudaError_t status = cudaMemcpy(out_params_full_precision, params_full_precision, static_cast<std::size_t>(param_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy trainable params download failed: "} + cudaGetErrorString(status)};
    }

    void upload_trainable_parameters(const std::uint32_t param_count, const float* const params_full_precision, float* const out_params_full_precision, std::uint16_t* const out_params, std::uint16_t* const out_param_gradients, float* const optimizer_first_moments, float* const optimizer_second_moments, std::uint32_t* const optimizer_param_steps) {
        if (param_count == 0u) return;
        if (params_full_precision == nullptr || out_params_full_precision == nullptr || out_params == nullptr || out_param_gradients == nullptr || optimizer_first_moments == nullptr || optimizer_second_moments == nullptr || optimizer_param_steps == nullptr) throw std::runtime_error{"invalid trainable parameter upload input."};

        const std::size_t param_bytes = static_cast<std::size_t>(param_count) * sizeof(float);
        if (const cudaError_t status = cudaMemcpy(out_params_full_precision, params_full_precision, param_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy trainable params upload failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_param_gradients, 0, static_cast<std::size_t>(param_count) * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded param gradients failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_first_moments, 0, param_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_second_moments, 0, param_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(optimizer_param_steps, 0, static_cast<std::size_t>(param_count) * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loaded optimizer param steps failed: "} + cudaGetErrorString(status)};

        const std::uint32_t blocks = (param_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        cast_params_to_half_kernel<<<blocks, THREADS_PER_BLOCK>>>(param_count, out_params_full_precision, reinterpret_cast<__half*>(out_params));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"cast_params_to_half_kernel loaded params failed: "} + cudaGetErrorString(status)};
    }

    void evaluate_network(const std::uint32_t sample_count, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset, std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const network_output) {
        if (sample_count == 0u) return;
        if (sample_count % (16u * MLP_FORWARD_ITERS) != 0u || sample_coords == nullptr || grid_offsets == nullptr || params == nullptr || density_input == nullptr || rgb_input == nullptr || network_output == nullptr) throw std::runtime_error{"invalid network inference input."};

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * MLP_FORWARD_ITERS) * (config::MLP_WIDTH + MLP_SKEW);
        constexpr dim3 threads{32u, MLP_WIDTH_BLOCKS, 1u};
        if (const cudaError_t status = cudaFuncSetAttribute(mlp_forward_64_relu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, forward_shmem); status != cudaSuccess) throw std::runtime_error{std::string{"cudaFuncSetAttribute mlp_forward_64_relu_kernel failed: "} + cudaGetErrorString(status)};

        for (std::uint32_t offset = 0u; offset < sample_count; offset += config::NETWORK_BATCH_SIZE) {
            const std::uint32_t chunk              = ::cuda::std::min(config::NETWORK_BATCH_SIZE, sample_count - offset);
            const float* const chunk_sample_coords = sample_coords + static_cast<std::uint64_t>(offset) * SAMPLE_COORD_FLOATS;

            const dim3 grid_blocks{(chunk + GRID_FORWARD_THREADS - 1u) / GRID_FORWARD_THREADS, config::GRID_N_LEVELS, 1u};
            encode_grid_forward_kernel<<<grid_blocks, GRID_FORWARD_THREADS>>>(chunk, grid_offsets[0u], grid_offsets[1u], grid_offsets[2u], grid_offsets[3u], grid_offsets[4u], grid_offsets[5u], grid_offsets[6u], grid_offsets[7u], grid_offsets[8u], chunk_sample_coords, reinterpret_cast<const __half*>(params + grid_param_offset), reinterpret_cast<__half*>(density_input));
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_grid_forward_kernel inference failed: "} + cudaGetErrorString(status)};

            const std::uint32_t linear_blocks = (chunk + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
            encode_spherical_harmonics_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(chunk, chunk_sample_coords, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(MLP_OUTPUT_WIDTH) * chunk);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_spherical_harmonics_kernel failed: "} + cudaGetErrorString(status)};

            const dim3 blocks{chunk / (16u * MLP_FORWARD_ITERS), 1u, 1u};
            mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(chunk, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + density_param_offset), nullptr, reinterpret_cast<__half*>(rgb_input), false, config::DENSITY_HIDDEN_LAYERS);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"density mlp inference failed: "} + cudaGetErrorString(status)};

            mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + rgb_param_offset), nullptr, reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * MLP_OUTPUT_WIDTH, true, config::RGB_HIDDEN_LAYERS);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"rgb mlp inference failed: "} + cudaGetErrorString(status)};

            extract_density_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * MLP_OUTPUT_WIDTH);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"extract_density_kernel inference failed: "} + cudaGetErrorString(status)};
        }
    }

    void update_density_grid(const float* const camera, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const std::uint32_t current_step, const std::uint32_t* const grid_offsets, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t grid_param_offset, float* const sample_coords, std::uint16_t* const density_input, std::uint16_t* const density_grid_output, float* const density_grid_values, float* const density_grid_scratch, std::uint32_t* const density_grid_indices, float* const density_grid_mean, std::uint32_t* const density_grid_occupied_count, std::uint8_t* const occupancy, std::uint32_t& density_grid_ema_step, Pcg32& density_grid_rng, float& out_elapsed_ms) {
        out_elapsed_ms                  = 0.0f;
        std::uint32_t density_grid_skip = current_step / DENSITY_GRID_SKIP_INTERVAL;
        if (density_grid_skip < 1u) density_grid_skip = 1u;
        if (density_grid_skip > DENSITY_GRID_MAX_SKIP) density_grid_skip = DENSITY_GRID_MAX_SKIP;
        if (current_step % density_grid_skip != 0u) return;

        if (frame_count == 0u || width == 0u || height == 0u || focal_length <= 0.0f || camera == nullptr || grid_offsets == nullptr || params == nullptr || sample_coords == nullptr || density_input == nullptr || density_grid_output == nullptr || density_grid_values == nullptr || density_grid_scratch == nullptr || density_grid_indices == nullptr || density_grid_mean == nullptr || density_grid_occupied_count == nullptr || occupancy == nullptr) throw std::runtime_error{"invalid density grid update input."};

        const auto start                            = std::chrono::steady_clock::now();
        const std::uint32_t uniform_sample_count    = current_step < DENSITY_GRID_WARMUP_STEPS ? DENSITY_GRID_WARMUP_SAMPLES : DENSITY_GRID_STEADY_UNIFORM_SAMPLES;
        const std::uint32_t nonuniform_sample_count = current_step < DENSITY_GRID_WARMUP_STEPS ? 0u : DENSITY_GRID_STEADY_NONUNIFORM_SAMPLES;
        const std::uint32_t sample_count            = uniform_sample_count + nonuniform_sample_count;
        if (sample_count == 0u || sample_count > config::MAX_SAMPLES || sample_count % (16u * MLP_FORWARD_ITERS) != 0u) throw std::runtime_error{"invalid density grid sample count."};

        if (current_step == 0u) {
            density_grid_ema_step = 0u;
            mark_untrained_density_grid_kernel<<<(NERF_GRID_CELLS + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(density_grid_values, frame_count, width, height, focal_length, camera);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"mark_untrained_density_grid_kernel failed: "} + cudaGetErrorString(status)};
        }

        if (const cudaError_t status = cudaMemset(density_grid_scratch, 0, static_cast<std::size_t>(NERF_GRID_CELLS) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid scratch failed: "} + cudaGetErrorString(status)};

        if (uniform_sample_count > 0u) {
            generate_density_grid_samples_kernel<<<(uniform_sample_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(uniform_sample_count, density_grid_rng, density_grid_ema_step, -0.01f, density_grid_values, sample_coords, density_grid_indices);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate uniform density grid samples failed: "} + cudaGetErrorString(status)};
        }
        density_grid_rng.advance(1ull << 32u);

        if (nonuniform_sample_count > 0u) {
            generate_density_grid_samples_kernel<<<(nonuniform_sample_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(nonuniform_sample_count, density_grid_rng, density_grid_ema_step, NERF_MIN_OPTICAL_THICKNESS, density_grid_values, sample_coords + static_cast<std::uint64_t>(uniform_sample_count) * SAMPLE_COORD_FLOATS, density_grid_indices + uniform_sample_count);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate nonuniform density grid samples failed: "} + cudaGetErrorString(status)};
        }
        density_grid_rng.advance(1ull << 32u);

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * MLP_FORWARD_ITERS) * (config::MLP_WIDTH + MLP_SKEW);
        constexpr dim3 threads{32u, MLP_WIDTH_BLOCKS, 1u};
        if (const cudaError_t status = cudaFuncSetAttribute(mlp_forward_64_relu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, forward_shmem); status != cudaSuccess) throw std::runtime_error{std::string{"cudaFuncSetAttribute density grid mlp failed: "} + cudaGetErrorString(status)};

        for (std::uint32_t offset = 0u; offset < sample_count; offset += config::NETWORK_BATCH_SIZE) {
            const std::uint32_t chunk = sample_count - offset < config::NETWORK_BATCH_SIZE ? sample_count - offset : config::NETWORK_BATCH_SIZE;
            if (chunk % (16u * MLP_FORWARD_ITERS) != 0u) throw std::runtime_error{"invalid density grid chunk size."};

            const float* chunk_sample_coords = sample_coords + static_cast<std::uint64_t>(offset) * SAMPLE_COORD_FLOATS;
            const dim3 grid_blocks{(chunk + GRID_FORWARD_THREADS - 1u) / GRID_FORWARD_THREADS, config::GRID_N_LEVELS, 1u};
            encode_grid_forward_kernel<<<grid_blocks, GRID_FORWARD_THREADS>>>(chunk, grid_offsets[0u], grid_offsets[1u], grid_offsets[2u], grid_offsets[3u], grid_offsets[4u], grid_offsets[5u], grid_offsets[6u], grid_offsets[7u], grid_offsets[8u], chunk_sample_coords, reinterpret_cast<const __half*>(params + grid_param_offset), reinterpret_cast<__half*>(density_input));
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_grid_forward_kernel density grid failed: "} + cudaGetErrorString(status)};

            const dim3 blocks{chunk / (16u * MLP_FORWARD_ITERS), 1u, 1u};
            mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(chunk, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + density_param_offset), nullptr, reinterpret_cast<__half*>(density_grid_output), false, config::DENSITY_HIDDEN_LAYERS);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"density grid mlp inference failed: "} + cudaGetErrorString(status)};

            splat_density_grid_samples_kernel<<<(chunk + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(chunk, density_grid_indices + offset, reinterpret_cast<const __half*>(density_grid_output), density_grid_scratch);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"splat_density_grid_samples_kernel failed: "} + cudaGetErrorString(status)};
        }

        update_density_grid_ema_kernel<<<(NERF_GRID_CELLS + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(density_grid_scratch, density_grid_values);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"update_density_grid_ema_kernel failed: "} + cudaGetErrorString(status)};
        ++density_grid_ema_step;

        if (const cudaError_t status = cudaMemset(density_grid_mean, 0, sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid mean failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(density_grid_occupied_count, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset density grid occupied count failed: "} + cudaGetErrorString(status)};

        reduce_density_grid_mean_kernel<<<(NERF_GRID_CELLS / 4u + 1023u) / 1024u, 1024u>>>(density_grid_values, density_grid_mean);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"reduce_density_grid_mean_kernel failed: "} + cudaGetErrorString(status)};

        build_density_grid_bitfield_kernel<<<(NERF_GRID_CELLS / 8u + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(density_grid_values, density_grid_mean, occupancy, density_grid_occupied_count);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"build_density_grid_bitfield_kernel failed: "} + cudaGetErrorString(status)};

        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"density grid update synchronization failed: "} + cudaGetErrorString(status)};
        out_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count();
    }

    void run_validation(const std::uint8_t* const validation_pixels, const float* const validation_camera, const std::uint32_t validation_frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const std::uint8_t* const occupancy, const std::uint32_t* const grid_offsets, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset, float* const sample_coords, std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const network_output, std::uint32_t* const validation_numsteps, std::uint32_t* const validation_sample_counter, std::uint32_t* const validation_overflow_counter, double* const validation_loss_sum, double& out_loss_sum) {
        out_loss_sum = 0.0;
        if (validation_pixels == nullptr || validation_camera == nullptr || validation_frame_count == 0u || width == 0u || height == 0u || focal_length <= 0.0f || occupancy == nullptr || grid_offsets == nullptr || params == nullptr || sample_coords == nullptr || density_input == nullptr || rgb_input == nullptr || network_output == nullptr || validation_numsteps == nullptr || validation_sample_counter == nullptr || validation_overflow_counter == nullptr || validation_loss_sum == nullptr) throw std::runtime_error{"invalid validation input."};

        const std::uint64_t total_pixels_64 = static_cast<std::uint64_t>(width) * height;
        if (total_pixels_64 > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"validation image has too many pixels."};
        const auto total_pixels = static_cast<std::uint32_t>(total_pixels_64);

        if (const cudaError_t status = cudaMemset(validation_loss_sum, 0, sizeof(double)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset validation loss sum failed: "} + cudaGetErrorString(status)};

        for (std::uint32_t validation_image_index = 0u; validation_image_index < validation_frame_count; ++validation_image_index) {
            for (std::uint32_t pixel_offset = 0u; pixel_offset < total_pixels; pixel_offset += VALIDATION_TILE_RAYS) {
                const std::uint32_t tile_pixels = ::cuda::std::min(VALIDATION_TILE_RAYS, total_pixels - pixel_offset);

                if (const cudaError_t status = cudaMemset(validation_numsteps, 0, static_cast<std::size_t>(VALIDATION_TILE_RAYS) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset validation numsteps failed: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMemset(validation_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset validation sample counter failed: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMemset(validation_overflow_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset validation overflow counter failed: "} + cudaGetErrorString(status)};

                generate_validation_samples_kernel<<<(tile_pixels + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(tile_pixels, pixel_offset, VALIDATION_MAX_SAMPLES, width, height, focal_length, validation_camera, validation_image_index, occupancy, validation_sample_counter, validation_overflow_counter, validation_numsteps, sample_coords);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate_validation_samples_kernel failed: "} + cudaGetErrorString(status)};

                std::uint32_t used_samples    = 0u;
                std::uint32_t overflowed_rays = 0u;
                if (const cudaError_t status = cudaMemcpy(&used_samples, validation_sample_counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy validation sample counter failed: "} + cudaGetErrorString(status)};
                if (const cudaError_t status = cudaMemcpy(&overflowed_rays, validation_overflow_counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy validation overflow counter failed: "} + cudaGetErrorString(status)};

                if (overflowed_rays != 0u) throw std::runtime_error{std::format("Validation sample budget overflowed for {} rays.", overflowed_rays)};
                if (used_samples > VALIDATION_MAX_SAMPLES) throw std::runtime_error{"validation used sample count exceeded validation sample budget."};

                if (used_samples > 0u) {
                    const std::uint32_t padded_used_samples = ((used_samples + config::NETWORK_BATCH_GRANULARITY - 1u) / config::NETWORK_BATCH_GRANULARITY) * config::NETWORK_BATCH_GRANULARITY;
                    if (padded_used_samples > VALIDATION_MAX_SAMPLES) throw std::runtime_error{"validation padded sample count exceeded validation sample budget."};

                    const std::uint32_t coord_elements = padded_used_samples * SAMPLE_COORD_FLOATS;
                    pad_validation_rollover_coords_kernel<<<(coord_elements + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(used_samples, padded_used_samples, sample_coords);
                    if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"pad_validation_rollover_coords_kernel failed: "} + cudaGetErrorString(status)};

                    evaluate_network(padded_used_samples, sample_coords, grid_offsets, params, density_param_offset, rgb_param_offset, grid_param_offset, density_input, rgb_input, network_output);
                }

                accumulate_validation_loss_kernel<<<(tile_pixels + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(tile_pixels, pixel_offset, validation_image_index, width, height, validation_pixels, validation_numsteps, sample_coords, reinterpret_cast<const __half*>(network_output), validation_loss_sum);
                if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"accumulate_validation_loss_kernel failed: "} + cudaGetErrorString(status)};
            }
        }

        if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"validation synchronization failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemcpy(&out_loss_sum, validation_loss_sum, sizeof(double), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy validation loss sum failed: "} + cudaGetErrorString(status)};
    }

    void forward_network(const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset, std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const density_forward_hidden, std::uint16_t* const rgb_forward_hidden, std::uint16_t* const network_output) {
        if (sample_coords == nullptr || grid_offsets == nullptr || params == nullptr || density_input == nullptr || rgb_input == nullptr || density_forward_hidden == nullptr || rgb_forward_hidden == nullptr || network_output == nullptr) throw std::runtime_error{"invalid network forward input."};

        constexpr dim3 grid_blocks{(config::NETWORK_BATCH_SIZE + GRID_FORWARD_THREADS - 1u) / GRID_FORWARD_THREADS, config::GRID_N_LEVELS, 1u};
        encode_grid_forward_kernel<<<grid_blocks, GRID_FORWARD_THREADS>>>(config::NETWORK_BATCH_SIZE, grid_offsets[0u], grid_offsets[1u], grid_offsets[2u], grid_offsets[3u], grid_offsets[4u], grid_offsets[5u], grid_offsets[6u], grid_offsets[7u], grid_offsets[8u], sample_coords, reinterpret_cast<const __half*>(params + grid_param_offset), reinterpret_cast<__half*>(density_input));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_grid_forward_kernel failed: "} + cudaGetErrorString(status)};

        constexpr std::uint32_t linear_blocks = (config::NETWORK_BATCH_SIZE + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        encode_spherical_harmonics_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(config::NETWORK_BATCH_SIZE, sample_coords, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(MLP_OUTPUT_WIDTH) * config::NETWORK_BATCH_SIZE);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_spherical_harmonics_kernel failed: "} + cudaGetErrorString(status)};

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * MLP_FORWARD_ITERS) * (config::MLP_WIDTH + MLP_SKEW);
        constexpr dim3 threads{32u, MLP_WIDTH_BLOCKS, 1u};
        constexpr dim3 blocks{config::NETWORK_BATCH_SIZE / (16u * MLP_FORWARD_ITERS), 1u, 1u};

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_forward_64_relu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, forward_shmem); status != cudaSuccess) throw std::runtime_error{std::string{"cudaFuncSetAttribute mlp_forward_64_relu_kernel failed: "} + cudaGetErrorString(status)};
        mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + density_param_offset), reinterpret_cast<__half*>(density_forward_hidden), reinterpret_cast<__half*>(rgb_input), false, config::DENSITY_HIDDEN_LAYERS);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"density mlp forward failed: "} + cudaGetErrorString(status)};

        mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + rgb_param_offset), reinterpret_cast<__half*>(rgb_forward_hidden), reinterpret_cast<__half*>(network_output), true, config::RGB_HIDDEN_LAYERS);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"rgb mlp forward failed: "} + cudaGetErrorString(status)};

        extract_density_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"extract_density_kernel failed: "} + cudaGetErrorString(status)};
    }

    void backward_network(const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint16_t* const params, std::uint16_t* const gradients, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset, const std::uint16_t* const density_input, const std::uint16_t* const rgb_input, const std::uint16_t* const density_forward_hidden, const std::uint16_t* const rgb_forward_hidden, const std::uint16_t* const network_output, const std::uint16_t* const network_output_gradients, std::uint16_t* const rgb_output_gradients, std::uint16_t* const rgb_input_gradients, std::uint16_t* const density_input_gradients, std::uint16_t* const density_backward_hidden, std::uint16_t* const rgb_backward_hidden, void* const cublaslt_handle, std::uint8_t* const cublaslt_workspace) {
        if (sample_coords == nullptr || grid_offsets == nullptr || params == nullptr || gradients == nullptr || density_input == nullptr || rgb_input == nullptr || density_forward_hidden == nullptr || rgb_forward_hidden == nullptr || network_output == nullptr || network_output_gradients == nullptr || rgb_output_gradients == nullptr || rgb_input_gradients == nullptr || density_input_gradients == nullptr || density_backward_hidden == nullptr || rgb_backward_hidden == nullptr || cublaslt_handle == nullptr || cublaslt_workspace == nullptr) throw std::runtime_error{"invalid network backward input."};

        constexpr std::uint32_t linear_blocks       = (config::NETWORK_BATCH_SIZE + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        constexpr std::uint64_t hidden_layer_stride = static_cast<std::uint64_t>(config::MLP_WIDTH) * config::NETWORK_BATCH_SIZE;
        constexpr int batch                         = static_cast<int>(config::NETWORK_BATCH_SIZE);
        const auto cublaslt                         = static_cast<cublasLtHandle_t>(cublaslt_handle);

        extract_rgb_gradients_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_output_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"extract_rgb_gradients_kernel failed: "} + cudaGetErrorString(status)};

        constexpr int backward_shmem = sizeof(__half) * (16u * MLP_FORWARD_ITERS) * (config::MLP_WIDTH + MLP_SKEW);
        constexpr dim3 threads{32u, MLP_WIDTH_BLOCKS, 1u};
        constexpr dim3 blocks{config::NETWORK_BATCH_SIZE / (16u * MLP_FORWARD_ITERS), 1u, 1u};

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, backward_shmem); status != cudaSuccess) throw std::runtime_error{std::string{"cudaFuncSetAttribute rgb mlp backward failed: "} + cudaGetErrorString(status)};
        mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::row_major><<<blocks, threads, backward_shmem>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(rgb_output_gradients), reinterpret_cast<const __half*>(params + rgb_param_offset), reinterpret_cast<const __half*>(rgb_forward_hidden), reinterpret_cast<__half*>(rgb_backward_hidden), MLP_OUTPUT_WIDTH, config::RGB_HIDDEN_LAYERS);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"rgb mlp backward hidden failed: "} + cudaGetErrorString(status)};

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_ROW;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate rgb last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, MLP_OUTPUT_WIDTH, batch, MLP_OUTPUT_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb last gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, batch, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb last gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, MLP_OUTPUT_WIDTH, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb last gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb last gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb last gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb last gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate rgb last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute rgb last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic rgb last weight gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt rgb last weight gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(rgb_output_gradients), lt.a_desc, reinterpret_cast<const __half*>(rgb_forward_hidden) + (config::RGB_HIDDEN_LAYERS - 1u) * hidden_layer_stride, lt.b_desc, &beta, reinterpret_cast<__half*>(gradients + rgb_param_offset + MLP_FIRST_LAYER_PARAMS + (config::RGB_HIDDEN_LAYERS - 1u) * MLP_HIDDEN_LAYER_PARAMS), lt.d_desc, reinterpret_cast<__half*>(gradients + rgb_param_offset + MLP_FIRST_LAYER_PARAMS + (config::RGB_HIDDEN_LAYERS - 1u) * MLP_HIDDEN_LAYER_PARAMS), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error{std::string{"cublasLtMatmul rgb last weight gradients failed: "} + cublasGetStatusString(status)};
            }
        }

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_ROW;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate rgb hidden weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, config::MLP_WIDTH, batch, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb hidden gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, batch, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb hidden gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, config::MLP_WIDTH, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb hidden gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb hidden gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb hidden gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb hidden gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate rgb hidden weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute rgb hidden weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic rgb hidden weight gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt rgb hidden weight gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(rgb_backward_hidden), lt.a_desc, reinterpret_cast<const __half*>(rgb_forward_hidden), lt.b_desc, &beta, reinterpret_cast<__half*>(gradients + rgb_param_offset + MLP_FIRST_LAYER_PARAMS), lt.d_desc, reinterpret_cast<__half*>(gradients + rgb_param_offset + MLP_FIRST_LAYER_PARAMS), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error{std::string{"cublasLtMatmul rgb hidden weight gradients failed: "} + cublasGetStatusString(status)};
            }
        }

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate rgb first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, config::MLP_WIDTH, batch, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb first gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, batch, MLP_INPUT_WIDTH, batch); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb first gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, config::MLP_WIDTH, MLP_INPUT_WIDTH, MLP_INPUT_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb first gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb first gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb first gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb first gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate rgb first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute rgb first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic rgb first weight gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt rgb first weight gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(rgb_backward_hidden) + (config::RGB_HIDDEN_LAYERS - 1u) * hidden_layer_stride, lt.a_desc, reinterpret_cast<const __half*>(rgb_input), lt.b_desc, &beta, reinterpret_cast<__half*>(gradients + rgb_param_offset), lt.d_desc, reinterpret_cast<__half*>(gradients + rgb_param_offset), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error{std::string{"cublasLtMatmul rgb first weight gradients failed: "} + cublasGetStatusString(status)};
            }
        }

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate rgb input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, MLP_INPUT_WIDTH, config::MLP_WIDTH, MLP_INPUT_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb input gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, config::MLP_WIDTH, batch, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb input gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, MLP_INPUT_WIDTH, batch, batch); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate rgb input gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb input gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb input gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute rgb input gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate rgb input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute rgb input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic rgb input gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt rgb input gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(params + rgb_param_offset), lt.a_desc, reinterpret_cast<const __half*>(rgb_backward_hidden) + (config::RGB_HIDDEN_LAYERS - 1u) * hidden_layer_stride, lt.b_desc, &beta, reinterpret_cast<__half*>(rgb_input_gradients), lt.d_desc, reinterpret_cast<__half*>(rgb_input_gradients), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error{std::string{"cublasLtMatmul rgb input gradients failed: "} + cublasGetStatusString(status)};
            }
        }

        add_density_gradient_kernel<<<linear_blocks, THREADS_PER_BLOCK>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_input_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"add_density_gradient_kernel failed: "} + cudaGetErrorString(status)};

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, backward_shmem); status != cudaSuccess) throw std::runtime_error{std::string{"cudaFuncSetAttribute density mlp backward failed: "} + cudaGetErrorString(status)};
        mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::col_major><<<blocks, threads, backward_shmem>>>(config::NETWORK_BATCH_SIZE, reinterpret_cast<const __half*>(rgb_input_gradients), reinterpret_cast<const __half*>(params + density_param_offset), reinterpret_cast<const __half*>(density_forward_hidden), reinterpret_cast<__half*>(density_backward_hidden), config::NETWORK_BATCH_SIZE, config::DENSITY_HIDDEN_LAYERS);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"density mlp backward hidden failed: "} + cudaGetErrorString(status)};

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_ROW;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_ROW;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate density last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, MLP_OUTPUT_WIDTH, batch, batch); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density last gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, batch, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density last gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, MLP_OUTPUT_WIDTH, config::MLP_WIDTH, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density last gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density last gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density last gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density last gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate density last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute density last weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic density last weight gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt density last weight gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(rgb_input_gradients), lt.a_desc, reinterpret_cast<const __half*>(density_forward_hidden), lt.b_desc, &beta, reinterpret_cast<__half*>(gradients + density_param_offset + MLP_FIRST_LAYER_PARAMS), lt.d_desc, reinterpret_cast<__half*>(gradients + density_param_offset + MLP_FIRST_LAYER_PARAMS), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) {
                throw std::runtime_error{std::string{"cublasLtMatmul density last weight gradients failed: "} + cublasGetStatusString(status)};
            }
        }

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate density first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, config::MLP_WIDTH, batch, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density first gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, batch, MLP_INPUT_WIDTH, batch); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density first gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, config::MLP_WIDTH, MLP_INPUT_WIDTH, MLP_INPUT_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density first gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density first gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density first gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density first gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate density first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute density first weight gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic density first weight gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt density first weight gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(density_backward_hidden), lt.a_desc, reinterpret_cast<const __half*>(density_input), lt.b_desc, &beta, reinterpret_cast<__half*>(gradients + density_param_offset), lt.d_desc, reinterpret_cast<__half*>(gradients + density_param_offset), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmul density first weight gradients failed: "} + cublasGetStatusString(status)};
        }

        {
            CublasLtMatmulResources lt;
            const auto alpha                          = static_cast<__half>(1.0f);
            const auto beta                           = static_cast<__half>(0.0f);
            constexpr cublasLtOrder_t a_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t b_order         = CUBLASLT_ORDER_COL;
            constexpr cublasLtOrder_t d_order         = CUBLASLT_ORDER_ROW;
            constexpr std::size_t max_workspace_bytes = CUBLASLT_WORKSPACE_BYTES;
            cublasLtMatmulHeuristicResult_t heuristic = {};
            int returned_algo_count                   = 0;

            if (const cublasStatus_t status = cublasLtMatmulDescCreate(&lt.operation_desc, CUBLAS_COMPUTE_16F, CUDA_R_16F); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulDescCreate density input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.a_desc, CUDA_R_16F, MLP_INPUT_WIDTH, config::MLP_WIDTH, MLP_INPUT_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density input gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.b_desc, CUDA_R_16F, config::MLP_WIDTH, batch, config::MLP_WIDTH); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density input gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutCreate(&lt.d_desc, CUDA_R_16F, MLP_INPUT_WIDTH, batch, batch); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutCreate density input gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &a_order, sizeof(a_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density input gradients A failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &b_order, sizeof(b_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density input gradients B failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatrixLayoutSetAttribute(lt.d_desc, CUBLASLT_MATRIX_LAYOUT_ORDER, &d_order, sizeof(d_order)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatrixLayoutSetAttribute density input gradients D failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceCreate(&lt.preference); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceCreate density input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(lt.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &max_workspace_bytes, sizeof(max_workspace_bytes)); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulPreferenceSetAttribute density input gradients failed: "} + cublasGetStatusString(status)};
            if (const cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(cublaslt, lt.operation_desc, lt.a_desc, lt.b_desc, lt.d_desc, lt.d_desc, lt.preference, 1, &heuristic, &returned_algo_count); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmulAlgoGetHeuristic density input gradients failed: "} + cublasGetStatusString(status)};
            if (returned_algo_count == 0 || heuristic.state != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{"cublasLt density input gradients returned no supported algorithm."};
            if (const cublasStatus_t status = cublasLtMatmul(cublaslt, lt.operation_desc, &alpha, reinterpret_cast<const __half*>(params + density_param_offset), lt.a_desc, reinterpret_cast<const __half*>(density_backward_hidden), lt.b_desc, &beta, reinterpret_cast<__half*>(density_input_gradients), lt.d_desc, reinterpret_cast<__half*>(density_input_gradients), lt.d_desc, &heuristic.algo, cublaslt_workspace, CUBLASLT_WORKSPACE_BYTES, nullptr); status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error{std::string{"cublasLtMatmul density input gradients failed: "} + cublasGetStatusString(status)};
        }

        if (const cudaError_t status = cudaMemset(gradients + grid_param_offset, 0, static_cast<std::size_t>(grid_offsets[config::GRID_N_LEVELS]) * config::GRID_FEATURES_PER_LEVEL * sizeof(__half)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset grid param gradients failed: "} + cudaGetErrorString(status)};

        constexpr std::uint32_t grid_threads = (config::NETWORK_BATCH_SIZE * config::GRID_FEATURES_PER_LEVEL / GRID_BACKWARD_FEATURES + GRID_BACKWARD_THREADS - 1u) / GRID_BACKWARD_THREADS;
        constexpr dim3 grid_blocks{grid_threads, config::GRID_N_LEVELS, 1u};
        encode_grid_backward_kernel<<<grid_blocks, GRID_BACKWARD_THREADS>>>(config::NETWORK_BATCH_SIZE, grid_offsets[0u], grid_offsets[1u], grid_offsets[2u], grid_offsets[3u], grid_offsets[4u], grid_offsets[5u], grid_offsets[6u], grid_offsets[7u], grid_offsets[8u], sample_coords, reinterpret_cast<const __half*>(density_input_gradients), reinterpret_cast<__half*>(gradients + grid_param_offset));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"encode_grid_backward_kernel failed: "} + cudaGetErrorString(status)};
    }

    void allocate_adam_state(const std::uint32_t param_count, float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps) {
        out_first_moments  = nullptr;
        out_second_moments = nullptr;
        out_param_steps    = nullptr;

        if (param_count == 0u) throw std::runtime_error{"optimizer param count is zero."};

        if (const cudaError_t status = cudaMalloc(&out_first_moments, param_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_second_moments, param_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMalloc(&out_param_steps, param_count * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMalloc optimizer param steps failed: "} + cudaGetErrorString(status)};

        if (const cudaError_t status = cudaMemset(out_first_moments, 0, param_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer first moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_second_moments, 0, param_count * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer second moments failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(out_param_steps, 0, param_count * sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset optimizer param steps failed: "} + cudaGetErrorString(status)};
    }

    void sample_training_batch(const float* const camera, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const std::uint32_t current_step, const std::uint32_t rays_per_batch, const std::uint32_t sample_limit, const std::uint8_t* const occupancy, float* const sample_coords, float* const rays, std::uint32_t* const ray_indices, std::uint32_t* const numsteps, std::uint32_t* const ray_counter, std::uint32_t* const sample_counter) {
        if (sample_limit == 0u || sample_limit > config::MAX_SAMPLES || frame_count == 0u || width == 0u || height == 0u || focal_length <= 0.0f || (rays_per_batch != 0u && (camera == nullptr || occupancy == nullptr || sample_coords == nullptr || rays == nullptr || ray_indices == nullptr || numsteps == nullptr || ray_counter == nullptr || sample_counter == nullptr))) throw std::runtime_error{"invalid sampler input."};
        if (rays_per_batch == 0u) return;

        if (const cudaError_t status = cudaMemset(ray_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler ray counter failed: "} + cudaGetErrorString(status)};
        if (const cudaError_t status = cudaMemset(sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset sampler sample counter failed: "} + cudaGetErrorString(status)};

        const std::uint32_t blocks = (rays_per_batch + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        generate_training_samples_kernel<<<blocks, THREADS_PER_BLOCK>>>(rays_per_batch, sample_limit, current_step, frame_count, width, height, focal_length, camera, occupancy, ray_counter, sample_counter, ray_indices, rays, numsteps, sample_coords);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"generate_training_samples_kernel failed: "} + cudaGetErrorString(status)};
    }

    void compute_training_loss_and_compact_samples(const std::uint32_t rays_per_batch, const std::uint32_t current_step, const std::uint32_t* const ray_counter, const std::uint8_t* const pixels, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const std::uint16_t* const network_output, std::uint32_t* const compacted_sample_counter, const std::uint32_t* const ray_indices, const float* const rays, std::uint32_t* const numsteps, const float* const sample_coords, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients, float* const loss_values) {
        if (rays_per_batch == 0u) return;
        if (frame_count == 0u || width == 0u || height == 0u || ray_counter == nullptr || pixels == nullptr || network_output == nullptr || compacted_sample_counter == nullptr || ray_indices == nullptr || rays == nullptr || numsteps == nullptr || sample_coords == nullptr || compacted_sample_coords == nullptr || network_output_gradients == nullptr) throw std::runtime_error{"invalid loss and compaction input."};

        if (const cudaError_t status = cudaMemset(compacted_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset compacted sample counter failed: "} + cudaGetErrorString(status)};
        if (loss_values != nullptr)
            if (const cudaError_t status = cudaMemset(loss_values, 0, static_cast<std::size_t>(rays_per_batch) * sizeof(float)); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemset loss values failed: "} + cudaGetErrorString(status)};

        const std::uint32_t blocks = (rays_per_batch + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        compute_training_loss_and_compact_kernel<<<blocks, THREADS_PER_BLOCK>>>(rays_per_batch, current_step, ray_counter, pixels, frame_count, width, height, reinterpret_cast<const __half*>(network_output), compacted_sample_counter, ray_indices, rays, numsteps, sample_coords, compacted_sample_coords, reinterpret_cast<__half*>(network_output_gradients), loss_values);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"compute_training_loss_and_compact_kernel failed: "} + cudaGetErrorString(status)};
    }

    void pad_compacted_training_batch(const std::uint32_t* const compacted_sample_counter, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients) {
        if (compacted_sample_counter == nullptr || compacted_sample_coords == nullptr || network_output_gradients == nullptr) throw std::runtime_error{"rollover buffers are null."};

        constexpr std::uint32_t gradient_elements = config::NETWORK_BATCH_SIZE * MLP_OUTPUT_WIDTH;
        pad_rollover_network_output_gradients_kernel<<<(gradient_elements + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(compacted_sample_counter, reinterpret_cast<__half*>(network_output_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"pad_rollover_network_output_gradients_kernel failed: "} + cudaGetErrorString(status)};

        constexpr std::uint32_t coord_elements = config::NETWORK_BATCH_SIZE * SAMPLE_COORD_FLOATS;
        pad_rollover_coords_kernel<<<(coord_elements + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(compacted_sample_counter, compacted_sample_coords);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"pad_rollover_coords_kernel failed: "} + cudaGetErrorString(status)};
    }

    void read_counter(const std::uint32_t* const counter, std::uint32_t& out_value) {
        if (counter == nullptr) throw std::runtime_error{"counter is null."};
        if (const cudaError_t status = cudaMemcpy(&out_value, counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy counter failed: "} + cudaGetErrorString(status)};
    }

    void read_loss_sum(const float* const loss_values, const std::uint32_t loss_count, float& out_loss_sum) {
        out_loss_sum = 0.0f;
        if (loss_count == 0u) return;
        if (loss_values == nullptr) throw std::runtime_error{"loss values are null."};

        std::vector<float> host_loss(loss_count);
        if (const cudaError_t status = cudaMemcpy(host_loss.data(), loss_values, static_cast<std::size_t>(loss_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy loss values failed: "} + cudaGetErrorString(status)};
        for (const float loss : host_loss) out_loss_sum += loss;
    }

    void step_optimizer(const std::uint32_t param_count, const std::uint32_t mlp_param_count, float* const params_full_precision, std::uint16_t* const params, const std::uint16_t* const gradients, float* const first_moments, float* const second_moments, std::uint32_t* const param_steps) {
        if (param_count == 0u) return;
        if (mlp_param_count > param_count || params_full_precision == nullptr || params == nullptr || gradients == nullptr || first_moments == nullptr || second_moments == nullptr || param_steps == nullptr) throw std::runtime_error{"invalid optimizer input."};

        const std::uint32_t blocks = (param_count + THREADS_PER_BLOCK - 1u) / THREADS_PER_BLOCK;
        adam_step_kernel<<<blocks, THREADS_PER_BLOCK>>>(param_count, mlp_param_count, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<const __half*>(gradients), first_moments, second_moments, param_steps);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) throw std::runtime_error{std::string{"adam_step_kernel failed: "} + cudaGetErrorString(status)};
    }
} // namespace ngp::cuda
