#include "ngp.train.h"
#include <cmath>
#include <cuda/std/algorithm>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_splitk_parallel.h>
#include <cutlass/gemm/threadblock/threadblock_swizzle.h>
#include <cutlass/half.h>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <mma.h>
#include <vector>

namespace ngp::cuda {
    // Device memory.
    void free_device_data(void** const pointers, const std::size_t count) noexcept {
        for (std::size_t i = 0; i < count; ++i) {
            if (pointers[i] != nullptr) cudaFree(pointers[i]);
            pointers[i] = nullptr;
        }
    }

    namespace {
        using namespace config;

        // Launch configuration.
        inline constexpr std::uint32_t threads_per_block = 128u;

        // Sampler.
        inline constexpr std::uint32_t nerf_grid_size             = 128u;
        inline constexpr std::uint32_t nerf_grid_cells            = nerf_grid_size * nerf_grid_size * nerf_grid_size;
        inline constexpr std::uint32_t nerf_steps                 = 1024u;
        inline constexpr std::uint32_t max_random_samples_per_ray = 16u;
        inline constexpr std::uint32_t random_values_per_thread   = 4u;
        inline constexpr std::uint32_t sample_coord_floats        = 7u;
        inline constexpr std::uint32_t ray_floats                 = 6u;
        inline constexpr float min_cone_stepsize                  = 1.73205080757f / static_cast<float>(nerf_steps);

        // Grid encoding.
        inline constexpr std::uint32_t grid_forward_threads   = 512u;
        inline constexpr std::uint32_t grid_backward_threads  = 256u;
        inline constexpr std::uint32_t grid_backward_features = 2u;

        // Fully fused MLP.
        inline constexpr std::uint32_t mlp_forward_iters       = 8u;
        inline constexpr std::uint32_t mlp_width_blocks        = mlp_width / 16u;
        inline constexpr std::uint32_t mlp_skew                = 8u;
        inline constexpr std::uint32_t mlp_input_skew          = 8u;
        inline constexpr std::uint32_t mlp_first_layer_params  = mlp_width * mlp_input_width;
        inline constexpr std::uint32_t mlp_hidden_layer_params = mlp_width * mlp_width;
        inline constexpr std::uint32_t mlp_last_layer_params   = mlp_output_width * mlp_width;
        inline constexpr std::uint32_t density_network_params  = mlp_first_layer_params + (density_hidden_layers - 1u) * mlp_hidden_layer_params + mlp_last_layer_params;
        inline constexpr std::uint32_t rgb_network_params      = mlp_first_layer_params + (rgb_hidden_layers - 1u) * mlp_hidden_layer_params + mlp_last_layer_params;

        // Random number generation.
        inline constexpr std::uint64_t pcg32_default_state  = 0x853c49e6748fea9bULL;
        inline constexpr std::uint64_t pcg32_default_stream = 0xda3e39cb94b95bdbULL;
        inline constexpr std::uint64_t pcg32_mult           = 0x5851f42d4c957f2dULL;

        // Error reporting.
        std::string cuda_error(const char* operation, const cudaError_t status) {
            return std::string{operation} + " failed: " + cudaGetErrorString(status);
        }

        // Small POD helpers.
        struct GridOffsetTable final {
            std::uint32_t data[grid_n_levels + 1u] = {};
        };

        struct Pcg32 final {
            std::uint64_t state = pcg32_default_state;
            std::uint64_t inc   = pcg32_default_stream;

            __host__ __device__ explicit Pcg32(const std::uint64_t initstate, const std::uint64_t initseq = 1u) {
                this->seed(initstate, initseq);
            }

            __host__ __device__ void seed(const std::uint64_t initstate, const std::uint64_t initseq) {
                this->state = 0u;
                this->inc   = (initseq << 1u) | 1u;
                this->next_uint();
                this->state += initstate;
                this->next_uint();
            }

            __host__ __device__ std::uint32_t next_uint() {
                const std::uint64_t oldstate   = this->state;
                this->state                    = oldstate * pcg32_mult + this->inc;
                const std::uint32_t xorshifted = static_cast<std::uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
                const std::uint32_t rot        = static_cast<std::uint32_t>(oldstate >> 59u);
                return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31u));
            }

            __host__ __device__ float next_float() {
                union {
                    std::uint32_t bits;
                    float value;
                } result    = {};
                result.bits = (this->next_uint() >> 9u) | 0x3f800000u;
                return result.value - 1.0f;
            }

            __host__ __device__ void advance(std::uint64_t delta) {
                std::uint64_t cur_mult = pcg32_mult;
                std::uint64_t cur_plus = this->inc;
                std::uint64_t acc_mult = 1u;
                std::uint64_t acc_plus = 0u;

                while (delta > 0u) {
                    if ((delta & 1u) != 0u) {
                        acc_mult *= cur_mult;
                        acc_plus = acc_plus * cur_mult + cur_plus;
                    }

                    cur_plus = (cur_mult + 1u) * cur_plus;
                    cur_mult *= cur_mult;
                    delta >>= 1u;
                }

                this->state = acc_mult * this->state + acc_plus;
            }
        };

        // Grid encoding helpers.
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

        __device__ void pos_fract(const float input, float& pos, std::uint32_t& pos_grid, const float scale) {
            pos                 = fmaf(scale, input, 0.5f);
            const float floored = floorf(pos);
            pos_grid            = static_cast<std::uint32_t>(static_cast<int>(floored));
            pos -= floored;
        }

        // Sampler helpers.
        __device__ bool contains_unit_aabb(const float3 pos) {
            return pos.x >= 0.0f && pos.x <= 1.0f && pos.y >= 0.0f && pos.y <= 1.0f && pos.z >= 0.0f && pos.z <= 1.0f;
        }

        __device__ bool ray_intersect_unit_aabb(const float3 origin, const float3 direction, float& out_tmin) {
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

        __device__ bool density_grid_occupied_at(const float3 pos, const std::uint8_t* occupancy) {
            const int x = static_cast<int>(pos.x * static_cast<float>(nerf_grid_size));
            const int y = static_cast<int>(pos.y * static_cast<float>(nerf_grid_size));
            const int z = static_cast<int>(pos.z * static_cast<float>(nerf_grid_size));
            if (x < 0 || x >= static_cast<int>(nerf_grid_size) || y < 0 || y >= static_cast<int>(nerf_grid_size) || z < 0 || z >= static_cast<int>(nerf_grid_size)) return false;
            std::uint32_t morton_x    = static_cast<std::uint32_t>(x);
            std::uint32_t morton_y    = static_cast<std::uint32_t>(y);
            std::uint32_t morton_z    = static_cast<std::uint32_t>(z);
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

        __device__ float advance_to_next_voxel(const float t, const float3 pos, const float3 direction, const float3 inv_direction) {
            constexpr float scale = static_cast<float>(nerf_grid_size);
            const float3 p        = {(pos.x - 0.5f) * scale, (pos.y - 0.5f) * scale, (pos.z - 0.5f) * scale};
            const float tx        = (floorf(p.x + 0.5f + 0.5f * copysignf(1.0f, direction.x)) - p.x) * inv_direction.x;
            const float ty        = (floorf(p.y + 0.5f + 0.5f * copysignf(1.0f, direction.y)) - p.y) * inv_direction.y;
            const float tz        = (floorf(p.z + 0.5f + 0.5f * copysignf(1.0f, direction.z)) - p.z) * inv_direction.z;
            const float t_target  = t + fmaxf(fminf(fminf(tx, ty), tz) / scale, 0.0f);
            return t + ceilf(fmaxf((t_target - t) / min_cone_stepsize, 0.5f)) * min_cone_stepsize;
        }

        // Rendering and loss helpers.
        __device__ float logistic(const float x) {
            return 1.0f / (1.0f + expf(-x));
        }

        __device__ float network_to_density(const float value) {
            return expf(value);
        }

        __device__ float network_to_rgb_derivative(const float value) {
            const float rgb = logistic(value);
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

        __device__ float4 read_rgba(const std::uint32_t pixel_x, const std::uint32_t pixel_y, const std::uint32_t image, const std::uint32_t width, const std::uint32_t height, const std::uint8_t* pixels) {
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

        __device__ std::uint32_t image_index(const std::uint32_t ray_index, const std::uint32_t rays_per_batch, const std::uint32_t frame_count) {
            return static_cast<std::uint32_t>((static_cast<std::uint64_t>(ray_index) * frame_count) / rays_per_batch) % frame_count;
        }

        __device__ void random_image_pos_training(Pcg32& rng, const std::uint32_t width, const std::uint32_t height, const bool snap_to_pixel_centers, float& out_u, float& out_v) {
            out_u = rng.next_float();
            out_v = rng.next_float();
            if (!snap_to_pixel_centers) return;

            const std::uint32_t pixel_x = ::cuda::std::min(static_cast<std::uint32_t>(out_u * static_cast<float>(width)), width - 1u);
            const std::uint32_t pixel_y = ::cuda::std::min(static_cast<std::uint32_t>(out_v * static_cast<float>(height)), height - 1u);
            out_u                       = (static_cast<float>(pixel_x) + 0.5f) / static_cast<float>(width);
            out_v                       = (static_cast<float>(pixel_y) + 0.5f) / static_cast<float>(height);
        }

        // Sampler kernels.
        __global__ void generate_training_samples(const std::uint32_t rays_per_batch, const std::uint32_t max_samples, const std::uint64_t seed, const std::uint32_t current_step, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const bool snap_to_pixel_centers, const float* __restrict__ camera, const std::uint8_t* __restrict__ occupancy, std::uint32_t* __restrict__ ray_counter,
            std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ ray_indices_out, float* __restrict__ rays_out, std::uint32_t* __restrict__ numsteps_out, float* __restrict__ coords_out) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= rays_per_batch) return;

            const std::uint32_t image = static_cast<std::uint32_t>((static_cast<std::uint64_t>(i) * frame_count) / rays_per_batch) % frame_count;
            const float* frame_camera = camera + static_cast<std::uint64_t>(image) * 12u;

            Pcg32 rng{seed};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(i) * max_random_samples_per_ray);

            float u = rng.next_float();
            float v = rng.next_float();
            if (snap_to_pixel_centers) {
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
            if (!ray_intersect_unit_aabb(ray_origin, ray_direction_normalized, tmin)) return;

            constexpr float dt         = min_cone_stepsize;
            const float start_t        = tmin + rng.next_float() * dt;
            const float3 inv_direction = {1.0f / ray_direction_normalized.x, 1.0f / ray_direction_normalized.y, 1.0f / ray_direction_normalized.z};

            std::uint32_t numsteps = 0u;
            float t                = start_t;
            float3 pos             = {};

            while (numsteps < nerf_steps) {
                pos = {ray_origin.x + ray_direction_normalized.x * t, ray_origin.y + ray_direction_normalized.y * t, ray_origin.z + ray_direction_normalized.z * t};
                if (!contains_unit_aabb(pos)) break;

                if (density_grid_occupied_at(pos, occupancy)) {
                    ++numsteps;
                    t += dt;
                } else {
                    t = advance_to_next_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }

            if (numsteps == 0u) return;

            const std::uint32_t base = atomicAdd(sample_counter, numsteps);
            if (base + numsteps > max_samples) return;

            const std::uint32_t ray_index = atomicAdd(ray_counter, 1u);
            ray_indices_out[ray_index]    = i;

            float* ray_out = rays_out + static_cast<std::uint64_t>(ray_index) * ray_floats;
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
                if (!contains_unit_aabb(pos)) break;

                if (density_grid_occupied_at(pos, occupancy)) {
                    float* coord = coords_out + static_cast<std::uint64_t>(base + j) * sample_coord_floats;
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
                    t = advance_to_next_voxel(t, pos, ray_direction_normalized, inv_direction);
                }
            }
        }

        // Loss and compaction kernels.
        __global__ void compute_loss_and_compact_kernel(const std::uint32_t rays_per_batch, const std::uint32_t batch_size, const std::uint64_t seed, const std::uint32_t current_step, const std::uint32_t* __restrict__ ray_counter, const std::uint8_t* __restrict__ pixels, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const bool snap_to_pixel_centers, const __half* __restrict__ network_output,
            std::uint32_t* __restrict__ compacted_sample_counter, const std::uint32_t* __restrict__ ray_indices_in, const float* __restrict__ rays_in, std::uint32_t* __restrict__ numsteps_in, const float* __restrict__ coords_in, float* __restrict__ coords_out, __half* __restrict__ dloss_doutput, float* __restrict__ loss_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= *ray_counter) return;

            std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
            std::uint32_t base     = numsteps_in[i * 2u + 1u];

            const float* coord_in = coords_in + static_cast<std::uint64_t>(base) * sample_coord_floats;
            const __half* output  = network_output + static_cast<std::uint64_t>(base) * mlp_output_width;

            float transmittance                   = 1.0f;
            constexpr float transmittance_epsilon = 1e-4f;
            float3 rgb_ray                        = {};
            std::uint32_t compacted_numsteps      = 0u;
            const float* ray                      = rays_in + static_cast<std::uint64_t>(i) * ray_floats;
            const float3 ray_origin               = {ray[0], ray[1], ray[2]};

            for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
                if (transmittance < transmittance_epsilon) break;

                const float rgb_x   = logistic(__half2float(output[0u]));
                const float rgb_y   = logistic(__half2float(output[1u]));
                const float rgb_z   = logistic(__half2float(output[2u]));
                const float density = network_to_density(__half2float(output[3u]));
                const float alpha   = 1.0f - __expf(-density * coord_in[3u]);
                const float weight  = alpha * transmittance;
                rgb_ray.x += weight * rgb_x;
                rgb_ray.y += weight * rgb_y;
                rgb_ray.z += weight * rgb_z;
                transmittance *= 1.0f - alpha;

                output += mlp_output_width;
                coord_in += sample_coord_floats;
            }

            const std::uint32_t ray_index = ray_indices_in[i];
            Pcg32 rng{seed};
            rng.advance(static_cast<std::uint64_t>(current_step) << 32u);
            rng.advance(static_cast<std::uint64_t>(ray_index) * max_random_samples_per_ray);

            const std::uint32_t image = image_index(ray_index, rays_per_batch, frame_count);
            float u                   = 0.0f;
            float v                   = 0.0f;
            random_image_pos_training(rng, width, height, snap_to_pixel_centers, u, v);

            const float3 background_color  = {rng.next_float(), rng.next_float(), rng.next_float()};
            const std::uint32_t pixel_x    = ::cuda::std::min(static_cast<std::uint32_t>(u * static_cast<float>(width)), width - 1u);
            const std::uint32_t pixel_y    = ::cuda::std::min(static_cast<std::uint32_t>(v * static_cast<float>(height)), height - 1u);
            const float4 texel             = read_rgba(pixel_x, pixel_y, image, width, height, pixels);
            const float3 background_linear = srgb_to_linear(background_color);
            const float3 rgb_target        = linear_to_srgb({texel.x + (1.0f - texel.w) * background_linear.x, texel.y + (1.0f - texel.w) * background_linear.y, texel.z + (1.0f - texel.w) * background_linear.z});

            if (compacted_numsteps == numsteps) {
                rgb_ray.x += transmittance * background_color.x;
                rgb_ray.y += transmittance * background_color.y;
                rgb_ray.z += transmittance * background_color.z;
            }

            output -= static_cast<std::uint64_t>(compacted_numsteps) * mlp_output_width;
            coord_in -= static_cast<std::uint64_t>(compacted_numsteps) * sample_coord_floats;

            std::uint32_t compacted_base = atomicAdd(compacted_sample_counter, compacted_numsteps);
            compacted_numsteps           = ::cuda::std::min(batch_size - ::cuda::std::min(batch_size, compacted_base), compacted_numsteps);
            numsteps_in[i * 2u + 0u]     = compacted_numsteps;
            numsteps_in[i * 2u + 1u]     = compacted_base;
            if (compacted_numsteps == 0u) return;

            coords_out += static_cast<std::uint64_t>(compacted_base) * sample_coord_floats;
            dloss_doutput += static_cast<std::uint64_t>(compacted_base) * mlp_output_width;

            const float3 difference = {rgb_ray.x - rgb_target.x, rgb_ray.y - rgb_target.y, rgb_ray.z - rgb_target.z};
            const float3 gradient   = {2.0f * difference.x, 2.0f * difference.y, 2.0f * difference.z};
            if (loss_output != nullptr) loss_output[i] = (difference.x * difference.x + difference.y * difference.y + difference.z * difference.z) / (3.0f * static_cast<float>(rays_per_batch));

            const float scaled_loss = 128.0f / static_cast<float>(rays_per_batch);
            float3 rgb_ray2         = {};
            transmittance           = 1.0f;

            for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
                float* coord_out   = coords_out + static_cast<std::uint64_t>(j) * sample_coord_floats;
                const float* coord = coord_in + static_cast<std::uint64_t>(j) * sample_coord_floats;
                for (std::uint32_t k = 0u; k < sample_coord_floats; ++k) coord_out[k] = coord[k];

                const float3 pos        = {coord[0], coord[1], coord[2]};
                const float depth       = norm3df(pos.x - ray_origin.x, pos.y - ray_origin.y, pos.z - ray_origin.z);
                const float dt          = coord[3u];
                const float mlp_rgb_x   = __half2float(output[0u]);
                const float mlp_rgb_y   = __half2float(output[1u]);
                const float mlp_rgb_z   = __half2float(output[2u]);
                const float mlp_density = __half2float(output[3u]);
                const float3 rgb        = {logistic(mlp_rgb_x), logistic(mlp_rgb_y), logistic(mlp_rgb_z)};
                const float density     = network_to_density(mlp_density);
                const float alpha       = 1.0f - __expf(-density * dt);
                const float weight      = alpha * transmittance;
                rgb_ray2.x += weight * rgb.x;
                rgb_ray2.y += weight * rgb.y;
                rgb_ray2.z += weight * rgb.z;
                transmittance *= 1.0f - alpha;

                const float3 suffix        = {rgb_ray.x - rgb_ray2.x, rgb_ray.y - rgb_ray2.y, rgb_ray.z - rgb_ray2.z};
                const float3 dloss_by_drgb = {weight * gradient.x, weight * gradient.y, weight * gradient.z};

                dloss_doutput[0u] = __float2half(scaled_loss * (dloss_by_drgb.x * network_to_rgb_derivative(mlp_rgb_x)));
                dloss_doutput[1u] = __float2half(scaled_loss * (dloss_by_drgb.y * network_to_rgb_derivative(mlp_rgb_y)));
                dloss_doutput[2u] = __float2half(scaled_loss * (dloss_by_drgb.z * network_to_rgb_derivative(mlp_rgb_z)));

                const float density_derivative = expf(::cuda::std::clamp(mlp_density, -15.0f, 15.0f));
                const float dloss_by_dmlp      = density_derivative * (dt * (gradient.x * (transmittance * rgb.x - suffix.x) + gradient.y * (transmittance * rgb.y - suffix.y) + gradient.z * (transmittance * rgb.z - suffix.z)));
                dloss_doutput[3u]              = __float2half(scaled_loss * dloss_by_dmlp + (mlp_density > -10.0f && depth < 0.1f ? 1e-4f : 0.0f));

                dloss_doutput += mlp_output_width;
                output += mlp_output_width;
            }
        }

        // Rollover kernels.
        template <typename T>
        __global__ void fill_rollover_kernel(const std::uint32_t sample_count, const std::uint32_t stride, const std::uint32_t* __restrict__ input_count, T* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * stride || i >= sample_count * stride || n == 0u) return;
            inout[i] = inout[i % (n * stride)];
        }

        __global__ void fill_rollover_and_rescale_half_kernel(const std::uint32_t sample_count, const std::uint32_t stride, const std::uint32_t* __restrict__ input_count, __half* __restrict__ inout) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n = *input_count;
            if (i < n * stride || i >= sample_count * stride || n == 0u) return;
            inout[i] = __float2half(__half2float(inout[i % (n * stride)]) * static_cast<float>(n) / static_cast<float>(sample_count));
        }

        // Encoding kernels.
        __global__ void encode_grid_forward_kernel(const std::uint32_t sample_count, const GridOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, const float* __restrict__ sample_coords, const __half* __restrict__ grid, __half* __restrict__ encoded_positions) {
            const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i >= sample_count) return;

            const std::uint32_t level = blockIdx.y;
            grid += offset_table.data[level] * grid_features_per_level;
            const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
            const float scale                = exp2f(static_cast<float>(level) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
            const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
            const float* sample              = sample_coords + static_cast<std::uint64_t>(i) * sample_coord_floats;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            pos_fract(sample[0], pos_x, grid_x, scale);
            pos_fract(sample[1], pos_y, grid_y, scale);
            pos_fract(sample[2], pos_z, grid_z, scale);

            __half result0 = 0.0f;
            __half result1 = 0.0f;
            __half result2 = 0.0f;
            __half result3 = 0.0f;

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * grid_features_per_level;
                const __half weight_half  = weight;
                result0                   = __hfma(weight_half, grid[index + 0u], result0);
                result1                   = __hfma(weight_half, grid[index + 1u], result1);
                result2                   = __hfma(weight_half, grid[index + 2u], result2);
                result3                   = __hfma(weight_half, grid[index + 3u], result3);
            }

            encoded_positions[i + (level * grid_features_per_level + 0u) * sample_count] = result0;
            encoded_positions[i + (level * grid_features_per_level + 1u) * sample_count] = result1;
            encoded_positions[i + (level * grid_features_per_level + 2u) * sample_count] = result2;
            encoded_positions[i + (level * grid_features_per_level + 3u) * sample_count] = result3;
        }

        __global__ void encode_grid_backward_kernel(const std::uint32_t sample_count, const GridOffsetTable offset_table, const std::uint32_t base_resolution, const float log2_per_level_scale, const float* __restrict__ sample_coords, const __half* __restrict__ encoded_position_gradients, __half* __restrict__ grid_gradients) {
            const std::uint32_t thread = blockIdx.x * blockDim.x + threadIdx.x;
            const std::uint32_t i      = (thread * grid_backward_features) / grid_features_per_level;
            if (i >= sample_count) return;

            const std::uint32_t level   = blockIdx.y;
            const std::uint32_t feature = thread * grid_backward_features - i * grid_features_per_level;
            grid_gradients += offset_table.data[level] * grid_features_per_level;
            const std::uint32_t hashmap_size = offset_table.data[level + 1u] - offset_table.data[level];
            const float scale                = exp2f(static_cast<float>(level) * log2_per_level_scale) * static_cast<float>(base_resolution) - 1.0f;
            const std::uint32_t resolution   = static_cast<std::uint32_t>(ceilf(scale)) + 1u;
            const float* sample              = sample_coords + static_cast<std::uint64_t>(i) * sample_coord_floats;

            float pos_x          = 0.0f;
            float pos_y          = 0.0f;
            float pos_z          = 0.0f;
            std::uint32_t grid_x = 0u;
            std::uint32_t grid_y = 0u;
            std::uint32_t grid_z = 0u;
            pos_fract(sample[0], pos_x, grid_x, scale);
            pos_fract(sample[1], pos_y, grid_y, scale);
            pos_fract(sample[2], pos_z, grid_z, scale);

            const __half grad0 = encoded_position_gradients[i + (level * grid_features_per_level + feature + 0u) * sample_count];
            const __half grad1 = encoded_position_gradients[i + (level * grid_features_per_level + feature + 1u) * sample_count];

            for (std::uint32_t corner = 0u; corner < 8u; ++corner) {
                const bool high_x         = (corner & 1u) != 0u;
                const bool high_y         = (corner & 2u) != 0u;
                const bool high_z         = (corner & 4u) != 0u;
                const float weight        = (high_x ? pos_x : 1.0f - pos_x) * (high_y ? pos_y : 1.0f - pos_y) * (high_z ? pos_z : 1.0f - pos_z);
                const std::uint32_t index = grid_index(hashmap_size, resolution, high_x ? grid_x + 1u : grid_x, high_y ? grid_y + 1u : grid_y, high_z ? grid_z + 1u : grid_z) * grid_features_per_level + feature;
                const __half weight_half  = weight;
                atomicAdd(reinterpret_cast<__half2*>(grid_gradients + index), __halves2half2(__hmul(weight_half, grad0), __hmul(weight_half, grad1)));
            }
        }

        __global__ void encode_spherical_harmonics_kernel(const std::uint32_t sample_count, const float* __restrict__ sample_coords, __half* __restrict__ output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= sample_count) return;

            const float* coord = sample_coords + static_cast<std::uint64_t>(i) * sample_coord_floats;
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

        // Parameter initialization kernels.
        __global__ void cast_params_to_half_kernel(const std::uint32_t param_count, const float* __restrict__ params_full_precision, __half* __restrict__ params) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;
            params[i] = static_cast<__half>(params_full_precision[i]);
        }

        __global__ void initialize_grid_params_kernel(const std::uint32_t param_count, const std::uint64_t seed, const std::uint64_t rng_offset, float* __restrict__ params_full_precision, __half* __restrict__ params, __half* __restrict__ param_gradients) {
            const std::uint32_t i         = threadIdx.x + blockIdx.x * blockDim.x;
            const std::uint32_t n_threads = blockDim.x * gridDim.x;
            Pcg32 rng{seed};
            rng.advance(rng_offset + static_cast<std::uint64_t>(i) * random_values_per_thread);

            for (std::uint32_t j = 0u; j < random_values_per_thread; ++j) {
                const std::uint32_t idx = i + n_threads * j;
                if (idx >= param_count) return;

                const float value          = rng.next_float() * 2e-4f - 1e-4f;
                params_full_precision[idx] = value;
                params[idx]                = static_cast<__half>(value);
                param_gradients[idx]       = static_cast<__half>(0.0f);
            }
        }

        // Fully fused MLP helpers.
        template <typename ThreadBlock, typename Warp>
        struct CutlassLayerConfig {
            using thread_block_shape = ThreadBlock;
            using warp_shape         = Warp;
        };

        struct CutlassFullLayer : CutlassLayerConfig<cutlass::gemm::GemmShape<128, 128, 32>, cutlass::gemm::GemmShape<64, 64, 32>> {};
        struct CutlassLastLayer final : CutlassFullLayer {};
        struct CutlassFullLayerK : CutlassLayerConfig<cutlass::gemm::GemmShape<64, 64, 32>, cutlass::gemm::GemmShape<32, 32, 32>> {};
        struct CutlassLastLayerK final : CutlassFullLayerK {};

        template <typename T>
        inline constexpr int cutlass_vector_elements = 128 / cutlass::sizeof_bits<T>::value;

        std::string cutlass_error(const char* operation, const cutlass::Status status) {
            if (status == cutlass::Status::kSuccess) return {};
            return std::string{operation} + " failed: " + cutlassGetStatusString(status);
        }

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        using CutlassGemm = cutlass::gemm::device::Gemm<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t, LayoutD, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::epilogue::thread::LinearCombination<cutlass::half_t, cutlass_vector_elements<cutlass::half_t>, cutlass::half_t, cutlass::half_t>,
            cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        using CutlassSplitKGemm = cutlass::gemm::device::GemmSplitKParallel<cutlass::half_t, LayoutA, cutlass::half_t, LayoutB, cutlass::half_t, LayoutD, cutlass::half_t, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, typename Config::thread_block_shape, typename Config::warp_shape, cutlass::gemm::GemmShape<16, 8, 8>, cutlass::epilogue::thread::LinearCombination<cutlass::half_t, cutlass_vector_elements<cutlass::half_t>, cutlass::half_t, cutlass::half_t>>;

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        std::size_t cutlass_gemm_workspace_size(const int m, const int n, const int k, const int lda, const int ldb, const int ldd) {
            using Gemm = CutlassGemm<Config, LayoutA, LayoutB, LayoutD>;
            typename Gemm::Arguments args{{m, n, k}, {nullptr, lda}, {nullptr, ldb}, {nullptr, ldd}, {nullptr, ldd}, {cutlass::half_t{1.0f}, cutlass::half_t{0.0f}}, 1};
            return Gemm::get_workspace_size(args);
        }

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        std::size_t cutlass_split_k_workspace_size(const int m, const int n, const int k, const int lda, const int ldb, const int ldd, const int split_k) {
            using Gemm = CutlassSplitKGemm<Config, LayoutA, LayoutB, LayoutD>;
            typename Gemm::Arguments args{{m, n, k}, {nullptr, lda}, {nullptr, ldb}, {nullptr, ldd}, {nullptr, ldd}, {cutlass::half_t{1.0f}, cutlass::half_t{0.0f}}, split_k};
            return Gemm::get_workspace_size(args);
        }

        std::size_t cutlass_workspace_size(const std::uint32_t batch_size) {
            const int batch   = static_cast<int>(batch_size);
            const int split_k = static_cast<int>(batch_size / ::cuda::std::min(1u << 12u, batch_size));
            std::size_t bytes = 0u;
            bytes             = ::cuda::std::max(bytes, cutlass_split_k_workspace_size<CutlassLastLayerK, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(16, 64, batch, 16, 64, 64, split_k));
            bytes             = ::cuda::std::max(bytes, cutlass_split_k_workspace_size<CutlassLastLayerK, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(16, 64, batch, batch, 64, 64, split_k));
            bytes             = ::cuda::std::max(bytes, cutlass_split_k_workspace_size<CutlassFullLayerK, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(64, 64, batch, 64, 64, 64, split_k));
            bytes             = ::cuda::std::max(bytes, cutlass_split_k_workspace_size<CutlassFullLayerK, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(64, 32, batch, 64, batch, 32, split_k));
            bytes             = ::cuda::std::max(bytes, cutlass_gemm_workspace_size<CutlassFullLayer, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(32, batch, 64, 32, 64, batch));
            return bytes;
        }

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        std::string run_cutlass_gemm(const int m, const int n, const int k, const __half* const a, const int lda, const __half* const b, const int ldb, __half* const d, const int ldd, void* const workspace) {
            using Gemm = CutlassGemm<Config, LayoutA, LayoutB, LayoutD>;
            typename Gemm::Arguments args{{m, n, k}, {reinterpret_cast<cutlass::half_t*>(const_cast<__half*>(a)), lda}, {reinterpret_cast<cutlass::half_t*>(const_cast<__half*>(b)), ldb}, {reinterpret_cast<cutlass::half_t*>(d), ldd}, {reinterpret_cast<cutlass::half_t*>(d), ldd}, {cutlass::half_t{1.0f}, cutlass::half_t{0.0f}}, 1};
            Gemm gemm;
            if (std::string error = cutlass_error("cutlass gemm initialize", gemm.initialize(args, workspace, nullptr)); !error.empty()) return error;
            if (std::string error = cutlass_error("cutlass gemm", gemm(nullptr)); !error.empty()) return error;
            return {};
        }

        template <typename Config, typename LayoutA, typename LayoutB, typename LayoutD>
        std::string run_cutlass_split_k(const int m, const int n, const int k, const __half* const a, const int lda, const __half* const b, const int ldb, __half* const d, const int ldd, void* const workspace, const int split_k) {
            using Gemm = CutlassSplitKGemm<Config, LayoutA, LayoutB, LayoutD>;
            typename Gemm::Arguments args{{m, n, k}, {reinterpret_cast<cutlass::half_t*>(const_cast<__half*>(a)), lda}, {reinterpret_cast<cutlass::half_t*>(const_cast<__half*>(b)), ldb}, {reinterpret_cast<cutlass::half_t*>(d), ldd}, {reinterpret_cast<cutlass::half_t*>(d), ldd}, {cutlass::half_t{1.0f}, cutlass::half_t{0.0f}}, split_k};
            Gemm gemm;
            if (std::string error = cutlass_error("cutlass split-k gemm initialize", gemm.initialize(args, workspace)); !error.empty()) return error;
            if (std::string error = cutlass_error("cutlass split-k gemm", gemm(nullptr)); !error.empty()) return error;
            return {};
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
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[mlp_forward_iters];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __half* __restrict__ weights_shmem        = act_shmem + 16u * (mlp_input_width + mlp_input_skew);
            constexpr std::uint32_t n_elems_per_load      = mlp_width_blocks * 32u * 8u;
            const std::uint32_t thread_elem_idx       = (li + wi * 32u) * 8u;
            constexpr std::uint32_t n_weight_elements = mlp_width * mlp_input_width;

            for (std::uint32_t idx = thread_elem_idx; idx < n_weight_elements; idx += n_elems_per_load) {
                const std::uint32_t idx_skewed                       = idx + idx / mlp_input_width * mlp_input_skew;
                *reinterpret_cast<int4*>(&weights_shmem[idx_skewed]) = *reinterpret_cast<const int4*>(&weights_this_layer[idx]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < mlp_input_width / 16u; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, input_threadblock + 16u * i * batch_size + 16u * l, batch_size);
                    nvcuda::wmma::load_matrix_sync(weights_frag, weights_shmem + 16u * i + weights_col * (mlp_input_width + mlp_input_skew), mlp_input_width + mlp_input_skew);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (mlp_width + mlp_skew), result_frag[l], mlp_width + mlp_skew, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < mlp_forward_iters; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (mlp_width + mlp_skew)]);
        }

        __device__ void mlp_hidden_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ hidden_threadblock) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[mlp_forward_iters];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i + weights_col * mlp_width, mlp_width);

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (mlp_width + mlp_skew), mlp_width + mlp_skew);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }
                relu_fragment(result_frag[l]);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + l * 16u * (mlp_width + mlp_skew), result_frag[l], mlp_width + mlp_skew, nvcuda::wmma::mem_row_major);

            __syncthreads();

            if (hidden_threadblock != nullptr)
                for (std::uint32_t i = 0u; i < mlp_forward_iters; ++i) *reinterpret_cast<int4*>(&hidden_threadblock[lane_offset + (row + 16u * i) * mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (mlp_width + mlp_skew)]);
        }

        __device__ void mlp_last_layer_forward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, __half* __restrict__ out, const std::uint32_t output_stride, const nvcuda::wmma::layout_t output_layout) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> weights_frag[mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag;

            const std::uint32_t li = threadIdx.x;
            const std::uint32_t wi = threadIdx.y;

            __half* __restrict__ weights_shmem = act_shmem + mlp_forward_iters * 16u * (mlp_width + mlp_skew);
            const std::uint32_t weights_row    = (8u * li) % mlp_width;
            const std::uint32_t weights_col    = (8u * li + 8u * 32u * wi) / mlp_width;

            *reinterpret_cast<int4*>(&weights_shmem[weights_row + weights_col * (mlp_width + mlp_skew)]) = *reinterpret_cast<const int4*>(&weights_this_layer[weights_row + weights_col * mlp_width]);
            __syncthreads();

            for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_shmem + 16u * i, mlp_width + mlp_skew);

            for (std::uint32_t idx = wi; idx < mlp_forward_iters; idx += mlp_width_blocks) {
                nvcuda::wmma::fill_fragment(result_frag, 0.0f);
                for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * idx) * (mlp_width + mlp_skew), mlp_width + mlp_skew);
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
            const std::uint32_t elem_idx = 16u * blockIdx.x * mlp_forward_iters;

            mlp_input_layer_forward(shmem, input + elem_idx, weights, hidden == nullptr ? nullptr : hidden + elem_idx * mlp_width, batch_size);
            if (hidden_layers == 2u) mlp_hidden_layer_forward(shmem, weights + mlp_first_layer_params, hidden == nullptr ? nullptr : hidden + static_cast<std::uint64_t>(mlp_width) * batch_size + elem_idx * mlp_width);

            const __half* last_weights = weights + mlp_first_layer_params + (hidden_layers - 1u) * mlp_hidden_layer_params;
            if (output_row_major)
                mlp_last_layer_forward(shmem, last_weights, output + elem_idx * mlp_output_width, mlp_output_width, nvcuda::wmma::mem_row_major);
            else
                mlp_last_layer_forward(shmem, last_weights, output + elem_idx, batch_size, nvcuda::wmma::mem_col_major);
        }

        __device__ void mlp_hidden_layer_backward(__half* __restrict__ act_shmem, const __half* __restrict__ weights_this_layer, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden) {
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag[mlp_width_blocks];
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[mlp_forward_iters];

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t wi          = threadIdx.y;
            const std::uint32_t lane_offset = (8u * li) % mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / mlp_width;
            const std::uint32_t weights_col = 16u * wi;

            __syncthreads();

            for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) nvcuda::wmma::load_matrix_sync(weights_frag[i], weights_this_layer + 16u * i * mlp_width + weights_col, mlp_width);

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);
                for (std::uint32_t i = 0u; i < mlp_width_blocks; ++i) {
                    nvcuda::wmma::load_matrix_sync(act_frag, act_shmem + 16u * i + (16u * l) * (mlp_width + mlp_skew), mlp_width + mlp_skew);
                    nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag[i], result_frag[l]);
                }

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_hidden + weights_col + l * 16u * mlp_width, mlp_width);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (mlp_width + mlp_skew), result_frag[l], mlp_width + mlp_skew, nvcuda::wmma::mem_row_major);

            __syncthreads();

            for (std::uint32_t i = 0u; i < mlp_forward_iters; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + i * 16u) * mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (mlp_width + mlp_skew)]);
        }

        template <typename OutputLayout>
        __global__ void mlp_backward_hidden_64_relu_kernel(const std::uint32_t batch_size, const __half* __restrict__ dloss_doutput, const __half* __restrict__ weights, const __half* __restrict__ forward_hidden, __half* __restrict__ backward_hidden, const std::uint32_t output_stride, const std::uint32_t hidden_layers) {
            const std::uint32_t wi            = threadIdx.y;
            const std::uint32_t elem_idx_base = 16u * blockIdx.x * mlp_forward_iters;

            extern __shared__ __half shmem[];
            __half* act_shmem = shmem;

            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, OutputLayout> act_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> weights_frag;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, __half> result_frag[mlp_forward_iters];

            const std::uint32_t weights_col = 16u * wi;
            const __half* last_weights      = weights + mlp_first_layer_params + (hidden_layers - 1u) * mlp_hidden_layer_params;
            const __half* forward_last      = forward_hidden + static_cast<std::uint64_t>(hidden_layers - 1u) * mlp_width * batch_size;
            nvcuda::wmma::load_matrix_sync(weights_frag, last_weights + weights_col, mlp_width);

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) {
                nvcuda::wmma::fill_fragment(result_frag[l], 0.0f);

                if constexpr (std::is_same_v<OutputLayout, nvcuda::wmma::row_major>)
                    nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + (elem_idx_base + 16u * l) * output_stride, output_stride);
                else
                    nvcuda::wmma::load_matrix_sync(act_frag, dloss_doutput + elem_idx_base + 16u * l, output_stride);

                nvcuda::wmma::mma_sync(result_frag[l], act_frag, weights_frag, result_frag[l]);

                nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> forward_frag;
                nvcuda::wmma::load_matrix_sync(forward_frag, forward_last + weights_col + (elem_idx_base + l * 16u) * mlp_width, mlp_width);
                relu_backward_fragment(result_frag[l], forward_frag);
            }

            __syncthreads();

            for (std::uint32_t l = 0u; l < mlp_forward_iters; ++l) nvcuda::wmma::store_matrix_sync(act_shmem + weights_col + (16u * l) * (mlp_width + mlp_skew), result_frag[l], mlp_width + mlp_skew, nvcuda::wmma::mem_row_major);

            __syncthreads();

            const std::uint32_t li          = threadIdx.x;
            const std::uint32_t lane_offset = (8u * li) % mlp_width;
            const std::uint32_t row         = (8u * li + wi * 8u * 32u) / mlp_width;

            for (std::uint32_t i = 0u; i < mlp_forward_iters; ++i) *reinterpret_cast<int4*>(&backward_hidden[lane_offset + (row + elem_idx_base + i * 16u) * mlp_width]) = *reinterpret_cast<int4*>(&act_shmem[lane_offset + (row + 16u * i) * (mlp_width + mlp_skew)]);

            if (hidden_layers == 2u) mlp_hidden_layer_backward(act_shmem, weights + mlp_first_layer_params, forward_hidden + elem_idx_base * mlp_width, backward_hidden + static_cast<std::uint64_t>(mlp_width) * batch_size + elem_idx_base * mlp_width);
        }

        // Network bridge kernels.
        __global__ void extract_density_kernel(const std::uint32_t batch_size, const __half* __restrict__ density_output, __half* __restrict__ network_output) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            network_output[static_cast<std::uint64_t>(i) * mlp_output_width + 3u] = density_output[i];
        }

        __global__ void extract_rgb_gradients_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ rgb_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;

            const __half zero = 0.0f;
            for (std::uint32_t j = 0u; j < mlp_output_width; ++j) rgb_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + j] = zero;
            rgb_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 0u] = network_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 0u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 1u] = network_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 1u];
            rgb_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 2u] = network_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 2u];
        }

        __global__ void add_density_gradient_kernel(const std::uint32_t batch_size, const __half* __restrict__ network_output_gradients, __half* __restrict__ density_output_gradients) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= batch_size) return;
            density_output_gradients[i] = density_output_gradients[i] + network_output_gradients[static_cast<std::uint64_t>(i) * mlp_output_width + 3u];
        }

        // Optimizer kernels.
        __global__ void adam_step(const std::uint32_t param_count, const std::uint32_t mlp_param_count, const float loss_scale, const float learning_rate, const float beta1, const float beta2, const float epsilon, const float l2_reg, float* __restrict__ params_full_precision, __half* __restrict__ params, const __half* __restrict__ gradients, float* __restrict__ first_moments, float* __restrict__ second_moments, std::uint32_t* __restrict__ param_steps) {
            const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
            if (i >= param_count) return;

            float gradient = static_cast<float>(gradients[i]) / loss_scale;
            if (i >= mlp_param_count && gradient == 0.0f) return;

            const float param = params_full_precision[i];
            if (i < mlp_param_count) gradient += l2_reg * param;

            const float gradient_sq  = gradient * gradient;
            const float first_moment = first_moments[i] = beta1 * first_moments[i] + (1.0f - beta1) * gradient;
            const float second_moment = second_moments[i] = beta2 * second_moments[i] + (1.0f - beta2) * gradient_sq;
            const std::uint32_t step                      = ++param_steps[i];
            const float corrected_lr                      = learning_rate * sqrtf(1.0f - powf(beta2, static_cast<float>(step))) / (1.0f - powf(beta1, static_cast<float>(step)));
            const float updated_param                     = param - corrected_lr * first_moment / (sqrtf(second_moment) + epsilon);

            params_full_precision[i] = updated_param;
            params[i]                = static_cast<__half>(updated_param);
        }
    } // namespace

    // Dataset.
    std::string copy_dataset_to_device_once(const std::uint8_t* const pixels, const std::size_t pixels_bytes, const float* const camera, const std::size_t camera_count, const std::uint8_t*& out_pixels, const float*& out_camera) {
        out_pixels = nullptr;
        out_camera = nullptr;

        if (pixels == nullptr || pixels_bytes == 0) return "pixels is empty.";
        if (camera == nullptr || camera_count == 0) return "camera is empty.";

        void* uploaded_pixels = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_pixels, pixels_bytes); status != cudaSuccess) return cuda_error("cudaMalloc pixels", status);
        out_pixels = static_cast<std::uint8_t*>(uploaded_pixels);

        if (const cudaError_t status = cudaMemcpy(const_cast<std::uint8_t*>(out_pixels), pixels, pixels_bytes, cudaMemcpyHostToDevice); status != cudaSuccess) {
            free_device_data(out_pixels, out_camera);
            return cuda_error("cudaMemcpy pixels", status);
        }

        void* uploaded_camera = nullptr;
        if (const cudaError_t status = cudaMalloc(&uploaded_camera, camera_count * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_pixels, out_camera);
            return cuda_error("cudaMalloc camera", status);
        }
        out_camera = static_cast<float*>(uploaded_camera);

        if (const cudaError_t status = cudaMemcpy(const_cast<float*>(out_camera), camera, camera_count * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) {
            free_device_data(out_pixels, out_camera);
            return cuda_error("cudaMemcpy camera", status);
        }

        return {};
    }

    // Sampler.
    std::string allocate_sampler_once(const std::uint32_t rays_per_batch, const std::uint32_t max_samples, float*& out_sample_coords, float*& out_rays, std::uint32_t*& out_ray_indices, std::uint32_t*& out_numsteps, std::uint32_t*& out_ray_counter, std::uint32_t*& out_sample_counter, std::uint8_t*& out_occupancy) {
        out_sample_coords  = nullptr;
        out_rays           = nullptr;
        out_ray_indices    = nullptr;
        out_numsteps       = nullptr;
        out_ray_counter    = nullptr;
        out_sample_counter = nullptr;
        out_occupancy      = nullptr;

        if (rays_per_batch == 0u) return "sampler rays per batch is zero.";
        if (max_samples == 0u) return "sampler max samples is zero.";

        if (const cudaError_t status = cudaMalloc(&out_sample_coords, static_cast<std::size_t>(max_samples) * sample_coord_floats * sizeof(float)); status != cudaSuccess) return cuda_error("cudaMalloc sampler sample coords", status);
        if (const cudaError_t status = cudaMalloc(&out_rays, static_cast<std::size_t>(rays_per_batch) * ray_floats * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler rays", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_ray_indices, static_cast<std::size_t>(rays_per_batch) * sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler ray indices", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_numsteps, static_cast<std::size_t>(rays_per_batch) * 2u * sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler numsteps", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_ray_counter, sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler ray counter", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler sample counter", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_occupancy, nerf_grid_cells / 8u); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMalloc sampler occupancy", status);
        }
        if (const cudaError_t status = cudaMemset(out_occupancy, 0xFF, nerf_grid_cells / 8u); status != cudaSuccess) {
            free_device_data(out_sample_coords, out_rays, out_ray_indices, out_numsteps, out_ray_counter, out_sample_counter, out_occupancy);
            return cuda_error("cudaMemset sampler occupancy", status);
        }

        return {};
    }

    // Network buffers.
    std::string allocate_network_once(const std::uint32_t batch_size, const std::uint32_t max_samples, std::uint16_t*& out_density_input, std::uint16_t*& out_rgb_input, std::uint16_t*& out_network_output, std::uint16_t*& out_network_output_gradients, std::uint16_t*& out_rgb_output_gradients, std::uint16_t*& out_rgb_input_gradients, std::uint16_t*& out_density_input_gradients, std::uint16_t*& out_density_forward_hidden, std::uint16_t*& out_rgb_forward_hidden,
        std::uint16_t*& out_density_backward_hidden, std::uint16_t*& out_rgb_backward_hidden, std::uint8_t*& out_cutlass_workspace) {
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
        out_cutlass_workspace        = nullptr;

        if (batch_size == 0u) return "network batch size is zero.";
        if (max_samples == 0u) return "network max samples is zero.";
        if (batch_size % (16u * mlp_forward_iters) != 0u) return "network batch size does not match the fully fused MLP tile size.";
        if (max_samples % (16u * mlp_forward_iters) != 0u) return "network max samples does not match the fully fused MLP tile size.";

        if (const cudaError_t status = cudaMalloc(&out_density_input, static_cast<std::size_t>(mlp_input_width) * batch_size * sizeof(__half)); status != cudaSuccess) return cuda_error("cudaMalloc density network input", status);
        if (const cudaError_t status = cudaMalloc(&out_rgb_input, static_cast<std::size_t>(mlp_input_width) * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc rgb network input", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_network_output, static_cast<std::size_t>(mlp_output_width) * max_samples * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc network output", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_network_output_gradients, static_cast<std::size_t>(mlp_output_width) * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc network output gradients", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_rgb_output_gradients, static_cast<std::size_t>(mlp_output_width) * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc rgb output gradients", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_rgb_input_gradients, static_cast<std::size_t>(mlp_input_width) * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc rgb input gradients", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_density_input_gradients, static_cast<std::size_t>(mlp_input_width) * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc density input gradients", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_density_forward_hidden, static_cast<std::size_t>(density_hidden_layers) * mlp_width * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc density forward hidden", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_rgb_forward_hidden, static_cast<std::size_t>(rgb_hidden_layers) * mlp_width * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc rgb forward hidden", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_density_backward_hidden, static_cast<std::size_t>(density_hidden_layers) * mlp_width * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc density backward hidden", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_rgb_backward_hidden, static_cast<std::size_t>(rgb_hidden_layers) * mlp_width * batch_size * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
            return cuda_error("cudaMalloc rgb backward hidden", status);
        }

        const std::size_t workspace_bytes = cutlass_workspace_size(batch_size);
        if (workspace_bytes > 0u) {
            if (const cudaError_t status = cudaMalloc(&out_cutlass_workspace, workspace_bytes); status != cudaSuccess) {
                free_device_data(out_density_input, out_rgb_input, out_network_output, out_network_output_gradients, out_rgb_output_gradients, out_rgb_input_gradients, out_density_input_gradients, out_density_forward_hidden, out_rgb_forward_hidden, out_density_backward_hidden, out_rgb_backward_hidden, out_cutlass_workspace);
                return cuda_error("cudaMalloc cutlass workspace", status);
            }
        }

        return {};
    }

    // Loss and compaction buffers.
    std::string allocate_training_loss_once(const std::uint32_t batch_size, const std::uint32_t rays_per_batch, std::uint32_t*& out_compacted_sample_counter, float*& out_compacted_sample_coords, float*& out_loss_values) {
        out_compacted_sample_counter = nullptr;
        out_compacted_sample_coords  = nullptr;
        out_loss_values              = nullptr;

        if (batch_size == 0u) return "training batch size is zero.";
        if (rays_per_batch == 0u) return "training rays per batch is zero.";

        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_counter, sizeof(std::uint32_t)); status != cudaSuccess) return cuda_error("cudaMalloc compacted sample counter", status);
        if (const cudaError_t status = cudaMalloc(&out_compacted_sample_coords, static_cast<std::size_t>(batch_size) * sample_coord_floats * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_compacted_sample_counter, out_compacted_sample_coords, out_loss_values);
            return cuda_error("cudaMalloc compacted sample coords", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_loss_values, static_cast<std::size_t>(rays_per_batch) * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_compacted_sample_counter, out_compacted_sample_coords, out_loss_values);
            return cuda_error("cudaMalloc loss values", status);
        }

        return {};
    }

    // Trainable parameters.
    std::string allocate_trainable_params_once(const std::uint32_t param_count, float*& out_params_full_precision, std::uint16_t*& out_params, std::uint16_t*& out_param_gradients) {
        out_params_full_precision = nullptr;
        out_params                = nullptr;
        out_param_gradients       = nullptr;

        if (param_count == 0u) return {};

        if (const cudaError_t status = cudaMalloc(&out_params_full_precision, static_cast<std::size_t>(param_count) * sizeof(float)); status != cudaSuccess) return cuda_error("cudaMalloc trainable params full precision", status);
        if (const cudaError_t status = cudaMalloc(&out_params, static_cast<std::size_t>(param_count) * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_params_full_precision, out_params, out_param_gradients);
            return cuda_error("cudaMalloc trainable params", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_param_gradients, static_cast<std::size_t>(param_count) * sizeof(__half)); status != cudaSuccess) {
            free_device_data(out_params_full_precision, out_params, out_param_gradients);
            return cuda_error("cudaMalloc trainable param gradients", status);
        }

        return {};
    }

    // Parameter initialization.
    std::string initialize_mlp_params_once(const std::uint64_t seed, const std::uint32_t density_input_width, const std::uint32_t density_output_width, const std::uint32_t density_layers, const std::uint32_t density_param_offset, const std::uint32_t rgb_input_width, const std::uint32_t rgb_output_width, const std::uint32_t rgb_layers, const std::uint32_t rgb_param_offset, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {
        if (params_full_precision == nullptr) return "mlp full precision params are null.";
        if (params == nullptr) return "mlp params are null.";
        if (param_gradients == nullptr) return "mlp param gradients are null.";
        if (density_input_width != mlp_input_width || rgb_input_width != mlp_input_width) return "mlp input width does not match the compiled fully fused MLP.";
        if (density_output_width != mlp_output_width || rgb_output_width != mlp_output_width) return "mlp output width does not match the compiled fully fused MLP.";
        if (density_layers != density_hidden_layers || rgb_layers != rgb_hidden_layers) return "mlp hidden layer count does not match the compiled fully fused MLP.";

        const std::uint32_t mlp_param_count = ::cuda::std::max(density_param_offset + density_network_params, rgb_param_offset + rgb_network_params);
        std::vector host_params(mlp_param_count, 0.0f);
        Pcg32 rng{seed};

        auto initialize_matrix = [&](const std::uint32_t offset, const std::uint32_t rows, const std::uint32_t cols) {
            const float scale = std::sqrt(6.0f / static_cast<float>(rows + cols));
            for (std::uint32_t i = 0u; i < rows * cols; ++i) host_params[offset + i] = rng.next_float() * 2.0f * scale - scale;
        };

        initialize_matrix(density_param_offset, mlp_width, mlp_input_width);
        initialize_matrix(density_param_offset + mlp_first_layer_params, mlp_output_width, mlp_width);
        initialize_matrix(rgb_param_offset, mlp_width, mlp_input_width);
        initialize_matrix(rgb_param_offset + mlp_first_layer_params, mlp_width, mlp_width);
        initialize_matrix(rgb_param_offset + mlp_first_layer_params + mlp_hidden_layer_params, mlp_output_width, mlp_width);

        if (const cudaError_t status = cudaMemcpy(params_full_precision, host_params.data(), host_params.size() * sizeof(float), cudaMemcpyHostToDevice); status != cudaSuccess) return cuda_error("cudaMemcpy mlp full precision params", status);
        if (const cudaError_t status = cudaMemset(param_gradients, 0, host_params.size() * sizeof(__half)); status != cudaSuccess) return cuda_error("cudaMemset mlp gradients", status);

        const std::uint32_t blocks = (mlp_param_count + threads_per_block - 1u) / threads_per_block;
        cast_params_to_half_kernel<<<blocks, threads_per_block>>>(mlp_param_count, params_full_precision, reinterpret_cast<__half*>(params));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("cast_params_to_half_kernel", status);
        return {};
    }

    std::string initialize_grid_params_once(const std::uint32_t param_count, const std::uint64_t seed, const std::uint64_t rng_offset, float* const params_full_precision, std::uint16_t* const params, std::uint16_t* const param_gradients) {
        if (param_count == 0u) return {};
        if (params_full_precision == nullptr) return "grid full precision params are null.";
        if (params == nullptr) return "grid params are null.";
        if (param_gradients == nullptr) return "grid param gradients are null.";

        const std::uint32_t n_threads = (param_count + random_values_per_thread - 1u) / random_values_per_thread;
        const std::uint32_t blocks    = (n_threads + threads_per_block - 1u) / threads_per_block;
        initialize_grid_params_kernel<<<blocks, threads_per_block>>>(param_count, seed, rng_offset, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<__half*>(param_gradients));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("initialize_grid_params_kernel", status);
        return {};
    }

    // Encoding execution.
    std::string encode_grid_forward(const std::uint32_t sample_count, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint32_t grid_levels, const std::uint32_t features_per_level, const std::uint32_t base_resolution, const float per_level_scale, const std::uint16_t* const grid_params, std::uint16_t* const encoded_positions) {
        if (sample_count == 0u) return {};
        if (sample_coords == nullptr) return "grid sample coords are null.";
        if (grid_offsets == nullptr) return "grid offsets are null.";
        if (grid_levels != grid_n_levels) return "grid level count does not match the compiled grid encoder.";
        if (features_per_level != grid_features_per_level) return "grid features per level does not match the compiled grid encoder.";
        if (base_resolution == 0u) return "grid base resolution is zero.";
        if (per_level_scale <= 0.0f) return "grid per level scale must be positive.";
        if (grid_params == nullptr) return "grid params are null.";
        if (encoded_positions == nullptr) return "grid encoded positions are null.";

        GridOffsetTable offset_table = {};
        for (std::uint32_t i = 0u; i <= grid_n_levels; ++i) offset_table.data[i] = grid_offsets[i];

        const dim3 blocks{(sample_count + grid_forward_threads - 1u) / grid_forward_threads, grid_n_levels, 1u};
        encode_grid_forward_kernel<<<blocks, grid_forward_threads>>>(sample_count, offset_table, base_resolution, log2f(per_level_scale), sample_coords, reinterpret_cast<const __half*>(grid_params), reinterpret_cast<__half*>(encoded_positions));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("encode_grid_forward_kernel", status);
        return {};
    }

    std::string encode_grid_backward(const std::uint32_t sample_count, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint32_t grid_levels, const std::uint32_t features_per_level, const std::uint32_t base_resolution, const float per_level_scale, const std::uint16_t* const encoded_position_gradients, std::uint16_t* const grid_param_gradients) {
        if (sample_count == 0u) return {};
        if (sample_coords == nullptr) return "grid sample coords are null.";
        if (grid_offsets == nullptr) return "grid offsets are null.";
        if (grid_levels != grid_n_levels) return "grid level count does not match the compiled grid encoder.";
        if (features_per_level != grid_features_per_level) return "grid features per level does not match the compiled grid encoder.";
        if (base_resolution == 0u) return "grid base resolution is zero.";
        if (per_level_scale <= 0.0f) return "grid per level scale must be positive.";
        if (encoded_position_gradients == nullptr) return "grid encoded position gradients are null.";
        if (grid_param_gradients == nullptr) return "grid param gradients are null.";

        GridOffsetTable offset_table = {};
        for (std::uint32_t i = 0u; i <= grid_n_levels; ++i) offset_table.data[i] = grid_offsets[i];

        if (const cudaError_t status = cudaMemset(grid_param_gradients, 0, static_cast<std::size_t>(offset_table.data[grid_n_levels]) * grid_features_per_level * sizeof(__half)); status != cudaSuccess) return cuda_error("cudaMemset grid param gradients", status);

        const std::uint32_t threads = (sample_count * grid_features_per_level / grid_backward_features + grid_backward_threads - 1u) / grid_backward_threads;
        const dim3 blocks{threads, grid_n_levels, 1u};
        encode_grid_backward_kernel<<<blocks, grid_backward_threads>>>(sample_count, offset_table, base_resolution, log2f(per_level_scale), sample_coords, reinterpret_cast<const __half*>(encoded_position_gradients), reinterpret_cast<__half*>(grid_param_gradients));

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("encode_grid_backward_kernel", status);
        return {};
    }

    // Network execution.
    std::string network_inference_once(const std::uint32_t sample_count, const std::uint32_t batch_size, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint32_t grid_levels, const std::uint32_t features_per_level, const std::uint32_t base_resolution, const float per_level_scale, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset,
        std::uint16_t* const density_input, std::uint16_t* const rgb_input, std::uint16_t* const network_output) {
        if (sample_count == 0u) return {};
        if (batch_size == 0u) return "network inference batch size is zero.";
        if (sample_count % (16u * mlp_forward_iters) != 0u) return "network inference sample count does not match the fully fused MLP tile size.";
        if (batch_size % (16u * mlp_forward_iters) != 0u) return "network inference batch size does not match the fully fused MLP tile size.";
        if (sample_coords == nullptr) return "network inference sample coords are null.";
        if (params == nullptr) return "network inference params are null.";
        if (density_input == nullptr) return "network inference density input is null.";
        if (rgb_input == nullptr) return "network inference rgb input is null.";
        if (network_output == nullptr) return "network inference output is null.";

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * mlp_forward_iters) * (mlp_width + mlp_skew);
        constexpr dim3 threads{32u, mlp_width_blocks, 1u};
        if (const cudaError_t status = cudaFuncSetAttribute(mlp_forward_64_relu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, forward_shmem); status != cudaSuccess) return cuda_error("cudaFuncSetAttribute mlp_forward_64_relu_kernel", status);

        for (std::uint32_t offset = 0u; offset < sample_count; offset += batch_size) {
            const std::uint32_t chunk = ::cuda::std::min(batch_size, sample_count - offset);
            if (chunk % (16u * mlp_forward_iters) != 0u) return "network inference chunk size does not match the fully fused MLP tile size.";

            if (const std::string error = encode_grid_forward(chunk, sample_coords + static_cast<std::uint64_t>(offset) * sample_coord_floats, grid_offsets, grid_levels, features_per_level, base_resolution, per_level_scale, params + grid_param_offset, density_input); !error.empty()) return error;

            const std::uint32_t linear_blocks = (chunk + threads_per_block - 1u) / threads_per_block;
            encode_spherical_harmonics_kernel<<<linear_blocks, threads_per_block>>>(chunk, sample_coords + static_cast<std::uint64_t>(offset) * sample_coord_floats, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(mlp_output_width) * chunk);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("encode_spherical_harmonics_kernel", status);

            const dim3 blocks{chunk / (16u * mlp_forward_iters), 1u, 1u};
            mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(chunk, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + density_param_offset), nullptr, reinterpret_cast<__half*>(rgb_input), false, density_hidden_layers);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("density mlp inference", status);

            mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + rgb_param_offset), nullptr, reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * mlp_output_width, true, rgb_hidden_layers);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("rgb mlp inference", status);

            extract_density_kernel<<<linear_blocks, threads_per_block>>>(chunk, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output) + static_cast<std::uint64_t>(offset) * mlp_output_width);
            if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("extract_density_kernel inference", status);
        }

        return {};
    }

    std::string network_forward_once(const std::uint32_t batch_size, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint32_t grid_levels, const std::uint32_t features_per_level, const std::uint32_t base_resolution, const float per_level_scale, const std::uint16_t* const params, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset, std::uint16_t* const density_input,
        std::uint16_t* const rgb_input, std::uint16_t* const density_forward_hidden, std::uint16_t* const rgb_forward_hidden, std::uint16_t* const network_output) {
        if (batch_size == 0u) return {};
        if (batch_size % (16u * mlp_forward_iters) != 0u) return "network batch size does not match the fully fused MLP tile size.";
        if (sample_coords == nullptr) return "network sample coords are null.";
        if (params == nullptr) return "network params are null.";
        if (density_input == nullptr) return "density network input is null.";
        if (rgb_input == nullptr) return "rgb network input is null.";
        if (density_forward_hidden == nullptr) return "density forward hidden is null.";
        if (rgb_forward_hidden == nullptr) return "rgb forward hidden is null.";
        if (network_output == nullptr) return "network output is null.";

        if (const std::string error = encode_grid_forward(batch_size, sample_coords, grid_offsets, grid_levels, features_per_level, base_resolution, per_level_scale, params + grid_param_offset, density_input); !error.empty()) return error;

        const std::uint32_t linear_blocks = (batch_size + threads_per_block - 1u) / threads_per_block;
        encode_spherical_harmonics_kernel<<<linear_blocks, threads_per_block>>>(batch_size, sample_coords, reinterpret_cast<__half*>(rgb_input) + static_cast<std::uint64_t>(mlp_output_width) * batch_size);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("encode_spherical_harmonics_kernel", status);

        constexpr int forward_shmem = sizeof(__half) * (16u + 16u * mlp_forward_iters) * (mlp_width + mlp_skew);
        constexpr dim3 threads{32u, mlp_width_blocks, 1u};
        const dim3 blocks{batch_size / (16u * mlp_forward_iters), 1u, 1u};

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_forward_64_relu_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, forward_shmem); status != cudaSuccess) return cuda_error("cudaFuncSetAttribute mlp_forward_64_relu_kernel", status);
        mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(batch_size, reinterpret_cast<const __half*>(density_input), reinterpret_cast<const __half*>(params + density_param_offset), reinterpret_cast<__half*>(density_forward_hidden), reinterpret_cast<__half*>(rgb_input), false, density_hidden_layers);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("density mlp forward", status);

        mlp_forward_64_relu_kernel<<<blocks, threads, forward_shmem>>>(batch_size, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<const __half*>(params + rgb_param_offset), reinterpret_cast<__half*>(rgb_forward_hidden), reinterpret_cast<__half*>(network_output), true, rgb_hidden_layers);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("rgb mlp forward", status);

        extract_density_kernel<<<linear_blocks, threads_per_block>>>(batch_size, reinterpret_cast<const __half*>(rgb_input), reinterpret_cast<__half*>(network_output));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("extract_density_kernel", status);
        return {};
    }

    std::string network_backward_once(const std::uint32_t batch_size, const float* const sample_coords, const std::uint32_t* const grid_offsets, const std::uint32_t grid_levels, const std::uint32_t features_per_level, const std::uint32_t base_resolution, const float per_level_scale, const std::uint16_t* const params, std::uint16_t* const gradients, const std::uint32_t density_param_offset, const std::uint32_t rgb_param_offset, const std::uint32_t grid_param_offset,
        const std::uint16_t* const density_input, const std::uint16_t* const rgb_input, const std::uint16_t* const density_forward_hidden, const std::uint16_t* const rgb_forward_hidden, const std::uint16_t* const network_output, const std::uint16_t* const network_output_gradients, std::uint16_t* const rgb_output_gradients, std::uint16_t* const rgb_input_gradients, std::uint16_t* const density_input_gradients, std::uint16_t* const density_backward_hidden,
        std::uint16_t* const rgb_backward_hidden, std::uint8_t* const cutlass_workspace) {
        if (batch_size == 0u) return {};
        if (batch_size % (16u * mlp_forward_iters) != 0u) return "network batch size does not match the fully fused MLP tile size.";
        if (sample_coords == nullptr) return "network sample coords are null.";
        if (params == nullptr) return "network params are null.";
        if (gradients == nullptr) return "network gradients are null.";
        if (density_input == nullptr) return "density input is null.";
        if (rgb_input == nullptr) return "rgb input is null.";
        if (density_forward_hidden == nullptr) return "density forward hidden is null.";
        if (rgb_forward_hidden == nullptr) return "rgb forward hidden is null.";
        if (network_output == nullptr) return "network output is null.";
        if (network_output_gradients == nullptr) return "network output gradients are null.";
        if (rgb_output_gradients == nullptr) return "rgb output gradients are null.";
        if (rgb_input_gradients == nullptr) return "rgb input gradients are null.";
        if (density_input_gradients == nullptr) return "density input gradients are null.";
        if (density_backward_hidden == nullptr) return "density backward hidden is null.";
        if (rgb_backward_hidden == nullptr) return "rgb backward hidden is null.";
        if (cutlass_workspace == nullptr && cutlass_workspace_size(batch_size) > 0u) return "cutlass workspace is null.";

        const std::uint32_t linear_blocks       = (batch_size + threads_per_block - 1u) / threads_per_block;
        const std::uint64_t hidden_layer_stride = static_cast<std::uint64_t>(mlp_width) * batch_size;
        const int batch                         = static_cast<int>(batch_size);
        const int split_k                       = static_cast<int>(batch_size / ::cuda::std::min(1u << 12u, batch_size));
        void* const workspace                   = cutlass_workspace;

        extract_rgb_gradients_kernel<<<linear_blocks, threads_per_block>>>(batch_size, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_output_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("extract_rgb_gradients_kernel", status);

        constexpr int backward_shmem = sizeof(__half) * (16u * mlp_forward_iters) * (mlp_width + mlp_skew);
        constexpr dim3 threads{32u, mlp_width_blocks, 1u};
        const dim3 blocks{batch_size / (16u * mlp_forward_iters), 1u, 1u};

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::row_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, backward_shmem); status != cudaSuccess) return cuda_error("cudaFuncSetAttribute rgb mlp backward", status);
        mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::row_major><<<blocks, threads, backward_shmem>>>(batch_size, reinterpret_cast<const __half*>(rgb_output_gradients), reinterpret_cast<const __half*>(params + rgb_param_offset), reinterpret_cast<const __half*>(rgb_forward_hidden), reinterpret_cast<__half*>(rgb_backward_hidden), mlp_output_width, rgb_hidden_layers);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("rgb mlp backward hidden", status);

        if (std::string error = run_cutlass_split_k<CutlassLastLayerK, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(
                mlp_output_width, mlp_width, batch, reinterpret_cast<const __half*>(rgb_output_gradients), mlp_output_width, reinterpret_cast<const __half*>(rgb_forward_hidden) + (rgb_hidden_layers - 1u) * hidden_layer_stride, mlp_width, reinterpret_cast<__half*>(gradients + rgb_param_offset + mlp_first_layer_params + (rgb_hidden_layers - 1u) * mlp_hidden_layer_params), mlp_width, workspace, split_k);
            !error.empty())
            return error;
        if (std::string error = run_cutlass_split_k<CutlassFullLayerK, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(mlp_width, mlp_width, batch, reinterpret_cast<const __half*>(rgb_backward_hidden), mlp_width, reinterpret_cast<const __half*>(rgb_forward_hidden), mlp_width, reinterpret_cast<__half*>(gradients + rgb_param_offset + mlp_first_layer_params), mlp_width, workspace, split_k); !error.empty()) return error;
        if (std::string error = run_cutlass_split_k<CutlassFullLayerK, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(mlp_width, mlp_input_width, batch, reinterpret_cast<const __half*>(rgb_backward_hidden) + (rgb_hidden_layers - 1u) * hidden_layer_stride, mlp_width, reinterpret_cast<const __half*>(rgb_input), batch, reinterpret_cast<__half*>(gradients + rgb_param_offset), mlp_input_width, workspace, split_k); !error.empty())
            return error;
        if (std::string error = run_cutlass_gemm<CutlassFullLayer, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(mlp_input_width, batch, mlp_width, reinterpret_cast<const __half*>(params + rgb_param_offset), mlp_input_width, reinterpret_cast<const __half*>(rgb_backward_hidden) + (rgb_hidden_layers - 1u) * hidden_layer_stride, mlp_width, reinterpret_cast<__half*>(rgb_input_gradients), batch, workspace); !error.empty()) return error;

        add_density_gradient_kernel<<<linear_blocks, threads_per_block>>>(batch_size, reinterpret_cast<const __half*>(network_output_gradients), reinterpret_cast<__half*>(rgb_input_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("add_density_gradient_kernel", status);

        if (const cudaError_t status = cudaFuncSetAttribute(mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::col_major>, cudaFuncAttributeMaxDynamicSharedMemorySize, backward_shmem); status != cudaSuccess) return cuda_error("cudaFuncSetAttribute density mlp backward", status);
        mlp_backward_hidden_64_relu_kernel<nvcuda::wmma::col_major><<<blocks, threads, backward_shmem>>>(batch_size, reinterpret_cast<const __half*>(rgb_input_gradients), reinterpret_cast<const __half*>(params + density_param_offset), reinterpret_cast<const __half*>(density_forward_hidden), reinterpret_cast<__half*>(density_backward_hidden), batch_size, density_hidden_layers);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("density mlp backward hidden", status);

        if (std::string error = run_cutlass_split_k<CutlassLastLayerK, cutlass::layout::RowMajor, cutlass::layout::RowMajor, cutlass::layout::RowMajor>(mlp_output_width, mlp_width, batch, reinterpret_cast<const __half*>(rgb_input_gradients), batch, reinterpret_cast<const __half*>(density_forward_hidden), mlp_width, reinterpret_cast<__half*>(gradients + density_param_offset + mlp_first_layer_params), mlp_width, workspace, split_k); !error.empty()) return error;
        if (std::string error = run_cutlass_split_k<CutlassFullLayerK, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(mlp_width, mlp_input_width, batch, reinterpret_cast<const __half*>(density_backward_hidden), mlp_width, reinterpret_cast<const __half*>(density_input), batch, reinterpret_cast<__half*>(gradients + density_param_offset), mlp_input_width, workspace, split_k); !error.empty()) return error;
        if (std::string error = run_cutlass_gemm<CutlassFullLayer, cutlass::layout::ColumnMajor, cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>(mlp_input_width, batch, mlp_width, reinterpret_cast<const __half*>(params + density_param_offset), mlp_input_width, reinterpret_cast<const __half*>(density_backward_hidden), mlp_width, reinterpret_cast<__half*>(density_input_gradients), batch, workspace); !error.empty()) return error;

        return encode_grid_backward(batch_size, sample_coords, grid_offsets, grid_levels, features_per_level, base_resolution, per_level_scale, density_input_gradients, gradients + grid_param_offset);
    }

    // Optimizer.
    std::string allocate_adam_state_once(const std::uint32_t param_count, float*& out_first_moments, float*& out_second_moments, std::uint32_t*& out_param_steps) {
        out_first_moments  = nullptr;
        out_second_moments = nullptr;
        out_param_steps    = nullptr;

        if (param_count == 0u) return "optimizer param count is zero.";

        if (const cudaError_t status = cudaMalloc(&out_first_moments, param_count * sizeof(float)); status != cudaSuccess) return cuda_error("cudaMalloc optimizer first moments", status);
        if (const cudaError_t status = cudaMalloc(&out_second_moments, param_count * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_first_moments, out_second_moments, out_param_steps);
            return cuda_error("cudaMalloc optimizer second moments", status);
        }
        if (const cudaError_t status = cudaMalloc(&out_param_steps, param_count * sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_first_moments, out_second_moments, out_param_steps);
            return cuda_error("cudaMalloc optimizer param steps", status);
        }

        if (const cudaError_t status = cudaMemset(out_first_moments, 0, param_count * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_first_moments, out_second_moments, out_param_steps);
            return cuda_error("cudaMemset optimizer first moments", status);
        }
        if (const cudaError_t status = cudaMemset(out_second_moments, 0, param_count * sizeof(float)); status != cudaSuccess) {
            free_device_data(out_first_moments, out_second_moments, out_param_steps);
            return cuda_error("cudaMemset optimizer second moments", status);
        }
        if (const cudaError_t status = cudaMemset(out_param_steps, 0, param_count * sizeof(std::uint32_t)); status != cudaSuccess) {
            free_device_data(out_first_moments, out_second_moments, out_param_steps);
            return cuda_error("cudaMemset optimizer param steps", status);
        }

        return {};
    }

    // Training step sampler.
    std::string sample_training_batch(const float* const camera, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const float focal_length, const std::uint32_t current_step, const std::uint64_t seed, const std::uint32_t rays_per_batch, const std::uint32_t max_samples, const bool snap_to_pixel_centers, const std::uint8_t* const occupancy, float* const sample_coords, float* const rays, std::uint32_t* const ray_indices,
        std::uint32_t* const numsteps, std::uint32_t* const ray_counter, std::uint32_t* const sample_counter) {
        if (camera == nullptr) return "sampler camera is null.";
        if (frame_count == 0u) return "sampler frame count is zero.";
        if (width == 0u || height == 0u) return "sampler resolution is zero.";
        if (focal_length <= 0.0f) return "sampler focal length must be positive.";
        if (rays_per_batch == 0u) return {};
        if (max_samples == 0u) return "sampler max samples is zero.";
        if (occupancy == nullptr) return "sampler occupancy is null.";
        if (sample_coords == nullptr) return "sampler sample coords are null.";
        if (rays == nullptr) return "sampler rays are null.";
        if (ray_indices == nullptr) return "sampler ray indices are null.";
        if (numsteps == nullptr) return "sampler numsteps are null.";
        if (ray_counter == nullptr) return "sampler ray counter is null.";
        if (sample_counter == nullptr) return "sampler sample counter is null.";

        if (const cudaError_t status = cudaMemset(ray_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) return cuda_error("cudaMemset sampler ray counter", status);
        if (const cudaError_t status = cudaMemset(sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) return cuda_error("cudaMemset sampler sample counter", status);

        const std::uint32_t blocks = (rays_per_batch + threads_per_block - 1u) / threads_per_block;
        generate_training_samples<<<blocks, threads_per_block>>>(rays_per_batch, max_samples, seed, current_step, frame_count, width, height, focal_length, snap_to_pixel_centers, camera, occupancy, ray_counter, sample_counter, ray_indices, rays, numsteps, sample_coords);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("generate_training_samples", status);
        return {};
    }

    // Loss and compaction execution.
    std::string compute_loss_and_compact_once(const std::uint32_t rays_per_batch, const std::uint32_t batch_size, const std::uint64_t seed, const std::uint32_t current_step, const std::uint32_t* const ray_counter, const std::uint8_t* const pixels, const std::uint32_t frame_count, const std::uint32_t width, const std::uint32_t height, const bool snap_to_pixel_centers, const std::uint16_t* const network_output, std::uint32_t* const compacted_sample_counter,
        const std::uint32_t* const ray_indices, const float* const rays, std::uint32_t* const numsteps, const float* const sample_coords, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients, float* const loss_values) {
        if (rays_per_batch == 0u) return {};
        if (batch_size == 0u) return "loss batch size is zero.";
        if (ray_counter == nullptr) return "loss ray counter is null.";
        if (pixels == nullptr) return "loss pixels are null.";
        if (frame_count == 0u) return "loss frame count is zero.";
        if (width == 0u || height == 0u) return "loss resolution is zero.";
        if (network_output == nullptr) return "loss network output is null.";
        if (compacted_sample_counter == nullptr) return "loss compacted sample counter is null.";
        if (ray_indices == nullptr) return "loss ray indices are null.";
        if (rays == nullptr) return "loss rays are null.";
        if (numsteps == nullptr) return "loss numsteps are null.";
        if (sample_coords == nullptr) return "loss sample coords are null.";
        if (compacted_sample_coords == nullptr) return "loss compacted sample coords are null.";
        if (network_output_gradients == nullptr) return "loss network output gradients are null.";

        if (const cudaError_t status = cudaMemset(compacted_sample_counter, 0, sizeof(std::uint32_t)); status != cudaSuccess) return cuda_error("cudaMemset compacted sample counter", status);
        if (loss_values != nullptr)
            if (const cudaError_t status = cudaMemset(loss_values, 0, static_cast<std::size_t>(rays_per_batch) * sizeof(float)); status != cudaSuccess) return cuda_error("cudaMemset loss values", status);

        const std::uint32_t blocks = (rays_per_batch + threads_per_block - 1u) / threads_per_block;
        compute_loss_and_compact_kernel<<<blocks, threads_per_block>>>(rays_per_batch, batch_size, seed, current_step, ray_counter, pixels, frame_count, width, height, snap_to_pixel_centers, reinterpret_cast<const __half*>(network_output), compacted_sample_counter, ray_indices, rays, numsteps, sample_coords, compacted_sample_coords, reinterpret_cast<__half*>(network_output_gradients), loss_values);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("compute_loss_and_compact_kernel", status);
        return {};
    }

    std::string fill_rollover_once(const std::uint32_t batch_size, const std::uint32_t* const compacted_sample_counter, float* const compacted_sample_coords, std::uint16_t* const network_output_gradients) {
        if (batch_size == 0u) return {};
        if (compacted_sample_counter == nullptr) return "rollover compacted sample counter is null.";
        if (compacted_sample_coords == nullptr) return "rollover compacted sample coords are null.";
        if (network_output_gradients == nullptr) return "rollover network output gradients are null.";

        const std::uint32_t gradient_elements = batch_size * mlp_output_width;
        fill_rollover_and_rescale_half_kernel<<<(gradient_elements + threads_per_block - 1u) / threads_per_block, threads_per_block>>>(batch_size, mlp_output_width, compacted_sample_counter, reinterpret_cast<__half*>(network_output_gradients));
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("fill_rollover_and_rescale_half_kernel", status);

        const std::uint32_t coord_elements = batch_size * sample_coord_floats;
        fill_rollover_kernel<float><<<(coord_elements + threads_per_block - 1u) / threads_per_block, threads_per_block>>>(batch_size, sample_coord_floats, compacted_sample_counter, compacted_sample_coords);
        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("fill_rollover_kernel compacted coords", status);
        return {};
    }

    // Host readback.
    std::string read_counter_once(const std::uint32_t* const counter, std::uint32_t& out_value) {
        if (counter == nullptr) return "counter is null.";
        if (const cudaError_t status = cudaMemcpy(&out_value, counter, sizeof(std::uint32_t), cudaMemcpyDeviceToHost); status != cudaSuccess) return cuda_error("cudaMemcpy counter", status);
        return {};
    }

    std::string read_loss_sum_once(const float* const loss_values, const std::uint32_t loss_count, float& out_loss_sum) {
        out_loss_sum = 0.0f;
        if (loss_count == 0u) return {};
        if (loss_values == nullptr) return "loss values are null.";

        std::vector<float> host_loss(loss_count);
        if (const cudaError_t status = cudaMemcpy(host_loss.data(), loss_values, static_cast<std::size_t>(loss_count) * sizeof(float), cudaMemcpyDeviceToHost); status != cudaSuccess) return cuda_error("cudaMemcpy loss values", status);
        for (const float loss : host_loss) out_loss_sum += loss;
        return {};
    }

    // Optimizer execution.
    std::string optimize(const std::uint32_t param_count, const std::uint32_t mlp_param_count, const float loss_scale, const float learning_rate, const float beta1, const float beta2, const float epsilon, const float l2_reg, float* const params_full_precision, std::uint16_t* const params, const std::uint16_t* const gradients, float* const first_moments, float* const second_moments, std::uint32_t* const param_steps) {
        if (param_count == 0u) return {};
        if (mlp_param_count > param_count) return "optimizer mlp param count exceeds param count.";
        if (loss_scale == 0.0f) return "optimizer loss scale is zero.";
        if (params_full_precision == nullptr) return "optimizer full precision params are null.";
        if (params == nullptr) return "optimizer params are null.";
        if (gradients == nullptr) return "optimizer gradients are null.";
        if (first_moments == nullptr) return "optimizer first moments are null.";
        if (second_moments == nullptr) return "optimizer second moments are null.";
        if (param_steps == nullptr) return "optimizer param steps are null.";

        const std::uint32_t blocks = (param_count + threads_per_block - 1u) / threads_per_block;
        adam_step<<<blocks, threads_per_block>>>(param_count, mlp_param_count, loss_scale, learning_rate, beta1, beta2, epsilon, l2_reg, params_full_precision, reinterpret_cast<__half*>(params), reinterpret_cast<const __half*>(gradients), first_moments, second_moments, param_steps);

        if (const cudaError_t status = cudaGetLastError(); status != cudaSuccess) return cuda_error("adam_step", status);
        return {};
    }
} // namespace ngp::cuda
