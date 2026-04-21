#include "network.cuh"
#include "stb/stb_image_write.h"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <sstream>
#include <type_traits>

namespace ngp {

    void delete_trainer_state(network::TrainerState<__half>* trainer) {
        delete trainer;
    }

    struct TrainingStepWorkspace final {
        legacy::GpuAllocation alloc                             = {};
        std::uint32_t* ray_indices                              = nullptr;
        void* rays_unnormalized                                 = nullptr;
        std::uint32_t* numsteps                                 = nullptr;
        float* coords                                           = nullptr;
        __half* mlp_out                                         = nullptr;
        __half* dloss_dmlp_out                                  = nullptr;
        float* coords_compacted                                 = nullptr;
        std::uint32_t* ray_counter                              = nullptr;
        std::uint32_t max_samples                               = 0u;
        std::uint32_t max_inference                             = 0u;
        std::uint32_t floats_per_coord                          = 0u;
        std::uint32_t padded_output_width                       = 0u;
        std::uint32_t n_rays_total                              = 0u;
        legacy::GPUMatrixDynamic<float> coords_matrix           = {};
        legacy::GPUMatrixDynamic<__half> rgbsigma_matrix        = {};
        legacy::GPUMatrixDynamic<float> compacted_coords_matrix = {};
        legacy::GPUMatrixDynamic<__half> gradient_matrix        = {};
        legacy::GPUMatrixDynamic<__half> compacted_output       = {};
    };

    struct Ray final {
        legacy::math::vec3 o = {};
        legacy::math::vec3 d = {};

        __host__ __device__ legacy::math::vec3 operator()(const float t) const {
            return o + t * d;
        }

        __host__ __device__ void advance(const float t) {
            o += d * t;
        }

        __host__ __device__ bool is_valid() const {
            return d != legacy::math::vec3(0.0f);
        }
    };

    struct ValidationRenderWorkspace final {
        std::uint32_t total_pixels                        = 0u;
        std::uint32_t padded_output_width                 = 0u;
        std::uint32_t floats_per_coord                    = 0u;
        std::uint32_t max_samples                         = 0u;
        legacy::GpuBuffer<legacy::math::vec3> rendered    = {};
        legacy::GpuBuffer<std::uint32_t> tile_numsteps    = {};
        legacy::GpuBuffer<float> tile_coords              = {};
        legacy::GpuBuffer<__half> tile_mlp_out            = {};
        legacy::GpuBuffer<std::uint32_t> sample_counter   = {};
        legacy::GpuBuffer<std::uint32_t> overflow_counter = {};
    };

    inline __host__ __device__ Ray uv_to_ray(const std::uint32_t spp, const legacy::math::vec2& uv, const legacy::math::ivec2& resolution, const float focal_length, const legacy::math::mat4x3& camera_matrix, const float near_distance = 0.0f) {
        (void) spp;
        legacy::math::vec3 dir    = {(uv.x - 0.5f) * (float) resolution.x / focal_length, (uv.y - 0.5f) * (float) resolution.y / focal_length, 1.0f};
        dir                       = legacy::math::mat3(camera_matrix) * dir;
        legacy::math::vec3 origin = camera_matrix[3];
        origin += dir * near_distance;
        return {origin, dir};
    }

    inline __host__ __device__ legacy::math::vec2 pos_to_uv(const legacy::math::vec3& pos, const legacy::math::ivec2& resolution, const float focal_length, const legacy::math::mat4x3& camera_matrix) {
        legacy::math::vec3 dir = ngp::legacy::math::inverse(legacy::math::mat3(camera_matrix)) * (pos - camera_matrix[3]);
        dir /= dir.z;
        return dir.xy() * focal_length / legacy::math::vec2(resolution) + legacy::math::vec2(0.5f);
    }

    inline __host__ __device__ float network_to_density(const float val) {
        return expf(val);
    }

    constexpr __host__ __device__ float SQRT3() {
        return 1.73205080757f;
    }

    constexpr __host__ __device__ std::uint32_t NERF_STEPS() {
        return 1024u;
    }

    constexpr __host__ __device__ float STEPSIZE() {
        return SQRT3() / (float) NERF_STEPS();
    }

    constexpr __host__ __device__ float MIN_CONE_STEPSIZE() {
        return STEPSIZE();
    }

    constexpr __host__ __device__ std::uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() {
        return 16u;
    }

    constexpr __host__ __device__ float NERF_MIN_OPTICAL_THICKNESS() {
        return 0.01f;
    }

    struct DensityGridReduceOp final {
        std::uint32_t base_grid_elements = 0u;

        __device__ float operator()(const float val) const {
            return fmaxf(val, 0.0f) / base_grid_elements;
        }
    };

    struct SumIdentityOp final {
        __device__ float operator()(const float val) const {
            return val;
        }
    };

    inline __host__ __device__ legacy::math::vec3 warp_position(const legacy::math::vec3& pos, const legacy::BoundingBox& aabb) {
        return aabb.relative_pos(pos);
    }

    inline __host__ __device__ legacy::math::vec3 warp_direction(const legacy::math::vec3& dir) {
        return (dir + 1.0f) * 0.5f;
    }

    inline __host__ __device__ float warp_dt(const float dt) {
        (void) dt;
        return 0.0f;
    }

    inline __host__ __device__ std::uint32_t density_grid_idx_at(const legacy::math::vec3& pos) {
        const legacy::math::ivec3 i = pos * (float) legacy::NERF_GRIDSIZE();
        if (i.x < 0 || i.x >= (int) legacy::NERF_GRIDSIZE() || i.y < 0 || i.y >= (int) legacy::NERF_GRIDSIZE() || i.z < 0 || i.z >= (int) legacy::NERF_GRIDSIZE()) return 0xFFFFFFFFu;
        return network::detail::morton3D(i.x, i.y, i.z);
    }

    inline __host__ __device__ bool density_grid_occupied_at(const legacy::math::vec3& pos, const std::uint8_t* density_grid_bitfield) {
        const std::uint32_t idx = density_grid_idx_at(pos);
        if (idx == 0xFFFFFFFFu) return false;
        return density_grid_bitfield[idx / 8u] & (1u << (idx % 8u));
    }

    inline __host__ __device__ float distance_to_next_voxel(const legacy::math::vec3& pos, const legacy::math::vec3& dir, const legacy::math::vec3& idir) {
        const legacy::math::vec3 p = (float) legacy::NERF_GRIDSIZE() * (pos - 0.5f);
        const float tx             = (floorf(p.x + 0.5f + 0.5f * legacy::math::sign(dir.x)) - p.x) * idir.x;
        const float ty             = (floorf(p.y + 0.5f + 0.5f * legacy::math::sign(dir.y)) - p.y) * idir.y;
        const float tz             = (floorf(p.z + 0.5f + 0.5f * legacy::math::sign(dir.z)) - p.z) * idir.z;
        const float t              = fminf(fminf(tx, ty), tz);
        return fmaxf(t / (float) legacy::NERF_GRIDSIZE(), 0.0f);
    }

    inline __host__ __device__ float advance_n_steps(const float t, const float n) {
        return t + n * MIN_CONE_STEPSIZE();
    }

    inline __host__ __device__ float calc_dt() {
        return MIN_CONE_STEPSIZE();
    }

    inline __host__ __device__ float advance_to_next_voxel(const float t, const legacy::math::vec3& pos, const legacy::math::vec3& dir, const legacy::math::vec3& idir) {
        const float t_target = t + distance_to_next_voxel(pos, dir, idir);
        return t + ceilf(fmaxf((t_target - t) / MIN_CONE_STEPSIZE(), 0.5f)) * MIN_CONE_STEPSIZE();
    }

    inline __device__ legacy::math::vec2 nerf_random_image_pos_training(legacy::math::pcg32& rng, const legacy::math::ivec2& resolution, const bool snap_to_pixel_centers) {
        legacy::math::vec2 uv = network::detail::random_val_2d(rng);
        if (snap_to_pixel_centers) uv = (legacy::math::vec2(ngp::legacy::math::clamp(legacy::math::ivec2(uv * legacy::math::vec2(resolution)), 0, resolution - 1)) + 0.5f) / legacy::math::vec2(resolution);
        return uv;
    }

    inline __host__ __device__ std::uint32_t image_idx(const std::uint32_t base_idx, const std::uint32_t n_rays, const std::uint32_t n_rays_total, const std::uint32_t n_training_images) {
        (void) n_rays_total;
        return ((base_idx * n_training_images) / n_rays) % n_training_images;
    }

    inline __host__ __device__ float srgb_to_linear(const float srgb) {
        if (srgb <= 0.04045f) return srgb / 12.92f;
        return powf((srgb + 0.055f) / 1.055f, 2.4f);
    }

    inline __host__ __device__ legacy::math::vec3 srgb_to_linear(const legacy::math::vec3& x) {
        return {srgb_to_linear(x.x), srgb_to_linear(x.y), srgb_to_linear(x.z)};
    }

    inline __host__ __device__ float linear_to_srgb(const float linear) {
        if (linear < 0.0031308f) return 12.92f * linear;
        return 1.055f * powf(linear, 0.41666f) - 0.055f;
    }

    inline __host__ __device__ legacy::math::vec3 linear_to_srgb(const legacy::math::vec3& x) {
        return {linear_to_srgb(x.x), linear_to_srgb(x.y), linear_to_srgb(x.z)};
    }

    inline __host__ __device__ legacy::math::ivec2 image_pos(const legacy::math::vec2& pos, const legacy::math::ivec2& resolution) {
        return ngp::legacy::math::clamp(legacy::math::ivec2(pos * legacy::math::vec2(resolution)), 0, resolution - 1);
    }

    inline __host__ __device__ std::uint64_t pixel_idx(const legacy::math::ivec2& px, const legacy::math::ivec2& resolution, const std::uint32_t img) {
        return px.x + px.y * resolution.x + img * (std::uint64_t) resolution.x * resolution.y;
    }

    inline __host__ __device__ legacy::math::vec4 read_rgba(const legacy::math::ivec2& px, const legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        const std::uint32_t rgba32 = ((const std::uint32_t*) pixels)[pixel_idx(px, resolution, img)];
        legacy::math::vec4 result  = {
            ((rgba32 & 0x000000FFu) >> 0u) * (1.0f / 255.0f),
            ((rgba32 & 0x0000FF00u) >> 8u) * (1.0f / 255.0f),
            ((rgba32 & 0x00FF0000u) >> 16u) * (1.0f / 255.0f),
            ((rgba32 & 0xFF000000u) >> 24u) * (1.0f / 255.0f),
        };
        result.rgb() = srgb_to_linear(result.rgb()) * result.a;
        return result;
    }

    inline __host__ __device__ legacy::math::vec4 read_rgba(const legacy::math::vec2& pos, const legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        return read_rgba(image_pos(pos, resolution), resolution, pixels, img);
    }

    struct LossAndGradient final {
        legacy::math::vec3 loss     = {};
        legacy::math::vec3 gradient = {};

        __host__ __device__ LossAndGradient operator*(const float scalar) {
            return {loss * scalar, gradient * scalar};
        }

        __host__ __device__ LossAndGradient operator/(const float scalar) {
            return {loss / scalar, gradient / scalar};
        }
    };

    inline __host__ __device__ LossAndGradient l2_loss(const legacy::math::vec3& target, const legacy::math::vec3& prediction) {
        const legacy::math::vec3 difference = prediction - target;
        return {difference * difference, 2.0f * difference};
    }

    inline __host__ __device__ LossAndGradient loss_and_gradient(const legacy::math::vec3& target, const legacy::math::vec3& prediction) {
        return l2_loss(target, prediction);
    }

    inline __host__ __device__ float network_to_rgb(const float val) {
        return network::detail::logistic(val);
    }

    inline __host__ __device__ float network_to_rgb_derivative(const float val) {
        const float rgb = network::detail::logistic(val);
        return rgb * (1.0f - rgb);
    }

    template <typename T>
    __host__ __device__ legacy::math::vec3 network_to_rgb_vec(const T& val) {
        return {
            network_to_rgb(float(val[0])),
            network_to_rgb(float(val[1])),
            network_to_rgb(float(val[2])),
        };
    }

    inline __host__ __device__ float network_to_density_derivative(const float val) {
        return expf(legacy::math::clamp(val, -15.0f, 15.0f));
    }

    inline __host__ __device__ legacy::math::vec3 unwarp_position(const legacy::math::vec3& pos, const legacy::BoundingBox& aabb) {
        return aabb.min + pos * aabb.diag();
    }

    inline __host__ __device__ float unwarp_dt(const float dt) {
        (void) dt;
        return MIN_CONE_STEPSIZE();
    }

    inline __host__ __device__ legacy::math::vec3 clamp_rgb01(const legacy::math::vec3& value) {
        return {
            legacy::math::clamp(value.x, 0.0f, 1.0f),
            legacy::math::clamp(value.y, 0.0f, 1.0f),
            legacy::math::clamp(value.z, 0.0f, 1.0f),
        };
    }

    template <typename T>
    __global__ void fill_rollover(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i                = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;
        inout[i] = inout[i % (n_input_elements * stride)];
    }

    template <typename T>
    __global__ void fill_rollover_and_rescale(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i                = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;

        T result = inout[i % (n_input_elements * stride)];
        result   = (T) ((float) result * n_input_elements / n_elements);
        inout[i] = result;
    }

    template <typename T>
    inline __device__ T warp_reduce(T val) {
        TCNN_PRAGMA_UNROLL
        for (int offset = warpSize / 2; offset > 0; offset /= 2) val += __shfl_xor_sync(0xffffffff, val, offset);
        return val;
    }

    template <typename T, typename T_OUT, typename F>
    __global__ void block_reduce(const std::uint32_t n_elements, const F fun, const T* __restrict__ input, T_OUT* __restrict__ output, const std::uint32_t n_blocks) {
        const std::uint32_t sum_idx        = blockIdx.x / n_blocks;
        const std::uint32_t sub_blocks_idx = blockIdx.x % n_blocks;
        const std::uint32_t i              = threadIdx.x + sub_blocks_idx * blockDim.x;
        const std::uint32_t block_offset   = sum_idx * n_elements;

        static __shared__ T_OUT sdata[32];

        const int lane = threadIdx.x % warpSize;
        const int wid  = threadIdx.x / warpSize;

        T_OUT val = {};
        if constexpr (std::is_same_v<std::decay_t<T>, __half> || std::is_same_v<std::decay_t<T>, half>) {
            if (i < n_elements) {
                half vals[8];
                *(int4*) &vals[0] = *((int4*) input + i + block_offset);
                val               = fun((T) vals[0]) + fun((T) vals[1]) + fun((T) vals[2]) + fun((T) vals[3]) + fun((T) vals[4]) + fun((T) vals[5]) + fun((T) vals[6]) + fun((T) vals[7]);
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, float>) {
            if (i < n_elements) {
                const float4 vals = *((float4*) input + i + block_offset);
                val               = fun((T) vals.x) + fun((T) vals.y) + fun((T) vals.z) + fun((T) vals.w);
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
            if (i < n_elements) {
                const double2 vals = *((double2*) input + i + block_offset);
                val                = fun((T) vals.x) + fun((T) vals.y);
            }
        } else {
            assert(false);
            return;
        }

        val = warp_reduce(val);

        if (lane == 0) sdata[wid] = val;
        __syncthreads();

        if (wid == 0) {
            val = threadIdx.x < blockDim.x / warpSize ? sdata[lane] : static_cast<T_OUT>(0);
            val = warp_reduce(val);
            if (lane == 0) atomicAdd(&output[sum_idx], val);
        }
    }

    template <typename T, typename T_OUT, typename F>
    void reduce_sum(T* device_pointer, F fun, T_OUT* workspace, std::uint32_t n_elements, cudaStream_t stream, const std::uint32_t n_sums = 1u) {
        const std::uint32_t threads          = 1024u;
        const std::uint32_t n_elems_per_load = 16u / sizeof(T);

        if (n_elements % n_elems_per_load != 0u) throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
        if (((std::size_t) device_pointer) % 16u != 0u) throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};

        n_elements /= n_elems_per_load;
        const std::uint32_t blocks = (n_elements + threads - 1u) / threads;
        block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
    }

    inline std::uint32_t reduce_sum_workspace_size(const std::uint32_t n_elements) {
        return (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
    }

    __global__ void mark_untrained_density_grid(const std::uint32_t n_elements, float* __restrict__ grid_out, const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t x = network::detail::morton3D_invert(i >> 0u);
        const std::uint32_t y = network::detail::morton3D_invert(i >> 1u);
        const std::uint32_t z = network::detail::morton3D_invert(i >> 2u);

        const float voxel_size       = 1.0f / legacy::NERF_GRIDSIZE();
        const legacy::math::vec3 pos = legacy::math::vec3{(float) x, (float) y, (float) z} / (float) legacy::NERF_GRIDSIZE();

        legacy::math::vec3 corners[8] = {
            pos + legacy::math::vec3{0.0f, 0.0f, 0.0f},
            pos + legacy::math::vec3{voxel_size, 0.0f, 0.0f},
            pos + legacy::math::vec3{0.0f, voxel_size, 0.0f},
            pos + legacy::math::vec3{voxel_size, voxel_size, 0.0f},
            pos + legacy::math::vec3{0.0f, 0.0f, voxel_size},
            pos + legacy::math::vec3{voxel_size, 0.0f, voxel_size},
            pos + legacy::math::vec3{0.0f, voxel_size, voxel_size},
            pos + legacy::math::vec3{voxel_size, voxel_size, voxel_size},
        };

        const std::uint32_t min_count = 1u;
        std::uint32_t count           = 0u;

        for (std::uint32_t j = 0u; j < n_training_images && count < min_count; ++j) {
            const auto& frame = frames[j];
            const auto& xform = frame.camera;

            for (std::uint32_t k = 0u; k < 8u; ++k) {
                const legacy::math::vec3 dir = ngp::legacy::math::normalize(corners[k] - xform[3]);
                if (ngp::legacy::math::dot(dir, xform[2]) < 1e-4f) continue;

                const legacy::math::vec2 uv = pos_to_uv(corners[k], frame.resolution, frame.focal_length, xform);
                const Ray ray               = uv_to_ray(0u, uv, frame.resolution, frame.focal_length, xform);
                if (ngp::legacy::math::distance(ngp::legacy::math::normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
                    ++count;
                    break;
                }
            }
        }

        grid_out[i] = count >= min_count ? 0.0f : -1.0f;
    }

    __global__ void generate_grid_samples_nerf_nonuniform(const std::uint32_t n_elements, legacy::math::pcg32 rng, const std::uint32_t step, legacy::BoundingBox aabb, const float* __restrict__ grid_in, legacy::NerfPosition* __restrict__ out, std::uint32_t* __restrict__ indices, const float thresh) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        rng.advance(i * 4u);
        std::uint32_t idx = 0u;
        for (std::uint32_t j = 0u; j < 10u; ++j) {
            idx = ((i + step * n_elements) * 56924617u + j * 19349663u + 96925573u) % legacy::NERF_GRID_N_CELLS();
            if (grid_in[idx] > thresh) break;
        }

        const std::uint32_t x        = network::detail::morton3D_invert(idx >> 0u);
        const std::uint32_t y        = network::detail::morton3D_invert(idx >> 1u);
        const std::uint32_t z        = network::detail::morton3D_invert(idx >> 2u);
        const legacy::math::vec3 pos = (legacy::math::vec3{(float) x, (float) y, (float) z} + network::detail::random_val_3d(rng)) / (float) legacy::NERF_GRIDSIZE();

        out[i]     = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE())};
        indices[i] = idx;
    }

    __global__ void splat_grid_samples_nerf_max_nearest_neighbor(const std::uint32_t n_elements, const std::uint32_t* __restrict__ indices, const __half* __restrict__ network_output, float* __restrict__ grid_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t idx = indices[i];
        const float mlp         = network_to_density(float(network_output[i]));
        const float thickness   = mlp * MIN_CONE_STEPSIZE();
        atomicMax((std::uint32_t*) &grid_out[idx], __float_as_uint(thickness));
    }

    __global__ void ema_grid_samples_nerf(const std::uint32_t n_elements, const float decay, const std::uint32_t count, float* __restrict__ grid_out, const float* __restrict__ grid_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        (void) count;
        const float importance = grid_in[i];
        const float prev_val   = grid_out[i];
        const float val        = prev_val < 0.0f ? prev_val : fmaxf(prev_val * decay, importance);
        grid_out[i]            = val;
    }

    __global__ void grid_to_bitfield(const std::uint32_t n_elements, const float* __restrict__ grid, std::uint8_t* __restrict__ grid_bitfield, const float* __restrict__ mean_density_ptr) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        std::uint8_t bits  = 0u;
        const float thresh = fminf(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

        TCNN_PRAGMA_UNROLL
        for (std::uint8_t j = 0u; j < 8u; ++j) bits |= grid[i * 8u + j] > thresh ? ((std::uint8_t) 1u << j) : 0u;

        grid_bitfield[i] = bits;
    }

    __global__ void generate_training_samples_nerf(const std::uint32_t n_rays, const legacy::BoundingBox aabb, const std::uint32_t max_samples, const std::uint32_t n_rays_total, legacy::math::pcg32 rng, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ numsteps_counter, std::uint32_t* __restrict__ ray_indices_out, Ray* __restrict__ rays_out_unnormalized, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out,
        const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames, const std::uint8_t* __restrict__ density_grid, const bool snap_to_pixel_centers) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;

        const std::uint32_t img              = image_idx(i, n_rays, n_rays_total, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
        const legacy::math::vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);

        const float focal_length         = frame.focal_length;
        const legacy::math::mat4x3 xform = frame.camera;

        Ray ray_unnormalized = uv_to_ray(0u, uv, resolution, focal_length, xform);
        if (!ray_unnormalized.is_valid()) ray_unnormalized = {xform[3], xform[2]};

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);

        const float startt            = advance_n_steps(tminmax.x, network::detail::random_val(rng));
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t j = 0u;
        float t         = startt;
        legacy::math::vec3 pos;

        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS()) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
        if (j == 0u) return;

        const std::uint32_t numsteps = j;
        const std::uint32_t base     = atomicAdd(numsteps_counter, numsteps);
        if (base + numsteps > max_samples) return;

        coords_out += base;

        const std::uint32_t ray_idx     = atomicAdd(ray_counter, 1u);
        ray_indices_out[ray_idx]        = i;
        rays_out_unnormalized[ray_idx]  = ray_unnormalized;
        numsteps_out[ray_idx * 2u + 0u] = numsteps;
        numsteps_out[ray_idx * 2u + 1u] = base;

        const legacy::math::vec3 warped_dir = warp_direction(ray_d_normalized);
        t                                   = startt;
        j                                   = 0u;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, warp_dt(dt));
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void compute_loss_kernel_train_nerf(const std::uint32_t n_rays, const legacy::BoundingBox aabb, const std::uint32_t n_rays_total, legacy::math::pcg32 rng, const std::uint32_t max_samples_compacted, const std::uint32_t* __restrict__ rays_counter, float loss_scale, const int padded_output_width, const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames, const __half* network_output, std::uint32_t* __restrict__ numsteps_counter,
        const std::uint32_t* __restrict__ ray_indices_in, const Ray* __restrict__ rays_in_unnormalized, std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const legacy::NerfCoordinate> coords_in, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out, __half* dloss_doutput, float* __restrict__ loss_output, const bool snap_to_pixel_centers, const float* __restrict__ mean_density_ptr, const float near_distance) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= *rays_counter) return;

        std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
        std::uint32_t base     = numsteps_in[i * 2u + 1u];

        coords_in += base;
        network_output += base * padded_output_width;

        float T                 = 1.0f;
        constexpr float epsilon = 1e-4f;

        legacy::math::vec3 rgb_ray       = legacy::math::vec3(0.0f);
        std::uint32_t compacted_numsteps = 0u;
        const legacy::math::vec3 ray_o   = rays_in_unnormalized[i].o;
        for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
            if (T < epsilon) break;

            const legacy::math::tvec<__half, 4u> local_network_output = *(legacy::math::tvec<__half, 4u>*) network_output;
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const legacy::math::vec3 pos                              = unwarp_position(coords_in.ptr->pos.p, aabb);
            const float dt                                            = unwarp_dt(coords_in.ptr->dt);
            const float density                                       = network_to_density(float(local_network_output[3]));

            const float alpha  = 1.0f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            T *= (1.0f - alpha);

            network_output += padded_output_width;
            coords_in += 1u;
        }

        const std::uint32_t ray_idx = ray_indices_in[i];
        rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

        const std::uint32_t img              = image_idx(ray_idx, n_rays, n_rays_total, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        const legacy::math::vec2 uv               = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);
        const legacy::math::vec3 background_color = network::detail::random_val_3d(rng);
        const legacy::math::vec4 texsamp          = read_rgba(uv, resolution, frame.pixels);
        const legacy::math::vec3 rgbtarget        = linear_to_srgb(texsamp.rgb() + (1.0f - texsamp.a) * srgb_to_linear(background_color));

        if (compacted_numsteps == numsteps) rgb_ray += T * background_color;

        network_output -= padded_output_width * compacted_numsteps;
        coords_in -= compacted_numsteps;

        std::uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps);
        compacted_numsteps           = std::min(max_samples_compacted - std::min(max_samples_compacted, compacted_base), compacted_numsteps);
        numsteps_in[i * 2u + 0u]     = compacted_numsteps;
        numsteps_in[i * 2u + 1u]     = compacted_base;
        if (compacted_numsteps == 0u) return;

        coords_out += compacted_base;
        dloss_doutput += compacted_base * padded_output_width;

        LossAndGradient lg    = loss_and_gradient(rgbtarget, rgb_ray);
        const float mean_loss = ngp::legacy::math::mean(lg.loss);
        if (loss_output) loss_output[i] = mean_loss / (float) n_rays;

        loss_scale /= n_rays;

        const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

        legacy::math::vec3 rgb_ray2 = {0.0f, 0.0f, 0.0f};
        T                           = 1.0f;
        for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
            legacy::NerfCoordinate* coord_out      = coords_out(j);
            const legacy::NerfCoordinate* coord_in = coords_in(j);
            *coord_out                             = *coord_in;

            const legacy::math::vec3 pos                              = unwarp_position(coord_in->pos.p, aabb);
            const float depth                                         = ngp::legacy::math::distance(pos, ray_o);
            const float dt                                            = unwarp_dt(coord_in->dt);
            const legacy::math::tvec<__half, 4u> local_network_output = *(legacy::math::tvec<__half, 4u>*) network_output;
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const float density                                       = network_to_density(float(local_network_output[3]));
            const float alpha                                         = 1.0f - __expf(-density * dt);
            const float weight                                        = alpha * T;
            rgb_ray2 += weight * rgb;
            T *= (1.0f - alpha);

            const legacy::math::vec3 suffix        = rgb_ray - rgb_ray2;
            const legacy::math::vec3 dloss_by_drgb = weight * lg.gradient;

            legacy::math::tvec<__half, 4u> local_dL_doutput;
            local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0]));
            local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1]));
            local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2]));

            const float density_derivative = network_to_density_derivative(float(local_network_output[3]));
            const float dloss_by_dmlp      = density_derivative * (dt * ngp::legacy::math::dot(lg.gradient, T * rgb - suffix));
            local_dL_doutput[3]            = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) + (float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);

            *(legacy::math::tvec<__half, 4u>*) dloss_doutput = local_dL_doutput;

            dloss_doutput += padded_output_width;
            network_output += padded_output_width;
        }
    }

    __global__ void generate_validation_samples_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, legacy::BoundingBox aabb, const std::uint32_t max_samples, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ overflow_counter, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<legacy::NerfCoordinate> coords_out, InstantNGP::GpuFrame frame, const std::uint8_t* __restrict__ density_grid) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        numsteps_out[i * 2u + 0u] = 0u;
        numsteps_out[i * 2u + 1u] = 0u;

        const std::uint32_t global_pixel     = pixel_offset + i;
        const legacy::math::ivec2 resolution = frame.resolution;
        const legacy::math::ivec2 px         = {(int) (global_pixel % (std::uint32_t) resolution.x), (int) (global_pixel / (std::uint32_t) resolution.x)};
        const legacy::math::vec2 uv          = (legacy::math::vec2{(float) px.x + 0.5f, (float) px.y + 0.5f}) / legacy::math::vec2(resolution);
        const legacy::math::mat4x3 xform     = frame.camera;

        Ray ray_unnormalized = uv_to_ray(0u, uv, resolution, frame.focal_length, xform);
        if (!ray_unnormalized.is_valid()) return;

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);
        if (tminmax.y <= tminmax.x) return;

        float t                       = advance_n_steps(tminmax.x, 0.5f);
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t numsteps = 0u;
        legacy::math::vec3 pos;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && numsteps < NERF_STEPS()) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                ++numsteps;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
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

        coords_out += base;
        const legacy::math::vec3 warped_dir = warp_direction(ray_d_normalized);
        t                                   = advance_n_steps(tminmax.x, 0.5f);

        std::uint32_t j = 0u;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < numsteps) {
            const float dt = calc_dt();
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, warp_dt(dt));
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void composite_validation_kernel_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, const std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const legacy::NerfCoordinate> coords_in, const __half* __restrict__ network_output, const std::uint32_t padded_output_width, legacy::math::vec3 background_color, legacy::math::vec3* __restrict__ image_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        const std::uint32_t global_pixel = pixel_offset + i;
        const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
        const std::uint32_t base         = numsteps_in[i * 2u + 1u];
        float T                          = 1.0f;
        legacy::math::vec3 rgb_ray       = legacy::math::vec3(0.0f);

        if (numsteps > 0u) {
            coords_in += base;
            network_output += base * padded_output_width;

            for (std::uint32_t j = 0u; j < numsteps; ++j) {
                const legacy::math::tvec<__half, 4u> local_network_output = *(const legacy::math::tvec<__half, 4u>*) network_output;
                const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
                const float dt                                            = unwarp_dt(coords_in.ptr->dt);
                const float density                                       = network_to_density(float(local_network_output[3]));
                const float alpha                                         = 1.0f - __expf(-density * dt);
                const float weight                                        = alpha * T;

                rgb_ray += weight * rgb;
                T *= (1.0f - alpha);

                coords_in += 1;
                network_output += padded_output_width;
                if (T < 1e-4f) break;
            }
        }

        image_out[global_pixel] = clamp_rgb01(rgb_ray + T * background_color);
    }

    InstantNGP::InstantNGP(const NetworkConfig& network_config_) {
        spec.network_config = network_config_;
        std::printf("Making training plan at training step %u.\n", training.step);

        spec.plan                             = {};
        spec.plan.training.batch_size         = 1u << 18;
        spec.plan.training.floats_per_coord   = sizeof(legacy::NerfCoordinate) / sizeof(float);
        spec.plan.validation.floats_per_coord = spec.plan.training.floats_per_coord;
        spec.plan.validation.max_samples      = spec.plan.validation.tile_rays * spec.plan.validation.max_samples_per_ray;

        constexpr std::uint32_t n_pos_dims    = sizeof(legacy::NerfPosition) / sizeof(float);
        spec.plan.network.n_pos_dims          = n_pos_dims;
        spec.plan.network.n_dir_dims          = 3u;
        spec.plan.network.dir_offset          = n_pos_dims + 1u;
        spec.plan.network.density_alignment   = 16u;
        spec.plan.network.density_output_dims = 16u;
        spec.plan.network.rgb_alignment       = 16u;
        spec.plan.network.rgb_output_dims     = 3u;

        const std::uint32_t encoding_output_dims   = spec.network_config.encoding.n_levels * spec.network_config.encoding.n_features_per_level;
        spec.plan.network.density_input_dims       = legacy::next_multiple(encoding_output_dims, legacy::lcm(spec.plan.network.density_alignment, spec.network_config.encoding.n_features_per_level));
        const std::uint32_t dir_output_dims        = spec.network_config.direction_encoding.sh_degree * spec.network_config.direction_encoding.sh_degree;
        spec.plan.network.dir_encoding_output_dims = legacy::next_multiple(dir_output_dims, spec.plan.network.rgb_alignment);
        spec.plan.network.rgb_input_dims           = legacy::next_multiple(spec.plan.network.density_output_dims + spec.plan.network.dir_encoding_output_dims, spec.plan.network.rgb_alignment);

        spec.plan.training.padded_output_width     = std::max(legacy::next_multiple(spec.plan.network.rgb_output_dims, 16u), 4u);
        spec.plan.training.max_samples             = spec.plan.training.batch_size * 16u;
        spec.plan.prep.uniform_samples_warmup      = legacy::NERF_GRID_N_CELLS();
        spec.plan.prep.uniform_samples_steady      = legacy::NERF_GRID_N_CELLS() / 4u;
        spec.plan.prep.nonuniform_samples_steady   = legacy::NERF_GRID_N_CELLS() / 4u;
        spec.plan.density_grid.padded_output_width = spec.plan.network.density_output_dims;
        spec.plan.density_grid.query_batch_size    = legacy::NERF_GRID_N_CELLS() * 2u;
        spec.plan.density_grid.n_elements          = legacy::NERF_GRID_N_CELLS();
        spec.plan.validation.padded_output_width   = spec.plan.training.padded_output_width;

        cudaStream_t created_stream = {};
        legacy::cuda_check(cudaStreamCreate(&created_stream));
        device.stream = created_stream;

        try {
            training.rng        = legacy::math::pcg32{spec.seed};
            sampler.density_rng = legacy::math::pcg32{training.rng.next_uint()};
            device.trainer      = std::unique_ptr<network::TrainerState<__half>, void (*)(network::TrainerState<__half>*)>{
                new network::TrainerState<__half>(spec.network_config, spec.plan, spec.seed, device.stream),
                delete_trainer_state,
            };
        } catch (...) {
            cudaStreamDestroy(created_stream);
            device.stream = nullptr;
            throw;
        }
    }

    InstantNGP::~InstantNGP() noexcept {
        device.trainer.reset();
        if (!device.stream) return;

        network::detail::free_aux_stream_pool(device.stream);
        cudaStreamDestroy(device.stream);
    }

    InstantNGP::InstantNGP(InstantNGP&& other) noexcept {
        *this = std::move(other);
    }

    InstantNGP& InstantNGP::operator=(InstantNGP&& other) noexcept {
        if (this == &other) return *this;

        device.trainer.reset();
        if (device.stream) {
            network::detail::free_aux_stream_pool(device.stream);
            cudaStreamDestroy(device.stream);
        }

        spec                = other.spec;
        dataset             = std::move(other.dataset);
        sampler             = std::move(other.sampler);
        training            = std::move(other.training);
        device              = std::move(other.device);
        other.device.stream = nullptr;
        return *this;
    }

    auto InstantNGP::read_train_stats() const -> TrainStats {
        TrainStats stats{};
        stats.loss                                  = training.last_loss;
        stats.train_ms                              = training.last_train_ms;
        stats.prep_ms                               = training.last_prep_ms;
        stats.training_step                         = training.step;
        stats.batch_size                            = spec.plan.training.batch_size;
        stats.rays_per_batch                        = training.counters.rays_per_batch;
        stats.measured_batch_size                   = training.counters.measured_batch_size;
        stats.measured_batch_size_before_compaction = training.counters.measured_batch_size_before_compaction;
        return stats;
    }

    auto InstantNGP::render_validation_image(const std::filesystem::path& output_path, const std::uint32_t validation_image_index) -> ValidationResult {
        if (output_path.empty()) throw std::invalid_argument{"validation output path must not be empty."};
        if (!device.trainer) throw std::runtime_error{"Validation rendering requires an initialized network."};

        const std::size_t validation_count = dataset.cpu.validation.size();
        if (validation_count == 0u) throw std::runtime_error{"No validation images are available in the current dataset."};
        if (validation_image_index >= validation_count) throw std::runtime_error{"Validation image index is out of range."};

        const Dataset::CPU::Frame& source = dataset.cpu.validation[validation_image_index];
        if (source.width == 0u || source.height == 0u) throw std::runtime_error{"Validation frame has zero resolution."};
        if (!std::isfinite(source.focal_length_x) || source.focal_length_x <= 0.0f) throw std::runtime_error{"Validation frame has an invalid focal_length_x."};
        if (!std::isfinite(source.focal_length_y) || source.focal_length_y <= 0.0f) throw std::runtime_error{"Validation frame has an invalid focal_length_y."};

        const float focal_length_difference = std::fabs(source.focal_length_x - source.focal_length_y);
        const float focal_length_scale      = std::max(1.0f, std::max(std::fabs(source.focal_length_x), std::fabs(source.focal_length_y)));
        if (focal_length_difference > 1e-6f * focal_length_scale) throw std::runtime_error{"Validation currently requires focal_length_x and focal_length_y to match."};

        GpuFrame frame{};
        frame.resolution   = legacy::math::ivec2{(int) source.width, (int) source.height};
        frame.focal_length = source.focal_length_x;
        for (std::size_t row = 0u; row < 3u; ++row) {
            for (std::size_t column = 0u; column < 4u; ++column) frame.camera[column][row] = source.transform_matrix_4x4[row * 4u + column];
        }
        frame.camera[1] *= -1.0f;
        frame.camera[2] *= -1.0f;
        frame.camera[3]                      = frame.camera[3] * 0.33f + legacy::math::vec3(0.5f);
        const legacy::math::vec4 camera_row0 = ngp::legacy::math::row(frame.camera, 0);
        frame.camera                         = ngp::legacy::math::row(frame.camera, 0, ngp::legacy::math::row(frame.camera, 1));
        frame.camera                         = ngp::legacy::math::row(frame.camera, 1, ngp::legacy::math::row(frame.camera, 2));
        frame.camera                         = ngp::legacy::math::row(frame.camera, 2, camera_row0);

        const legacy::math::ivec2 resolution = frame.resolution;
        const legacy::math::vec3 background  = legacy::math::vec3(1.0f);

        ValidationRenderWorkspace workspace{};
        workspace.total_pixels        = (std::uint32_t) ngp::legacy::math::product(resolution);
        workspace.padded_output_width = spec.plan.validation.padded_output_width;
        workspace.floats_per_coord    = spec.plan.validation.floats_per_coord;
        workspace.max_samples         = spec.plan.validation.max_samples;
        workspace.rendered            = ngp::legacy::GpuBuffer<legacy::math::vec3>{workspace.total_pixels};
        workspace.tile_numsteps       = ngp::legacy::GpuBuffer<std::uint32_t>{spec.plan.validation.tile_rays * 2u};
        workspace.tile_coords         = ngp::legacy::GpuBuffer<float>{workspace.max_samples * workspace.floats_per_coord};
        workspace.tile_mlp_out        = ngp::legacy::GpuBuffer<__half>{workspace.max_samples * workspace.padded_output_width};
        workspace.sample_counter      = ngp::legacy::GpuBuffer<std::uint32_t>{1u};
        workspace.overflow_counter    = ngp::legacy::GpuBuffer<std::uint32_t>{1u};

        for (std::uint32_t pixel_offset = 0u; pixel_offset < workspace.total_pixels; pixel_offset += spec.plan.validation.tile_rays) {
            const std::uint32_t tile_pixels = std::min(spec.plan.validation.tile_rays, workspace.total_pixels - pixel_offset);
            legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), device.stream));

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, sampler.aabb, workspace.max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<legacy::NerfCoordinate>{(legacy::NerfCoordinate*) workspace.tile_coords.data(), 1u}, frame, sampler.density.occupancy_bits.data());
            }

            std::uint32_t used_samples    = 0u;
            std::uint32_t overflowed_rays = 0u;
            legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
            legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, device.stream));
            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            if (overflowed_rays != 0u) {
                std::ostringstream message;
                message << "Validation render sample budget overflowed for " << overflowed_rays << " rays. Reduce tile size or increase the sample budget.";
                throw std::runtime_error{message.str()};
            }

            if (used_samples > 0u) {
                const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                const std::uint32_t coord_elements      = padded_used_samples * workspace.floats_per_coord;
                fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(padded_used_samples, workspace.floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                legacy::GPUMatrixDynamic<float> coords_matrix((float*) workspace.tile_coords.data(), workspace.floats_per_coord, padded_used_samples, legacy::CM);
                legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), workspace.padded_output_width, padded_used_samples, legacy::CM);
                device.trainer->model.inference(device.stream, coords_matrix, rgbsigma_matrix);
            }

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const legacy::NerfCoordinate>{(const legacy::NerfCoordinate*) workspace.tile_coords.data(), 1u}, workspace.tile_mlp_out.data(), workspace.padded_output_width, background, workspace.rendered.data());
            }
        }

        legacy::cuda_check(cudaStreamSynchronize(device.stream));

        std::vector<legacy::math::vec3> rendered_host(workspace.total_pixels);
        workspace.rendered.copy_to_host(rendered_host);

        std::vector<std::uint8_t> png_rgb((std::size_t) workspace.total_pixels * 3u);
        double total_squared_error = 0.0;
        for (int y = 0; y < resolution.y; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                const std::size_t pixel_index       = (std::size_t) x + (std::size_t) y * (std::size_t) resolution.x;
                const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                const legacy::math::vec4 gt         = read_rgba(legacy::math::ivec2{x, y}, resolution, source.rgba.data());
                const legacy::math::vec3 target     = clamp_rgb01(linear_to_srgb(gt.rgb() + (1.0f - gt.a) * background));
                const legacy::math::vec3 diff       = prediction - target;
                total_squared_error += (double) ngp::legacy::math::mean(diff * diff);

                png_rgb[pixel_index * 3u + 0u] = (std::uint8_t) lrintf(prediction.x * 255.0f);
                png_rgb[pixel_index * 3u + 1u] = (std::uint8_t) lrintf(prediction.y * 255.0f);
                png_rgb[pixel_index * 3u + 2u] = (std::uint8_t) lrintf(prediction.z * 255.0f);
            }
        }

        const std::filesystem::path parent_dir = output_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create image directory '" + parent_dir.string() + "'."};
        if (stbi_write_png(output_path.string().c_str(), resolution.x, resolution.y, 3, png_rgb.data(), resolution.x * 3) == 0) throw std::runtime_error{"Failed to write PNG image '" + output_path.string() + "'."};

        ValidationResult validation{};
        validation.width       = (std::uint32_t) resolution.x;
        validation.height      = (std::uint32_t) resolution.y;
        validation.mse         = (float) (total_squared_error / (double) workspace.total_pixels);
        validation.psnr        = validation.mse > 0.0f ? -10.0f * std::log10(validation.mse) : std::numeric_limits<float>::infinity();
        validation.image_index = (std::int32_t) validation_image_index;
        return validation;
    }

    void InstantNGP::train(const std::int32_t iters) {
        if (!device.trainer || dataset.gpu.frames.size() == 0u) throw std::runtime_error{"training data must be loaded before train()."};

        for (std::int32_t i = 0; i < iters; ++i) {
            // Training prep.
            std::uint32_t n_prep_to_skip = training.step / spec.plan.prep.skip_growth_interval;
            if (n_prep_to_skip < 1u) n_prep_to_skip = 1u;
            if (n_prep_to_skip > spec.plan.prep.max_skip) n_prep_to_skip = spec.plan.prep.max_skip;
            if (training.step % n_prep_to_skip == 0u) {
                const auto prep_start = std::chrono::steady_clock::now();
                legacy::ScopeGuard prep_timing_guard{[&] { training.last_prep_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prep_start).count() / n_prep_to_skip; }};

                const std::uint32_t n_uniform_density_grid_samples    = training.step < spec.plan.prep.warmup_steps ? spec.plan.prep.uniform_samples_warmup : spec.plan.prep.uniform_samples_steady;
                const std::uint32_t n_nonuniform_density_grid_samples = training.step < spec.plan.prep.warmup_steps ? 0u : spec.plan.prep.nonuniform_samples_steady;
                const std::uint32_t n_elements                        = spec.plan.density_grid.n_elements;

                sampler.density.values.resize(n_elements);
                const std::uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;
                const std::uint32_t padded_output_width    = spec.plan.density_grid.padded_output_width;

                const std::size_t positions_bytes        = legacy::align_to_cacheline(n_density_grid_samples * sizeof(legacy::NerfPosition));
                const std::size_t indices_bytes          = legacy::align_to_cacheline(n_elements * sizeof(std::uint32_t));
                const std::size_t density_tmp_bytes      = legacy::align_to_cacheline(n_elements * sizeof(float));
                const std::size_t mlp_out_bytes          = legacy::align_to_cacheline(n_density_grid_samples * padded_output_width * sizeof(__half));
                legacy::GpuAllocation density_grid_alloc = network::detail::allocate_workspace(device.stream, positions_bytes + indices_bytes + density_tmp_bytes + mlp_out_bytes);

                std::uint8_t* density_grid_base = density_grid_alloc.data();
                std::size_t density_grid_offset = 0u;
                auto* density_grid_positions    = reinterpret_cast<legacy::NerfPosition*>(density_grid_base + density_grid_offset);
                density_grid_offset += positions_bytes;
                auto* density_grid_indices = reinterpret_cast<std::uint32_t*>(density_grid_base + density_grid_offset);
                density_grid_offset += indices_bytes;
                auto* density_grid_tmp = reinterpret_cast<float*>(density_grid_base + density_grid_offset);
                density_grid_offset += density_tmp_bytes;
                auto* density_grid_mlp_out = reinterpret_cast<__half*>(density_grid_base + density_grid_offset);

                if (training.step == 0u) {
                    sampler.density.ema_step = 0u;
                    if (n_elements > 0u) {
                        const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                        mark_untrained_density_grid<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_elements, sampler.density.values.data(), static_cast<std::uint32_t>(dataset.gpu.frames.size()), dataset.gpu.frames.data());
                    }
                }

                legacy::cuda_check(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float) * n_elements, device.stream));

                if (n_uniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_uniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_uniform_density_grid_samples, sampler.density_rng, sampler.density.ema_step, sampler.aabb, sampler.density.values.data(), density_grid_positions, density_grid_indices, -0.01f);
                }
                sampler.density_rng.advance();

                if (n_nonuniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_nonuniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_nonuniform_density_grid_samples, sampler.density_rng, sampler.density.ema_step, sampler.aabb, sampler.density.values.data(), density_grid_positions + n_uniform_density_grid_samples, density_grid_indices + n_uniform_density_grid_samples, NERF_MIN_OPTICAL_THICKNESS());
                }
                sampler.density_rng.advance();

                const std::size_t density_batch_size = spec.plan.density_grid.query_batch_size;
                for (std::size_t density_batch_offset = 0u; density_batch_offset < n_density_grid_samples; density_batch_offset += density_batch_size) {
                    const std::size_t density_query_size = std::min(density_batch_size, static_cast<std::size_t>(n_density_grid_samples) - density_batch_offset);
                    legacy::GPUMatrixDynamic<__half> density_matrix(density_grid_mlp_out + density_batch_offset, padded_output_width, density_query_size, legacy::RM);
                    legacy::GPUMatrixDynamic<float> density_position_matrix((float*) (density_grid_positions + density_batch_offset), sizeof(legacy::NerfPosition) / sizeof(float), density_query_size, legacy::CM);
                    device.trainer->model.density(device.stream, density_position_matrix, density_matrix);
                }

                if (n_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    splat_grid_samples_nerf_max_nearest_neighbor<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_density_grid_samples, density_grid_indices, density_grid_mlp_out, density_grid_tmp);
                }

                if (n_elements > 0u) {
                    const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    ema_grid_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(n_elements, sampler.density.ema_decay, sampler.density.ema_step, sampler.density.values.data(), density_grid_tmp);
                }
                ++sampler.density.ema_step;

                const std::uint32_t base_grid_elements = legacy::NERF_GRID_N_CELLS();
                sampler.density.occupancy_bits.enlarge(base_grid_elements / 8u);
                sampler.density.reduction_workspace.enlarge(reduce_sum_workspace_size(base_grid_elements));

                legacy::cuda_check(cudaMemsetAsync(sampler.density.reduction_workspace.data(), 0, sizeof(float), device.stream));
                reduce_sum(sampler.density.values.data(), DensityGridReduceOp{base_grid_elements}, sampler.density.reduction_workspace.data(), base_grid_elements, device.stream);

                if (base_grid_elements / 8u > 0u) {
                    const std::uint32_t blocks = ((base_grid_elements / 8u) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    grid_to_bitfield<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(base_grid_elements / 8u, sampler.density.values.data(), sampler.density.occupancy_bits.data(), sampler.density.reduction_workspace.data());
                }

                legacy::cuda_check(cudaStreamSynchronize(device.stream));
            }

            // Per-step hyperparameters and timing.
            device.trainer->optimizer.update_hyperparams(spec.network_config.optimizer);
            const bool get_loss_scalar = training.step % 16u == 0u;
            const auto train_start     = std::chrono::steady_clock::now();
            legacy::ScopeGuard train_timing_guard{[&] { training.last_train_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count(); }};

            // Step workspace.
            auto& counters                 = training.counters;
            const std::uint32_t batch_size = spec.plan.training.batch_size;

            counters.numsteps_counter.enlarge(1u);
            counters.numsteps_counter_compacted.enlarge(1u);
            counters.loss.enlarge(counters.rays_per_batch);
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter_compacted.data(), 0, sizeof(std::uint32_t), device.stream));
            legacy::cuda_check(cudaMemsetAsync(counters.loss.data(), 0, sizeof(float) * counters.rays_per_batch, device.stream));

            TrainingStepWorkspace workspace{};
            workspace.padded_output_width = spec.plan.training.padded_output_width;
            workspace.floats_per_coord    = spec.plan.training.floats_per_coord;
            workspace.max_samples         = spec.plan.training.max_samples;

            const std::size_t ray_indices_bytes       = legacy::align_to_cacheline(counters.rays_per_batch * sizeof(std::uint32_t));
            const std::size_t rays_unnormalized_bytes = legacy::align_to_cacheline(counters.rays_per_batch * sizeof(Ray));
            const std::size_t numsteps_bytes          = legacy::align_to_cacheline(counters.rays_per_batch * 2u * sizeof(std::uint32_t));
            const std::size_t coords_bytes            = legacy::align_to_cacheline(workspace.max_samples * workspace.floats_per_coord * sizeof(float));
            const std::size_t mlp_out_bytes           = legacy::align_to_cacheline(std::max(batch_size, workspace.max_samples) * workspace.padded_output_width * sizeof(__half));
            const std::size_t dloss_bytes             = legacy::align_to_cacheline(batch_size * workspace.padded_output_width * sizeof(__half));
            const std::size_t compacted_coords_bytes  = legacy::align_to_cacheline(batch_size * workspace.floats_per_coord * sizeof(float));
            const std::size_t ray_counter_bytes       = legacy::align_to_cacheline(sizeof(std::uint32_t));
            const std::size_t total_bytes             = ray_indices_bytes + rays_unnormalized_bytes + numsteps_bytes + coords_bytes + mlp_out_bytes + dloss_bytes + compacted_coords_bytes + ray_counter_bytes;

            workspace.alloc              = network::detail::allocate_workspace(device.stream, total_bytes);
            std::uint8_t* workspace_base = workspace.alloc.data();
            std::size_t workspace_offset = 0u;

            workspace.ray_indices = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);
            workspace_offset += ray_indices_bytes;
            workspace.rays_unnormalized = workspace_base + workspace_offset;
            workspace_offset += rays_unnormalized_bytes;
            workspace.numsteps = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);
            workspace_offset += numsteps_bytes;
            workspace.coords = reinterpret_cast<float*>(workspace_base + workspace_offset);
            workspace_offset += coords_bytes;
            workspace.mlp_out = reinterpret_cast<__half*>(workspace_base + workspace_offset);
            workspace_offset += mlp_out_bytes;
            workspace.dloss_dmlp_out = reinterpret_cast<__half*>(workspace_base + workspace_offset);
            workspace_offset += dloss_bytes;
            workspace.coords_compacted = reinterpret_cast<float*>(workspace_base + workspace_offset);
            workspace_offset += compacted_coords_bytes;
            workspace.ray_counter = reinterpret_cast<std::uint32_t*>(workspace_base + workspace_offset);

            workspace.max_inference = counters.measured_batch_size_before_compaction == 0u ? workspace.max_samples : legacy::next_multiple(std::min(counters.measured_batch_size_before_compaction, workspace.max_samples), network::detail::batch_size_granularity);
            if (counters.measured_batch_size_before_compaction == 0u) counters.measured_batch_size_before_compaction = workspace.max_inference;

            workspace.coords_matrix           = legacy::GPUMatrixDynamic<float>{workspace.coords, workspace.floats_per_coord, workspace.max_inference, legacy::CM};
            workspace.rgbsigma_matrix         = legacy::GPUMatrixDynamic<__half>{workspace.mlp_out, workspace.padded_output_width, workspace.max_inference, legacy::CM};
            workspace.compacted_coords_matrix = legacy::GPUMatrixDynamic<float>{workspace.coords_compacted, workspace.floats_per_coord, batch_size, legacy::CM};
            workspace.gradient_matrix         = legacy::GPUMatrixDynamic<__half>{workspace.dloss_dmlp_out, workspace.padded_output_width, batch_size, legacy::CM};
            workspace.compacted_output        = legacy::GPUMatrixDynamic<__half>{spec.plan.training.padded_output_width, batch_size, device.stream, legacy::CM};

            if (training.step == 0u) counters.n_rays_total = 0u;
            workspace.n_rays_total = counters.n_rays_total;
            counters.n_rays_total += counters.rays_per_batch;

            // Sample generation and first inference pass.
            legacy::cuda_check(cudaMemsetAsync(workspace.ray_counter, 0, sizeof(std::uint32_t), device.stream));
            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_training_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(counters.rays_per_batch, sampler.aabb, workspace.max_inference, workspace.n_rays_total, training.rng, workspace.ray_counter, counters.numsteps_counter.data(), workspace.ray_indices, reinterpret_cast<Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords), 1u},
                    static_cast<std::uint32_t>(dataset.gpu.pixels.size()), dataset.gpu.frames.data(), sampler.density.occupancy_bits.data(), sampler.snap_to_pixel_centers);
            }
            device.trainer->model.inference(device.stream, workspace.coords_matrix, workspace.rgbsigma_matrix);

            // Loss compaction.
            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                compute_loss_kernel_train_nerf<<<blocks, network::detail::n_threads_linear, 0, device.stream>>>(counters.rays_per_batch, sampler.aabb, workspace.n_rays_total, training.rng, batch_size, workspace.ray_counter, network::detail::default_loss_scale<__half>(), workspace.padded_output_width, static_cast<std::uint32_t>(dataset.gpu.pixels.size()), dataset.gpu.frames.data(), workspace.mlp_out, counters.numsteps_counter_compacted.data(), workspace.ray_indices,
                    reinterpret_cast<const Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<const legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords), 1u}, legacy::PitchedPtr<legacy::NerfCoordinate>{reinterpret_cast<legacy::NerfCoordinate*>(workspace.coords_compacted), 1u}, workspace.dloss_dmlp_out, counters.loss.data(), sampler.snap_to_pixel_centers, sampler.density.reduction_workspace.data(), sampler.near_distance);
            }

            const std::uint32_t dloss_elements  = batch_size * workspace.padded_output_width;
            const std::uint32_t coords_elements = batch_size * workspace.floats_per_coord;
            fill_rollover_and_rescale<__half><<<((dloss_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(batch_size, workspace.padded_output_width, counters.numsteps_counter_compacted.data(), workspace.dloss_dmlp_out);
            fill_rollover<float><<<((coords_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, device.stream>>>(batch_size, workspace.floats_per_coord, counters.numsteps_counter_compacted.data(), workspace.coords_compacted);

            // Backward and optimizer.
            legacy::run_graph_capture(device.trainer->graph, device.stream, [&]() {
                device.trainer->model.forward(device.stream, workspace.compacted_coords_matrix, &workspace.compacted_output, device.trainer->scratch);
                device.trainer->model.backward(device.stream, device.trainer->scratch, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix, network::detail::GradientMode::Overwrite);
            });
            device.trainer->optimizer.step(device.stream, network::detail::default_loss_scale<__half>(), device.trainer->params.full_precision, device.trainer->params.values, device.trainer->params.gradients);
            ++training.step;

            // Host-visible step finalization.
            legacy::cuda_check(cudaStreamSynchronize(device.stream));

            std::uint32_t measured_batch_size_before_compaction = 0u;
            std::uint32_t measured_batch_size                   = 0u;
            counters.numsteps_counter.copy_to_host(&measured_batch_size_before_compaction, 1u);
            counters.numsteps_counter_compacted.copy_to_host(&measured_batch_size, 1u);
            counters.measured_batch_size_before_compaction = 0u;
            counters.measured_batch_size                   = 0u;

            float last_loss = 0.0f;
            if (measured_batch_size_before_compaction != 0u && measured_batch_size != 0u) {
                counters.measured_batch_size_before_compaction = measured_batch_size_before_compaction;
                counters.measured_batch_size                   = measured_batch_size;

                if (get_loss_scalar) {
                    legacy::GpuAllocation reduction_alloc = network::detail::allocate_workspace(device.stream, reduce_sum_workspace_size(counters.rays_per_batch) * sizeof(float));
                    float* reduction_workspace            = reinterpret_cast<float*>(reduction_alloc.data());
                    legacy::cuda_check(cudaMemsetAsync(reduction_workspace, 0, sizeof(float), device.stream));
                    reduce_sum(counters.loss.data(), SumIdentityOp{}, reduction_workspace, counters.rays_per_batch, device.stream);
                    legacy::cuda_check(cudaMemcpyAsync(&last_loss, reduction_workspace, sizeof(float), cudaMemcpyDeviceToHost, device.stream));
                    legacy::cuda_check(cudaStreamSynchronize(device.stream));
                    last_loss *= (float) counters.measured_batch_size / (float) batch_size;
                }

                counters.rays_per_batch = (std::uint32_t) ((float) counters.rays_per_batch * (float) batch_size / (float) counters.measured_batch_size);
                counters.rays_per_batch = std::min(legacy::next_multiple(counters.rays_per_batch, network::detail::batch_size_granularity), 1u << 18);
            }

            if (get_loss_scalar) training.last_loss = last_loss;

            if (counters.measured_batch_size == 0u) {
                training.last_loss = 0.0f;
                std::fprintf(stderr, "Nerf training generated 0 samples. Aborting training.\n");
                throw std::runtime_error{"Training stopped unexpectedly."};
            }

            training.rng.advance();
        }
    }

} // namespace ngp
