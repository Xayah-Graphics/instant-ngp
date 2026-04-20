#include "network.cuh"
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <sstream>
#include <type_traits>

namespace ngp {

    void TrainerStateDeleter::operator()(ngp::network::TrainerState<__half>* trainer) const {
        delete trainer;
    }

    struct Ray final {
        ngp::legacy::math::vec3 o = {};
        ngp::legacy::math::vec3 d = {};

        __host__ __device__ ngp::legacy::math::vec3 operator()(const float t) const {
            return o + t * d;
        }

        __host__ __device__ void advance(const float t) {
            o += d * t;
        }

        __host__ __device__ bool is_valid() const {
            return d != ngp::legacy::math::vec3(0.0f);
        }
    };

    inline __host__ __device__ Ray uv_to_ray(const std::uint32_t spp, const ngp::legacy::math::vec2& uv, const ngp::legacy::math::ivec2& resolution, const float focal_length, const ngp::legacy::math::mat4x3& camera_matrix, const float near_distance = 0.0f) {
        (void) spp;
        ngp::legacy::math::vec3 dir = {(uv.x - 0.5f) * (float) resolution.x / focal_length, (uv.y - 0.5f) * (float) resolution.y / focal_length, 1.0f};
        dir = ngp::legacy::math::mat3(camera_matrix) * dir;
        ngp::legacy::math::vec3 origin = camera_matrix[3];
        origin += dir * near_distance;
        return {origin, dir};
    }

    inline __host__ __device__ ngp::legacy::math::vec2 pos_to_uv(const ngp::legacy::math::vec3& pos, const ngp::legacy::math::ivec2& resolution, const float focal_length, const ngp::legacy::math::mat4x3& camera_matrix) {
        ngp::legacy::math::vec3 dir = ngp::legacy::math::inverse(ngp::legacy::math::mat3(camera_matrix)) * (pos - camera_matrix[3]);
        dir /= dir.z;
        return dir.xy() * focal_length / ngp::legacy::math::vec2(resolution) + ngp::legacy::math::vec2(0.5f);
    }

    inline __host__ __device__ float network_to_density(const float val) {
        return expf(val);
    }

    inline constexpr __host__ __device__ float SQRT3() {
        return 1.73205080757f;
    }

    inline constexpr __host__ __device__ std::uint32_t NERF_STEPS() {
        return 1024u;
    }

    inline constexpr __host__ __device__ float STEPSIZE() {
        return SQRT3() / (float) NERF_STEPS();
    }

    inline constexpr __host__ __device__ float MIN_CONE_STEPSIZE() {
        return STEPSIZE();
    }

    inline constexpr __host__ __device__ std::uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY() {
        return 16u;
    }

    inline constexpr __host__ __device__ float NERF_MIN_OPTICAL_THICKNESS() {
        return 0.01f;
    }

    struct DensityGridReduceOp final {
        std::uint32_t base_grid_elements = 0u;

        __device__ float operator()(const float val) const {
            return fmaxf(val, 0.0f) / base_grid_elements;
        }
    };

    inline __host__ __device__ ngp::legacy::math::vec3 warp_position(const ngp::legacy::math::vec3& pos, const ngp::legacy::BoundingBox& aabb) {
        return aabb.relative_pos(pos);
    }

    inline __host__ __device__ ngp::legacy::math::vec3 warp_direction(const ngp::legacy::math::vec3& dir) {
        return (dir + 1.0f) * 0.5f;
    }

    inline __host__ __device__ float warp_dt(const float dt) {
        (void) dt;
        return 0.0f;
    }

    inline __host__ __device__ std::uint32_t density_grid_idx_at(const ngp::legacy::math::vec3& pos) {
        const ngp::legacy::math::ivec3 i = pos * (float) ngp::legacy::NERF_GRIDSIZE();
        if (i.x < 0 || i.x >= (int) ngp::legacy::NERF_GRIDSIZE() || i.y < 0 || i.y >= (int) ngp::legacy::NERF_GRIDSIZE() || i.z < 0 || i.z >= (int) ngp::legacy::NERF_GRIDSIZE()) return 0xFFFFFFFFu;
        return ngp::network::detail::morton3D(i.x, i.y, i.z);
    }

    inline __host__ __device__ bool density_grid_occupied_at(const ngp::legacy::math::vec3& pos, const std::uint8_t* density_grid_bitfield) {
        const std::uint32_t idx = density_grid_idx_at(pos);
        if (idx == 0xFFFFFFFFu) return false;
        return density_grid_bitfield[idx / 8u] & (1u << (idx % 8u));
    }

    inline __host__ __device__ float distance_to_next_voxel(const ngp::legacy::math::vec3& pos, const ngp::legacy::math::vec3& dir, const ngp::legacy::math::vec3& idir) {
        const ngp::legacy::math::vec3 p = (float) ngp::legacy::NERF_GRIDSIZE() * (pos - 0.5f);
        const float tx = (floorf(p.x + 0.5f + 0.5f * ngp::legacy::math::sign(dir.x)) - p.x) * idir.x;
        const float ty = (floorf(p.y + 0.5f + 0.5f * ngp::legacy::math::sign(dir.y)) - p.y) * idir.y;
        const float tz = (floorf(p.z + 0.5f + 0.5f * ngp::legacy::math::sign(dir.z)) - p.z) * idir.z;
        const float t = fminf(fminf(tx, ty), tz);
        return fmaxf(t / (float) ngp::legacy::NERF_GRIDSIZE(), 0.0f);
    }

    inline __host__ __device__ float advance_n_steps(const float t, const float n) {
        return t + n * MIN_CONE_STEPSIZE();
    }

    inline __host__ __device__ float calc_dt() {
        return MIN_CONE_STEPSIZE();
    }

    inline __host__ __device__ float advance_to_next_voxel(const float t, const ngp::legacy::math::vec3& pos, const ngp::legacy::math::vec3& dir, const ngp::legacy::math::vec3& idir) {
        const float t_target = t + distance_to_next_voxel(pos, dir, idir);
        return t + ceilf(fmaxf((t_target - t) / MIN_CONE_STEPSIZE(), 0.5f)) * MIN_CONE_STEPSIZE();
    }

    inline __device__ ngp::legacy::math::vec2 nerf_random_image_pos_training(ngp::legacy::math::pcg32& rng, const ngp::legacy::math::ivec2& resolution, const bool snap_to_pixel_centers) {
        ngp::legacy::math::vec2 uv = ngp::network::detail::random_val_2d(rng);
        if (snap_to_pixel_centers) uv = (ngp::legacy::math::vec2(ngp::legacy::math::clamp(ngp::legacy::math::ivec2(uv * ngp::legacy::math::vec2(resolution)), 0, resolution - 1)) + 0.5f) / ngp::legacy::math::vec2(resolution);
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

    inline __host__ __device__ ngp::legacy::math::vec3 srgb_to_linear(const ngp::legacy::math::vec3& x) {
        return {srgb_to_linear(x.x), srgb_to_linear(x.y), srgb_to_linear(x.z)};
    }

    inline __host__ __device__ float linear_to_srgb(const float linear) {
        if (linear < 0.0031308f) return 12.92f * linear;
        return 1.055f * powf(linear, 0.41666f) - 0.055f;
    }

    inline __host__ __device__ ngp::legacy::math::vec3 linear_to_srgb(const ngp::legacy::math::vec3& x) {
        return {linear_to_srgb(x.x), linear_to_srgb(x.y), linear_to_srgb(x.z)};
    }

    inline __host__ __device__ ngp::legacy::math::ivec2 image_pos(const ngp::legacy::math::vec2& pos, const ngp::legacy::math::ivec2& resolution) {
        return ngp::legacy::math::clamp(ngp::legacy::math::ivec2(pos * ngp::legacy::math::vec2(resolution)), 0, resolution - 1);
    }

    inline __host__ __device__ std::uint64_t pixel_idx(const ngp::legacy::math::ivec2& px, const ngp::legacy::math::ivec2& resolution, const std::uint32_t img) {
        return px.x + px.y * resolution.x + img * (std::uint64_t) resolution.x * resolution.y;
    }

    inline __host__ __device__ ngp::legacy::math::vec4 read_rgba(const ngp::legacy::math::ivec2& px, const ngp::legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        const std::uint32_t rgba32 = ((const std::uint32_t*) pixels)[pixel_idx(px, resolution, img)];
        ngp::legacy::math::vec4 result = {
            ((rgba32 & 0x000000FFu) >> 0u) * (1.0f / 255.0f),
            ((rgba32 & 0x0000FF00u) >> 8u) * (1.0f / 255.0f),
            ((rgba32 & 0x00FF0000u) >> 16u) * (1.0f / 255.0f),
            ((rgba32 & 0xFF000000u) >> 24u) * (1.0f / 255.0f),
        };
        result.rgb() = srgb_to_linear(result.rgb()) * result.a;
        return result;
    }

    inline __host__ __device__ ngp::legacy::math::vec4 read_rgba(const ngp::legacy::math::vec2& pos, const ngp::legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        return read_rgba(image_pos(pos, resolution), resolution, pixels, img);
    }

    struct LossAndGradient final {
        ngp::legacy::math::vec3 loss = {};
        ngp::legacy::math::vec3 gradient = {};

        __host__ __device__ LossAndGradient operator*(const float scalar) {
            return {loss * scalar, gradient * scalar};
        }

        __host__ __device__ LossAndGradient operator/(const float scalar) {
            return {loss / scalar, gradient / scalar};
        }
    };

    inline __host__ __device__ LossAndGradient l2_loss(const ngp::legacy::math::vec3& target, const ngp::legacy::math::vec3& prediction) {
        const ngp::legacy::math::vec3 difference = prediction - target;
        return {difference * difference, 2.0f * difference};
    }

    inline __host__ __device__ LossAndGradient loss_and_gradient(const ngp::legacy::math::vec3& target, const ngp::legacy::math::vec3& prediction) {
        return l2_loss(target, prediction);
    }

    inline __host__ __device__ float network_to_rgb(const float val) {
        return ngp::network::detail::logistic(val);
    }

    inline __host__ __device__ float network_to_rgb_derivative(const float val) {
        const float rgb = ngp::network::detail::logistic(val);
        return rgb * (1.0f - rgb);
    }

    template <typename T>
    __host__ __device__ ngp::legacy::math::vec3 network_to_rgb_vec(const T& val) {
        return {
            network_to_rgb(float(val[0])),
            network_to_rgb(float(val[1])),
            network_to_rgb(float(val[2])),
        };
    }

    inline __host__ __device__ float network_to_density_derivative(const float val) {
        return expf(ngp::legacy::math::clamp(val, -15.0f, 15.0f));
    }

    inline __host__ __device__ ngp::legacy::math::vec3 unwarp_position(const ngp::legacy::math::vec3& pos, const ngp::legacy::BoundingBox& aabb) {
        return aabb.min + pos * aabb.diag();
    }

    inline __host__ __device__ float unwarp_dt(const float dt) {
        (void) dt;
        return MIN_CONE_STEPSIZE();
    }

    template <typename T>
    __global__ void fill_rollover(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;
        inout[i] = inout[i % (n_input_elements * stride)];
    }

    template <typename T>
    __global__ void fill_rollover_and_rescale(const std::uint32_t n_elements, const std::uint32_t stride, const std::uint32_t* n_input_elements_ptr, T* inout) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        const std::uint32_t n_input_elements = *n_input_elements_ptr;
        if (i < (n_input_elements * stride) || i >= (n_elements * stride) || n_input_elements == 0u) return;

        T result = inout[i % (n_input_elements * stride)];
        result = (T) ((float) result * n_input_elements / n_elements);
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
        const std::uint32_t sum_idx = blockIdx.x / n_blocks;
        const std::uint32_t sub_blocks_idx = blockIdx.x % n_blocks;
        const std::uint32_t i = threadIdx.x + sub_blocks_idx * blockDim.x;
        const std::uint32_t block_offset = sum_idx * n_elements;

        static __shared__ T_OUT sdata[32];

        const int lane = threadIdx.x % warpSize;
        const int wid = threadIdx.x / warpSize;

        T_OUT val = {};
        if constexpr (std::is_same_v<std::decay_t<T>, __half> || std::is_same_v<std::decay_t<T>, ::half>) {
            if (i < n_elements) {
                ::half vals[8];
                *(int4*) &vals[0] = *((int4*) input + i + block_offset);
                val = fun((T) vals[0]) + fun((T) vals[1]) + fun((T) vals[2]) + fun((T) vals[3]) + fun((T) vals[4]) + fun((T) vals[5]) + fun((T) vals[6]) + fun((T) vals[7]);
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, float>) {
            if (i < n_elements) {
                const float4 vals = *((float4*) input + i + block_offset);
                val = fun((T) vals.x) + fun((T) vals.y) + fun((T) vals.z) + fun((T) vals.w);
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
            if (i < n_elements) {
                const double2 vals = *((double2*) input + i + block_offset);
                val = fun((T) vals.x) + fun((T) vals.y);
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
        const std::uint32_t threads = 1024u;
        const std::uint32_t n_elems_per_load = 16u / sizeof(T);

        if (n_elements % n_elems_per_load != 0u) throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
        if (((std::size_t) device_pointer) % 16u != 0u) throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};

        n_elements /= n_elems_per_load;
        const std::uint32_t blocks = (n_elements + threads - 1u) / threads;
        block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
    }

    inline std::uint32_t reduce_sum_workspace_size(const std::uint32_t n_elements) {
        return (n_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
    }

    __global__ void mark_untrained_density_grid(const std::uint32_t n_elements, float* __restrict__ grid_out, const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t x = ngp::network::detail::morton3D_invert(i >> 0u);
        const std::uint32_t y = ngp::network::detail::morton3D_invert(i >> 1u);
        const std::uint32_t z = ngp::network::detail::morton3D_invert(i >> 2u);

        const float voxel_size = 1.0f / ngp::legacy::NERF_GRIDSIZE();
        const ngp::legacy::math::vec3 pos = ngp::legacy::math::vec3{(float) x, (float) y, (float) z} / (float) ngp::legacy::NERF_GRIDSIZE();

        ngp::legacy::math::vec3 corners[8] = {
            pos + ngp::legacy::math::vec3{0.0f, 0.0f, 0.0f},
            pos + ngp::legacy::math::vec3{voxel_size, 0.0f, 0.0f},
            pos + ngp::legacy::math::vec3{0.0f, voxel_size, 0.0f},
            pos + ngp::legacy::math::vec3{voxel_size, voxel_size, 0.0f},
            pos + ngp::legacy::math::vec3{0.0f, 0.0f, voxel_size},
            pos + ngp::legacy::math::vec3{voxel_size, 0.0f, voxel_size},
            pos + ngp::legacy::math::vec3{0.0f, voxel_size, voxel_size},
            pos + ngp::legacy::math::vec3{voxel_size, voxel_size, voxel_size},
        };

        const std::uint32_t min_count = 1u;
        std::uint32_t count = 0u;

        for (std::uint32_t j = 0u; j < n_training_images && count < min_count; ++j) {
            const auto& frame = frames[j];
            const auto& xform = frame.camera;

            for (std::uint32_t k = 0u; k < 8u; ++k) {
                const ngp::legacy::math::vec3 dir = ngp::legacy::math::normalize(corners[k] - xform[3]);
                if (ngp::legacy::math::dot(dir, xform[2]) < 1e-4f) continue;

                const ngp::legacy::math::vec2 uv = pos_to_uv(corners[k], frame.resolution, frame.focal_length, xform);
                const Ray ray = uv_to_ray(0u, uv, frame.resolution, frame.focal_length, xform);
                if (ngp::legacy::math::distance(ngp::legacy::math::normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
                    ++count;
                    break;
                }
            }
        }

        grid_out[i] = count >= min_count ? 0.0f : -1.0f;
    }

    __global__ void generate_grid_samples_nerf_nonuniform(const std::uint32_t n_elements, ngp::legacy::math::pcg32 rng, const std::uint32_t step, ngp::legacy::BoundingBox aabb, const float* __restrict__ grid_in, ngp::legacy::NerfPosition* __restrict__ out, std::uint32_t* __restrict__ indices, const float thresh) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        rng.advance(i * 4u);
        std::uint32_t idx = 0u;
        for (std::uint32_t j = 0u; j < 10u; ++j) {
            idx = ((i + step * n_elements) * 56924617u + j * 19349663u + 96925573u) % ngp::legacy::NERF_GRID_N_CELLS();
            if (grid_in[idx] > thresh) break;
        }

        const std::uint32_t x = ngp::network::detail::morton3D_invert(idx >> 0u);
        const std::uint32_t y = ngp::network::detail::morton3D_invert(idx >> 1u);
        const std::uint32_t z = ngp::network::detail::morton3D_invert(idx >> 2u);
        const ngp::legacy::math::vec3 pos = (ngp::legacy::math::vec3{(float) x, (float) y, (float) z} + ngp::network::detail::random_val_3d(rng)) / (float) ngp::legacy::NERF_GRIDSIZE();

        out[i] = {warp_position(pos, aabb), warp_dt(MIN_CONE_STEPSIZE())};
        indices[i] = idx;
    }

    __global__ void splat_grid_samples_nerf_max_nearest_neighbor(const std::uint32_t n_elements, const std::uint32_t* __restrict__ indices, const __half* __restrict__ network_output, float* __restrict__ grid_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t idx = indices[i];
        const float mlp = network_to_density(float(network_output[i]));
        const float thickness = mlp * MIN_CONE_STEPSIZE();
        atomicMax((std::uint32_t*) &grid_out[idx], __float_as_uint(thickness));
    }

    __global__ void ema_grid_samples_nerf(const std::uint32_t n_elements, const float decay, const std::uint32_t count, float* __restrict__ grid_out, const float* __restrict__ grid_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        (void) count;
        const float importance = grid_in[i];
        const float prev_val = grid_out[i];
        const float val = prev_val < 0.0f ? prev_val : fmaxf(prev_val * decay, importance);
        grid_out[i] = val;
    }

    __global__ void grid_to_bitfield(const std::uint32_t n_elements, const float* __restrict__ grid, std::uint8_t* __restrict__ grid_bitfield, const float* __restrict__ mean_density_ptr) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        std::uint8_t bits = 0u;
        const float thresh = fminf(NERF_MIN_OPTICAL_THICKNESS(), *mean_density_ptr);

        TCNN_PRAGMA_UNROLL
        for (std::uint8_t j = 0u; j < 8u; ++j) bits |= grid[i * 8u + j] > thresh ? ((std::uint8_t) 1u << j) : 0u;

        grid_bitfield[i] = bits;
    }

    __global__ void generate_training_samples_nerf(const std::uint32_t n_rays, const ngp::legacy::BoundingBox aabb, const std::uint32_t max_samples, const std::uint32_t n_rays_total, ngp::legacy::math::pcg32 rng, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ numsteps_counter, std::uint32_t* __restrict__ ray_indices_out, Ray* __restrict__ rays_out_unnormalized,
        std::uint32_t* __restrict__ numsteps_out, ngp::legacy::PitchedPtr<ngp::legacy::NerfCoordinate> coords_out, const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames, const std::uint8_t* __restrict__ density_grid, const bool snap_to_pixel_centers) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;

        const std::uint32_t img = image_idx(i, n_rays, n_rays_total, n_training_images);
        const auto& frame = frames[img];
        const ngp::legacy::math::ivec2 resolution = frame.resolution;

        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY());
        const ngp::legacy::math::vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);

        const float focal_length = frame.focal_length;
        const ngp::legacy::math::mat4x3 xform = frame.camera;

        Ray ray_unnormalized = uv_to_ray(0u, uv, resolution, focal_length, xform);
        if (!ray_unnormalized.is_valid()) ray_unnormalized = {xform[3], xform[2]};

        const ngp::legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        ngp::legacy::math::vec2 tminmax = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x = fmaxf(tminmax.x, 0.0f);

        const float startt = advance_n_steps(tminmax.x, ngp::network::detail::random_val(rng));
        const ngp::legacy::math::vec3 idir = ngp::legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t j = 0u;
        float t = startt;
        ngp::legacy::math::vec3 pos;

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
        const std::uint32_t base = atomicAdd(numsteps_counter, numsteps);
        if (base + numsteps > max_samples) return;

        coords_out += base;

        const std::uint32_t ray_idx = atomicAdd(ray_counter, 1u);
        ray_indices_out[ray_idx] = i;
        rays_out_unnormalized[ray_idx] = ray_unnormalized;
        numsteps_out[ray_idx * 2u + 0u] = numsteps;
        numsteps_out[ray_idx * 2u + 1u] = base;

        const ngp::legacy::math::vec3 warped_dir = warp_direction(ray_d_normalized);
        t = startt;
        j = 0u;
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

    __global__ void compute_loss_kernel_train_nerf(const std::uint32_t n_rays, const ngp::legacy::BoundingBox aabb, const std::uint32_t n_rays_total, ngp::legacy::math::pcg32 rng, const std::uint32_t max_samples_compacted, const std::uint32_t* __restrict__ rays_counter, float loss_scale, const int padded_output_width, const std::uint32_t n_training_images, const InstantNGP::GpuFrame* __restrict__ frames,
        const __half* network_output, std::uint32_t* __restrict__ numsteps_counter, const std::uint32_t* __restrict__ ray_indices_in, const Ray* __restrict__ rays_in_unnormalized, std::uint32_t* __restrict__ numsteps_in, ngp::legacy::PitchedPtr<const ngp::legacy::NerfCoordinate> coords_in, ngp::legacy::PitchedPtr<ngp::legacy::NerfCoordinate> coords_out,
        __half* dloss_doutput, float* __restrict__ loss_output, const bool snap_to_pixel_centers, const float* __restrict__ mean_density_ptr, const float near_distance) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= *rays_counter) return;

        std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
        std::uint32_t base = numsteps_in[i * 2u + 1u];

        coords_in += base;
        network_output += base * padded_output_width;

        float T = 1.0f;
        constexpr float epsilon = 1e-4f;

        ngp::legacy::math::vec3 rgb_ray = ngp::legacy::math::vec3(0.0f);
        std::uint32_t compacted_numsteps = 0u;
        const ngp::legacy::math::vec3 ray_o = rays_in_unnormalized[i].o;
        for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
            if (T < epsilon) break;

            const ngp::legacy::math::tvec<__half, 4u> local_network_output = *(ngp::legacy::math::tvec<__half, 4u>*) network_output;
            const ngp::legacy::math::vec3 rgb = network_to_rgb_vec(local_network_output);
            const ngp::legacy::math::vec3 pos = unwarp_position(coords_in.ptr->pos.p, aabb);
            const float dt = unwarp_dt(coords_in.ptr->dt);
            const float density = network_to_density(float(local_network_output[3]));

            const float alpha = 1.0f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            T *= (1.0f - alpha);

            network_output += padded_output_width;
            coords_in += 1u;
        }

        const std::uint32_t ray_idx = ray_indices_in[i];
        rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY());

        const std::uint32_t img = image_idx(ray_idx, n_rays, n_rays_total, n_training_images);
        const auto& frame = frames[img];
        const ngp::legacy::math::ivec2 resolution = frame.resolution;

        const ngp::legacy::math::vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);
        const ngp::legacy::math::vec3 background_color = ngp::network::detail::random_val_3d(rng);
        const ngp::legacy::math::vec4 texsamp = read_rgba(uv, resolution, frame.pixels);
        const ngp::legacy::math::vec3 rgbtarget = linear_to_srgb(texsamp.rgb() + (1.0f - texsamp.a) * srgb_to_linear(background_color));

        if (compacted_numsteps == numsteps) rgb_ray += T * background_color;

        network_output -= padded_output_width * compacted_numsteps;
        coords_in -= compacted_numsteps;

        std::uint32_t compacted_base = atomicAdd(numsteps_counter, compacted_numsteps);
        compacted_numsteps = std::min(max_samples_compacted - std::min(max_samples_compacted, compacted_base), compacted_numsteps);
        numsteps_in[i * 2u + 0u] = compacted_numsteps;
        numsteps_in[i * 2u + 1u] = compacted_base;
        if (compacted_numsteps == 0u) return;

        coords_out += compacted_base;
        dloss_doutput += compacted_base * padded_output_width;

        LossAndGradient lg = loss_and_gradient(rgbtarget, rgb_ray);
        const float mean_loss = ngp::legacy::math::mean(lg.loss);
        if (loss_output) loss_output[i] = mean_loss / (float) n_rays;

        loss_scale /= n_rays;

        const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS() ? 1e-4f : 0.0f;

        ngp::legacy::math::vec3 rgb_ray2 = {0.0f, 0.0f, 0.0f};
        T = 1.0f;
        for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
            ngp::legacy::NerfCoordinate* coord_out = coords_out(j);
            const ngp::legacy::NerfCoordinate* coord_in = coords_in(j);
            *coord_out = *coord_in;

            const ngp::legacy::math::vec3 pos = unwarp_position(coord_in->pos.p, aabb);
            const float depth = ngp::legacy::math::distance(pos, ray_o);
            const float dt = unwarp_dt(coord_in->dt);
            const ngp::legacy::math::tvec<__half, 4u> local_network_output = *(ngp::legacy::math::tvec<__half, 4u>*) network_output;
            const ngp::legacy::math::vec3 rgb = network_to_rgb_vec(local_network_output);
            const float density = network_to_density(float(local_network_output[3]));
            const float alpha = 1.0f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray2 += weight * rgb;
            T *= (1.0f - alpha);

            const ngp::legacy::math::vec3 suffix = rgb_ray - rgb_ray2;
            const ngp::legacy::math::vec3 dloss_by_drgb = weight * lg.gradient;

            ngp::legacy::math::tvec<__half, 4u> local_dL_doutput;
            local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0]));
            local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1]));
            local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2]));

            const float density_derivative = network_to_density_derivative(float(local_network_output[3]));
            const float dloss_by_dmlp = density_derivative * (dt * ngp::legacy::math::dot(lg.gradient, T * rgb - suffix));
            local_dL_doutput[3] = loss_scale * dloss_by_dmlp + (float(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) + (float(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);

            *(ngp::legacy::math::tvec<__half, 4u>*) dloss_doutput = local_dL_doutput;

            dloss_doutput += padded_output_width;
            network_output += padded_output_width;
        }
    }

    InstantNGP::InstantNGP(const NetworkConfig& network_config_) : network_config{network_config_} {
        std::printf("Making training plan at training step %u.\n", training_step);

        plan = {};
        plan.training.batch_size = 1u << 18;
        plan.training.floats_per_coord = sizeof(ngp::legacy::NerfCoordinate) / sizeof(float);
        plan.validation.floats_per_coord = plan.training.floats_per_coord;
        plan.validation.max_samples = plan.validation.tile_rays * plan.validation.max_samples_per_ray;

        constexpr std::uint32_t n_pos_dims = sizeof(ngp::legacy::NerfPosition) / sizeof(float);
        plan.network.n_pos_dims = n_pos_dims;
        plan.network.n_dir_dims = 3u;
        plan.network.dir_offset = n_pos_dims + 1u;
        plan.network.density_alignment = 16u;
        plan.network.density_output_dims = 16u;
        plan.network.rgb_alignment = 16u;
        plan.network.rgb_output_dims = 3u;

        const std::uint32_t encoding_output_dims = network_config.encoding.n_levels * network_config.encoding.n_features_per_level;
        plan.network.density_input_dims = ngp::legacy::next_multiple(encoding_output_dims, ngp::legacy::lcm(plan.network.density_alignment, network_config.encoding.n_features_per_level));
        const std::uint32_t dir_output_dims = network_config.direction_encoding.sh_degree * network_config.direction_encoding.sh_degree;
        plan.network.dir_encoding_output_dims = ngp::legacy::next_multiple(dir_output_dims, plan.network.rgb_alignment);
        plan.network.rgb_input_dims = ngp::legacy::next_multiple(plan.network.density_output_dims + plan.network.dir_encoding_output_dims, plan.network.rgb_alignment);

        plan.training.padded_output_width = std::max(ngp::legacy::next_multiple(plan.network.rgb_output_dims, 16u), 4u);
        plan.training.max_samples = plan.training.batch_size * 16u;
        plan.prep.uniform_samples_warmup = ngp::legacy::NERF_GRID_N_CELLS();
        plan.prep.uniform_samples_steady = ngp::legacy::NERF_GRID_N_CELLS() / 4u;
        plan.prep.nonuniform_samples_steady = ngp::legacy::NERF_GRID_N_CELLS() / 4u;
        plan.density_grid.padded_output_width = plan.network.density_output_dims;
        plan.density_grid.query_batch_size = ngp::legacy::NERF_GRID_N_CELLS() * 2u;
        plan.density_grid.n_elements = ngp::legacy::NERF_GRID_N_CELLS();
        plan.validation.padded_output_width = plan.training.padded_output_width;

        rng = ngp::legacy::math::pcg32{seed};
        density_grid_rng = ngp::legacy::math::pcg32{rng.next_uint()};
        trainer = std::unique_ptr<ngp::network::TrainerState<__half>, ngp::TrainerStateDeleter>{new ngp::network::TrainerState<__half>(network_config, plan, seed, stream.stream)};
    }

    InstantNGP::~InstantNGP() = default;

    void InstantNGP::run_training_prep() {
        const std::uint32_t n_prep_to_skip = std::clamp(training_step / plan.prep.skip_growth_interval, 1u, plan.prep.max_skip);
        if (training_step % n_prep_to_skip != 0u) return;

        const auto start = std::chrono::steady_clock::now();
        ngp::legacy::ScopeGuard timing_guard{[&] { training_prep_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - start).count() / n_prep_to_skip; }};
        update_density_grid();
        ngp::legacy::cuda_check(cudaStreamSynchronize(stream.stream));
    }

    auto InstantNGP::begin_training_step() -> InstantNGP::TrainingStepWorkspace {
        auto& counters = counters_rgb;
        const std::uint32_t batch_size = plan.training.batch_size;

        counters.numsteps_counter.enlarge(1u);
        counters.numsteps_counter_compacted.enlarge(1u);
        counters.loss.enlarge(counters.rays_per_batch);
        ngp::legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter.data(), 0, sizeof(std::uint32_t), stream.stream));
        ngp::legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter_compacted.data(), 0, sizeof(std::uint32_t), stream.stream));
        ngp::legacy::cuda_check(cudaMemsetAsync(counters.loss.data(), 0, sizeof(float) * counters.rays_per_batch, stream.stream));

        TrainingStepWorkspace workspace{};
        workspace.padded_output_width = plan.training.padded_output_width;
        workspace.floats_per_coord = plan.training.floats_per_coord;
        workspace.max_samples = plan.training.max_samples;

        const std::size_t ray_indices_bytes = ngp::legacy::align_to_cacheline(counters.rays_per_batch * sizeof(std::uint32_t));
        const std::size_t rays_unnormalized_bytes = ngp::legacy::align_to_cacheline(counters.rays_per_batch * sizeof(Ray));
        const std::size_t numsteps_bytes = ngp::legacy::align_to_cacheline(counters.rays_per_batch * 2u * sizeof(std::uint32_t));
        const std::size_t coords_bytes = ngp::legacy::align_to_cacheline(workspace.max_samples * workspace.floats_per_coord * sizeof(float));
        const std::size_t mlp_out_bytes = ngp::legacy::align_to_cacheline(std::max(batch_size, workspace.max_samples) * workspace.padded_output_width * sizeof(__half));
        const std::size_t dloss_bytes = ngp::legacy::align_to_cacheline(batch_size * workspace.padded_output_width * sizeof(__half));
        const std::size_t compacted_coords_bytes = ngp::legacy::align_to_cacheline(batch_size * workspace.floats_per_coord * sizeof(float));
        const std::size_t ray_counter_bytes = ngp::legacy::align_to_cacheline(sizeof(std::uint32_t));
        const std::size_t total_bytes = ray_indices_bytes + rays_unnormalized_bytes + numsteps_bytes + coords_bytes + mlp_out_bytes + dloss_bytes + compacted_coords_bytes + ray_counter_bytes;

        workspace.alloc = ngp::network::detail::allocate_workspace(stream.stream, total_bytes);
        std::uint8_t* base = workspace.alloc.data();
        std::size_t offset = 0u;

        workspace.ray_indices = reinterpret_cast<std::uint32_t*>(base + offset);
        offset += ray_indices_bytes;
        workspace.rays_unnormalized = base + offset;
        offset += rays_unnormalized_bytes;
        workspace.numsteps = reinterpret_cast<std::uint32_t*>(base + offset);
        offset += numsteps_bytes;
        workspace.coords = reinterpret_cast<float*>(base + offset);
        offset += coords_bytes;
        workspace.mlp_out = reinterpret_cast<__half*>(base + offset);
        offset += mlp_out_bytes;
        workspace.dloss_dmlp_out = reinterpret_cast<__half*>(base + offset);
        offset += dloss_bytes;
        workspace.coords_compacted = reinterpret_cast<float*>(base + offset);
        offset += compacted_coords_bytes;
        workspace.ray_counter = reinterpret_cast<std::uint32_t*>(base + offset);

        workspace.max_inference = counters.measured_batch_size_before_compaction == 0u ? workspace.max_samples
                                                                                        : ngp::legacy::next_multiple(std::min(counters.measured_batch_size_before_compaction, workspace.max_samples), ngp::network::detail::batch_size_granularity);
        if (counters.measured_batch_size_before_compaction == 0u) counters.measured_batch_size_before_compaction = workspace.max_inference;

        workspace.coords_matrix = ngp::legacy::GPUMatrixDynamic<float>{workspace.coords, workspace.floats_per_coord, workspace.max_inference, ngp::legacy::CM};
        workspace.rgbsigma_matrix = ngp::legacy::GPUMatrixDynamic<__half>{workspace.mlp_out, workspace.padded_output_width, workspace.max_inference, ngp::legacy::CM};
        workspace.compacted_coords_matrix = ngp::legacy::GPUMatrixDynamic<float>{workspace.coords_compacted, workspace.floats_per_coord, batch_size, ngp::legacy::CM};
        workspace.gradient_matrix = ngp::legacy::GPUMatrixDynamic<__half>{workspace.dloss_dmlp_out, workspace.padded_output_width, batch_size, ngp::legacy::CM};
        workspace.compacted_output = ngp::legacy::GPUMatrixDynamic<__half>{plan.training.padded_output_width, batch_size, stream.stream, ngp::legacy::CM};

        if (training_step == 0u) counters.n_rays_total = 0u;
        workspace.n_rays_total = counters.n_rays_total;
        counters.n_rays_total += counters.rays_per_batch;

        return workspace;
    }

    void InstantNGP::generate_training_batch(TrainingStepWorkspace& workspace) {
        auto& counters = counters_rgb;

        ngp::legacy::cuda_check(cudaMemsetAsync(workspace.ray_counter, 0, sizeof(std::uint32_t), stream.stream));
        if (counters.rays_per_batch > 0u) {
            const std::uint32_t blocks = (counters.rays_per_batch + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            generate_training_samples_nerf<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(
                counters.rays_per_batch,
                aabb,
                workspace.max_inference,
                workspace.n_rays_total,
                rng,
                workspace.ray_counter,
                counters.numsteps_counter.data(),
                workspace.ray_indices,
                reinterpret_cast<Ray*>(workspace.rays_unnormalized),
                workspace.numsteps,
                ngp::legacy::PitchedPtr<ngp::legacy::NerfCoordinate>{reinterpret_cast<ngp::legacy::NerfCoordinate*>(workspace.coords), 1u},
                static_cast<std::uint32_t>(dataset.gpu.train.pixels.size()),
                dataset.gpu.train.frames.data(),
                density_grid_bitfield.data(),
                snap_to_pixel_centers
            );
        }

        trainer->model.inference(stream.stream, workspace.coords_matrix, workspace.rgbsigma_matrix);
    }

    void InstantNGP::compact_training_batch(TrainingStepWorkspace& workspace) {
        auto& counters = counters_rgb;
        const std::uint32_t batch_size = plan.training.batch_size;

        if (counters.rays_per_batch > 0u) {
            const std::uint32_t blocks = (counters.rays_per_batch + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            compute_loss_kernel_train_nerf<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(
                counters.rays_per_batch,
                aabb,
                workspace.n_rays_total,
                rng,
                batch_size,
                workspace.ray_counter,
                ngp::network::detail::default_loss_scale<__half>(),
                workspace.padded_output_width,
                static_cast<std::uint32_t>(dataset.gpu.train.pixels.size()),
                dataset.gpu.train.frames.data(),
                workspace.mlp_out,
                counters.numsteps_counter_compacted.data(),
                workspace.ray_indices,
                reinterpret_cast<const Ray*>(workspace.rays_unnormalized),
                workspace.numsteps,
                ngp::legacy::PitchedPtr<const ngp::legacy::NerfCoordinate>{reinterpret_cast<ngp::legacy::NerfCoordinate*>(workspace.coords), 1u},
                ngp::legacy::PitchedPtr<ngp::legacy::NerfCoordinate>{reinterpret_cast<ngp::legacy::NerfCoordinate*>(workspace.coords_compacted), 1u},
                workspace.dloss_dmlp_out,
                counters.loss.data(),
                snap_to_pixel_centers,
                density_grid_mean.data(),
                near_distance
            );
        }

        const std::uint32_t dloss_elements = batch_size * workspace.padded_output_width;
        const std::uint32_t coords_elements = batch_size * workspace.floats_per_coord;
        fill_rollover_and_rescale<__half><<<((dloss_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear), ngp::network::detail::n_threads_linear, 0, stream.stream>>>(batch_size, workspace.padded_output_width, counters.numsteps_counter_compacted.data(),
            workspace.dloss_dmlp_out);
        fill_rollover<float><<<((coords_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear), ngp::network::detail::n_threads_linear, 0, stream.stream>>>(batch_size, workspace.floats_per_coord, counters.numsteps_counter_compacted.data(), workspace.coords_compacted);
    }

    void InstantNGP::update_density_grid() {
        const std::uint32_t n_uniform_density_grid_samples = training_step < plan.prep.warmup_steps ? plan.prep.uniform_samples_warmup : plan.prep.uniform_samples_steady;
        const std::uint32_t n_nonuniform_density_grid_samples = training_step < plan.prep.warmup_steps ? 0u : plan.prep.nonuniform_samples_steady;
        const std::uint32_t n_elements = plan.density_grid.n_elements;

        density_grid.resize(n_elements);
        const std::uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;
        const std::uint32_t padded_output_width = plan.density_grid.padded_output_width;

        const std::size_t positions_bytes = ngp::legacy::align_to_cacheline(n_density_grid_samples * sizeof(ngp::legacy::NerfPosition));
        const std::size_t indices_bytes = ngp::legacy::align_to_cacheline(n_elements * sizeof(std::uint32_t));
        const std::size_t density_tmp_bytes = ngp::legacy::align_to_cacheline(n_elements * sizeof(float));
        const std::size_t mlp_out_bytes = ngp::legacy::align_to_cacheline(n_density_grid_samples * padded_output_width * sizeof(__half));
        ngp::legacy::GpuAllocation alloc = ngp::network::detail::allocate_workspace(stream.stream, positions_bytes + indices_bytes + density_tmp_bytes + mlp_out_bytes);

        std::uint8_t* base = alloc.data();
        std::size_t offset = 0u;
        auto* density_grid_positions = reinterpret_cast<ngp::legacy::NerfPosition*>(base + offset);
        offset += positions_bytes;
        auto* density_grid_indices = reinterpret_cast<std::uint32_t*>(base + offset);
        offset += indices_bytes;
        auto* density_grid_tmp = reinterpret_cast<float*>(base + offset);
        offset += density_tmp_bytes;
        auto* mlp_out = reinterpret_cast<__half*>(base + offset);

        if (training_step == 0u) {
            density_grid_ema_step = 0u;
            if (n_elements > 0u) {
                const std::uint32_t blocks = (n_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
                mark_untrained_density_grid<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(n_elements, density_grid.data(), static_cast<std::uint32_t>(dataset.gpu.train.frames.size()), dataset.gpu.train.frames.data());
            }
        }

        ngp::legacy::cuda_check(cudaMemsetAsync(density_grid_tmp, 0, sizeof(float) * n_elements, stream.stream));

        if (n_uniform_density_grid_samples > 0u) {
            const std::uint32_t blocks = (n_uniform_density_grid_samples + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            generate_grid_samples_nerf_nonuniform<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(n_uniform_density_grid_samples, density_grid_rng, density_grid_ema_step, aabb, density_grid.data(), density_grid_positions, density_grid_indices, -0.01f);
        }
        density_grid_rng.advance();

        if (n_nonuniform_density_grid_samples > 0u) {
            const std::uint32_t blocks = (n_nonuniform_density_grid_samples + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            generate_grid_samples_nerf_nonuniform<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(n_nonuniform_density_grid_samples, density_grid_rng, density_grid_ema_step, aabb, density_grid.data(), density_grid_positions + n_uniform_density_grid_samples, density_grid_indices + n_uniform_density_grid_samples,
                NERF_MIN_OPTICAL_THICKNESS());
        }
        density_grid_rng.advance();

        const std::size_t density_batch_size = plan.density_grid.query_batch_size;
        for (std::size_t i = 0u; i < n_density_grid_samples; i += density_batch_size) {
            const std::size_t batch_size = std::min(density_batch_size, static_cast<std::size_t>(n_density_grid_samples) - i);
            ngp::legacy::GPUMatrixDynamic<__half> density_matrix(mlp_out + i, padded_output_width, batch_size, ngp::legacy::RM);
            ngp::legacy::GPUMatrixDynamic<float> density_grid_position_matrix((float*) (density_grid_positions + i), sizeof(ngp::legacy::NerfPosition) / sizeof(float), batch_size, ngp::legacy::CM);
            trainer->model.density(stream.stream, density_grid_position_matrix, density_matrix);
        }

        if (n_density_grid_samples > 0u) {
            const std::uint32_t blocks = (n_density_grid_samples + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            splat_grid_samples_nerf_max_nearest_neighbor<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(n_density_grid_samples, density_grid_indices, mlp_out, density_grid_tmp);
        }

        if (n_elements > 0u) {
            const std::uint32_t blocks = (n_elements + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            ema_grid_samples_nerf<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(n_elements, density_grid_decay, density_grid_ema_step, density_grid.data(), density_grid_tmp);
        }
        ++density_grid_ema_step;

        const std::uint32_t base_grid_elements = ngp::legacy::NERF_GRID_N_CELLS();
        density_grid_bitfield.enlarge(base_grid_elements / 8u);
        density_grid_mean.enlarge(reduce_sum_workspace_size(base_grid_elements));

        ngp::legacy::cuda_check(cudaMemsetAsync(density_grid_mean.data(), 0, sizeof(float), stream.stream));
        reduce_sum(density_grid.data(), DensityGridReduceOp{base_grid_elements}, density_grid_mean.data(), base_grid_elements, stream.stream);

        if (base_grid_elements / 8u > 0u) {
            const std::uint32_t blocks = ((base_grid_elements / 8u) + ngp::network::detail::n_threads_linear - 1u) / ngp::network::detail::n_threads_linear;
            grid_to_bitfield<<<blocks, ngp::network::detail::n_threads_linear, 0, stream.stream>>>(base_grid_elements / 8u, density_grid.data(), density_grid_bitfield.data(), density_grid_mean.data());
        }
    }

    void InstantNGP::train(const std::int32_t iters) {
        for (std::int32_t i = 0; i < iters; ++i) {
            std::printf("Training iteration %d.\n", i);
            run_training_prep();
            trainer->optimizer.update_hyperparams(network_config.optimizer);
            ++training_step;
        }
    }

} // namespace ngp
