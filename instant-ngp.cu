#include "encoder.cuh"
#include "fully-fused-mlp.cuh"
#include "instant-ngp.h"
#include "optimizer.cuh"
#include <algorithm>
#include <limits>
#include <memory>
#include <sstream>
#include <stack>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace ngp::network {

    template <typename T>
    __global__ void extract_density(const std::uint32_t n_elements, const std::uint32_t density_stride, const std::uint32_t rgbd_stride, const T* __restrict__ density, T* __restrict__ rgbd) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;
        rgbd[i * rgbd_stride] = density[i * density_stride];
    }

    template <typename T>
    __global__ void extract_rgb(const std::uint32_t n_elements, const std::uint32_t rgb_stride, const std::uint32_t output_stride, const T* __restrict__ rgbd, T* __restrict__ rgb) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t elem_idx         = i / 3u;
        const std::uint32_t dim_idx          = i - elem_idx * 3u;
        rgb[elem_idx * rgb_stride + dim_idx] = rgbd[elem_idx * output_stride + dim_idx];
    }

    template <typename T>
    __global__ void add_density_gradient(const std::uint32_t n_elements, const std::uint32_t rgbd_stride, const T* __restrict__ rgbd, const std::uint32_t density_stride, T* __restrict__ density) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;
        density[i * density_stride] += rgbd[i * rgbd_stride + 3u];
    }

} // namespace ngp::network

#include "stb/stb_image_write.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <utility>

namespace ngp::network::detail {

    struct AuxStreamSlot final {
        AuxStreamSlot() {
            legacy::cuda_check(cudaStreamCreate(&stream));
            legacy::cuda_check(cudaEventCreate(&event));
        }

        ~AuxStreamSlot() {
            if (event) cudaEventDestroy(event);
            if (stream) cudaStreamDestroy(stream);
        }

        AuxStreamSlot& operator=(const AuxStreamSlot&) = delete;
        AuxStreamSlot(const AuxStreamSlot&)            = delete;

        cudaStream_t stream = {};
        cudaEvent_t event   = {};
    };

    std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot>>>& aux_stream_pools() {
        static auto* pools = new std::unordered_map<cudaStream_t, std::stack<std::unique_ptr<AuxStreamSlot>>>{};
        return *pools;
    }

    void free_aux_stream_pool(const cudaStream_t parent_stream) {
        legacy::check_or_throw(parent_stream != nullptr);
        aux_stream_pools().erase(parent_stream);
    }

    SyncedStreamReservation::SyncedStreamReservation(const cudaStream_t stream, const std::size_t n_streams) : main_stream{stream} {
        if (n_streams == 0u) throw std::runtime_error{"SyncedStreamReservation: must request at least one stream"};
        if (n_streams == 1u) return;
        if (n_streams != 2u) throw std::runtime_error{"SyncedStreamReservation: this repository only supports a single auxiliary stream"};

        legacy::check_or_throw(main_stream != nullptr);
        auto& pool = aux_stream_pools()[main_stream];
        if (pool.empty()) pool.push(std::make_unique<AuxStreamSlot>());
        aux_stream_slot = pool.top().release();
        pool.pop();
        aux_stream = aux_stream_slot->stream;
        legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, main_stream));
        legacy::cuda_check(cudaStreamWaitEvent(aux_stream_slot->stream, aux_stream_slot->event, 0));
    }

    SyncedStreamReservation::~SyncedStreamReservation() {
        if (!aux_stream_slot) return;

        legacy::cuda_check(cudaEventRecord(aux_stream_slot->event, aux_stream_slot->stream));
        legacy::cuda_check(cudaStreamWaitEvent(main_stream, aux_stream_slot->event, 0));
        auto& pools = aux_stream_pools();
        if (!pools.contains(main_stream)) std::terminate();
        pools[main_stream].push(std::unique_ptr<AuxStreamSlot>{aux_stream_slot});
        aux_stream_slot = nullptr;
        aux_stream      = nullptr;
    }

    SyncedStreamReservation& SyncedStreamReservation::operator=(SyncedStreamReservation&& other) noexcept {
        std::swap(aux_stream_slot, other.aux_stream_slot);
        std::swap(aux_stream, other.aux_stream);
        std::swap(main_stream, other.main_stream);
        return *this;
    }

    SyncedStreamReservation::SyncedStreamReservation(SyncedStreamReservation&& other) noexcept {
        *this = std::move(other);
    }

} // namespace ngp::network::detail


namespace ngp {

    inline constexpr std::uint32_t NERF_GRIDSIZE                = 128u;
    inline constexpr std::uint32_t NERF_GRID_N_CELLS            = NERF_GRIDSIZE * NERF_GRIDSIZE * NERF_GRIDSIZE;
    inline constexpr std::uint32_t NERF_STEPS                   = 1024u;
    inline constexpr float MIN_CONE_STEPSIZE                    = 1.73205080757f / static_cast<float>(NERF_STEPS);
    inline constexpr std::uint32_t N_MAX_RANDOM_SAMPLES_PER_RAY = 16u;
    inline constexpr float NERF_MIN_OPTICAL_THICKNESS           = 0.01f;

    struct BoundingBox final {
        __host__ __device__ BoundingBox() = default;
        __host__ __device__ BoundingBox(const legacy::math::vec3& min, const legacy::math::vec3& max) : min{min}, max{max} {}

        __device__ legacy::math::vec3 diag() const {
            return max - min;
        }

        __device__ legacy::math::vec3 relative_pos(const legacy::math::vec3& pos) const {
            return (pos - min) / diag();
        }

        __device__ legacy::math::vec2 ray_intersect(const legacy::math::vec3& pos, const legacy::math::vec3& dir) const {
            float tmin = (min.x - pos.x) / dir.x;
            float tmax = (max.x - pos.x) / dir.x;

            if (tmin > tmax) cuda::std::swap(tmin, tmax);

            float tymin = (min.y - pos.y) / dir.y;
            float tymax = (max.y - pos.y) / dir.y;

            if (tymin > tymax) cuda::std::swap(tymin, tymax);
            if (tmin > tymax || tymin > tmax) return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            if (tymin > tmin) tmin = tymin;
            if (tymax < tmax) tmax = tymax;

            float tzmin = (min.z - pos.z) / dir.z;
            float tzmax = (max.z - pos.z) / dir.z;

            if (tzmin > tzmax) cuda::std::swap(tzmin, tzmax);
            if (tmin > tzmax || tzmin > tmax) return {std::numeric_limits<float>::max(), std::numeric_limits<float>::max()};
            if (tzmin > tmin) tmin = tzmin;
            if (tzmax < tmax) tmax = tzmax;

            return {tmin, tmax};
        }

        __device__ bool contains(const legacy::math::vec3& pos) const {
            return pos.x >= min.x && pos.x <= max.x && pos.y >= min.y && pos.y <= max.y && pos.z >= min.z && pos.z <= max.z;
        }

        legacy::math::vec3 min = legacy::math::vec3(0.0f);
        legacy::math::vec3 max = legacy::math::vec3(1.0f);
    };

    struct NerfPosition final {
        __device__ NerfPosition() = default;
        __device__ NerfPosition(const legacy::math::vec3& pos, float dt) : p{pos} {}
        legacy::math::vec3 p = {};
    };

    struct NerfDirection final {
        __device__ NerfDirection() = default;
        __device__ NerfDirection(const legacy::math::vec3& dir, float dt) : d{dir} {}
        legacy::math::vec3 d = {};
    };

    struct NerfCoordinate final {
        __device__ NerfCoordinate() = default;
        __device__ NerfCoordinate(const legacy::math::vec3& pos, const legacy::math::vec3& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}

        __device__ void set(const legacy::math::vec3& pos, const legacy::math::vec3& dir, float dt) {
            this->dt  = dt;
            this->pos = NerfPosition{pos, dt};
            this->dir = NerfDirection{dir, dt};
        }

        NerfPosition pos  = {};
        float dt          = 0.0f;
        NerfDirection dir = {};
    };

    template <typename T>
    inline constexpr float default_loss_scale = 1.0f;

#ifdef __CUDACC__
    template <>
    inline constexpr float default_loss_scale<__half> = 128.0f;
#endif

    __device__ inline std::uint32_t morton3D(const std::uint32_t x, const std::uint32_t y, const std::uint32_t z) {
        std::uint32_t xx = x;
        std::uint32_t yy = y;
        std::uint32_t zz = z;
        xx               = (xx * 0x00010001u) & 0xFF0000FFu;
        xx               = (xx * 0x00000101u) & 0x0F00F00Fu;
        xx               = (xx * 0x00000011u) & 0xC30C30C3u;
        xx               = (xx * 0x00000005u) & 0x49249249u;
        yy               = (yy * 0x00010001u) & 0xFF0000FFu;
        yy               = (yy * 0x00000101u) & 0x0F00F00Fu;
        yy               = (yy * 0x00000011u) & 0xC30C30C3u;
        yy               = (yy * 0x00000005u) & 0x49249249u;
        zz               = (zz * 0x00010001u) & 0xFF0000FFu;
        zz               = (zz * 0x00000101u) & 0x0F00F00Fu;
        zz               = (zz * 0x00000011u) & 0xC30C30C3u;
        zz               = (zz * 0x00000005u) & 0x49249249u;
        return xx | (yy << 1u) | (zz << 2u);
    }

    __device__ inline std::uint32_t morton3D_invert(std::uint32_t value) {
        value = value & 0x49249249u;
        value = (value | (value >> 2u)) & 0xC30C30C3u;
        value = (value | (value >> 4u)) & 0x0F00F00Fu;
        value = (value | (value >> 8u)) & 0xFF0000FFu;
        value = (value | (value >> 16u)) & 0x0000FFFFu;
        return value;
    }

    template <typename RNG>
    __device__ legacy::math::vec3 random_val_3d(RNG& rng) {
        return {rng.next_float(), rng.next_float(), rng.next_float()};
    }

    __device__ inline float logistic(const float x) {
        return 1.0f / (1.0f + expf(-x));
    }

    inline mlp::Activation activation_from_config(const InstantNGP::ActivationMode activation) {
        switch (activation) {
        case InstantNGP::ActivationMode::None: return mlp::Activation::None;
        case InstantNGP::ActivationMode::ReLU: return mlp::Activation::ReLU;
        case InstantNGP::ActivationMode::Exponential: return mlp::Activation::Exponential;
        case InstantNGP::ActivationMode::Sigmoid: return mlp::Activation::Sigmoid;
        case InstantNGP::ActivationMode::Squareplus: return mlp::Activation::Squareplus;
        case InstantNGP::ActivationMode::Softplus: return mlp::Activation::Softplus;
        case InstantNGP::ActivationMode::Tanh: return mlp::Activation::Tanh;
        case InstantNGP::ActivationMode::LeakyReLU: return mlp::Activation::LeakyReLU;
        default: throw std::runtime_error{"Unsupported public activation mode."};
        }
    }

    inline __device__ legacy::Ray uv_to_ray(const legacy::math::vec2& uv, const legacy::math::ivec2& resolution, const float focal_length, const legacy::math::mat4x3& camera_matrix, const float near_distance = 0.0f) {
        legacy::math::vec3 dir    = {(uv.x - 0.5f) * static_cast<float>(resolution.x) / focal_length, (uv.y - 0.5f) * static_cast<float>(resolution.y) / focal_length, 1.0f};
        dir                       = camera_matrix[0] * dir.x + camera_matrix[1] * dir.y + camera_matrix[2] * dir.z;
        legacy::math::vec3 origin = camera_matrix[3];
        origin += dir * near_distance;
        return {origin, dir};
    }

    inline __device__ float network_to_density(const float val) {
        return expf(val);
    }

    struct DensityGridReduceOp final {
        std::uint32_t base_grid_elements = 0u;

        __device__ float operator()(const float val) const {
            return fmaxf(val, 0.0f) / static_cast<float>(base_grid_elements);
        }
    };

    struct SumIdentityOp final {
        __device__ float operator()(const float val) const {
            return val;
        }
    };

    inline __device__ legacy::math::vec3 warp_position(const legacy::math::vec3& pos, const BoundingBox& aabb) {
        return aabb.relative_pos(pos);
    }

    inline __device__ legacy::math::vec3 warp_direction(const legacy::math::vec3& dir) {
        return (dir + 1.0f) * 0.5f;
    }

    inline __device__ std::uint32_t density_grid_idx_at(const legacy::math::vec3& pos) {
        const legacy::math::ivec3 i = pos * static_cast<float>(NERF_GRIDSIZE);
        if (i.x < 0 || i.x >= static_cast<int>(NERF_GRIDSIZE) || i.y < 0 || i.y >= static_cast<int>(NERF_GRIDSIZE) || i.z < 0 || i.z >= static_cast<int>(NERF_GRIDSIZE)) return 0xFFFFFFFFu;
        return morton3D(i.x, i.y, i.z);
    }

    inline __device__ bool density_grid_occupied_at(const legacy::math::vec3& pos, const std::uint8_t* density_grid_bitfield) {
        const std::uint32_t idx = density_grid_idx_at(pos);
        if (idx == 0xFFFFFFFFu) return false;
        return density_grid_bitfield[idx / 8u] & (1u << (idx % 8u));
    }

    inline __device__ float advance_n_steps(const float t, const float n) {
        return t + n * MIN_CONE_STEPSIZE;
    }

    inline __device__ float advance_to_next_voxel(const float t, const legacy::math::vec3& pos, const legacy::math::vec3& dir, const legacy::math::vec3& idir) {
        const legacy::math::vec3 p = static_cast<float>(NERF_GRIDSIZE) * (pos - 0.5f);
        const float tx             = (floorf(p.x + 0.5f + 0.5f * legacy::math::sign(dir.x)) - p.x) * idir.x;
        const float ty             = (floorf(p.y + 0.5f + 0.5f * legacy::math::sign(dir.y)) - p.y) * idir.y;
        const float tz             = (floorf(p.z + 0.5f + 0.5f * legacy::math::sign(dir.z)) - p.z) * idir.z;
        const float t_target       = t + fmaxf(fminf(fminf(tx, ty), tz) / static_cast<float>(NERF_GRIDSIZE), 0.0f);
        return t + ceilf(fmaxf((t_target - t) / MIN_CONE_STEPSIZE, 0.5f)) * MIN_CONE_STEPSIZE;
    }

    inline __device__ legacy::math::vec2 nerf_random_image_pos_training(legacy::math::pcg32& rng, const legacy::math::ivec2& resolution, const bool snap_to_pixel_centers) {
        legacy::math::vec2 uv = {rng.next_float(), rng.next_float()};
        if (snap_to_pixel_centers) uv = (legacy::math::vec2(ngp::legacy::math::clamp(legacy::math::ivec2(uv * legacy::math::vec2(resolution)), 0, resolution - 1)) + 0.5f) / legacy::math::vec2(resolution);
        return uv;
    }

    inline __device__ std::uint32_t image_idx(const std::uint32_t base_idx, const std::uint32_t n_rays, const std::uint32_t n_training_images) {
        return ((base_idx * n_training_images) / n_rays) % n_training_images;
    }

    inline __host__ __device__ legacy::math::vec3 srgb_to_linear(const legacy::math::vec3& x) {
        return {
            x.x <= 0.04045f ? x.x / 12.92f : powf((x.x + 0.055f) / 1.055f, 2.4f),
            x.y <= 0.04045f ? x.y / 12.92f : powf((x.y + 0.055f) / 1.055f, 2.4f),
            x.z <= 0.04045f ? x.z / 12.92f : powf((x.z + 0.055f) / 1.055f, 2.4f),
        };
    }

    inline __host__ __device__ legacy::math::vec3 linear_to_srgb(const legacy::math::vec3& x) {
        return {
            x.x < 0.0031308f ? 12.92f * x.x : 1.055f * powf(x.x, 0.41666f) - 0.055f,
            x.y < 0.0031308f ? 12.92f * x.y : 1.055f * powf(x.y, 0.41666f) - 0.055f,
            x.z < 0.0031308f ? 12.92f * x.z : 1.055f * powf(x.z, 0.41666f) - 0.055f,
        };
    }

    inline __host__ __device__ legacy::math::vec4 read_rgba(const legacy::math::ivec2& px, const legacy::math::ivec2& resolution, const std::uint8_t* pixels, const std::uint32_t img = 0u) {
        const std::uint32_t rgba32 = reinterpret_cast<const std::uint32_t*>(pixels)[px.x + px.y * resolution.x + img * static_cast<std::uint64_t>(resolution.x) * resolution.y];
        legacy::math::vec4 result  = {
            static_cast<float>((rgba32 & 0x000000FFu) >> 0u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0x0000FF00u) >> 8u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0x00FF0000u) >> 16u) * (1.0f / 255.0f),
            static_cast<float>((rgba32 & 0xFF000000u) >> 24u) * (1.0f / 255.0f),
        };
        const legacy::math::vec3 linear_rgb = srgb_to_linear({result.x, result.y, result.z}) * result.a;
        result.x                            = linear_rgb.x;
        result.y                            = linear_rgb.y;
        result.z                            = linear_rgb.z;
        return result;
    }

    inline __device__ float network_to_rgb_derivative(const float val) {
        const float rgb = logistic(val);
        return rgb * (1.0f - rgb);
    }

    template <typename T>
    __device__ legacy::math::vec3 network_to_rgb_vec(const T& val) {
        return {
            logistic(static_cast<float>(val[0])),
            logistic(static_cast<float>(val[1])),
            logistic(static_cast<float>(val[2])),
        };
    }

    inline __host__ __device__ legacy::math::vec3 clamp_rgb01(const legacy::math::vec3& value) {
        return {
            cuda::std::clamp(value.x, 0.0f, 1.0f),
            cuda::std::clamp(value.y, 0.0f, 1.0f),
            cuda::std::clamp(value.z, 0.0f, 1.0f),
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
        result   = static_cast<T>(static_cast<float>(result) * static_cast<float>(n_input_elements) / static_cast<float>(n_elements));
        inout[i] = result;
    }

    template <typename T>
    __device__ T warp_reduce(T val) {
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

        const std::uint32_t lane = threadIdx.x % warpSize;
        const std::uint32_t wid  = threadIdx.x / warpSize;

        T_OUT val = {};
        if constexpr (std::is_same_v<std::decay_t<T>, __half> || std::is_same_v<std::decay_t<T>, half>) {
            if (i < n_elements) {
                half vals[8];
                *reinterpret_cast<int4*>(&vals[0]) = *(reinterpret_cast<const int4*>(input) + i + block_offset);
                val                                = fun(static_cast<T>(vals[0])) + fun(static_cast<T>(vals[1])) + fun(static_cast<T>(vals[2])) + fun(static_cast<T>(vals[3])) + fun(static_cast<T>(vals[4])) + fun(static_cast<T>(vals[5])) + fun(static_cast<T>(vals[6])) + fun(static_cast<T>(vals[7]));
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, float>) {
            if (i < n_elements) {
                const float4 vals = *(reinterpret_cast<const float4*>(input) + i + block_offset);
                val               = fun(static_cast<T>(vals.x)) + fun(static_cast<T>(vals.y)) + fun(static_cast<T>(vals.z)) + fun(static_cast<T>(vals.w));
            }
        } else if constexpr (std::is_same_v<std::decay_t<T>, double>) {
            if (i < n_elements) {
                const double2 vals = *(reinterpret_cast<const double2*>(input) + i + block_offset);
                val                = fun(static_cast<T>(vals.x)) + fun(static_cast<T>(vals.y));
            }
        } else {
            static_assert(std::is_same_v<std::decay_t<T>, __half> || std::is_same_v<std::decay_t<T>, half> || std::is_same_v<std::decay_t<T>, float> || std::is_same_v<std::decay_t<T>, double>, "block_reduce only supports __half, float, and double.");
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
        constexpr std::uint32_t threads      = 1024u;
        const std::uint32_t n_elems_per_load = 16u / sizeof(T);

        if (n_elements % n_elems_per_load != 0u) throw std::runtime_error{"Number of bytes to reduce_sum must be a multiple of 16."};
        if ((reinterpret_cast<std::size_t>(device_pointer) % 16u) != 0u) throw std::runtime_error{"Can only reduce_sum on 16-byte aligned memory."};

        n_elements /= n_elems_per_load;
        const std::uint32_t blocks = (n_elements + threads - 1u) / threads;
        block_reduce<T, T_OUT, F><<<blocks * n_sums, threads, 0, stream>>>(n_elements, fun, device_pointer, workspace, blocks);
    }

    inline std::uint32_t reduce_sum_workspace_size(const std::uint32_t n_elements) {
        return (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
    }

    __global__ void mark_untrained_density_grid(const std::uint32_t n_elements, float* __restrict__ grid_out, const std::uint32_t n_training_images, const InstantNGP::DatasetState::DeviceData::GpuFrame* __restrict__ frames) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t x = morton3D_invert(i >> 0u);
        const std::uint32_t y = morton3D_invert(i >> 1u);
        const std::uint32_t z = morton3D_invert(i >> 2u);

        constexpr float voxel_size   = 1.0f / static_cast<float>(NERF_GRIDSIZE);
        const legacy::math::vec3 pos = legacy::math::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} / static_cast<float>(NERF_GRIDSIZE);

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

        constexpr std::uint32_t min_count = 1u;
        std::uint32_t count               = 0u;

        for (std::uint32_t j = 0u; j < n_training_images && count < min_count; ++j) {
            const auto& frame = frames[j];
            const auto& xform = frame.camera;

            for (const legacy::math::vec3& corner : corners) {
                const legacy::math::vec3 dir = ngp::legacy::math::normalize(corner - xform[3]);
                if (ngp::legacy::math::dot(dir, xform[2]) < 1e-4f) continue;

                const legacy::math::vec3 offset = corner - xform[3];
                legacy::math::vec3 camera_dir   = {
                    ngp::legacy::math::dot(xform[0], offset),
                    ngp::legacy::math::dot(xform[1], offset),
                    ngp::legacy::math::dot(xform[2], offset),
                };
                camera_dir /= camera_dir.z;
                const legacy::math::vec2 uv = legacy::math::vec2{camera_dir.x, camera_dir.y} * frame.focal_length / legacy::math::vec2(frame.resolution) + legacy::math::vec2(0.5f);
                const legacy::Ray ray       = uv_to_ray(uv, frame.resolution, frame.focal_length, xform);
                if (ngp::legacy::math::distance(ngp::legacy::math::normalize(ray.d), dir) < 1e-3f && uv.x > 0.0f && uv.y > 0.0f && uv.x < 1.0f && uv.y < 1.0f) {
                    ++count;
                    break;
                }
            }
        }

        grid_out[i] = count >= min_count ? 0.0f : -1.0f;
    }

    __global__ void generate_grid_samples_nerf_nonuniform(const std::uint32_t n_elements, legacy::math::pcg32 rng, const std::uint32_t step, BoundingBox aabb, const float* __restrict__ grid_in, NerfPosition* __restrict__ out, std::uint32_t* __restrict__ indices, const float thresh) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        rng.advance(i * 4u);
        std::uint32_t idx = 0u;
        for (std::uint32_t j = 0u; j < 10u; ++j) {
            idx = ((i + step * n_elements) * 56924617u + j * 19349663u + 96925573u) % NERF_GRID_N_CELLS;
            if (grid_in[idx] > thresh) break;
        }

        const std::uint32_t x        = morton3D_invert(idx >> 0u);
        const std::uint32_t y        = morton3D_invert(idx >> 1u);
        const std::uint32_t z        = morton3D_invert(idx >> 2u);
        const legacy::math::vec3 pos = (legacy::math::vec3{static_cast<float>(x), static_cast<float>(y), static_cast<float>(z)} + random_val_3d(rng)) / static_cast<float>(NERF_GRIDSIZE);

        out[i]     = {warp_position(pos, aabb), MIN_CONE_STEPSIZE};
        indices[i] = idx;
    }

    __global__ void splat_grid_samples_nerf_max_nearest_neighbor(const std::uint32_t n_elements, const std::uint32_t* __restrict__ indices, const __half* __restrict__ network_output, float* __restrict__ grid_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const std::uint32_t idx = indices[i];
        const float mlp         = network_to_density(network_output[i]);
        const float thickness   = mlp * MIN_CONE_STEPSIZE;
        atomicMax(reinterpret_cast<std::uint32_t*>(&grid_out[idx]), __float_as_uint(thickness));
    }

    __global__ void ema_grid_samples_nerf(const std::uint32_t n_elements, const float decay, float* __restrict__ grid_out, const float* __restrict__ grid_in) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        const float importance = grid_in[i];
        const float prev_val   = grid_out[i];
        const float val        = prev_val < 0.0f ? prev_val : fmaxf(prev_val * decay, importance);
        grid_out[i]            = val;
    }

    __global__ void grid_to_bitfield(const std::uint32_t n_elements, const float* __restrict__ grid, std::uint8_t* __restrict__ grid_bitfield, const float* __restrict__ mean_density_ptr) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_elements) return;

        std::uint8_t bits  = 0u;
        const float thresh = fminf(NERF_MIN_OPTICAL_THICKNESS, *mean_density_ptr);

        TCNN_PRAGMA_UNROLL
        for (std::uint8_t j = 0u; j < 8u; ++j) bits |= grid[i * 8u + j] > thresh ? (static_cast<std::uint8_t>(1u) << j) : 0u;

        grid_bitfield[i] = bits;
    }

    __global__ void generate_training_samples_nerf(const std::uint32_t n_rays, const BoundingBox aabb, const std::uint32_t max_samples, legacy::math::pcg32 rng, std::uint32_t* __restrict__ ray_counter, std::uint32_t* __restrict__ numsteps_counter, std::uint32_t* __restrict__ ray_indices_out, legacy::Ray* __restrict__ rays_out_unnormalized, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<NerfCoordinate> coords_out, const std::uint32_t n_training_images,
        const InstantNGP::DatasetState::DeviceData::GpuFrame* __restrict__ frames, const std::uint8_t* __restrict__ density_grid, const bool snap_to_pixel_centers) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_rays) return;

        const std::uint32_t img              = image_idx(i, n_rays, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        rng.advance(i * N_MAX_RANDOM_SAMPLES_PER_RAY);
        const legacy::math::vec2 uv = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);

        const float focal_length         = frame.focal_length;
        const legacy::math::mat4x3 xform = frame.camera;

        legacy::Ray ray_unnormalized = uv_to_ray(uv, resolution, focal_length, xform);
        if (ray_unnormalized.d == legacy::math::vec3(0.0f)) ray_unnormalized = {xform[3], xform[2]};

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);

        const float startt            = advance_n_steps(tminmax.x, rng.next_float());
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t j = 0u;
        float t         = startt;
        legacy::math::vec3 pos;

        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && j < NERF_STEPS) {
            constexpr float dt = MIN_CONE_STEPSIZE;
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
            constexpr float dt = MIN_CONE_STEPSIZE;
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, dt);
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void compute_loss_kernel_train_nerf(const std::uint32_t n_rays, const BoundingBox aabb, legacy::math::pcg32 rng, const std::uint32_t max_samples_compacted, const std::uint32_t* __restrict__ rays_counter, float loss_scale, const int padded_output_width, const std::uint32_t n_training_images, const InstantNGP::DatasetState::DeviceData::GpuFrame* __restrict__ frames, const __half* network_output, std::uint32_t* __restrict__ numsteps_counter,
        const std::uint32_t* __restrict__ ray_indices_in, const legacy::Ray* __restrict__ rays_in_unnormalized, std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const NerfCoordinate> coords_in, legacy::PitchedPtr<NerfCoordinate> coords_out, __half* dloss_doutput, float* __restrict__ loss_output, const bool snap_to_pixel_centers, const float* __restrict__ mean_density_ptr, const float near_distance) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= *rays_counter) return;

        std::uint32_t numsteps = numsteps_in[i * 2u + 0u];
        std::uint32_t base     = numsteps_in[i * 2u + 1u];

        coords_in += base;
        network_output += base * padded_output_width;

        float T                 = 1.0f;
        constexpr float epsilon = 1e-4f;

        auto rgb_ray                     = legacy::math::vec3(0.0f);
        std::uint32_t compacted_numsteps = 0u;
        const legacy::math::vec3 ray_o   = rays_in_unnormalized[i].o;
        for (; compacted_numsteps < numsteps; ++compacted_numsteps) {
            if (T < epsilon) break;

            const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const float dt                                            = coords_in.ptr->dt;
            const float density                                       = network_to_density(local_network_output[3]);

            const float alpha  = 1.0f - __expf(-density * dt);
            const float weight = alpha * T;
            rgb_ray += weight * rgb;
            T *= (1.0f - alpha);

            network_output += padded_output_width;
            coords_in += 1u;
        }

        const std::uint32_t ray_idx = ray_indices_in[i];
        rng.advance(ray_idx * N_MAX_RANDOM_SAMPLES_PER_RAY);

        const std::uint32_t img              = image_idx(ray_idx, n_rays, n_training_images);
        const auto& frame                    = frames[img];
        const legacy::math::ivec2 resolution = frame.resolution;

        const legacy::math::vec2 uv               = nerf_random_image_pos_training(rng, resolution, snap_to_pixel_centers);
        const legacy::math::vec3 background_color = random_val_3d(rng);
        const legacy::math::ivec2 texel           = ngp::legacy::math::clamp(legacy::math::ivec2(uv * legacy::math::vec2(resolution)), 0, resolution - 1);
        const legacy::math::vec4 texsamp          = read_rgba(texel, resolution, frame.pixels);
        const legacy::math::vec3 rgbtarget        = linear_to_srgb(legacy::math::vec3{texsamp.x, texsamp.y, texsamp.z} + (1.0f - texsamp.a) * srgb_to_linear(background_color));

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

        const legacy::math::vec3 difference = rgb_ray - rgbtarget;
        const legacy::math::vec3 loss       = difference * difference;
        const legacy::math::vec3 gradient   = 2.0f * difference;
        const float mean_loss               = ngp::legacy::math::mean(loss);
        if (loss_output) loss_output[i] = mean_loss / static_cast<float>(n_rays);

        loss_scale /= static_cast<float>(n_rays);

        const float output_l1_reg_density = *mean_density_ptr < NERF_MIN_OPTICAL_THICKNESS ? 1e-4f : 0.0f;

        legacy::math::vec3 rgb_ray2 = {0.0f, 0.0f, 0.0f};
        T                           = 1.0f;
        for (std::uint32_t j = 0u; j < compacted_numsteps; ++j) {
            NerfCoordinate* coord_out      = coords_out(j);
            const NerfCoordinate* coord_in = coords_in(j);
            *coord_out                     = *coord_in;

            const legacy::math::vec3 pos                              = aabb.min + coord_in->pos.p * aabb.diag();
            const float depth                                         = ngp::legacy::math::distance(pos, ray_o);
            const float dt                                            = coord_in->dt;
            const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
            const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
            const float density                                       = network_to_density(local_network_output[3]);
            const float alpha                                         = 1.0f - __expf(-density * dt);
            const float weight                                        = alpha * T;
            rgb_ray2 += weight * rgb;
            T *= (1.0f - alpha);

            const legacy::math::vec3 suffix        = rgb_ray - rgb_ray2;
            const legacy::math::vec3 dloss_by_drgb = weight * gradient;

            legacy::math::tvec<__half, 4u> local_dL_doutput{};
            local_dL_doutput[0] = loss_scale * (dloss_by_drgb.x * network_to_rgb_derivative(local_network_output[0]));
            local_dL_doutput[1] = loss_scale * (dloss_by_drgb.y * network_to_rgb_derivative(local_network_output[1]));
            local_dL_doutput[2] = loss_scale * (dloss_by_drgb.z * network_to_rgb_derivative(local_network_output[2]));

            const float density_derivative = expf(cuda::std::clamp(static_cast<float>(local_network_output[3]), -15.0f, 15.0f));
            const float dloss_by_dmlp      = density_derivative * (dt * ngp::legacy::math::dot(gradient, T * rgb - suffix));
            local_dL_doutput[3]            = loss_scale * dloss_by_dmlp + (static_cast<float>(local_network_output[3]) < 0.0f ? -output_l1_reg_density : 0.0f) + (static_cast<float>(local_network_output[3]) > -10.0f && depth < near_distance ? 1e-4f : 0.0f);

            *reinterpret_cast<legacy::math::tvec<__half, 4u>*>(dloss_doutput) = local_dL_doutput;

            dloss_doutput += padded_output_width;
            network_output += padded_output_width;
        }
    }

    __global__ void generate_validation_samples_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, BoundingBox aabb, const std::uint32_t max_samples, std::uint32_t* __restrict__ sample_counter, std::uint32_t* __restrict__ overflow_counter, std::uint32_t* __restrict__ numsteps_out, legacy::PitchedPtr<NerfCoordinate> coords_out, InstantNGP::DatasetState::DeviceData::GpuFrame frame, const std::uint8_t* __restrict__ density_grid) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        numsteps_out[i * 2u + 0u] = 0u;
        numsteps_out[i * 2u + 1u] = 0u;

        const std::uint32_t global_pixel     = pixel_offset + i;
        const legacy::math::ivec2 resolution = frame.resolution;
        const legacy::math::ivec2 px         = {static_cast<int>(global_pixel % static_cast<std::uint32_t>(resolution.x)), static_cast<int>(global_pixel / static_cast<std::uint32_t>(resolution.x))};
        const legacy::math::vec2 uv          = (legacy::math::vec2{static_cast<float>(px.x) + 0.5f, static_cast<float>(px.y) + 0.5f}) / legacy::math::vec2(resolution);
        const legacy::math::mat4x3 xform     = frame.camera;

        legacy::Ray ray_unnormalized = uv_to_ray(uv, resolution, frame.focal_length, xform);
        if (ray_unnormalized.d == legacy::math::vec3(0.0f)) return;

        const legacy::math::vec3 ray_d_normalized = ngp::legacy::math::normalize(ray_unnormalized.d);
        legacy::math::vec2 tminmax                = aabb.ray_intersect(ray_unnormalized.o, ray_d_normalized);
        tminmax.x                                 = fmaxf(tminmax.x, 0.0f);
        if (tminmax.y <= tminmax.x) return;

        float t                       = advance_n_steps(tminmax.x, 0.5f);
        const legacy::math::vec3 idir = legacy::math::vec3(1.0f) / ray_d_normalized;

        std::uint32_t numsteps = 0u;
        legacy::math::vec3 pos;
        while (aabb.contains(pos = ray_unnormalized.o + t * ray_d_normalized) && numsteps < NERF_STEPS) {
            constexpr float dt = MIN_CONE_STEPSIZE;
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
            constexpr float dt = MIN_CONE_STEPSIZE;
            if (density_grid_occupied_at(pos, density_grid)) {
                coords_out(j)->set(warp_position(pos, aabb), warped_dir, dt);
                ++j;
                t += dt;
            } else {
                t = advance_to_next_voxel(t, pos, ray_d_normalized, idir);
            }
        }
    }

    __global__ void composite_validation_kernel_nerf(const std::uint32_t n_pixels, const std::uint32_t pixel_offset, const std::uint32_t* __restrict__ numsteps_in, legacy::PitchedPtr<const NerfCoordinate> coords_in, const __half* __restrict__ network_output, const std::uint32_t padded_output_width, const legacy::math::vec3 background_color, legacy::math::vec3* __restrict__ image_out) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n_pixels) return;

        const std::uint32_t global_pixel = pixel_offset + i;
        const std::uint32_t numsteps     = numsteps_in[i * 2u + 0u];
        const std::uint32_t base         = numsteps_in[i * 2u + 1u];
        float T                          = 1.0f;
        auto rgb_ray                     = legacy::math::vec3(0.0f);

        if (numsteps > 0u) {
            coords_in += base;
            network_output += base * padded_output_width;

            for (std::uint32_t j = 0u; j < numsteps; ++j) {
                const legacy::math::tvec<__half, 4u> local_network_output = *reinterpret_cast<const legacy::math::tvec<__half, 4u>*>(network_output);
                const legacy::math::vec3 rgb                              = network_to_rgb_vec(local_network_output);
                const float dt                                            = coords_in.ptr->dt;
                const float density                                       = network_to_density(local_network_output[3]);
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

    struct InstantNGP::ModelState final {
        struct Layout final {
            std::uint32_t pos_input_width   = 0u;
            std::uint32_t pos_output_width  = 0u;
            legacy::MatrixLayout pos_layout = legacy::CM;
            std::size_t pos_param_count     = 0u;
        };

        struct ParamLayout final {
            std::size_t density_network = 0u;
            std::size_t rgb_network     = 0u;
            std::size_t pos_encoding    = 0u;
            std::size_t total           = 0u;
        };

        std::variant<encoding::GridEncodingTemplated<__half, 3u, 1u>, encoding::GridEncodingTemplated<__half, 3u, 2u>, encoding::GridEncodingTemplated<__half, 3u, 4u>, encoding::GridEncodingTemplated<__half, 3u, 8u>> pos_encoding;
        encoding::SphericalHarmonicsEncoding<__half> dir_encoding;
        mlp::FullyFusedMLP<__half, density_network_width> density_network;
        mlp::FullyFusedMLP<__half, rgb_network_width> rgb_network;
        Layout layout      = {};
        ParamLayout params = {};
    };

    struct InstantNGP::ModelScratch final {
        legacy::GPUMatrixDynamic<__half> density_network_input                     = {};
        legacy::GPUMatrixDynamic<__half> density_network_output                    = {};
        legacy::GPUMatrixDynamic<__half> rgb_network_input                         = {};
        legacy::GPUMatrixDynamic<__half> rgb_network_output                        = {};
        legacy::GPUMatrixDynamic<__half> dL_drgb                                   = {};
        legacy::GPUMatrixDynamic<__half> dL_drgb_input                             = {};
        legacy::GPUMatrixDynamic<__half> dL_ddensity_input                         = {};
        mlp::FullyFusedMLP<__half, density_network_width>::Scratch density_network = {};
        mlp::FullyFusedMLP<__half, rgb_network_width>::Scratch rgb_network         = {};
    };

    void InstantNGP::density(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>& output) const {
        if (!model) throw std::runtime_error{"Network model must be initialized before density inference."};
        if (input.layout() != legacy::CM) throw std::runtime_error{"model density input must be in column major format."};

        auto& current_model            = *model;
        const std::uint32_t batch_size = output.n();
        legacy::GPUMatrixDynamic<__half> density_input{current_model.layout.pos_output_width, batch_size, stream, current_model.layout.pos_layout};
        std::visit([&](auto& impl) { impl.encode(stream, input.slice_rows(0u, current_model.layout.pos_input_width), density_input); }, current_model.pos_encoding);
        current_model.density_network.inference(stream, density_input, output);
    }

    void InstantNGP::inference(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>& output) const {
        if (!model) throw std::runtime_error{"Network model must be initialized before inference."};

        auto& current_model                           = *model;
        const std::uint32_t model_input_width         = train_plan.network.dir_offset + current_model.dir_encoding.input_width;
        const std::uint32_t model_padded_output_width = std::max(current_model.rgb_network.padded_output_width, 4u);
        legacy::check_or_throw(input.m() == model_input_width);
        legacy::check_or_throw(output.m() == model_padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());

        const std::uint32_t batch_size = input.n();
        legacy::GPUMatrixDynamic<__half> density_input{current_model.layout.pos_output_width, batch_size, stream, current_model.layout.pos_layout};
        legacy::GPUMatrixDynamic<__half> rgb_input{current_model.rgb_network.input_width, batch_size, stream, current_model.dir_encoding.preferred_output_layout};
        legacy::GPUMatrixDynamic<__half> density_output = rgb_input.slice_rows(0u, current_model.density_network.padded_output_width);
        legacy::GPUMatrixDynamic<__half> rgb_output{output.data(), current_model.rgb_network.padded_output_width, batch_size, output.layout()};

        std::visit([&](auto& impl) { impl.encode(stream, input.slice_rows(0u, current_model.layout.pos_input_width), density_input); }, current_model.pos_encoding);
        current_model.density_network.inference(stream, density_input, density_output);

        auto dir_output = rgb_input.slice_rows(current_model.density_network.padded_output_width, current_model.dir_encoding.output_width);
        current_model.dir_encoding.encode(stream, input.slice_rows(train_plan.network.dir_offset, current_model.dir_encoding.input_width), dir_output);
        current_model.rgb_network.inference(stream, rgb_input, rgb_output);

        if (batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            ngp::network::extract_density<__half><<<blocks, network::detail::n_threads_linear, 0, stream>>>(batch_size, density_output.layout() == legacy::AoS ? density_output.stride() : 1u, output.layout() == legacy::AoS ? model_padded_output_width : 1u, density_output.data(), output.data() + 3u * (output.layout() == legacy::AoS ? 1u : batch_size));
        }
    }

    void InstantNGP::forward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, legacy::GPUMatrixDynamic<__half>* output) {
        if (!model || !model_scratch) throw std::runtime_error{"Network model must be initialized before forward."};

        auto& current_model                           = *model;
        auto& scratch                                 = *model_scratch;
        const std::uint32_t model_input_width         = train_plan.network.dir_offset + current_model.dir_encoding.input_width;
        const std::uint32_t model_padded_output_width = std::max(current_model.rgb_network.padded_output_width, 4u);
        legacy::check_or_throw(input.m() == model_input_width);
        legacy::check_or_throw(!output || output->m() == model_padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(!output || input.n() == output->n());

        const std::uint32_t batch_size = input.n();
        std::visit([&](auto& impl) { impl.encode(stream, input.slice_rows(0u, current_model.layout.pos_input_width), scratch.density_network_input); }, current_model.pos_encoding);
        current_model.density_network.forward(stream, scratch.density_network_input, &scratch.density_network_output, scratch.density_network);

        auto dir_output = scratch.rgb_network_input.slice_rows(current_model.density_network.padded_output_width, current_model.dir_encoding.output_width);
        current_model.dir_encoding.encode(stream, input.slice_rows(train_plan.network.dir_offset, current_model.dir_encoding.input_width), dir_output);

        if (output) scratch.rgb_network_output = ngp::legacy::GPUMatrixDynamic<__half>{output->data(), current_model.rgb_network.padded_output_width, batch_size, output->layout()};
        current_model.rgb_network.forward(stream, scratch.rgb_network_input, output ? &scratch.rgb_network_output : nullptr, scratch.rgb_network);

        if (output && batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            ngp::network::extract_density<__half><<<blocks, network::detail::n_threads_linear, 0, stream>>>(batch_size, current_model.dir_encoding.preferred_output_layout == legacy::AoS ? scratch.density_network_output.stride() : 1u, model_padded_output_width, scratch.density_network_output.data(), output->data() + 3u);
        }
    }

    void InstantNGP::backward(cudaStream_t stream, const legacy::GPUMatrixDynamic<float>& input, const legacy::GPUMatrixDynamic<__half>& output, const legacy::GPUMatrixDynamic<__half>& dL_doutput) {
        if (!model || !model_scratch) throw std::runtime_error{"Network model must be initialized before backward."};

        auto& current_model                                         = *model;
        auto& scratch                                               = *model_scratch;
        constexpr network::detail::GradientMode model_gradient_mode = network::detail::GradientMode::Overwrite;
        const std::uint32_t model_input_width                       = train_plan.network.dir_offset + current_model.dir_encoding.input_width;
        const std::uint32_t model_padded_output_width               = std::max(current_model.rgb_network.padded_output_width, 4u);
        legacy::check_or_throw(input.m() == model_input_width);
        legacy::check_or_throw(output.m() == model_padded_output_width);
        legacy::check_or_throw(dL_doutput.m() == model_padded_output_width);
        legacy::check_or_throw(input.n() % network::detail::batch_size_granularity == 0u);
        legacy::check_or_throw(input.n() == output.n());
        legacy::check_or_throw(input.n() == dL_doutput.n());

        const std::uint32_t batch_size = input.n();
        legacy::cuda_check(cudaMemsetAsync(scratch.dL_drgb.data(), 0, scratch.dL_drgb.n_bytes(), stream));

        if (batch_size > 0u) {
            const std::uint32_t rgb_elements = batch_size * 3u;
            const std::uint32_t blocks       = (rgb_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            ngp::network::extract_rgb<__half><<<blocks, network::detail::n_threads_linear, 0, stream>>>(rgb_elements, scratch.dL_drgb.m(), dL_doutput.m(), dL_doutput.data(), scratch.dL_drgb.data());
        }

        const legacy::GPUMatrixDynamic<__half> rgb_output{reinterpret_cast<__half*>(output.data()), current_model.rgb_network.padded_output_width, batch_size, output.layout()};
        current_model.rgb_network.backward(stream, scratch.rgb_network, scratch.rgb_network_input, rgb_output, scratch.dL_drgb, &scratch.dL_drgb_input, model_gradient_mode);

        auto dL_ddensity_output = scratch.dL_drgb_input.slice_rows(0u, current_model.density_network.padded_output_width);
        if (batch_size > 0u) {
            const std::uint32_t blocks = (batch_size + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
            ngp::network::add_density_gradient<__half><<<blocks, network::detail::n_threads_linear, 0, stream>>>(batch_size, dL_doutput.m(), dL_doutput.data(), dL_ddensity_output.layout() == legacy::RM ? 1u : dL_ddensity_output.stride(), dL_ddensity_output.data());
        }

        if (current_model.layout.pos_param_count > 0u) {
            current_model.density_network.backward(stream, scratch.density_network, scratch.density_network_input, scratch.density_network_output, dL_ddensity_output, &scratch.dL_ddensity_input, model_gradient_mode);
            std::visit([&](auto& impl) { impl.backward(stream, input.slice_rows(0u, current_model.layout.pos_input_width), scratch.dL_ddensity_input, model_gradient_mode); }, current_model.pos_encoding);
        } else {
            current_model.density_network.backward(stream, scratch.density_network, scratch.density_network_input, scratch.density_network_output, dL_ddensity_output, nullptr, model_gradient_mode);
        }
    }

    InstantNGP::InstantNGP(const NetworkConfig& network_config) {
        this->network_config                   = network_config;
        train_plan                             = {};
        train_plan.training.batch_size         = 1u << 18;
        train_plan.training.floats_per_coord   = sizeof(NerfCoordinate) / sizeof(float);
        train_plan.validation.floats_per_coord = train_plan.training.floats_per_coord;
        train_plan.validation.max_samples      = train_plan.validation.tile_rays * train_plan.validation.max_samples_per_ray;

        constexpr std::uint32_t n_pos_dims     = sizeof(NerfPosition) / sizeof(float);
        train_plan.network.n_pos_dims          = n_pos_dims;
        train_plan.network.n_dir_dims          = 3u;
        train_plan.network.dir_offset          = n_pos_dims + 1u;
        train_plan.network.density_output_dims = 16u;
        train_plan.network.rgb_output_dims     = 3u;

        if (this->network_config.encoding.n_levels == 0u) throw std::runtime_error{"HashGrid encoding requires at least one level."};
        if (this->network_config.encoding.n_levels > encoding::max_n_levels) throw std::runtime_error{"HashGrid encoding n_levels exceeds the supported maximum."};
        if (this->network_config.encoding.n_features_per_level != 1u && this->network_config.encoding.n_features_per_level != 2u && this->network_config.encoding.n_features_per_level != 4u && this->network_config.encoding.n_features_per_level != 8u) throw std::runtime_error{"HashGrid encoding n_features_per_level must be 1, 2, 4, or 8."};
        const std::uint64_t encoding_output_dims_64 = static_cast<std::uint64_t>(this->network_config.encoding.n_levels) * this->network_config.encoding.n_features_per_level;
        if (encoding_output_dims_64 > std::numeric_limits<std::uint32_t>::max()) throw std::runtime_error{"HashGrid encoding output width exceeds uint32 range."};
        const std::uint32_t encoding_output_dims = static_cast<std::uint32_t>(encoding_output_dims_64);
        if (encoding_output_dims % 16u != 0u) {
            std::ostringstream stream;
            stream << "HashGrid encoding output width must be a multiple of 16 after encoder padding removal: n_levels=" << this->network_config.encoding.n_levels << " n_features_per_level=" << this->network_config.encoding.n_features_per_level << " output_width=" << encoding_output_dims << '.';
            throw std::runtime_error{stream.str()};
        }

        if (this->network_config.direction_encoding.sh_degree == 0u || this->network_config.direction_encoding.sh_degree > 8u) throw std::runtime_error{"Spherical harmonics degree must be in [1, 8]."};
        const std::uint32_t dir_output_dims = this->network_config.direction_encoding.sh_degree * this->network_config.direction_encoding.sh_degree;
        if (dir_output_dims % 16u != 0u) {
            std::ostringstream stream;
            stream << "Spherical harmonics output width must be a multiple of 16 after encoder padding removal: sh_degree=" << this->network_config.direction_encoding.sh_degree << " output_width=" << dir_output_dims << '.';
            throw std::runtime_error{stream.str()};
        }

        train_plan.network.density_input_dims = encoding_output_dims;
        train_plan.network.rgb_input_dims     = train_plan.network.density_output_dims + dir_output_dims;

        train_plan.training.padded_output_width     = std::max(legacy::next_multiple(train_plan.network.rgb_output_dims, 16u), 4u);
        train_plan.training.max_samples             = train_plan.training.batch_size * 16u;
        train_plan.prep.uniform_samples_warmup      = NERF_GRID_N_CELLS;
        train_plan.prep.uniform_samples_steady      = NERF_GRID_N_CELLS / 4u;
        train_plan.prep.nonuniform_samples_steady   = NERF_GRID_N_CELLS / 4u;
        train_plan.density_grid.padded_output_width = train_plan.network.density_output_dims;
        train_plan.density_grid.query_batch_size    = NERF_GRID_N_CELLS * 2u;
        train_plan.density_grid.n_elements          = NERF_GRID_N_CELLS;
        train_plan.validation.padded_output_width   = train_plan.training.padded_output_width;

        cudaStream_t created_stream = {};
        legacy::cuda_check(cudaStreamCreate(&created_stream));
        stream = created_stream;

        training.rng         = legacy::math::pcg32{seed};
        sampling.density_rng = legacy::math::pcg32{training.rng.next_uint()};
        try {
            model = new ModelState{
                ngp::encoding::create_position_encoding<__half>(train_plan.network.n_pos_dims, this->network_config.encoding),
                ngp::encoding::create_direction_encoding<__half>(train_plan.network.n_dir_dims, this->network_config.direction_encoding),
                mlp::FullyFusedMLP<__half, density_network_width>{
                    train_plan.network.density_input_dims,
                    train_plan.network.density_output_dims,
                    this->network_config.density_network.n_hidden_layers,
                    activation_from_config(this->network_config.density_network.activation),
                    activation_from_config(this->network_config.density_network.output_activation),
                },
                mlp::FullyFusedMLP<__half, rgb_network_width>{
                    train_plan.network.rgb_input_dims,
                    train_plan.network.rgb_output_dims,
                    this->network_config.rgb_network.n_hidden_layers,
                    activation_from_config(this->network_config.rgb_network.activation),
                    activation_from_config(this->network_config.rgb_network.output_activation),
                },
                {},
            };
            model_scratch = new ModelScratch{};
            optimizer     = new Optimizer{};

            auto& current_model                   = *model;
            current_model.layout.pos_input_width  = std::visit([](const auto& impl) { return impl.input_width; }, current_model.pos_encoding);
            current_model.layout.pos_output_width = std::visit([](const auto& impl) { return impl.output_width; }, current_model.pos_encoding);
            current_model.layout.pos_layout       = std::visit([](const auto& impl) { return impl.preferred_output_layout; }, current_model.pos_encoding);
            current_model.layout.pos_param_count  = std::visit([](const auto& impl) { return impl.n_params; }, current_model.pos_encoding);

            current_model.params.density_network = 0u;
            current_model.params.rgb_network     = current_model.params.density_network + current_model.density_network.n_params;
            current_model.params.pos_encoding    = current_model.params.rgb_network + current_model.rgb_network.n_params;
            current_model.params.total           = current_model.params.pos_encoding + std::visit([](const auto& impl) { return impl.n_params; }, current_model.pos_encoding);

            optimizer->beta1              = this->network_config.optimizer.beta1;
            optimizer->beta2              = this->network_config.optimizer.beta2;
            optimizer->epsilon            = this->network_config.optimizer.epsilon;
            optimizer->base_learning_rate = this->network_config.optimizer.learning_rate;
            optimizer->l2_reg             = this->network_config.optimizer.l2_reg;
            const std::size_t param_count = current_model.params.total;
            optimizer->allocate(static_cast<std::uint32_t>(param_count), static_cast<std::uint32_t>(current_model.params.pos_encoding));

            parameter_buffer.resize(sizeof(float) * param_count + sizeof(__half) * param_count * 2u);
            parameter_buffer.memset(0);
            full_precision_params   = reinterpret_cast<float*>(parameter_buffer.data());
            network_params          = reinterpret_cast<__half*>(parameter_buffer.data() + sizeof(float) * param_count);
            network_param_gradients = reinterpret_cast<__half*>(parameter_buffer.data() + sizeof(float) * param_count + sizeof(__half) * param_count);

            current_model.density_network.params    = network_params + current_model.params.density_network;
            current_model.density_network.gradients = network_param_gradients + current_model.params.density_network;
            std::size_t density_param_offset        = 0u;
            for (std::size_t i = 0u; i < current_model.density_network.weight_matrices.size(); ++i) {
                current_model.density_network.weight_matrices[i].set_data_unsafe(current_model.density_network.params + density_param_offset);
                current_model.density_network.gradient_matrices[i].set_data_unsafe(current_model.density_network.gradients + density_param_offset);
                density_param_offset += current_model.density_network.weight_matrices[i].n_elements();
            }

            current_model.rgb_network.params    = network_params + current_model.params.rgb_network;
            current_model.rgb_network.gradients = network_param_gradients + current_model.params.rgb_network;
            std::size_t rgb_param_offset        = 0u;
            for (std::size_t i = 0u; i < current_model.rgb_network.weight_matrices.size(); ++i) {
                current_model.rgb_network.weight_matrices[i].set_data_unsafe(current_model.rgb_network.params + rgb_param_offset);
                current_model.rgb_network.gradient_matrices[i].set_data_unsafe(current_model.rgb_network.gradients + rgb_param_offset);
                rgb_param_offset += current_model.rgb_network.weight_matrices[i].n_elements();
            }

            std::visit(
                [&](auto& impl) {
                    impl.params    = network_params + current_model.params.pos_encoding;
                    impl.gradients = network_param_gradients + current_model.params.pos_encoding;
                },
                current_model.pos_encoding);

            legacy::math::pcg32 init_rng{seed};
            current_model.density_network.initialize_params(init_rng, full_precision_params + current_model.params.density_network, 1.0f);
            current_model.rgb_network.initialize_params(init_rng, full_precision_params + current_model.params.rgb_network, 1.0f);
            std::visit([&](auto& impl) { impl.initialize_params(init_rng, full_precision_params + current_model.params.pos_encoding, 1.0f); }, current_model.pos_encoding);
            if (param_count > 0u) {
                const std::uint32_t blocks = (static_cast<std::uint32_t>(param_count) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                ngp::network::detail::cast<__half><<<blocks, network::detail::n_threads_linear, 0, nullptr>>>(static_cast<std::uint32_t>(param_count), full_precision_params, network_params);
            }

            legacy::cuda_check(cudaDeviceSynchronize());

            auto& scratch                  = *model_scratch;
            const std::uint32_t batch_size = train_plan.training.batch_size;
            scratch.density_network_input  = ngp::legacy::GPUMatrixDynamic<__half>{current_model.layout.pos_output_width, batch_size, stream, current_model.layout.pos_layout};
            scratch.rgb_network_input      = ngp::legacy::GPUMatrixDynamic<__half>{current_model.rgb_network.input_width, batch_size, stream, current_model.dir_encoding.preferred_output_layout};
            scratch.density_network_output = scratch.rgb_network_input.slice_rows(0u, current_model.density_network.padded_output_width);
            scratch.dL_drgb                = ngp::legacy::GPUMatrixDynamic<__half>{current_model.rgb_network.padded_output_width, batch_size, stream, legacy::CM};
            scratch.dL_drgb_input          = ngp::legacy::GPUMatrixDynamic<__half>{current_model.rgb_network.input_width, batch_size, stream, current_model.dir_encoding.preferred_output_layout};
            if (current_model.layout.pos_param_count > 0u)
                scratch.dL_ddensity_input = ngp::legacy::GPUMatrixDynamic<__half>{current_model.layout.pos_output_width, batch_size, stream, current_model.layout.pos_layout};
            else
                scratch.dL_ddensity_input = {};

            current_model.density_network.prepare_scratch(stream, batch_size, legacy::CM, scratch.density_network);
            current_model.rgb_network.prepare_scratch(stream, batch_size, legacy::CM, scratch.rgb_network);
        } catch (...) {
            delete optimizer;
            delete model_scratch;
            delete model;
            optimizer               = nullptr;
            model_scratch           = nullptr;
            model                   = nullptr;
            parameter_buffer        = {};
            full_precision_params   = nullptr;
            network_params          = nullptr;
            network_param_gradients = nullptr;
            render_workspace        = {};
            training                = {};
            sampling                = {};
            dataset                 = {};
            if (stream) {
                network::detail::free_aux_stream_pool(stream);
                cudaStreamDestroy(stream);
                stream = nullptr;
            }
            throw;
        }
    }

    InstantNGP::~InstantNGP() noexcept {
        try {
            if (graph) {
                legacy::cuda_check(cudaGraphDestroy(graph));
                graph = nullptr;
            }
            if (graph_instance) {
                legacy::cuda_check(cudaGraphExecDestroy(graph_instance));
                graph_instance = nullptr;
            }
        } catch (const std::runtime_error& error) {
            if (std::string{error.what()}.find("driver shutting down") == std::string::npos) std::fprintf(stderr, "Could not destroy cuda graph: %s\n", error.what());
        }

        delete optimizer;
        delete model_scratch;
        delete model;
        optimizer               = nullptr;
        model_scratch           = nullptr;
        model                   = nullptr;
        parameter_buffer        = {};
        full_precision_params   = nullptr;
        network_params          = nullptr;
        network_param_gradients = nullptr;
        render_workspace        = {};
        training                = {};
        sampling                = {};
        dataset                 = {};
        if (stream) {
            network::detail::free_aux_stream_pool(stream);
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    auto InstantNGP::train(const std::int32_t iters) -> TrainResult {
        if (!model || network_params == nullptr || dataset.device.frames.size() == 0u) throw std::runtime_error{"training data must be loaded before train()."};

        for (std::int32_t i = 0; i < iters; ++i) {
            std::uint32_t n_prep_to_skip = training.step / train_plan.prep.skip_growth_interval;
            if (n_prep_to_skip < 1u) n_prep_to_skip = 1u;
            if (n_prep_to_skip > train_plan.prep.max_skip) n_prep_to_skip = train_plan.prep.max_skip;
            if (training.step % n_prep_to_skip == 0u) {
                const auto prep_start = std::chrono::steady_clock::now();
                const BoundingBox aabb{sampling.aabb_min, sampling.aabb_max};

                const std::uint32_t n_uniform_density_grid_samples    = training.step < train_plan.prep.warmup_steps ? train_plan.prep.uniform_samples_warmup : train_plan.prep.uniform_samples_steady;
                const std::uint32_t n_nonuniform_density_grid_samples = training.step < train_plan.prep.warmup_steps ? 0u : train_plan.prep.nonuniform_samples_steady;
                const std::uint32_t n_elements                        = train_plan.density_grid.n_elements;

                sampling.density.values.resize(n_elements);
                const std::uint32_t n_density_grid_samples = n_uniform_density_grid_samples + n_nonuniform_density_grid_samples;
                const std::uint32_t padded_output_width    = train_plan.density_grid.padded_output_width;

                const std::size_t positions_bytes   = legacy::next_multiple(n_density_grid_samples * sizeof(NerfPosition), static_cast<std::size_t>(128));
                const std::size_t indices_bytes     = legacy::next_multiple(n_elements * sizeof(std::uint32_t), static_cast<std::size_t>(128));
                const std::size_t density_tmp_bytes = legacy::next_multiple(n_elements * sizeof(float), static_cast<std::size_t>(128));
                const std::size_t mlp_out_bytes     = legacy::next_multiple(n_density_grid_samples * padded_output_width * sizeof(__half), static_cast<std::size_t>(128));
                auto& update_workspace              = sampling.update;
                update_workspace.arena.enlarge(positions_bytes + indices_bytes + density_tmp_bytes + mlp_out_bytes, stream);

                std::uint8_t* density_grid_base = reinterpret_cast<std::uint8_t*>(update_workspace.arena.data());
                std::size_t density_grid_offset = 0u;
                update_workspace.positions      = reinterpret_cast<float*>(density_grid_base + density_grid_offset);
                density_grid_offset += positions_bytes;
                update_workspace.indices = reinterpret_cast<std::uint32_t*>(density_grid_base + density_grid_offset);
                density_grid_offset += indices_bytes;
                update_workspace.density_scratch = reinterpret_cast<float*>(density_grid_base + density_grid_offset);
                density_grid_offset += density_tmp_bytes;
                update_workspace.mlp_out = reinterpret_cast<__half*>(density_grid_base + density_grid_offset);

                if (training.step == 0u) {
                    sampling.density.ema_step = 0u;
                    if (n_elements > 0u) {
                        const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                        mark_untrained_density_grid<<<blocks, network::detail::n_threads_linear, 0, stream>>>(n_elements, sampling.density.values.data(), static_cast<std::uint32_t>(dataset.device.frames.size()), dataset.device.frames.data());
                    }
                }

                legacy::cuda_check(cudaMemsetAsync(update_workspace.density_scratch, 0, sizeof(float) * n_elements, stream));

                if (n_uniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_uniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, stream>>>(n_uniform_density_grid_samples, sampling.density_rng, sampling.density.ema_step, aabb, sampling.density.values.data(), reinterpret_cast<NerfPosition*>(update_workspace.positions), update_workspace.indices, -0.01f);
                }
                sampling.density_rng.advance();

                if (n_nonuniform_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_nonuniform_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_grid_samples_nerf_nonuniform<<<blocks, network::detail::n_threads_linear, 0, stream>>>(n_nonuniform_density_grid_samples, sampling.density_rng, sampling.density.ema_step, aabb, sampling.density.values.data(), reinterpret_cast<NerfPosition*>(update_workspace.positions) + n_uniform_density_grid_samples, update_workspace.indices + n_uniform_density_grid_samples, NERF_MIN_OPTICAL_THICKNESS);
                }
                sampling.density_rng.advance();

                const std::size_t density_batch_size = train_plan.density_grid.query_batch_size;
                for (std::size_t density_batch_offset = 0u; density_batch_offset < n_density_grid_samples; density_batch_offset += density_batch_size) {
                    const std::size_t density_query_size = std::min(density_batch_size, static_cast<std::size_t>(n_density_grid_samples) - density_batch_offset);
                    legacy::GPUMatrixDynamic<__half> density_matrix(update_workspace.mlp_out + density_batch_offset, padded_output_width, density_query_size, legacy::RM);
                    legacy::GPUMatrixDynamic<float> density_position_matrix(update_workspace.positions + density_batch_offset * (sizeof(NerfPosition) / sizeof(float)), sizeof(NerfPosition) / sizeof(float), density_query_size, legacy::CM);
                    density(stream, density_position_matrix, density_matrix);
                }

                if (n_density_grid_samples > 0u) {
                    const std::uint32_t blocks = (n_density_grid_samples + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    splat_grid_samples_nerf_max_nearest_neighbor<<<blocks, network::detail::n_threads_linear, 0, stream>>>(n_density_grid_samples, update_workspace.indices, update_workspace.mlp_out, update_workspace.density_scratch);
                }

                if (n_elements > 0u) {
                    const std::uint32_t blocks = (n_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    ema_grid_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(n_elements, sampling.density.ema_decay, sampling.density.values.data(), update_workspace.density_scratch);
                }
                ++sampling.density.ema_step;

                constexpr std::uint32_t base_grid_elements = NERF_GRID_N_CELLS;
                sampling.density.occupancy.enlarge(base_grid_elements / 8u);
                sampling.density.reduction.enlarge(reduce_sum_workspace_size(base_grid_elements));

                legacy::cuda_check(cudaMemsetAsync(sampling.density.reduction.data(), 0, sizeof(float), stream));
                reduce_sum(sampling.density.values.data(), DensityGridReduceOp{base_grid_elements}, sampling.density.reduction.data(), base_grid_elements, stream);

                if (base_grid_elements / 8u > 0u) {
                    constexpr std::uint32_t blocks = ((base_grid_elements / 8u) + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    grid_to_bitfield<<<blocks, network::detail::n_threads_linear, 0, stream>>>(base_grid_elements / 8u, sampling.density.values.data(), sampling.density.occupancy.data(), sampling.density.reduction.data());
                }

                legacy::cuda_check(cudaStreamSynchronize(stream));
                training.last_prep_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - prep_start).count() / static_cast<float>(n_prep_to_skip);
            }

            optimizer->beta1              = network_config.optimizer.beta1;
            optimizer->beta2              = network_config.optimizer.beta2;
            optimizer->epsilon            = network_config.optimizer.epsilon;
            optimizer->base_learning_rate = network_config.optimizer.learning_rate;
            optimizer->l2_reg             = network_config.optimizer.l2_reg;
            const bool get_loss_scalar    = training.step % 16u == 0u;
            const auto train_start        = std::chrono::steady_clock::now();
            const BoundingBox aabb{sampling.aabb_min, sampling.aabb_max};

            auto& counters                 = training.counters;
            auto& workspace                = training.workspace;
            const std::uint32_t batch_size = train_plan.training.batch_size;

            counters.numsteps_counter.enlarge(1u);
            counters.numsteps_counter_compacted.enlarge(1u);
            counters.loss.enlarge(counters.rays_per_batch);
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter.data(), 0, sizeof(std::uint32_t), stream));
            legacy::cuda_check(cudaMemsetAsync(counters.numsteps_counter_compacted.data(), 0, sizeof(std::uint32_t), stream));
            legacy::cuda_check(cudaMemsetAsync(counters.loss.data(), 0, sizeof(float) * counters.rays_per_batch, stream));

            workspace.padded_output_width = train_plan.training.padded_output_width;
            workspace.floats_per_coord    = train_plan.training.floats_per_coord;
            workspace.max_samples         = train_plan.training.max_samples;

            const std::size_t ray_indices_bytes       = legacy::next_multiple(counters.rays_per_batch * sizeof(std::uint32_t), static_cast<std::size_t>(128));
            const std::size_t rays_unnormalized_bytes = legacy::next_multiple(counters.rays_per_batch * sizeof(legacy::Ray), static_cast<std::size_t>(128));
            const std::size_t numsteps_bytes          = legacy::next_multiple(counters.rays_per_batch * 2u * sizeof(std::uint32_t), static_cast<std::size_t>(128));
            const std::size_t coords_bytes            = legacy::next_multiple(workspace.max_samples * workspace.floats_per_coord * sizeof(float), static_cast<std::size_t>(128));
            const std::size_t mlp_out_bytes           = legacy::next_multiple(std::max(batch_size, workspace.max_samples) * workspace.padded_output_width * sizeof(__half), static_cast<std::size_t>(128));
            const std::size_t dloss_bytes             = legacy::next_multiple(batch_size * workspace.padded_output_width * sizeof(__half), static_cast<std::size_t>(128));
            const std::size_t compacted_coords_bytes  = legacy::next_multiple(batch_size * workspace.floats_per_coord * sizeof(float), static_cast<std::size_t>(128));
            const std::size_t ray_counter_bytes       = legacy::next_multiple(sizeof(std::uint32_t), static_cast<std::size_t>(128));
            const std::size_t total_bytes             = ray_indices_bytes + rays_unnormalized_bytes + numsteps_bytes + coords_bytes + mlp_out_bytes + dloss_bytes + compacted_coords_bytes + ray_counter_bytes;

            workspace.arena.enlarge(total_bytes, stream);
            std::uint8_t* workspace_base = reinterpret_cast<std::uint8_t*>(workspace.arena.data());
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
            workspace.compacted_output        = legacy::GPUMatrixDynamic<__half>{train_plan.training.padded_output_width, batch_size, stream, legacy::CM};

            legacy::cuda_check(cudaMemsetAsync(workspace.ray_counter, 0, sizeof(std::uint32_t), stream));
            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_training_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(counters.rays_per_batch, aabb, workspace.max_inference, training.rng, workspace.ray_counter, counters.numsteps_counter.data(), workspace.ray_indices, static_cast<legacy::Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.coords), 1u},
                    static_cast<std::uint32_t>(dataset.device.pixels.size()), dataset.device.frames.data(), sampling.density.occupancy.data(), sampling.snap_to_pixel_centers);
            }
            inference(stream, workspace.coords_matrix, workspace.rgbsigma_matrix);

            if (counters.rays_per_batch > 0u) {
                const std::uint32_t blocks = (counters.rays_per_batch + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                compute_loss_kernel_train_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(counters.rays_per_batch, aabb, training.rng, batch_size, workspace.ray_counter, default_loss_scale<__half>, static_cast<int>(workspace.padded_output_width), static_cast<std::uint32_t>(dataset.device.pixels.size()), dataset.device.frames.data(), workspace.mlp_out, counters.numsteps_counter_compacted.data(), workspace.ray_indices,
                    static_cast<const legacy::Ray*>(workspace.rays_unnormalized), workspace.numsteps, legacy::PitchedPtr<const NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.coords), 1u}, legacy::PitchedPtr<NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.coords_compacted), 1u}, workspace.dloss_dmlp_out, counters.loss.data(), sampling.snap_to_pixel_centers, sampling.density.reduction.data(), sampling.near_distance);
            }

            const std::uint32_t dloss_elements  = batch_size * workspace.padded_output_width;
            const std::uint32_t coords_elements = batch_size * workspace.floats_per_coord;
            fill_rollover_and_rescale<__half><<<((dloss_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, stream>>>(batch_size, workspace.padded_output_width, counters.numsteps_counter_compacted.data(), workspace.dloss_dmlp_out);
            fill_rollover<float><<<((coords_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, stream>>>(batch_size, workspace.floats_per_coord, counters.numsteps_counter_compacted.data(), workspace.coords_compacted);

            bool launch_direct = stream == nullptr || stream == cudaStreamLegacy;
            if (!launch_direct) {
                cudaStreamCaptureStatus capture_status;
                legacy::cuda_check(cudaStreamIsCapturing(stream, &capture_status));
                if (capture_status != cudaStreamCaptureStatusNone) {
                    launch_direct = true;
                } else {
                    cudaError_t capture_result = cudaStreamIsCapturing(cudaStreamLegacy, &capture_status);
                    if (capture_result == cudaErrorStreamCaptureImplicit) {
                        launch_direct = true;
                    } else {
                        legacy::cuda_check(capture_result);
                        if (capture_status != cudaStreamCaptureStatusNone) {
                            launch_direct = true;
                        } else {
                            if (graph) {
                                legacy::cuda_check(cudaGraphDestroy(graph));
                                graph = nullptr;
                            }

                            legacy::cuda_check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));
                            legacy::current_graph_capture_sync_flags().push_back(&synchronize_when_capture_done);
                            try {
                                forward(stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                                backward(stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix);
                                legacy::cuda_check(cudaStreamEndCapture(stream, &graph));
                                if (legacy::current_graph_capture_sync_flags().back() != &synchronize_when_capture_done) throw std::runtime_error{"Graph capture must end in reverse order of creation."};
                                legacy::current_graph_capture_sync_flags().pop_back();
                            } catch (...) {
                                if (!legacy::current_graph_capture_sync_flags().empty() && legacy::current_graph_capture_sync_flags().back() == &synchronize_when_capture_done) legacy::current_graph_capture_sync_flags().pop_back();

                                cudaGraph_t aborted_graph      = nullptr;
                                cudaError_t end_capture_result = cudaStreamEndCapture(stream, &aborted_graph);
                                if (end_capture_result == cudaSuccess && aborted_graph)
                                    cudaGraphDestroy(aborted_graph);
                                else
                                    cudaGetLastError();

                                graph = nullptr;
                                throw;
                            }

                            if (synchronize_when_capture_done) {
                                legacy::cuda_check(cudaDeviceSynchronize());
                                synchronize_when_capture_done = false;
                            }

                            if (!graph) {
                                if (graph_instance) legacy::cuda_check(cudaGraphExecDestroy(graph_instance));
                                graph          = nullptr;
                                graph_instance = nullptr;
                            } else {
                                if (graph_instance) {
                                    cudaGraphExecUpdateResultInfo update_result;
                                    legacy::cuda_check(cudaGraphExecUpdate(graph_instance, graph, &update_result));
                                    if (update_result.result != cudaGraphExecUpdateSuccess) {
                                        legacy::cuda_check(cudaGraphExecDestroy(graph_instance));
                                        graph_instance = nullptr;
                                    }
                                }

                                if (!graph_instance) legacy::cuda_check(cudaGraphInstantiate(&graph_instance, graph, nullptr, nullptr, 0));
                                legacy::cuda_check(cudaGraphLaunch(graph_instance, stream));
                            }
                        }
                    }
                }
            }
            if (launch_direct) {
                forward(stream, workspace.compacted_coords_matrix, &workspace.compacted_output);
                backward(stream, workspace.compacted_coords_matrix, workspace.compacted_output, workspace.gradient_matrix);
            }
            optimizer->step(stream, default_loss_scale<__half>, full_precision_params, network_params, network_param_gradients);
            ++training.step;

            legacy::cuda_check(cudaStreamSynchronize(stream));

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
                    training.loss_reduction.enlarge(reduce_sum_workspace_size(counters.rays_per_batch), stream);
                    legacy::cuda_check(cudaMemsetAsync(training.loss_reduction.data(), 0, sizeof(float), stream));
                    reduce_sum(counters.loss.data(), SumIdentityOp{}, training.loss_reduction.data(), counters.rays_per_batch, stream);
                    legacy::cuda_check(cudaMemcpyAsync(&last_loss, training.loss_reduction.data(), sizeof(float), cudaMemcpyDeviceToHost, stream));
                    legacy::cuda_check(cudaStreamSynchronize(stream));
                    last_loss *= static_cast<float>(counters.measured_batch_size) / static_cast<float>(batch_size);
                }

                counters.rays_per_batch = static_cast<std::uint32_t>(static_cast<float>(counters.rays_per_batch) * static_cast<float>(batch_size) / static_cast<float>(counters.measured_batch_size));
                counters.rays_per_batch = std::min(legacy::next_multiple(counters.rays_per_batch, network::detail::batch_size_granularity), 1u << 18);
            }

            if (get_loss_scalar) training.last_loss = last_loss;

            if (counters.measured_batch_size == 0u) {
                training.last_loss = 0.0f;
                throw std::runtime_error{"Training stopped unexpectedly."};
            }

            training.rng.advance();
            training.last_train_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - train_start).count();
        }

        TrainResult result{};
        result.loss                                  = training.last_loss;
        result.train_ms                              = training.last_train_ms;
        result.prep_ms                               = training.last_prep_ms;
        result.step                                  = training.step;
        result.batch_size                            = train_plan.training.batch_size;
        result.rays_per_batch                        = training.counters.rays_per_batch;
        result.measured_batch_size                   = training.counters.measured_batch_size;
        result.measured_batch_size_before_compaction = training.counters.measured_batch_size_before_compaction;
        return result;
    }

    auto InstantNGP::validate(const std::filesystem::path& report_path) -> ValidateResult {
        if (report_path.empty()) throw std::invalid_argument{"validation benchmark report path must not be empty."};
        if (!model || network_params == nullptr) throw std::runtime_error{"Validation benchmark requires an initialized network."};

        const std::size_t validation_count = dataset.host.validation.size();
        if (validation_count == 0u) throw std::runtime_error{"No validation images are available in the current dataset."};

        const std::filesystem::path parent_dir = report_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create validation report directory '" + parent_dir.string() + "'."};

        std::ofstream report_stream{report_path, std::ios::binary};
        if (!report_stream.is_open()) throw std::runtime_error{"Failed to open validation benchmark report '" + report_path.string() + "'."};
        report_stream << std::setprecision(9);
        report_stream << "image_index,width,height,mse,psnr\n";

        ValidateResult result{};
        result.image_count = static_cast<std::uint32_t>(validation_count);
        result.min_psnr    = std::numeric_limits<float>::infinity();
        result.max_psnr    = -std::numeric_limits<float>::infinity();

        double total_squared_error = 0.0;
        double total_psnr          = 0.0;
        const auto benchmark_start = std::chrono::steady_clock::now();
        const BoundingBox aabb{sampling.aabb_min, sampling.aabb_max};

        for (std::size_t validation_image_index = 0u; validation_image_index < validation_count; ++validation_image_index) {
            const DatasetState::HostData::Frame& source = dataset.host.validation[validation_image_index];
            if (source.resolution.x <= 0 || source.resolution.y <= 0) throw std::runtime_error{"Validation frame has zero resolution."};
            if (!std::isfinite(source.focal_length) || source.focal_length <= 0.0f) throw std::runtime_error{"Validation frame has an invalid focal length."};

            InstantNGP::DatasetState::DeviceData::GpuFrame frame{};
            frame.resolution   = source.resolution;
            frame.focal_length = source.focal_length;
            frame.camera       = source.camera;

            const legacy::math::ivec2 resolution    = frame.resolution;
            const auto background                   = legacy::math::vec3(1.0f);
            const std::uint32_t total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
            const std::uint32_t padded_output_width = train_plan.validation.padded_output_width;
            const std::uint32_t floats_per_coord    = train_plan.validation.floats_per_coord;
            const std::uint32_t max_samples         = train_plan.validation.max_samples;
            auto& workspace                         = render_workspace;
            workspace.rendered.resize(total_pixels);
            workspace.tile_numsteps.resize(train_plan.validation.tile_rays * 2u);
            workspace.tile_coords.resize(max_samples * floats_per_coord);
            workspace.tile_mlp_out.resize(max_samples * padded_output_width);
            workspace.sample_counter.resize(1u);
            workspace.overflow_counter.resize(1u);

            for (std::uint32_t pixel_offset = 0u; pixel_offset < total_pixels; pixel_offset += train_plan.validation.tile_rays) {
                const std::uint32_t tile_pixels = std::min(train_plan.validation.tile_rays, total_pixels - pixel_offset);
                legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), stream));

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, aabb, max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
                }

                std::uint32_t used_samples    = 0u;
                std::uint32_t overflowed_rays = 0u;
                legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
                legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
                legacy::cuda_check(cudaStreamSynchronize(stream));

                if (overflowed_rays != 0u) {
                    std::ostringstream message;
                    message << "Validation benchmark sample budget overflowed for " << overflowed_rays << " rays at image " << validation_image_index << ". Reduce tile size or increase the sample budget.";
                    throw std::runtime_error{message.str()};
                }

                if (used_samples > 0u) {
                    const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                    const std::uint32_t coord_elements      = padded_used_samples * floats_per_coord;
                    fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, stream>>>(padded_used_samples, floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                    legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), floats_per_coord, padded_used_samples, legacy::CM);
                    legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), padded_output_width, padded_used_samples, legacy::CM);
                    inference(stream, coords_matrix, rgbsigma_matrix);
                }

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), padded_output_width, background, workspace.rendered.data());
                }
            }

            legacy::cuda_check(cudaStreamSynchronize(stream));

            std::vector<legacy::math::vec3> rendered_host(total_pixels);
            workspace.rendered.copy_to_host(rendered_host);

            double image_squared_error = 0.0;
            for (int y = 0; y < resolution.y; ++y) {
                for (int x = 0; x < resolution.x; ++x) {
                    const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                    const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                    const legacy::math::vec4 gt         = read_rgba(legacy::math::ivec2{x, y}, resolution, source.rgba.data());
                    const legacy::math::vec3 target     = clamp_rgb01(linear_to_srgb(legacy::math::vec3{gt.x, gt.y, gt.z} + (1.0f - gt.a) * background));
                    const legacy::math::vec3 diff       = prediction - target;
                    image_squared_error += static_cast<double>(ngp::legacy::math::mean(diff * diff));
                }
            }

            const auto mse   = static_cast<float>(image_squared_error / static_cast<double>(total_pixels));
            const float psnr = mse > 0.0f ? -10.0f * std::log10(mse) : std::numeric_limits<float>::infinity();

            report_stream << validation_image_index << ',' << resolution.x << ',' << resolution.y << ',' << mse << ',' << psnr << '\n';
            if (!report_stream.good()) throw std::runtime_error{"Failed while writing validation benchmark report '" + report_path.string() + "'."};

            total_squared_error += image_squared_error;
            total_psnr += static_cast<double>(psnr);
            result.total_pixels += static_cast<std::uint64_t>(total_pixels);
            result.min_psnr = std::min(result.min_psnr, psnr);
            result.max_psnr = std::max(result.max_psnr, psnr);
        }

        result.benchmark_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - benchmark_start).count();
        result.mean_mse     = result.total_pixels > 0u ? static_cast<float>(total_squared_error / static_cast<double>(result.total_pixels)) : 0.0f;
        result.mean_psnr    = result.image_count > 0u ? static_cast<float>(total_psnr / static_cast<double>(result.image_count)) : 0.0f;
        result.split_psnr   = result.mean_mse > 0.0f ? -10.0f * std::log10(result.mean_mse) : std::numeric_limits<float>::infinity();
        return result;
    }

    auto InstantNGP::inference(const std::filesystem::path& output_path, const InferenceCamera& camera) const -> InferenceResult {
        if (output_path.empty()) throw std::invalid_argument{"inference output path must not be empty."};
        if (!model || network_params == nullptr) throw std::runtime_error{"Inference requires an initialized network."};
        if (camera.resolution.x <= 0 || camera.resolution.y <= 0) throw std::runtime_error{"Inference camera has an invalid resolution."};
        if (!std::isfinite(camera.focal_length) || camera.focal_length <= 0.0f) throw std::runtime_error{"Inference camera has an invalid focal length."};
        for (std::size_t column = 0u; column < 4u; ++column) {
            for (std::size_t row = 0u; row < 3u; ++row) {
                if (!std::isfinite(camera.camera[column][row])) throw std::runtime_error{"Inference camera has a non-finite transform element."};
            }
        }

        const auto render_start = std::chrono::steady_clock::now();
        InstantNGP::DatasetState::DeviceData::GpuFrame frame{};
        frame.resolution   = camera.resolution;
        frame.focal_length = camera.focal_length;
        frame.camera       = camera.camera;

        const legacy::math::ivec2 resolution    = frame.resolution;
        const auto background                   = legacy::math::vec3(1.0f);
        const std::uint32_t total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
        const std::uint32_t padded_output_width = train_plan.validation.padded_output_width;
        const std::uint32_t floats_per_coord    = train_plan.validation.floats_per_coord;
        const std::uint32_t max_samples         = train_plan.validation.max_samples;
        const BoundingBox aabb{sampling.aabb_min, sampling.aabb_max};
        auto& workspace = render_workspace;
        workspace.rendered.resize(total_pixels);
        workspace.tile_numsteps.resize(train_plan.validation.tile_rays * 2u);
        workspace.tile_coords.resize(max_samples * floats_per_coord);
        workspace.tile_mlp_out.resize(max_samples * padded_output_width);
        workspace.sample_counter.resize(1u);
        workspace.overflow_counter.resize(1u);

        for (std::uint32_t pixel_offset = 0u; pixel_offset < total_pixels; pixel_offset += train_plan.validation.tile_rays) {
            const std::uint32_t tile_pixels = std::min(train_plan.validation.tile_rays, total_pixels - pixel_offset);
            legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), stream));
            legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), stream));

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, aabb, max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
            }

            std::uint32_t used_samples    = 0u;
            std::uint32_t overflowed_rays = 0u;
            legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
            legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
            legacy::cuda_check(cudaStreamSynchronize(stream));

            if (overflowed_rays != 0u) {
                std::ostringstream message;
                message << "Inference sample budget overflowed for " << overflowed_rays << " rays. Reduce tile size or increase the sample budget.";
                throw std::runtime_error{message.str()};
            }

            if (used_samples > 0u) {
                const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                const std::uint32_t coord_elements      = padded_used_samples * floats_per_coord;
                fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, stream>>>(padded_used_samples, floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), floats_per_coord, padded_used_samples, legacy::CM);
                legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), padded_output_width, padded_used_samples, legacy::CM);
                inference(stream, coords_matrix, rgbsigma_matrix);
            }

            if (tile_pixels > 0u) {
                const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), padded_output_width, background, workspace.rendered.data());
            }
        }

        legacy::cuda_check(cudaStreamSynchronize(stream));

        std::vector<legacy::math::vec3> rendered_host(total_pixels);
        workspace.rendered.copy_to_host(rendered_host);

        std::vector<std::uint8_t> png_rgb(static_cast<std::size_t>(total_pixels) * 3u);
        for (int y = 0; y < resolution.y; ++y) {
            for (int x = 0; x < resolution.x; ++x) {
                const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                png_rgb[pixel_index * 3u + 0u]      = static_cast<std::uint8_t>(lrintf(prediction.x * 255.0f));
                png_rgb[pixel_index * 3u + 1u]      = static_cast<std::uint8_t>(lrintf(prediction.y * 255.0f));
                png_rgb[pixel_index * 3u + 2u]      = static_cast<std::uint8_t>(lrintf(prediction.z * 255.0f));
            }
        }

        const std::filesystem::path parent_dir = output_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create image directory '" + parent_dir.string() + "'."};
        if (stbi_write_png(output_path.string().c_str(), resolution.x, resolution.y, 3, png_rgb.data(), resolution.x * 3) == 0) throw std::runtime_error{"Failed to write PNG image '" + output_path.string() + "'."};

        InferenceResult result{};
        result.width     = static_cast<std::uint32_t>(resolution.x);
        result.height    = static_cast<std::uint32_t>(resolution.y);
        result.render_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - render_start).count();
        return result;
    }

    auto InstantNGP::test(const std::filesystem::path& report_path) -> TestResult {
        if (report_path.empty()) throw std::invalid_argument{"test benchmark report path must not be empty."};
        if (!model || network_params == nullptr) throw std::runtime_error{"Test benchmark requires an initialized network."};

        const std::size_t test_count = dataset.host.test.size();
        if (test_count == 0u) throw std::runtime_error{"No test images are available in the current dataset."};

        const std::filesystem::path parent_dir = report_path.parent_path();
        if (!parent_dir.empty() && !std::filesystem::exists(parent_dir) && !std::filesystem::create_directories(parent_dir)) throw std::runtime_error{"Failed to create test report directory '" + parent_dir.string() + "'."};

        std::ofstream report_stream{report_path, std::ios::binary};
        if (!report_stream.is_open()) throw std::runtime_error{"Failed to open test benchmark report '" + report_path.string() + "'."};
        report_stream << std::setprecision(9);
        report_stream << "image_index,width,height,mse,psnr\n";

        TestResult result{};
        result.image_count = static_cast<std::uint32_t>(test_count);
        result.min_psnr    = std::numeric_limits<float>::infinity();
        result.max_psnr    = -std::numeric_limits<float>::infinity();

        double total_squared_error = 0.0;
        double total_psnr          = 0.0;
        const auto benchmark_start = std::chrono::steady_clock::now();
        const BoundingBox aabb{sampling.aabb_min, sampling.aabb_max};

        for (std::size_t test_image_index = 0u; test_image_index < test_count; ++test_image_index) {
            const DatasetState::HostData::Frame& source = dataset.host.test[test_image_index];
            if (source.resolution.x <= 0 || source.resolution.y <= 0) throw std::runtime_error{"Test frame has zero resolution."};
            if (!std::isfinite(source.focal_length) || source.focal_length <= 0.0f) throw std::runtime_error{"Test frame has an invalid focal length."};

            InstantNGP::DatasetState::DeviceData::GpuFrame frame{};
            frame.resolution   = source.resolution;
            frame.focal_length = source.focal_length;
            frame.camera       = source.camera;

            const legacy::math::ivec2 resolution    = frame.resolution;
            const auto background                   = legacy::math::vec3(1.0f);
            const std::uint32_t total_pixels        = static_cast<std::uint32_t>(ngp::legacy::math::product(resolution));
            const std::uint32_t padded_output_width = train_plan.validation.padded_output_width;
            const std::uint32_t floats_per_coord    = train_plan.validation.floats_per_coord;
            const std::uint32_t max_samples         = train_plan.validation.max_samples;
            auto& workspace                         = render_workspace;
            workspace.rendered.resize(total_pixels);
            workspace.tile_numsteps.resize(train_plan.validation.tile_rays * 2u);
            workspace.tile_coords.resize(max_samples * floats_per_coord);
            workspace.tile_mlp_out.resize(max_samples * padded_output_width);
            workspace.sample_counter.resize(1u);
            workspace.overflow_counter.resize(1u);

            for (std::uint32_t pixel_offset = 0u; pixel_offset < total_pixels; pixel_offset += train_plan.validation.tile_rays) {
                const std::uint32_t tile_pixels = std::min(train_plan.validation.tile_rays, total_pixels - pixel_offset);
                legacy::cuda_check(cudaMemsetAsync(workspace.tile_numsteps.data(), 0, workspace.tile_numsteps.size() * sizeof(std::uint32_t), stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.sample_counter.data(), 0, sizeof(std::uint32_t), stream));
                legacy::cuda_check(cudaMemsetAsync(workspace.overflow_counter.data(), 0, sizeof(std::uint32_t), stream));

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    generate_validation_samples_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, aabb, max_samples, workspace.sample_counter.data(), workspace.overflow_counter.data(), workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, frame, sampling.density.occupancy.data());
                }

                std::uint32_t used_samples    = 0u;
                std::uint32_t overflowed_rays = 0u;
                legacy::cuda_check(cudaMemcpyAsync(&used_samples, workspace.sample_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
                legacy::cuda_check(cudaMemcpyAsync(&overflowed_rays, workspace.overflow_counter.data(), sizeof(std::uint32_t), cudaMemcpyDeviceToHost, stream));
                legacy::cuda_check(cudaStreamSynchronize(stream));

                if (overflowed_rays != 0u) {
                    std::ostringstream message;
                    message << "Test benchmark sample budget overflowed for " << overflowed_rays << " rays at image " << test_image_index << ". Reduce tile size or increase the sample budget.";
                    throw std::runtime_error{message.str()};
                }

                if (used_samples > 0u) {
                    const std::uint32_t padded_used_samples = legacy::next_multiple(used_samples, network::detail::batch_size_granularity);
                    const std::uint32_t coord_elements      = padded_used_samples * floats_per_coord;
                    fill_rollover<float><<<((coord_elements + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear), network::detail::n_threads_linear, 0, stream>>>(padded_used_samples, floats_per_coord, workspace.sample_counter.data(), workspace.tile_coords.data());

                    legacy::GPUMatrixDynamic<float> coords_matrix(workspace.tile_coords.data(), floats_per_coord, padded_used_samples, legacy::CM);
                    legacy::GPUMatrixDynamic<__half> rgbsigma_matrix(workspace.tile_mlp_out.data(), padded_output_width, padded_used_samples, legacy::CM);
                    inference(stream, coords_matrix, rgbsigma_matrix);
                }

                if (tile_pixels > 0u) {
                    const std::uint32_t blocks = (tile_pixels + network::detail::n_threads_linear - 1u) / network::detail::n_threads_linear;
                    composite_validation_kernel_nerf<<<blocks, network::detail::n_threads_linear, 0, stream>>>(tile_pixels, pixel_offset, workspace.tile_numsteps.data(), ngp::legacy::PitchedPtr<const NerfCoordinate>{reinterpret_cast<NerfCoordinate*>(workspace.tile_coords.data()), 1u}, workspace.tile_mlp_out.data(), padded_output_width, background, workspace.rendered.data());
                }
            }

            legacy::cuda_check(cudaStreamSynchronize(stream));

            std::vector<legacy::math::vec3> rendered_host(total_pixels);
            workspace.rendered.copy_to_host(rendered_host);

            double image_squared_error = 0.0;
            for (int y = 0; y < resolution.y; ++y) {
                for (int x = 0; x < resolution.x; ++x) {
                    const std::size_t pixel_index       = static_cast<std::size_t>(x) + static_cast<std::size_t>(y) * static_cast<std::size_t>(resolution.x);
                    const legacy::math::vec3 prediction = clamp_rgb01(rendered_host[pixel_index]);
                    const legacy::math::vec4 gt         = read_rgba(legacy::math::ivec2{x, y}, resolution, source.rgba.data());
                    const legacy::math::vec3 target     = clamp_rgb01(linear_to_srgb(legacy::math::vec3{gt.x, gt.y, gt.z} + (1.0f - gt.a) * background));
                    const legacy::math::vec3 diff       = prediction - target;
                    image_squared_error += static_cast<double>(ngp::legacy::math::mean(diff * diff));
                }
            }

            const auto mse   = static_cast<float>(image_squared_error / static_cast<double>(total_pixels));
            const float psnr = mse > 0.0f ? -10.0f * std::log10(mse) : std::numeric_limits<float>::infinity();

            report_stream << test_image_index << ',' << resolution.x << ',' << resolution.y << ',' << mse << ',' << psnr << '\n';
            if (!report_stream.good()) throw std::runtime_error{"Failed while writing test benchmark report '" + report_path.string() + "'."};

            total_squared_error += image_squared_error;
            total_psnr += static_cast<double>(psnr);
            result.total_pixels += static_cast<std::uint64_t>(total_pixels);
            result.min_psnr = std::min(result.min_psnr, psnr);
            result.max_psnr = std::max(result.max_psnr, psnr);
        }

        result.benchmark_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - benchmark_start).count();
        result.mean_mse     = result.total_pixels > 0u ? static_cast<float>(total_squared_error / static_cast<double>(result.total_pixels)) : 0.0f;
        result.mean_psnr    = result.image_count > 0u ? static_cast<float>(total_psnr / static_cast<double>(result.image_count)) : 0.0f;
        result.split_psnr   = result.mean_mse > 0.0f ? -10.0f * std::log10(result.mean_mse) : std::numeric_limits<float>::infinity();
        return result;
    }

} // namespace ngp
