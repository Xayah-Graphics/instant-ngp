#ifndef NGP_LEGACY_CUH
#define NGP_LEGACY_CUH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <deque>
#include <memory>
#include <source_location>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(__CUDA_ARCH__)
#define TCNN_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TCNN_PRAGMA_UNROLL
#endif

namespace ngp::legacy {

    namespace math {
#define PCG32_DEFAULT_STATE  0x853c49e6748fea9bULL
#define PCG32_DEFAULT_STREAM 0xda3e39cb94b95bdbULL
#define PCG32_MULT           0x5851f42d4c957f2dULL

        struct pcg32 {
            __host__ __device__ pcg32() : state(PCG32_DEFAULT_STATE), inc(PCG32_DEFAULT_STREAM) {}

            __host__ __device__ pcg32(uint64_t initstate, uint64_t initseq = 1u) {
                seed(initstate, initseq);
            }

            __host__ __device__ void seed(uint64_t initstate, uint64_t initseq = 1) {
                state = 0U;
                inc   = (initseq << 1u) | 1u;
                next_uint();
                state += initstate;
                next_uint();
            }

            __host__ __device__ uint32_t next_uint() {
                uint64_t oldstate   = state;
                state               = oldstate * PCG32_MULT + inc;
                uint32_t xorshifted = (uint32_t) (((oldstate >> 18u) ^ oldstate) >> 27u);
                uint32_t rot        = (uint32_t) (oldstate >> 59u);
                return (xorshifted >> rot) | (xorshifted << ((~rot + 1u) & 31));
            }

            __host__ __device__ float next_float() {

                union {
                    uint32_t u;
                    float f;
                } x;
                x.u = (next_uint() >> 9) | 0x3f800000u;
                return x.f - 1.0f;
            }

            __host__ __device__ void advance(int64_t delta_ = (1ll << 32)) {
                uint64_t cur_mult = PCG32_MULT, cur_plus = inc, acc_mult = 1u, acc_plus = 0u;

                uint64_t delta = (uint64_t) delta_;

                while (delta > 0) {
                    if (delta & 1) {
                        acc_mult *= cur_mult;
                        acc_plus = acc_plus * cur_mult + cur_plus;
                    }
                    cur_plus = (cur_mult + 1) * cur_plus;
                    cur_mult *= cur_mult;
                    delta /= 2;
                }
                state = acc_mult * state + acc_plus;
            }

            uint64_t state; // RNG state.  All values are possible.
            uint64_t inc; // Controls which RNG sequence (stream) is selected. Must *always* be odd.
        };

        template <typename T, uint32_t N, size_t ALIGNMENT = sizeof(T)>
        struct alignas(ALIGNMENT != 0 && (ALIGNMENT & (ALIGNMENT - 1)) == 0 ? ALIGNMENT : alignof(T)) tvec final {
            static constexpr uint32_t size() {
                return N;
            }

            T elems[N] = {};

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T value) {
                TCNN_PRAGMA_UNROLL
                for (uint32_t i = 0u; i < N; ++i) elems[i] = value;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return elems[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return elems[index];
            }
        };

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator*(T value, const tvec<T, N, ALIGNMENT>& vector) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0u; i < N; ++i) result[i] = value * vector[i];
            return result;
        }

        struct vec2;
        struct vec3;

        struct ivec2 final {
            int x = 0;
            int y = 0;

            __host__ __device__ ivec2() = default;
            __host__ __device__ ivec2(int value) : x{value}, y{value} {}
            __host__ __device__ ivec2(int x, int y) : x{x}, y{y} {}
            __host__ __device__ ivec2(const vec2& value);

            __host__ __device__ int& operator[](uint32_t index) {
                return index == 0u ? x : y;
            }

            __host__ __device__ const int& operator[](uint32_t index) const {
                return index == 0u ? x : y;
            }
        };

        struct ivec3 final {
            int x = 0;
            int y = 0;
            int z = 0;

            __host__ __device__ ivec3() = default;
            __host__ __device__ ivec3(int value) : x{value}, y{value}, z{value} {}
            __host__ __device__ ivec3(int x, int y, int z) : x{x}, y{y}, z{z} {}
            __host__ __device__ ivec3(const vec3& value);
        };

        struct uvec3 final {
            uint32_t x = 0u;
            uint32_t y = 0u;
            uint32_t z = 0u;

            __host__ __device__ uvec3() = default;
            __host__ __device__ uvec3(uint32_t value) : x{value}, y{value}, z{value} {}
            __host__ __device__ uvec3(uint32_t x, uint32_t y, uint32_t z) : x{x}, y{y}, z{z} {}

            __host__ __device__ uint32_t& operator[](uint32_t index) {
                return index == 0u ? x : (index == 1u ? y : z);
            }

            __host__ __device__ const uint32_t& operator[](uint32_t index) const {
                return index == 0u ? x : (index == 1u ? y : z);
            }
        };

        struct vec2 final {
            float x = 0.0f;
            float y = 0.0f;

            __host__ __device__ vec2() = default;
            __host__ __device__ vec2(float value) : x{value}, y{value} {}
            __host__ __device__ vec2(float x, float y) : x{x}, y{y} {}
            __host__ __device__ vec2(const ivec2& value) : x{static_cast<float>(value.x)}, y{static_cast<float>(value.y)} {}

            __host__ __device__ float& operator[](uint32_t index) {
                return index == 0u ? x : y;
            }

            __host__ __device__ const float& operator[](uint32_t index) const {
                return index == 0u ? x : y;
            }
        };

        struct vec3 final {
            float x = 0.0f;
            float y = 0.0f;
            float z = 0.0f;

            __host__ __device__ vec3() = default;
            __host__ __device__ vec3(float value) : x{value}, y{value}, z{value} {}
            __host__ __device__ vec3(float x, float y, float z) : x{x}, y{y}, z{z} {}
            __host__ __device__ vec3(const ivec3& value) : x{static_cast<float>(value.x)}, y{static_cast<float>(value.y)}, z{static_cast<float>(value.z)} {}

            __host__ __device__ float& operator[](uint32_t index) {
                return index == 0u ? x : (index == 1u ? y : z);
            }

            __host__ __device__ const float& operator[](uint32_t index) const {
                return index == 0u ? x : (index == 1u ? y : z);
            }
        };

        struct vec4 final {
            union {
                float x = 0.0f;
                float r;
            };
            union {
                float y;
                float g;
            };
            union {
                float z;
                float b;
            };
            union {
                float w;
                float a;
            };

            __host__ __device__ vec4() : x{0.0f}, y{0.0f}, z{0.0f}, w{0.0f} {}
            __host__ __device__ vec4(float value) : x{value}, y{value}, z{value}, w{value} {}
            __host__ __device__ vec4(float x, float y, float z, float w) : x{x}, y{y}, z{z}, w{w} {}

            __host__ __device__ float& operator[](uint32_t index) {
                return index == 0u ? x : (index == 1u ? y : (index == 2u ? z : w));
            }

            __host__ __device__ const float& operator[](uint32_t index) const {
                return index == 0u ? x : (index == 1u ? y : (index == 2u ? z : w));
            }
        };

        __host__ __device__ inline ivec2::ivec2(const vec2& value) : x{static_cast<int>(value.x)}, y{static_cast<int>(value.y)} {}

        __host__ __device__ inline ivec3::ivec3(const vec3& value) : x{static_cast<int>(value.x)}, y{static_cast<int>(value.y)}, z{static_cast<int>(value.z)} {}

        struct mat4x3 final {
            vec3 columns[4] = {};

            __host__ __device__ vec3& operator[](uint32_t index) {
                return columns[index];
            }

            __host__ __device__ const vec3& operator[](uint32_t index) const {
                return columns[index];
            }
        };

        static_assert(sizeof(vec2) == sizeof(float) * 2u);
        static_assert(sizeof(vec3) == sizeof(float) * 3u);
        static_assert(sizeof(vec4) == sizeof(float) * 4u);
        static_assert(sizeof(ivec2) == sizeof(int) * 2u);
        static_assert(sizeof(ivec3) == sizeof(int) * 3u);
        static_assert(sizeof(uvec3) == sizeof(uint32_t) * 3u);
        static_assert(sizeof(mat4x3) == sizeof(float) * 12u);

        __host__ __device__ inline float sign(float value) {
            return value >= 0.0f ? 1.0f : -1.0f;
        }

        __host__ __device__ inline vec2 operator+(const vec2& a, const vec2& b) {
            return {a.x + b.x, a.y + b.y};
        }

        __host__ __device__ inline vec2 operator+(const vec2& a, float b) {
            return {a.x + b, a.y + b};
        }

        __host__ __device__ inline vec2 operator-(const vec2& a, const vec2& b) {
            return {a.x - b.x, a.y - b.y};
        }

        __host__ __device__ inline vec2 operator*(const vec2& a, const vec2& b) {
            return {a.x * b.x, a.y * b.y};
        }

        __host__ __device__ inline vec2 operator*(const vec2& a, float b) {
            return {a.x * b, a.y * b};
        }

        __host__ __device__ inline vec2 operator*(float a, const vec2& b) {
            return b * a;
        }

        __host__ __device__ inline vec2 operator/(const vec2& a, const vec2& b) {
            return {a.x / b.x, a.y / b.y};
        }

        __host__ __device__ inline ivec2 operator-(const ivec2& a, int b) {
            return {a.x - b, a.y - b};
        }

        __host__ __device__ inline vec3 operator+(const vec3& a, const vec3& b) {
            return {a.x + b.x, a.y + b.y, a.z + b.z};
        }

        __host__ __device__ inline vec3 operator+(const vec3& a, float b) {
            return {a.x + b, a.y + b, a.z + b};
        }

        __host__ __device__ inline vec3 operator+(float a, const vec3& b) {
            return b + a;
        }

        __host__ __device__ inline vec3 operator-(const vec3& a, const vec3& b) {
            return {a.x - b.x, a.y - b.y, a.z - b.z};
        }

        __host__ __device__ inline vec3 operator-(const vec3& a, float b) {
            return {a.x - b, a.y - b, a.z - b};
        }

        __host__ __device__ inline vec3 operator*(const vec3& a, const vec3& b) {
            return {a.x * b.x, a.y * b.y, a.z * b.z};
        }

        __host__ __device__ inline vec3 operator*(const vec3& a, float b) {
            return {a.x * b, a.y * b, a.z * b};
        }

        __host__ __device__ inline vec3 operator*(float a, const vec3& b) {
            return b * a;
        }

        __host__ __device__ inline vec3 operator/(const vec3& a, const vec3& b) {
            return {a.x / b.x, a.y / b.y, a.z / b.z};
        }

        __host__ __device__ inline vec3 operator/(const vec3& a, float b) {
            return {a.x / b, a.y / b, a.z / b};
        }

        __host__ __device__ inline bool operator==(const vec3& a, const vec3& b) {
            return a.x == b.x && a.y == b.y && a.z == b.z;
        }

        __host__ __device__ inline vec3& operator+=(vec3& a, const vec3& b) {
            a.x += b.x;
            a.y += b.y;
            a.z += b.z;
            return a;
        }

        __host__ __device__ inline vec3& operator*=(vec3& a, float b) {
            a.x *= b;
            a.y *= b;
            a.z *= b;
            return a;
        }

        __host__ __device__ inline vec3& operator/=(vec3& a, float b) {
            a.x /= b;
            a.y /= b;
            a.z /= b;
            return a;
        }

        __host__ __device__ inline ivec2 clamp(const ivec2& value, int low, const ivec2& high) {
            return {
                cuda::std::clamp(value.x, low, high.x),
                cuda::std::clamp(value.y, low, high.y),
            };
        }

        __host__ __device__ inline float dot(const vec3& a, const vec3& b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        __host__ __device__ inline float mean(const vec3& value) {
            return (value.x + value.y + value.z) / 3.0f;
        }

        __host__ __device__ inline int product(const ivec2& value) {
            return value.x * value.y;
        }

        __host__ __device__ inline float length(const vec3& value) {
            return static_cast<float>(cuda::std::sqrt(static_cast<double>(dot(value, value))));
        }

        __host__ __device__ inline float distance(const vec3& a, const vec3& b) {
            return length(a - b);
        }

        __host__ __device__ inline vec3 normalize(const vec3& value) {
            const float len = length(value);
            return len > 0.0f ? value / len : vec3{0.0f};
        }

#if defined(__CUDACC__)
        template <typename T, uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add_gmem(T* dst, const tvec<T, N, ALIGNMENT>& value) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0u; i < N; ++i) atomicAdd(dst + i, value[i]);
        }

        template <uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add_gmem(__half* dst, const tvec<__half, N, ALIGNMENT>& value) {
            static_assert(N % 2u == 0u, "Half vector atomics require an even number of elements.");

            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0u; i < N; i += 2u) atomicAdd(reinterpret_cast<__half2*>(dst + i), __halves2half2(value[i], value[i + 1u]));
        }
#endif
    } // namespace math

    struct Ray final {
        math::vec3 o = {};
        math::vec3 d = {};
    };
    static_assert(sizeof(Ray) == sizeof(float) * 6u);

    template <typename T>
    T next_multiple(T val, T divisor) {
        return ((val + divisor - 1) / divisor) * divisor;
    }

    [[noreturn]] inline void throw_runtime_error(std::string_view message, const std::source_location& location = std::source_location::current()) {
        std::ostringstream stream;
        stream << location.file_name() << ':' << location.line() << ' ' << message;
        throw std::runtime_error{stream.str()};
    }

    inline void check_or_throw(bool condition, std::string_view message = "check failed", const std::source_location& location = std::source_location::current()) {
        if (!condition) throw_runtime_error(message, location);
    }

    inline void cu_check(CUresult result, const std::source_location& location = std::source_location::current()) {
        if (result == CUDA_SUCCESS) return;

        const char* message = nullptr;
        cuGetErrorName(result, &message);
        std::ostringstream stream;
        stream << "CUDA driver call failed: " << (message ? message : "unknown error");
        throw_runtime_error(stream.str(), location);
    }

    inline void cuda_check(cudaError_t result, const std::source_location& location = std::source_location::current()) {
        if (result == cudaSuccess) return;

        std::ostringstream stream;
        stream << "CUDA runtime call failed: " << cudaGetErrorString(result);
        throw_runtime_error(stream.str(), location);
    }

    inline std::deque<bool*>& current_graph_capture_sync_flags() {
        static thread_local std::deque<bool*> s_current_captures;
        return s_current_captures;
    }

    class GpuHeap {
    private:
        template <typename T>
        struct Interval final {
            T start, end;

            [[nodiscard]] Interval intersect(const Interval& other) const {
                return {std::max(start, other.start), std::min(end, other.end)};
            }

            [[nodiscard]] bool empty() const {
                return end <= start;
            }

            [[nodiscard]] T size() const {
                return end - start;
            }
        };

    public:
        GpuHeap() {
            cuda_check(cudaGetDevice(&m_device));

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id         = m_device;

            CUresult granularity_result = cuMemGetAllocationGranularity(&m_granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
            if (granularity_result == CUDA_ERROR_NOT_SUPPORTED)
                m_granularity = 1;
            else
                cu_check(granularity_result);

            size_t free_bytes;
            size_t total_bytes;
            cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes));
            m_max_size              = (total_bytes / m_granularity) * m_granularity;
            m_global_free_intervals = {{0, m_max_size}};

            cu_check(cuMemAddressReserve(&m_base_address, m_max_size, 0, 0, 0));
        }

        GpuHeap(const GpuHeap&)            = delete;
        GpuHeap& operator=(const GpuHeap&) = delete;

        ~GpuHeap() {
            try {
                if (m_base_address) {
                    int previous_device;
                    cuda_check(cudaGetDevice(&previous_device));
                    cuda_check(cudaSetDevice(m_device));
                    try {
                        cuda_check(cudaDeviceSynchronize());
                        if (m_mapped_bytes) {
                            cu_check(cuMemUnmap(m_base_address, m_mapped_bytes));
                        }
                        for (const auto& handle : m_handles) {
                            cu_check(cuMemRelease(handle));
                        }
                        cu_check(cuMemAddressFree(m_base_address, m_max_size));
                    } catch (...) {
                        if (cudaSetDevice(previous_device) != cudaSuccess) std::terminate();
                        throw;
                    }
                    cuda_check(cudaSetDevice(previous_device));
                }
            } catch (const std::runtime_error& error) {
                if (std::string{error.what()}.find("driver shutting down") == std::string::npos) std::fprintf(stderr, "Could not free gpu heap: %s\n", error.what());
            }
        }

        [[nodiscard]] uint8_t* data() const {
            return reinterpret_cast<uint8_t*>(m_base_address);
        }

        size_t allocate(size_t n_bytes, cudaStream_t stream) {
            if (n_bytes == 0) return 0;

            n_bytes = next_multiple(n_bytes, static_cast<size_t>(128));

            if (stream && stream != cudaStreamLegacy) {
                auto& free_intervals             = m_stream_free_intervals[stream];
                Interval<size_t>* best_candidate = nullptr;
                for (auto& interval : free_intervals) {
                    if (interval.size() >= n_bytes && (!best_candidate || interval.size() < best_candidate->size())) best_candidate = &interval;
                }

                if (best_candidate) {
                    const size_t offset = best_candidate->start;
                    best_candidate->start += n_bytes;
                    if (best_candidate->start == best_candidate->end) std::erase_if(free_intervals, [](const Interval<size_t>& interval) { return interval.start == interval.end; });
                    return offset;
                }
            }

            Interval<size_t>* best_candidate = nullptr;
            for (auto& interval : m_global_free_intervals) {
                if (interval.size() >= n_bytes && (!best_candidate || interval.size() < best_candidate->size())) best_candidate = &interval;
            }

            if (!best_candidate) {
                std::ostringstream basic_ostringstream;
                basic_ostringstream << "GpuHeap: failed to allocate " << n_bytes << " bytes from a " << m_max_size << "-byte heap.";
                throw std::runtime_error{basic_ostringstream.str()};
            }

            const size_t offset = best_candidate->start;
            best_candidate->start += n_bytes;
            if (best_candidate->start == best_candidate->end) std::erase_if(m_global_free_intervals, [](const Interval<size_t>& interval) { return interval.start == interval.end; });

            grow(offset + n_bytes);
            return offset;
        }

        void release(size_t offset, size_t n_bytes, cudaStream_t stream) {
            if (n_bytes == 0) return;

            n_bytes = next_multiple(n_bytes, static_cast<size_t>(128));

            if (stream && stream != cudaStreamLegacy) {
                auto& free_intervals = m_stream_free_intervals[stream];
                auto pos             = std::upper_bound(free_intervals.begin(), free_intervals.end(), offset, [](size_t start, const Interval<size_t>& interval) { return start < interval.start; });
                free_intervals.insert(pos, {offset, offset + n_bytes});
                merge_adjacent(free_intervals);
                return;
            }

            cuda_check(cudaDeviceSynchronize());
            insert_global_free_interval({offset, offset + n_bytes});
        }

    private:
        void insert_global_free_interval(Interval<size_t> interval) {
            auto pos = std::upper_bound(m_global_free_intervals.begin(), m_global_free_intervals.end(), interval, [](const Interval<size_t>& a, const Interval<size_t>& b) { return a.start < b.start; });
            m_global_free_intervals.insert(pos, interval);
            merge_adjacent(m_global_free_intervals);
        }

        static void merge_adjacent(std::vector<Interval<size_t>>& intervals) {
            if (intervals.empty()) return;

            size_t j = 0;
            for (size_t i = 1; i < intervals.size(); ++i) {
                Interval<size_t>& prev = intervals[j];
                Interval<size_t>& cur  = intervals[i];
                if (prev.end == cur.start)
                    prev.end = cur.end;
                else {
                    ++j;
                    intervals[j] = cur;
                }
            }
            intervals.resize(j + 1);
        }

        void grow(size_t required_bytes) {
            if (required_bytes <= m_mapped_bytes) return;

            int current_device;
            cuda_check(cudaGetDevice(&current_device));
            if (current_device != m_device) {
                std::ostringstream stream;
                stream << "Attempted to use a GpuHeap of device " << m_device << " from the wrong device " << current_device << '.';
                throw std::runtime_error{stream.str()};
            }

            size_t n_bytes_to_allocate = required_bytes - m_mapped_bytes;
            n_bytes_to_allocate        = next_multiple(n_bytes_to_allocate, m_granularity);

            CUmemAllocationProp prop = {};
            prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id         = m_device;

            m_handles.emplace_back();
            cu_check(cuMemCreate(&m_handles.back(), n_bytes_to_allocate, &prop, 0));

            CUmemAccessDesc access_desc = {};
            access_desc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
            access_desc.location.id     = m_device;
            access_desc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

            cu_check(cuMemMap(m_base_address + m_mapped_bytes, n_bytes_to_allocate, 0, m_handles.back(), 0));
            cu_check(cuMemSetAccess(m_base_address + m_mapped_bytes, n_bytes_to_allocate, &access_desc, 1));
            m_mapped_bytes += n_bytes_to_allocate;

            if (!current_graph_capture_sync_flags().empty())
                *current_graph_capture_sync_flags().front() = true;
            else
                cuda_check(cudaDeviceSynchronize());
        }

        int m_device               = 0;
        CUdeviceptr m_base_address = {};
        size_t m_mapped_bytes      = 0;
        size_t m_granularity       = 0;
        size_t m_max_size          = 0;
        std::vector<CUmemGenericAllocationHandle> m_handles;
        std::vector<Interval<size_t>> m_global_free_intervals;
        std::unordered_map<cudaStream_t, std::vector<Interval<size_t>>> m_stream_free_intervals;
    };

    inline std::unordered_map<int, std::unique_ptr<GpuHeap>>& gpu_heaps() {
        static auto* gpu_heaps = new std::unordered_map<int, std::unique_ptr<GpuHeap>>{};
        return *gpu_heaps;
    }

    inline GpuHeap* gpu_heap() {
        int device;
        cuda_check(cudaGetDevice(&device));
        auto& heap = gpu_heaps()[device];
        if (!heap) heap = std::make_unique<GpuHeap>();
        return heap.get();
    }

    enum class MatrixLayout {
        RowMajor    = 0,
        SoA         = 0, // For data matrices TCNN's convention is RowMajor == SoA (struct of arrays)
        ColumnMajor = 1,
        AoS         = 1,
        Dynamic     = 2,
    };

    static constexpr MatrixLayout RM  = MatrixLayout::RowMajor;
    static constexpr MatrixLayout SoA = MatrixLayout::SoA;
    static constexpr MatrixLayout CM  = MatrixLayout::ColumnMajor;
    static constexpr MatrixLayout AoS = MatrixLayout::AoS;

    class GpuAllocation {
    public:
        GpuAllocation() = default;

        GpuAllocation(size_t n_bytes, cudaStream_t stream = nullptr) {
            if (n_bytes != 0) {
                m_handle          = acquire_handle();
                m_handle->heap    = gpu_heap();
                m_handle->stream  = stream;
                m_handle->n_bytes = next_multiple(n_bytes, static_cast<size_t>(128));
                m_handle->offset  = m_handle->heap->allocate(m_handle->n_bytes, stream);
            }
        }

        ~GpuAllocation() {
            reset();
        }

        GpuAllocation(const GpuAllocation& other) : m_handle{other.m_handle} {
            if (m_handle) ++m_handle->refs;
        }

        GpuAllocation& operator=(const GpuAllocation& other) {
            if (this != &other) {
                reset();
                m_handle = other.m_handle;
                if (m_handle) ++m_handle->refs;
            }
            return *this;
        }

        GpuAllocation(GpuAllocation&& other) noexcept {
            m_handle       = other.m_handle;
            other.m_handle = nullptr;
        }

        GpuAllocation& operator=(GpuAllocation&& other) noexcept {
            if (this != &other) {
                reset();
                m_handle       = other.m_handle;
                other.m_handle = nullptr;
            }
            return *this;
        }

        [[nodiscard]] uint8_t* data() const {
            return m_handle ? m_handle->heap->data() + m_handle->offset : nullptr;
        }

        [[nodiscard]] cudaStream_t stream() const {
            return m_handle ? m_handle->stream : nullptr;
        }

        [[nodiscard]] size_t bytes() const {
            return m_handle ? m_handle->n_bytes : 0;
        }

    private:
        struct Handle {
            uint32_t refs       = 0;
            GpuHeap* heap       = nullptr;
            cudaStream_t stream = nullptr;
            size_t n_bytes      = 0;
            size_t offset       = 0;
        };

        static std::vector<std::unique_ptr<Handle[]>>& handle_pool_chunks() {
            static auto* chunks = new std::vector<std::unique_ptr<Handle[]>>{};
            return *chunks;
        }

        static std::vector<Handle*>& handle_pool_free_handles() {
            static auto* free_handles = new std::vector<Handle*>{};
            return *free_handles;
        }

        static Handle* acquire_handle() {
            auto& chunks       = handle_pool_chunks();
            auto& free_handles = handle_pool_free_handles();

            if (free_handles.empty()) {
                constexpr size_t kChunkSize = 1024;
                auto chunk                  = std::make_unique<Handle[]>(kChunkSize);
                for (size_t i = 0; i < kChunkSize; ++i) {
                    free_handles.push_back(&chunk[i]);
                }
                chunks.push_back(std::move(chunk));
            }

            Handle* handle = free_handles.back();
            free_handles.pop_back();
            handle->refs    = 1;
            handle->heap    = nullptr;
            handle->stream  = nullptr;
            handle->n_bytes = 0;
            handle->offset  = 0;
            return handle;
        }

        static void recycle_handle(Handle* handle) {
            handle_pool_free_handles().push_back(handle);
        }

        void reset() {
            if (!m_handle) return;

            if (--m_handle->refs == 0) {
                if (m_handle->heap && m_handle->n_bytes) m_handle->heap->release(m_handle->offset, m_handle->n_bytes, m_handle->stream);
                recycle_handle(m_handle);
            }

            m_handle = nullptr;
        }

        Handle* m_handle = nullptr;

        template <typename T, MatrixLayout _layout>
        friend class GPUMatrix;
    };


    template <typename T>
    class GpuBuffer {
    public:
        GpuBuffer() = default;
        explicit GpuBuffer(size_t size, cudaStream_t stream = nullptr) {
            resize(size, stream);
        }

        GpuBuffer(GpuBuffer&&) noexcept            = default;
        GpuBuffer& operator=(GpuBuffer&&) noexcept = default;
        GpuBuffer(const GpuBuffer&)                = delete;
        GpuBuffer& operator=(const GpuBuffer&)     = delete;

        void resize(size_t size, cudaStream_t stream = nullptr) {
            if (m_size == size) return;

            m_allocation = size ? GpuAllocation{size * sizeof(T), stream} : GpuAllocation{};
            m_size       = size;
        }

        void enlarge(size_t size, cudaStream_t stream = nullptr) {
            if (size > m_size) resize(size, stream);
        }

        void memset(int value) {
            cuda_check(cudaMemset(data(), value, m_size * sizeof(T)));
        }

        void copy_from_host(const T* host_data, size_t num_elements) {
            if (num_elements > m_size) {
                std::ostringstream stream;
                stream << "Trying to copy " << num_elements << " elements, but memory size is only " << m_size << '.';
                throw std::runtime_error{stream.str()};
            }
            cuda_check(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
        }

        void copy_from_host(const std::vector<T>& host_data) {
            if (host_data.size() < m_size) {
                std::ostringstream stream;
                stream << "Trying to copy " << m_size << " elements, but vector size is only " << host_data.size() << '.';
                throw std::runtime_error{stream.str()};
            }
            copy_from_host(host_data.data(), m_size);
        }

        void copy_to_host(T* host_data, size_t num_elements) const {
            if (num_elements > m_size) {
                std::ostringstream stream;
                stream << "Trying to copy " << num_elements << " elements, but memory size is only " << m_size << '.';
                throw std::runtime_error{stream.str()};
            }
            cuda_check(cudaMemcpy(host_data, data(), num_elements * sizeof(T), cudaMemcpyDeviceToHost));
        }

        void copy_to_host(std::vector<T>& host_data) const {
            if (host_data.size() < m_size) {
                std::ostringstream stream;
                stream << "Trying to copy " << m_size << " elements, but vector size is only " << host_data.size() << '.';
                throw std::runtime_error{stream.str()};
            }
            copy_to_host(host_data.data(), m_size);
        }

        T* data() const {
            return reinterpret_cast<T*>(m_allocation.data());
        }

        [[nodiscard]] size_t size() const {
            return m_size;
        }

    private:
        size_t m_size = 0;
        GpuAllocation m_allocation;
    };

    template <typename T, MatrixLayout _layout>
    class GPUMatrix {
    public:
        static constexpr MatrixLayout static_layout            = _layout;
        static constexpr bool has_static_layout                = static_layout != MatrixLayout::Dynamic;
        static constexpr MatrixLayout static_transposed_layout = has_static_layout ? (static_layout == RM ? CM : RM) : MatrixLayout::Dynamic;

        GPUMatrix(uint32_t m, uint32_t n, MatrixLayout layout = CM) : m_rows{m}, m_cols{n}, m_layout{has_static_layout ? static_layout : layout} {
            m_allocation = GpuAllocation{m * n * sizeof(T)};
            m_data       = (T*) m_allocation.data();
            m_stride     = m_layout == CM ? m_rows : m_cols;
        }

        GPUMatrix(uint32_t m, uint32_t n, cudaStream_t stream, MatrixLayout layout = CM) : m_rows{m}, m_cols{n}, m_layout{has_static_layout ? static_layout : layout} {
            m_allocation = GpuAllocation{m * n * sizeof(T), stream};
            m_data       = (T*) m_allocation.data();
            m_stride     = m_layout == CM ? m_rows : m_cols;
        }

        explicit GPUMatrix(T* data, uint32_t m, uint32_t n, MatrixLayout layout = CM, uint32_t stride = 0, GpuAllocation allocation = {}) : m_data{data}, m_rows{m}, m_cols{n}, m_layout{has_static_layout ? static_layout : layout}, m_allocation{std::move(allocation)} {
            if (stride == 0)
                m_stride = m_layout == CM ? m_rows : m_cols;
            else
                m_stride = stride;
        }

        GPUMatrix() : GPUMatrix{nullptr, 0, 0} {}

        GPUMatrix<T, _layout>& operator=(GPUMatrix<T, _layout>&& other) {
            std::swap(m_data, other.m_data);
            std::swap(m_rows, other.m_rows);
            std::swap(m_cols, other.m_cols);
            std::swap(m_stride, other.m_stride);
            std::swap(m_layout, other.m_layout);
            std::swap(m_allocation, other.m_allocation);
            return *this;
        }

        GPUMatrix(GPUMatrix<T, _layout>&& other) {
            *this = std::move(other);
        }

        GPUMatrix(const GPUMatrix<T, _layout>& other)                        = delete;
        GPUMatrix<T, _layout>& operator=(const GPUMatrix<T, _layout>& other) = delete;

        void set_data_unsafe(void* data) {
            m_data = (T*) data;
        }
        void set_size_unsafe(uint32_t rows, uint32_t cols, uint32_t stride = 0) {
            m_rows = rows;
            m_cols = cols;

            if (stride == 0)
                m_stride = m_layout == CM ? m_rows : m_cols;
            else
                m_stride = stride;
        }

        GPUMatrix<T, _layout> slice_rows(uint32_t offset, uint32_t size) const {
            return GPUMatrix<T, _layout>{data() + (layout() == CM ? offset : offset * stride()), size, n(), layout(), stride(), m_allocation};
        }

        uint32_t m() const {
            return m_rows;
        }

        uint32_t n() const {
            return m_cols;
        }

        uint32_t stride() const {
            return m_stride;
        }
        uint32_t n_elements() const {
            return m_rows * m_cols;
        }
        size_t n_bytes() const {
            return n_elements() * sizeof(T);
        }

        MatrixLayout layout() const {
            return m_layout;
        }
        T* data() const {
            return m_data;
        }

        void initialize_xavier_uniform(math::pcg32& rnd, float scale = 1) {
            check_or_throw(data());
            check_or_throw(m_stride == (m_layout == CM ? m_rows : m_cols));

            scale *= std::sqrt(6.0f / (float) (m_cols + m_rows));

            std::vector<T> new_data(n_elements());

            for (size_t i = 0; i < new_data.size(); ++i) {
                new_data[i] = (T) (rnd.next_float() * 2.0f * scale - scale);
            }

            cuda_check(cudaMemcpy(data(), new_data.data(), n_bytes(), cudaMemcpyHostToDevice));
        }

        GPUMatrix<T, static_transposed_layout> transposed() const {
            return GPUMatrix<T, static_transposed_layout>(data(), n(), m(), m_layout == RM ? CM : RM, stride(), m_allocation);
        }

        GPUMatrix<T, RM> rm() const {
            check_or_throw(m_layout == RM);
            return GPUMatrix<T, RM>(data(), m(), n(), RM, stride(), m_allocation);
        }

        GPUMatrix<T, CM> cm() const {
            check_or_throw(m_layout == CM);
            return GPUMatrix<T, CM>(data(), m(), n(), CM, stride(), m_allocation);
        }

    private:
        T* m_data;
        uint32_t m_rows, m_cols, m_stride;
        MatrixLayout m_layout;

        GpuAllocation m_allocation;
    };

} // namespace ngp::legacy

namespace ngp::network::detail {

    enum class GradientMode {
        Ignore,
        Overwrite,
        Accumulate,
    };

    inline constexpr std::uint32_t batch_size_granularity = 256u;
    inline constexpr std::uint32_t n_threads_linear       = 128u;

#ifdef __CUDACC__
    template <typename T>
    __global__ void cast(const std::uint32_t num_elements, const float* __restrict__ full_precision, T* __restrict__ target) {
        const std::uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= num_elements) return;
        target[i] = static_cast<T>(full_precision[i]);
    }
#endif

    struct AuxStreamSlot;

    struct SyncedStreamReservation final {
        SyncedStreamReservation() = default;
        SyncedStreamReservation(cudaStream_t stream, std::size_t n_streams);
        ~SyncedStreamReservation();
        SyncedStreamReservation& operator=(const SyncedStreamReservation&) = delete;
        SyncedStreamReservation(const SyncedStreamReservation&)            = delete;
        SyncedStreamReservation& operator=(SyncedStreamReservation&& other) noexcept;
        SyncedStreamReservation(SyncedStreamReservation&& other) noexcept;

        AuxStreamSlot* aux_stream_slot = nullptr;
        cudaStream_t aux_stream        = nullptr;
        cudaStream_t main_stream       = nullptr;
    };

} // namespace ngp::network::detail

#endif // NGP_LEGACY_CUH
