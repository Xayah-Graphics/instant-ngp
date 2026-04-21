#ifndef NGP_LEGACY_CUH
#define NGP_LEGACY_CUH

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/utility>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <deque>
#include <functional>
#include <memory>
#include <numeric>
#include <source_location>
#include <sstream>
#include <stack>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(__CUDA_ARCH__)
#define TCNN_PRAGMA_UNROLL _Pragma("unroll")
#else
#define TCNN_PRAGMA_UNROLL
#endif

namespace ngp::legacy {

    inline constexpr std::uint32_t NERF_GRIDSIZE      = 128u;
    inline constexpr std::uint32_t NERF_GRID_N_CELLS  = NERF_GRIDSIZE * NERF_GRIDSIZE * NERF_GRIDSIZE;

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

            __host__ __device__ uint32_t next_uint(uint32_t bound) {

                uint32_t threshold = (~bound + 1u) % bound;

                for (;;) {
                    uint32_t r = next_uint();
                    if (r >= threshold) return r % bound;
                }
            }

            __host__ __device__ float next_float() {

                union {
                    uint32_t u;
                    float f;
                } x;
                x.u = (next_uint() >> 9) | 0x3f800000u;
                return x.f - 1.0f;
            }

            __host__ __device__ double next_double() {

                union {
                    uint64_t u;
                    double d;
                } x;
                x.u = ((uint64_t) next_uint() << 20) | 0x3ff0000000000000ULL;
                return x.d - 1.0;
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

            __host__ __device__ int64_t operator-(const pcg32& other) const {
                uint64_t cur_mult = PCG32_MULT, cur_plus = inc, cur_state = other.state, the_bit = 1u, distance = 0u;

                while (state != cur_state) {
                    if ((state & the_bit) != (cur_state & the_bit)) {
                        cur_state = cur_state * cur_mult + cur_plus;
                        distance |= the_bit;
                    }

                    the_bit <<= 1;
                    cur_plus = (cur_mult + 1ULL) * cur_plus;
                    cur_mult *= cur_mult;
                }

                return (int64_t) distance;
            }

            __host__ __device__ bool operator==(const pcg32& other) const {
                return state == other.state && inc == other.inc;
            }

            __host__ __device__ bool operator!=(const pcg32& other) const {
                return state != other.state || inc != other.inc;
            }

            uint64_t state; // RNG state.  All values are possible.
            uint64_t inc; // Controls which RNG sequence (stream) is selected. Must *always* be odd.
        };

        template <typename T, uint32_t N, size_t ALIGNMENT = sizeof(T)>
        struct tvec;

        template <size_t Requested, typename T>
        inline constexpr size_t vector_alignment_v = Requested != 0 && (Requested & (Requested - 1)) == 0 ? Requested : alignof(T);

        template <typename Vec>
        using vec_value_t = typename Vec::underlying_type;

        template <typename Vec>
        __host__ __device__ inline void fill_vec(Vec& out, vec_value_t<Vec> value) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < Vec::size(); ++i) {
                out[i] = value;
            }
        }

        template <typename ToVec, typename FromVec>
        __host__ __device__ inline void copy_vec(ToVec& out, const FromVec& in) {
            using To = vec_value_t<ToVec>;

            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < ToVec::size(); ++i) {
                out[i] = i < FromVec::size() ? (To) in[i] : (To) 0;
            }
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        struct alignas(vector_alignment_v<ALIGNMENT, T>) tvec {
            using underlying_type = T;

            static constexpr uint32_t size() {
                return N;
            }

            T elems[N];

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T scalar) {
                fill_vec(*this, scalar);
            }

            template <typename U, uint32_t M, size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<U, M, OTHER_ALIGNMENT>& other) {
                copy_vec(*this, other);
            }

            template <typename... Args>
                requires (sizeof...(Args) == N && (... && std::convertible_to<Args, T>) )
            __host__ __device__ tvec(Args... args) : elems{(T) args...} {}

            __host__ __device__ T* data() {
                return elems;
            }

            __host__ __device__ const T* data() const {
                return elems;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return elems[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return elems[index];
            }
        };

        template <typename T, size_t ALIGNMENT>
        struct alignas(vector_alignment_v<ALIGNMENT, T>) tvec<T, 1, ALIGNMENT> {
            using underlying_type = T;

            static constexpr uint32_t size() {
                return 1;
            }

            union {
                T x;
                T r;
            };

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T scalar) : x{scalar} {}

            template <typename U, uint32_t M, size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<U, M, OTHER_ALIGNMENT>& other) {
                copy_vec(*this, other);
            }

            __host__ __device__ T* data() {
                return &x;
            }

            __host__ __device__ const T* data() const {
                return &x;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return data()[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return data()[index];
            }
        };

        template <typename T, size_t ALIGNMENT>
        struct alignas(vector_alignment_v<ALIGNMENT, T>) tvec<T, 2, ALIGNMENT> {
            using underlying_type = T;

            static constexpr uint32_t size() {
                return 2;
            }

            union {
                T x;
                T r;
            };
            union {
                T y;
                T g;
            };

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T scalar) {
                fill_vec(*this, scalar);
            }

            __host__ __device__ tvec(T a, T b) : x{a}, y{b} {}

            template <typename U, uint32_t M, size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<U, M, OTHER_ALIGNMENT>& other) {
                copy_vec(*this, other);
            }

            __host__ __device__ T* data() {
                return &x;
            }

            __host__ __device__ const T* data() const {
                return &x;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return data()[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return data()[index];
            }

            __host__ __device__ tvec<T, 2, ALIGNMENT>& xy() {
                return *this;
            }

            __host__ __device__ const tvec<T, 2, ALIGNMENT>& xy() const {
                return *this;
            }
        };

        template <typename T, size_t ALIGNMENT>
        struct alignas(vector_alignment_v<ALIGNMENT, T>) tvec<T, 3, ALIGNMENT> {
            using underlying_type = T;

            static constexpr uint32_t size() {
                return 3;
            }

            union {
                T x;
                T r;
            };
            union {
                T y;
                T g;
            };
            union {
                T z;
                T b;
            };

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T scalar) {
                fill_vec(*this, scalar);
            }

            __host__ __device__ tvec(T a, T b, T c) : x{a}, y{b}, z{c} {}

            template <typename U, uint32_t M, size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<U, M, OTHER_ALIGNMENT>& other) {
                copy_vec(*this, other);
            }

            template <size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<T, 2, OTHER_ALIGNMENT>& a, T b) : x{a.x}, y{a.y}, z{b} {}

            template <size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(T a, const tvec<T, 2, OTHER_ALIGNMENT>& b) : x{a}, y{b.x}, z{b.y} {}

            __host__ __device__ T* data() {
                return &x;
            }

            __host__ __device__ const T* data() const {
                return &x;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return data()[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return data()[index];
            }

            __host__ __device__ tvec<T, 2>& xy() {
                return *reinterpret_cast<tvec<T, 2>*>(&x);
            }

            __host__ __device__ const tvec<T, 2>& xy() const {
                return *reinterpret_cast<const tvec<T, 2>*>(&x);
            }

            __host__ __device__ tvec<T, 3, ALIGNMENT>& rgb() {
                return *this;
            }

            __host__ __device__ const tvec<T, 3, ALIGNMENT>& rgb() const {
                return *this;
            }
        };

        template <typename T, size_t ALIGNMENT>
        struct alignas(vector_alignment_v<ALIGNMENT, T>) tvec<T, 4, ALIGNMENT> {
            using underlying_type = T;

            static constexpr uint32_t size() {
                return 4;
            }

            union {
                T x;
                T r;
            };
            union {
                T y;
                T g;
            };
            union {
                T z;
                T b;
            };
            union {
                T w;
                T a;
            };

            __host__ __device__ tvec() = default;

            __host__ __device__ tvec(T scalar) {
                fill_vec(*this, scalar);
            }

            __host__ __device__ tvec(T a0, T a1, T a2, T a3) : x{a0}, y{a1}, z{a2}, w{a3} {}

            template <typename U, uint32_t M, size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<U, M, OTHER_ALIGNMENT>& other) {
                copy_vec(*this, other);
            }

            template <size_t OTHER_ALIGNMENT>
            __host__ __device__ tvec(const tvec<T, 3, OTHER_ALIGNMENT>& a0, T a1) : x{a0.x}, y{a0.y}, z{a0.z}, w{a1} {}

            template <size_t ALIGNMENT_A, size_t ALIGNMENT_B>
            __host__ __device__ tvec(const tvec<T, 2, ALIGNMENT_A>& a0, const tvec<T, 2, ALIGNMENT_B>& a1) : x{a0.x}, y{a0.y}, z{a1.x}, w{a1.y} {}

            __host__ __device__ T* data() {
                return &x;
            }

            __host__ __device__ const T* data() const {
                return &x;
            }

            __host__ __device__ T& operator[](uint32_t index) {
                return data()[index];
            }

            __host__ __device__ const T& operator[](uint32_t index) const {
                return data()[index];
            }

            __host__ __device__ tvec<T, 2>& xy() {
                return *reinterpret_cast<tvec<T, 2>*>(&x);
            }

            __host__ __device__ const tvec<T, 2>& xy() const {
                return *reinterpret_cast<const tvec<T, 2>*>(&x);
            }

            __host__ __device__ tvec<T, 3>& rgb() {
                return *reinterpret_cast<tvec<T, 3>*>(&x);
            }

            __host__ __device__ const tvec<T, 3>& rgb() const {
                return *reinterpret_cast<const tvec<T, 3>*>(&x);
            }
        };

        template <typename T>
        __host__ __device__ inline T sign(T value) {
            return value >= (T) 0 ? (T) 1 : (T) -1;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator+(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] + b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator+(const tvec<T, N, ALIGNMENT>& a, T b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] + b;
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator+(T a, const tvec<T, N, ALIGNMENT>& b) {
            return b + a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator-(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] - b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator-(const tvec<T, N, ALIGNMENT>& a, T b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] - b;
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator-(T a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a - b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator*(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] * b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator*(const tvec<T, N, ALIGNMENT>& a, T b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] * b;
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator*(T a, const tvec<T, N, ALIGNMENT>& b) {
            return b * a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator/(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] / b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator/(const tvec<T, N, ALIGNMENT>& a, T b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = a[i] / b;
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> operator-(const tvec<T, N, ALIGNMENT>& value) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = -value[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT>& operator+=(tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                a[i] += b[i];
            }
            return a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT>& operator-=(tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                a[i] -= b[i];
            }
            return a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT>& operator*=(tvec<T, N, ALIGNMENT>& a, T b) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                a[i] *= b;
            }
            return a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT>& operator/=(tvec<T, N, ALIGNMENT>& a, T b) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                a[i] /= b;
            }
            return a;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline bool operator==(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                if (a[i] != b[i]) return false;
            }
            return true;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline bool operator!=(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            return !(a == b);
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> min(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = cuda::std::min(a[i], b[i]);
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> max(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = cuda::std::max(a[i], b[i]);
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> clamp(const tvec<T, N, ALIGNMENT>& value, T low, T high) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = cuda::std::clamp(value[i], low, high);
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> clamp(const tvec<T, N, ALIGNMENT>& value, T low, const tvec<T, N, ALIGNMENT>& high) {
            tvec<T, N, ALIGNMENT> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result[i] = cuda::std::clamp(value[i], low, high[i]);
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T dot(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            T result = (T) 0;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result += a[i] * b[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T sum(const tvec<T, N, ALIGNMENT>& value) {
            T result = (T) 0;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result += value[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T mean(const tvec<T, N, ALIGNMENT>& value) {
            return sum(value) / (T) N;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T product(const tvec<T, N, ALIGNMENT>& value) {
            T result = (T) 1;
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                result *= value[i];
            }
            return result;
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T length2(const tvec<T, N, ALIGNMENT>& value) {
            return dot(value, value);
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T length(const tvec<T, N, ALIGNMENT>& value) {
            return static_cast<T>(cuda::std::sqrt(static_cast<double>(length2(value))));
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline T distance(const tvec<T, N, ALIGNMENT>& a, const tvec<T, N, ALIGNMENT>& b) {
            return length(a - b);
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, N, ALIGNMENT> normalize(const tvec<T, N, ALIGNMENT>& value) {
            const T len = length(value);
            return len > (T) 0 ? value / len : tvec<T, N, ALIGNMENT>{(T) 0};
        }

        template <typename T, size_t ALIGNMENT>
        __host__ __device__ inline tvec<T, 3, ALIGNMENT> cross(const tvec<T, 3, ALIGNMENT>& a, const tvec<T, 3, ALIGNMENT>& b) {
            return {
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            };
        }

        template <uint32_t N>
        using vec = tvec<float, N>;

        template <uint32_t N>
        using ivec = tvec<int, N>;

        template <uint32_t N>
        using uvec = tvec<uint32_t, N>;

        using vec2  = vec<2>;
        using vec3  = vec<3>;
        using vec4  = vec<4>;
        using ivec2 = ivec<2>;
        using ivec3 = ivec<3>;
        using ivec4 = ivec<4>;
        using uvec2 = uvec<2>;
        using uvec3 = uvec<3>;
        using uvec4 = uvec<4>;

#if defined(__CUDACC__)
        template <typename T, uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add(T* dst, const tvec<T, N, ALIGNMENT>& value) {
            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; ++i) {
                atomicAdd(dst + i, value[i]);
            }
        }

        template <typename T, uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add_gmem(T* dst, const tvec<T, N, ALIGNMENT>& value) {
            atomic_add(dst, value);
        }

        template <uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add(__half* dst, const tvec<__half, N, ALIGNMENT>& value) {
            static_assert(N % 2 == 0, "Half vector atomics require an even number of elements.");

            TCNN_PRAGMA_UNROLL
            for (uint32_t i = 0; i < N; i += 2) {
                atomicAdd(reinterpret_cast<__half2*>(dst + i), __halves2half2(value[i], value[i + 1]));
            }
        }

        template <uint32_t N, size_t ALIGNMENT>
        __device__ inline void atomic_add_gmem(__half* dst, const tvec<__half, N, ALIGNMENT>& value) {
            atomic_add(dst, value);
        }
#endif

        template <typename T, uint32_t N, uint32_t M>
        struct tmat {
            using value_type = T;

            tvec<T, M> cols[N];

            __host__ __device__ tmat() = default;

            __host__ __device__ explicit tmat(T diagonal) {
                TCNN_PRAGMA_UNROLL
                for (uint32_t col = 0; col < N; ++col) {
                    TCNN_PRAGMA_UNROLL
                    for (uint32_t row_index = 0; row_index < M; ++row_index) {
                        cols[col][row_index] = col == row_index ? diagonal : (T) 0;
                    }
                }
            }

            template <typename... Args>
                requires (sizeof...(Args) == N * M && (... && std::convertible_to<Args, T>) )
            __host__ __device__ tmat(Args... args) {
                const T values[] = {(T) args...};
                TCNN_PRAGMA_UNROLL
                for (uint32_t col = 0; col < N; ++col) {
                    TCNN_PRAGMA_UNROLL
                    for (uint32_t row_index = 0; row_index < M; ++row_index) {
                        cols[col][row_index] = values[col * M + row_index];
                    }
                }
            }

            template <typename U, uint32_t OTHER_N, uint32_t OTHER_M>
            __host__ __device__ tmat(const tmat<U, OTHER_N, OTHER_M>& other) {
                TCNN_PRAGMA_UNROLL
                for (uint32_t col = 0; col < N; ++col) {
                    TCNN_PRAGMA_UNROLL
                    for (uint32_t row_index = 0; row_index < M; ++row_index) {
                        cols[col][row_index] = col < OTHER_N && row_index < OTHER_M ? (T) other[col][row_index] : (col == row_index ? (T) 1 : (T) 0);
                    }
                }
            }

            __host__ __device__ tvec<T, M>& operator[](uint32_t index) {
                return cols[index];
            }

            __host__ __device__ const tvec<T, M>& operator[](uint32_t index) const {
                return cols[index];
            }

            template <size_t ALIGNMENT>
            __host__ __device__ tvec<T, M> operator*(const tvec<T, N, ALIGNMENT>& value) const {
                tvec<T, M> result{(T) 0};
                TCNN_PRAGMA_UNROLL
                for (uint32_t col = 0; col < N; ++col) {
                    result += cols[col] * value[col];
                }
                return result;
            }
        };

        template <typename T, uint32_t N, uint32_t M>
        inline tvec<T, N> row(const tmat<T, N, M>& matrix, int row_index) {
            tvec<T, N> result;
            TCNN_PRAGMA_UNROLL
            for (uint32_t col = 0; col < N; ++col) {
                result[col] = matrix[col][row_index];
            }
            return result;
        }

        template <typename T, uint32_t N, uint32_t M, typename U, size_t ALIGNMENT>
        inline tmat<T, N, M> row(const tmat<T, N, M>& matrix, int row_index, const tvec<U, N, ALIGNMENT>& value) {
            tmat<T, N, M> result = matrix;
            TCNN_PRAGMA_UNROLL
            for (uint32_t col = 0; col < N; ++col) {
                result[col][row_index] = (T) value[col];
            }
            return result;
        }

        using mat3x3 = tmat<float, 3, 3>;
        using mat4x3 = tmat<float, 4, 3>;
        using mat3   = mat3x3;
    } // namespace math
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


    struct GraphCaptureState {
        cudaGraph_t graph                  = nullptr;
        cudaGraphExec_t graph_instance     = nullptr;
        bool synchronize_when_capture_done = false;
    };


    inline std::deque<GraphCaptureState*>& current_graph_captures() {
        static thread_local std::deque<GraphCaptureState*> s_current_captures;
        return s_current_captures;
    }

    inline std::atomic<size_t>& total_n_bytes_allocated() {
        static std::atomic<size_t> s_total_n_bytes_allocated{0};
        return s_total_n_bytes_allocated;
    }

    inline size_t align_to_cacheline(size_t bytes) {
        return next_multiple(bytes, static_cast<size_t>(128));
    }

    class ScopeGuard {
    public:
        explicit ScopeGuard(std::function<void()> callback) : m_callback{std::move(callback)} {}
        ScopeGuard& operator=(const ScopeGuard&) = delete;
        ScopeGuard(const ScopeGuard&)            = delete;
        ScopeGuard& operator=(ScopeGuard&& other) noexcept {
            std::swap(m_callback, other.m_callback);
            return *this;
        }
        ScopeGuard(ScopeGuard&& other) noexcept {
            *this = std::move(other);
        }
        ~ScopeGuard() {
            if (m_callback) m_callback();
        }

    private:
        std::function<void()> m_callback;
    };

    template <typename T>
    struct Interval {
        T start, end;

        bool operator<(const Interval& other) const {
            return end < other.end || (end == other.end && start < other.start);
        }

        [[nodiscard]] bool overlaps(const Interval& other) const {
            return !intersect(other).empty();
        }

        [[nodiscard]] Interval intersect(const Interval& other) const {
            return {std::max(start, other.start), std::min(end, other.end)};
        }

        [[nodiscard]] bool valid() const {
            return end >= start;
        }

        [[nodiscard]] bool empty() const {
            return end <= start;
        }

        [[nodiscard]] T size() const {
            return end - start;
        }
    };


    class GpuHeap {
    public:
        GpuHeap() {
            cuda_check(cudaGetDevice(&m_device));
            m_alignment = static_cast<size_t>(128);

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
                    ScopeGuard revert_device{[&]() { cuda_check(cudaSetDevice(previous_device)); }};

                    cuda_check(cudaDeviceSynchronize());
                    if (m_mapped_bytes) {
                        total_n_bytes_allocated() -= m_mapped_bytes;
                        cu_check(cuMemUnmap(m_base_address, m_mapped_bytes));
                    }
                    for (const auto& handle : m_handles) {
                        cu_check(cuMemRelease(handle));
                    }
                    cu_check(cuMemAddressFree(m_base_address, m_max_size));
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

            n_bytes = align_to_cacheline(n_bytes);

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

            n_bytes = align_to_cacheline(n_bytes);

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
            total_n_bytes_allocated() += n_bytes_to_allocate;

            if (!current_graph_captures().empty())
                current_graph_captures().front()->synchronize_when_capture_done = true;
            else
                cuda_check(cudaDeviceSynchronize());
        }

        int m_device               = 0;
        CUdeviceptr m_base_address = {};
        size_t m_mapped_bytes      = 0;
        size_t m_alignment         = 0;
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
                m_handle->n_bytes = align_to_cacheline(n_bytes);
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

        explicit operator bool() const {
            return m_handle != nullptr;
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

        void memset(int value, size_t num_elements, size_t offset = 0) {
            if (num_elements + offset > m_size) {
                std::ostringstream stream;
                stream << "Could not set memory: Number of elements " << num_elements << '+' << offset << " larger than allocated memory " << m_size << '.';
                throw std::runtime_error{stream.str()};
            }

            cuda_check(cudaMemset(data() + offset, value, num_elements * sizeof(T)));
        }

        void memset(int value) {
            memset(value, m_size);
        }

        void copy_from_host(const T* host_data, size_t num_elements) {
            if (num_elements > m_size) {
                std::ostringstream stream;
                stream << "Trying to copy " << num_elements << " elements, but memory size is only " << m_size << '.';
                throw std::runtime_error{stream.str()};
            }
            cuda_check(cudaMemcpy(data(), host_data, num_elements * sizeof(T), cudaMemcpyHostToDevice));
        }

        void copy_from_host(const T* host_data) {
            copy_from_host(host_data, m_size);
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

        void copy_to_host(T* host_data) const {
            copy_to_host(host_data, m_size);
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

    template <typename T>
    struct PitchedPtr {
        __host__ __device__ PitchedPtr() : ptr{nullptr}, stride_in_bytes{sizeof(T)} {}
        __host__ __device__ PitchedPtr(T* ptr, size_t stride_in_elements, size_t offset = 0, size_t extra_stride_bytes = 0) : ptr{ptr + offset}, stride_in_bytes{stride_in_elements * sizeof(T) + extra_stride_bytes} {}

        template <typename U>
        __host__ __device__ explicit PitchedPtr(PitchedPtr<U> other) : ptr{(T*) other.ptr}, stride_in_bytes{other.stride_in_bytes} {}

        __host__ __device__ T* operator()(uint32_t y) const {
            return (T*) ((const char*) ptr + y * stride_in_bytes);
        }

        __host__ __device__ void operator+=(uint32_t y) {
            ptr = (T*) ((const char*) ptr + y * stride_in_bytes);
        }

        __host__ __device__ void operator-=(uint32_t y) {
            ptr = (T*) ((const char*) ptr - y * stride_in_bytes);
        }

        __host__ __device__ explicit operator bool() const {
            return ptr;
        }

        T* ptr;
        size_t stride_in_bytes;
    };

    template <typename T, typename STRIDE_T = uint32_t>
    struct MatrixView {
        __host__ __device__ MatrixView() : data{nullptr}, stride_i{0}, stride_j{0} {}
        __host__ __device__ MatrixView(T* data, STRIDE_T stride_i, STRIDE_T stride_j) : data{data}, stride_i{stride_i}, stride_j{stride_j} {}
        __host__ __device__ MatrixView(const MatrixView<std::remove_const_t<T>>& other) : data{other.data}, stride_i{other.stride_i}, stride_j{other.stride_j} {}

        using signed_index_t   = std::make_signed_t<STRIDE_T>;
        using unsigned_index_t = std::make_unsigned_t<STRIDE_T>;

        __host__ __device__ T& operator()(signed_index_t i, signed_index_t j = 0) const {
            return data[i * (std::ptrdiff_t) stride_i + j * (std::ptrdiff_t) stride_j];
        }

        __host__ __device__ void advance(signed_index_t m, signed_index_t n) {
            data += m * (std::ptrdiff_t) stride_i + n * (std::ptrdiff_t) stride_j;
        }

        __host__ __device__ void advance_rows(signed_index_t m) {
            advance(m, 0);
        }

        __host__ __device__ void advance_cols(signed_index_t n) {
            advance(0, n);
        }

        __host__ __device__ T& operator()(unsigned_index_t i, unsigned_index_t j = 0) const {
            return data[i * (size_t) stride_i + j * (size_t) stride_j];
        }

        __host__ __device__ void advance(unsigned_index_t m, unsigned_index_t n) {
            data += m * (size_t) stride_i + n * (size_t) stride_j;
        }

        __host__ __device__ void advance_rows(unsigned_index_t m) {
            advance(m, (unsigned_index_t) 0);
        }

        __host__ __device__ void advance_cols(unsigned_index_t n) {
            advance((unsigned_index_t) 0, n);
        }

        template <uint32_t N>
        __host__ __device__ math::tvec<std::remove_const_t<T>, N> row(unsigned_index_t m) const {
            math::tvec<std::remove_const_t<T>, N> result;
            TCNN_PRAGMA_UNROLL
            for (unsigned_index_t i = 0; i < N; ++i) {
                result[i] = (*this)(m, i);
            }
            return result;
        }

        template <uint32_t N>
        __host__ __device__ math::tvec<std::remove_const_t<T>, N> col(unsigned_index_t n) const {
            math::tvec<std::remove_const_t<T>, N> result;
            TCNN_PRAGMA_UNROLL
            for (unsigned_index_t i = 0; i < N; ++i) {
                result[i] = (*this)(i, n);
            }
            return result;
        }

        template <typename U, uint32_t N, size_t A>
        __host__ __device__ void set_row(unsigned_index_t m, const math::tvec<U, N, A>& val) {
            TCNN_PRAGMA_UNROLL
            for (unsigned_index_t i = 0; i < N; ++i) {
                (*this)(m, i) = val[i];
            }
        }

        template <typename U, uint32_t N, size_t A>
        __host__ __device__ void set_col(unsigned_index_t n, const math::tvec<U, N, A>& val) {
            TCNN_PRAGMA_UNROLL
            for (unsigned_index_t i = 0; i < N; ++i) {
                (*this)(i, n) = val[i];
            }
        }

        __host__ __device__ explicit operator bool() const {
            return data;
        }

        T* data;
        STRIDE_T stride_i, stride_j;
    };

    template <typename T, MatrixLayout _layout>
    class GPUMatrix {
    public:
        static constexpr MatrixLayout static_layout            = _layout;
        static constexpr bool has_static_layout                = static_layout != MatrixLayout::Dynamic;
        static constexpr MatrixLayout static_transposed_layout = has_static_layout ? (static_layout == RM ? CM : RM) : MatrixLayout::Dynamic;

        using Type = T;

        GPUMatrix(uint32_t m, uint32_t n, MatrixLayout layout = CM) : m_rows{m}, m_cols{n}, m_layout{resolve_layout(layout)} {
            m_allocation = GpuAllocation{m * n * sizeof(T)};
            m_data       = (T*) m_allocation.data();
            set_stride_contiguous();
        }

        GPUMatrix(uint32_t m, uint32_t n, cudaStream_t stream, MatrixLayout layout = CM) : m_rows{m}, m_cols{n}, m_layout{resolve_layout(layout)} {
            m_allocation = GpuAllocation{m * n * sizeof(T), stream};
            m_data       = (T*) m_allocation.data();
            set_stride_contiguous();
        }

        explicit GPUMatrix(T* data, uint32_t m, uint32_t n, MatrixLayout layout = CM, uint32_t stride = 0, GpuAllocation allocation = {}) : m_data{data}, m_layout{resolve_layout(layout)}, m_allocation{std::move(allocation)} {
            set(data, m, n, stride);
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

        ~GPUMatrix() {}

        void set_data_unsafe(void* data) {
            m_data = (T*) data;
        }
        void set_size_unsafe(uint32_t rows, uint32_t cols, uint32_t stride = 0) {
            m_rows = rows;
            m_cols = cols;

            if (stride == 0)
                set_stride_contiguous();
            else
                m_stride = stride;
        }

        void set(T* data, uint32_t rows, uint32_t cols, uint32_t stride = 0) {
            set_data_unsafe(data);
            set_size_unsafe(rows, cols, stride);
        }

        void resize(uint32_t rows, uint32_t cols) {
            if (m_allocation || !data()) {
                cudaStream_t stream = m_allocation.stream();
                m_allocation        = GpuAllocation{rows * cols * sizeof(T), stream};
                m_data              = (T*) m_allocation.data();
            } else {
                throw std::runtime_error{"GPUMatrix::resize is not permitted when the underlying memory is not owned. Use GPUMatrix::set instead."};
            }

            set_size_unsafe(rows, cols);
        }

        uint32_t stride_contiguous() const {
            return m_layout == CM ? m() : n();
        }

        bool is_contiguous() const {
            return m_stride == stride_contiguous();
        }

        void set_stride_contiguous() {
            m_stride = stride_contiguous();
        }

        GPUMatrix<T, _layout> slice(uint32_t offset_rows, uint32_t new_rows, uint32_t offset_cols, uint32_t new_cols) const {
            return GPUMatrix<T, _layout>{
                data() + (layout() == CM ? (offset_rows + offset_cols * stride()) : (offset_cols + offset_rows * stride())),
                new_rows,
                new_cols,
                layout(),
                stride(),
                m_allocation,
            };
        }

        GPUMatrix<T, _layout> slice_rows(uint32_t offset, uint32_t size) const {
            return slice(offset, size, 0, cols());
        }

        MatrixView<T> view() const {
            return {data(), layout() == CM ? 1u : stride(), layout() == CM ? stride() : 1u};
        }

        uint32_t rows() const {
            return m_rows;
        }
        uint32_t fan_out() const {
            return m_rows;
        }
        uint32_t m() const {
            return m_rows;
        }

        uint32_t cols() const {
            return m_cols;
        }
        uint32_t fan_in() const {
            return m_cols;
        }
        uint32_t n() const {
            return m_cols;
        }

        uint32_t stride() const {
            return m_stride;
        }
        PitchedPtr<T> pitched_ptr() {
            return {data(), stride()};
        }
        PitchedPtr<const T> pitched_ptr() const {
            return {data(), stride()};
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
        MatrixLayout transposed_layout() const {
            return m_layout == RM ? CM : RM;
        }

        T* data() const {
            return m_data;
        }

        void memset(int value) {
            check_or_throw(data());
            check_or_throw(is_contiguous());
            cuda_check(cudaMemset(data(), value, n_bytes()));
        }

        void initialize_xavier_uniform(math::pcg32& rnd, float scale = 1) {
            check_or_throw(data());
            check_or_throw(is_contiguous());

            scale *= std::sqrt(6.0f / (float) (fan_in() + fan_out()));

            std::vector<T> new_data(n_elements());

            for (size_t i = 0; i < new_data.size(); ++i) {
                new_data[i] = (T) (rnd.next_float() * 2.0f * scale - scale);
            }

            cuda_check(cudaMemcpy(data(), new_data.data(), n_bytes(), cudaMemcpyHostToDevice));
        }

        GPUMatrix<T, static_transposed_layout> transposed() const {
            return GPUMatrix<T, static_transposed_layout>(data(), n(), m(), transposed_layout(), stride(), m_allocation);
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
        static constexpr MatrixLayout resolve_layout(MatrixLayout layout) {
            if constexpr (has_static_layout) {
                return static_layout;
            } else {
                return layout;
            }
        }

        T* m_data;
        uint32_t m_rows, m_cols, m_stride;
        MatrixLayout m_layout;

        GpuAllocation m_allocation;
    };

    template <typename T>
    using GPUMatrixDynamic = GPUMatrix<T, MatrixLayout::Dynamic>;


    struct BoundingBox {
        __host__ __device__ BoundingBox() {}

        __host__ __device__ BoundingBox(const math::vec3& a, const math::vec3& b) : min{a}, max{b} {}

        __host__ __device__ void enlarge(const BoundingBox& other) {
            min = math::min(min, other.min);
            max = math::max(max, other.max);
        }

        __host__ __device__ void enlarge(const math::vec3& point) {
            min = math::min(min, point);
            max = math::max(max, point);
        }

        __host__ __device__ void inflate(float amount) {
            min -= math::vec3(amount);
            max += math::vec3(amount);
        }

        __device__ math::vec3 diag() const {
            return max - min;
        }

        __device__ math::vec3 relative_pos(const math::vec3& pos) const {
            return (pos - min) / diag();
        }

        __device__ math::vec2 ray_intersect(const math::vec3& pos, const math::vec3& dir) const {
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

        __host__ __device__ bool is_empty() const {
            return max.x < min.x || max.y < min.y || max.z < min.z;
        }

        __device__ bool contains(const math::vec3& p) const {
            return p.x >= min.x && p.x <= max.x && p.y >= min.y && p.y <= max.y && p.z >= min.z && p.z <= max.z;
        }

        math::vec3 min = math::vec3(std::numeric_limits<float>::infinity());
        math::vec3 max = math::vec3(-std::numeric_limits<float>::infinity());
    };

    struct NerfPosition {
        __device__ NerfPosition(const math::vec3& pos, float dt) : p{pos} {}
        math::vec3 p;
    };

    struct NerfDirection {
        __device__ NerfDirection(const math::vec3& dir, float dt) : d{dir} {}
        math::vec3 d;
    };

    struct NerfCoordinate {
        __device__ NerfCoordinate(const math::vec3& pos, const math::vec3& dir, float dt) : pos{pos, dt}, dt{dt}, dir{dir, dt} {}

        __device__ void set(const math::vec3& pos, const math::vec3& dir, float dt) {
            this->dt  = dt;
            this->pos = NerfPosition(pos, dt);
            this->dir = NerfDirection(dir, dt);
        }

        NerfPosition pos;
        float dt;
        NerfDirection dir;
    };

} // namespace ngp::legacy

#endif // NGP_LEGACY_CUH
