#ifndef NGP_MATH_VEC_H
#define NGP_MATH_VEC_H

#include <bit>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#define NGP_MATH_VEC_HOST_DEVICE __host__ __device__
#else
#define NGP_MATH_VEC_HOST_DEVICE
#endif

namespace ngp {

    template <typename T, std::size_t Dimension, std::size_t Alignment = 0>
        requires (Dimension > 0)
    struct alignas(Alignment == 0 ? alignof(T) : Alignment) Vec final {
        static_assert(std::is_trivially_copyable_v<T>, "Vec element type must be trivially copyable.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Vec alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(T), "Vec alignment must be at least alignof(T).");

        T elements[Dimension] = {};

        NGP_MATH_VEC_HOST_DEVICE constexpr Vec() noexcept = default;

        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T scalar) noexcept {
            for (std::size_t index = 0; index < Dimension; ++index) {
                elements[index] = scalar;
            }
        }

        template <typename... Arguments>
            requires (sizeof...(Arguments) == Dimension) && (std::convertible_to<Arguments, T> && ...)
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(Arguments&&... arguments) noexcept : elements{static_cast<T>(std::forward<Arguments>(arguments))...} {}

        template <typename U, std::size_t OtherDimension, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, OtherDimension, OtherAlignment>& other) noexcept {
            for (std::size_t index = 0; index < Dimension; ++index) {
                if (index < OtherDimension) {
                    elements[index] = static_cast<T>(other[index]);
                } else {
                    elements[index] = T{};
                }
            }
        }

        [[nodiscard]] static constexpr std::size_t size() noexcept {
            return Dimension;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* data() noexcept {
            return elements;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* data() const noexcept {
            return elements;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T& operator[](std::size_t index) noexcept {
            return elements[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T& operator[](std::size_t index) const noexcept {
            return elements[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* begin() noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* begin() const noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* end() noexcept {
            return data() + Dimension;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* end() const noexcept {
            return data() + Dimension;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr bool operator==(const Vec&) const noexcept = default;
    };

    template <typename T, std::size_t Alignment>
    struct alignas(Alignment == 0 ? alignof(T) : Alignment) Vec<T, 1, Alignment> final {
        static_assert(std::is_trivially_copyable_v<T>, "Vec element type must be trivially copyable.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Vec alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(T), "Vec alignment must be at least alignof(T).");

        T x = {};

        NGP_MATH_VEC_HOST_DEVICE constexpr Vec() noexcept = default;

        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T scalar) noexcept : x{scalar} {}

        template <typename U, std::size_t OtherDimension, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, OtherDimension, OtherAlignment>& other) noexcept : x{OtherDimension > 0 ? static_cast<T>(other[0]) : T{}} {}

        [[nodiscard]] static constexpr std::size_t size() noexcept {
            return 1;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* data() noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* data() const noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T& operator[](std::size_t index) noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T& operator[](std::size_t index) const noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* begin() noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* begin() const noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* end() noexcept {
            return data() + 1;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* end() const noexcept {
            return data() + 1;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr bool operator==(const Vec&) const noexcept = default;
    };

    template <typename T, std::size_t Alignment>
    struct alignas(Alignment == 0 ? alignof(T) : Alignment) Vec<T, 2, Alignment> final {
        static_assert(std::is_trivially_copyable_v<T>, "Vec element type must be trivially copyable.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Vec alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(T), "Vec alignment must be at least alignof(T).");

        T x = {};
        T y = {};

        NGP_MATH_VEC_HOST_DEVICE constexpr Vec() noexcept = default;

        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T scalar) noexcept : x{scalar}, y{scalar} {}

        template <typename XArgument, typename YArgument>
            requires std::convertible_to<XArgument, T> && std::convertible_to<YArgument, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(XArgument&& x_argument, YArgument&& y_argument) noexcept : x{static_cast<T>(std::forward<XArgument>(x_argument))}, y{static_cast<T>(std::forward<YArgument>(y_argument))} {}

        template <typename U, std::size_t OtherDimension, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, OtherDimension, OtherAlignment>& other) noexcept {
            for (std::size_t index = 0; index < 2; ++index) {
                data()[index] = index < OtherDimension ? static_cast<T>(other[index]) : T{};
            }
        }

        [[nodiscard]] static constexpr std::size_t size() noexcept {
            return 2;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* data() noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* data() const noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T& operator[](std::size_t index) noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T& operator[](std::size_t index) const noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* begin() noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* begin() const noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* end() noexcept {
            return data() + 2;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* end() const noexcept {
            return data() + 2;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr bool operator==(const Vec&) const noexcept = default;
    };

    template <typename T, std::size_t Alignment>
    struct alignas(Alignment == 0 ? alignof(T) : Alignment) Vec<T, 3, Alignment> final {
        static_assert(std::is_trivially_copyable_v<T>, "Vec element type must be trivially copyable.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Vec alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(T), "Vec alignment must be at least alignof(T).");

        T x = {};
        T y = {};
        T z = {};

        NGP_MATH_VEC_HOST_DEVICE constexpr Vec() noexcept = default;

        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T scalar) noexcept : x{scalar}, y{scalar}, z{scalar} {}

        template <typename XArgument, typename YArgument, typename ZArgument>
            requires std::convertible_to<XArgument, T> && std::convertible_to<YArgument, T> && std::convertible_to<ZArgument, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(XArgument&& x_argument, YArgument&& y_argument, ZArgument&& z_argument) noexcept : x{static_cast<T>(std::forward<XArgument>(x_argument))}, y{static_cast<T>(std::forward<YArgument>(y_argument))}, z{static_cast<T>(std::forward<ZArgument>(z_argument))} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, 2, OtherAlignment>& first_two, T last) noexcept : x{static_cast<T>(first_two.x)}, y{static_cast<T>(first_two.y)}, z{last} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T first, const Vec<U, 2, OtherAlignment>& last_two) noexcept : x{first}, y{static_cast<T>(last_two.x)}, z{static_cast<T>(last_two.y)} {}

        template <typename U, std::size_t OtherDimension, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, OtherDimension, OtherAlignment>& other) noexcept {
            for (std::size_t index = 0; index < 3; ++index) {
                data()[index] = index < OtherDimension ? static_cast<T>(other[index]) : T{};
            }
        }

        [[nodiscard]] static constexpr std::size_t size() noexcept {
            return 3;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* data() noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* data() const noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T& operator[](std::size_t index) noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T& operator[](std::size_t index) const noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* begin() noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* begin() const noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* end() noexcept {
            return data() + 3;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* end() const noexcept {
            return data() + 3;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr bool operator==(const Vec&) const noexcept = default;
    };

    template <typename T, std::size_t Alignment>
    struct alignas(Alignment == 0 ? alignof(T) : Alignment) Vec<T, 4, Alignment> final {
        static_assert(std::is_trivially_copyable_v<T>, "Vec element type must be trivially copyable.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Vec alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(T), "Vec alignment must be at least alignof(T).");

        T x = {};
        T y = {};
        T z = {};
        T w = {};

        NGP_MATH_VEC_HOST_DEVICE constexpr Vec() noexcept = default;

        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T scalar) noexcept : x{scalar}, y{scalar}, z{scalar}, w{scalar} {}

        template <typename XArgument, typename YArgument, typename ZArgument, typename WArgument>
            requires std::convertible_to<XArgument, T> && std::convertible_to<YArgument, T> && std::convertible_to<ZArgument, T> && std::convertible_to<WArgument, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(XArgument&& x_argument, YArgument&& y_argument, ZArgument&& z_argument, WArgument&& w_argument) noexcept : x{static_cast<T>(std::forward<XArgument>(x_argument))}, y{static_cast<T>(std::forward<YArgument>(y_argument))}, z{static_cast<T>(std::forward<ZArgument>(z_argument))}, w{static_cast<T>(std::forward<WArgument>(w_argument))} {}

        template <typename U, std::size_t FirstAlignment, std::size_t SecondAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, 2, FirstAlignment>& first_two, const Vec<U, 2, SecondAlignment>& last_two) noexcept : x{static_cast<T>(first_two.x)}, y{static_cast<T>(first_two.y)}, z{static_cast<T>(last_two.x)}, w{static_cast<T>(last_two.y)} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, 3, OtherAlignment>& first_three, T last) noexcept : x{static_cast<T>(first_three.x)}, y{static_cast<T>(first_three.y)}, z{static_cast<T>(first_three.z)}, w{last} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T first, const Vec<U, 3, OtherAlignment>& last_three) noexcept : x{first}, y{static_cast<T>(last_three.x)}, z{static_cast<T>(last_three.y)}, w{static_cast<T>(last_three.z)} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, 2, OtherAlignment>& first_two, T third, T fourth) noexcept : x{static_cast<T>(first_two.x)}, y{static_cast<T>(first_two.y)}, z{third}, w{fourth} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T first, const Vec<U, 2, OtherAlignment>& middle_two, T last) noexcept : x{first}, y{static_cast<T>(middle_two.x)}, z{static_cast<T>(middle_two.y)}, w{last} {}

        template <typename U, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(T first, T second, const Vec<U, 2, OtherAlignment>& last_two) noexcept : x{first}, y{second}, z{static_cast<T>(last_two.x)}, w{static_cast<T>(last_two.y)} {}

        template <typename U, std::size_t OtherDimension, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_VEC_HOST_DEVICE explicit constexpr Vec(const Vec<U, OtherDimension, OtherAlignment>& other) noexcept {
            for (std::size_t index = 0; index < 4; ++index) {
                data()[index] = index < OtherDimension ? static_cast<T>(other[index]) : T{};
            }
        }

        [[nodiscard]] static constexpr std::size_t size() noexcept {
            return 4;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* data() noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* data() const noexcept {
            return &x;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T& operator[](std::size_t index) noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T& operator[](std::size_t index) const noexcept {
            return data()[index];
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* begin() noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* begin() const noexcept {
            return data();
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr T* end() noexcept {
            return data() + 4;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr const T* end() const noexcept {
            return data() + 4;
        }

        [[nodiscard]] NGP_MATH_VEC_HOST_DEVICE constexpr bool operator==(const Vec&) const noexcept = default;
    };

} // namespace ngp

#undef NGP_MATH_VEC_HOST_DEVICE

#endif // NGP_MATH_VEC_H
