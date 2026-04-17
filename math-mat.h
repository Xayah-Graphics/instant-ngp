#ifndef NGP_MATH_MAT_H
#define NGP_MATH_MAT_H

#include "math-vec.h"
#include <bit>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

#if defined(__CUDACC__)
#define NGP_MATH_MAT_HOST_DEVICE __host__ __device__
#else
#define NGP_MATH_MAT_HOST_DEVICE
#endif

namespace ngp {

    template <typename T, std::size_t ColumnCount, std::size_t RowCount, std::size_t Alignment = 0>
        requires (ColumnCount > 0) && (RowCount > 0)
    struct alignas(Alignment == 0 ? alignof(Vec<T, RowCount>) : Alignment) Mat final {
        static_assert(std::is_trivially_copyable_v<T>, "Mat scalar type must be trivially copyable.");
        static_assert(std::is_trivially_copyable_v<Vec<T, RowCount>>, "Mat column type must be trivially copyable.");
        static_assert(sizeof(Vec<T, RowCount>) == sizeof(T) * RowCount, "Mat requires tightly packed column vectors.");
        static_assert(Alignment == 0 || std::has_single_bit(Alignment), "Mat alignment must be 0 or a power of two.");
        static_assert(Alignment == 0 || Alignment >= alignof(Vec<T, RowCount>), "Mat alignment must be at least alignof(Vec<T, RowCount>).");

        Vec<T, RowCount> columns[ColumnCount] = {};

        NGP_MATH_MAT_HOST_DEVICE constexpr Mat() noexcept = default;

        NGP_MATH_MAT_HOST_DEVICE explicit constexpr Mat(T diagonal) noexcept {
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                for (std::size_t row_index = 0; row_index < RowCount; ++row_index) {
                    columns[column_index][row_index] = column_index == row_index ? diagonal : T{};
                }
            }
        }

        template <typename... Arguments>
            requires (sizeof...(Arguments) == ColumnCount * RowCount) && (std::convertible_to<Arguments, T> && ...)
        NGP_MATH_MAT_HOST_DEVICE explicit constexpr Mat(Arguments&&... arguments) noexcept {
            const T values[ColumnCount * RowCount] = {static_cast<T>(std::forward<Arguments>(arguments))...};
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                for (std::size_t row_index = 0; row_index < RowCount; ++row_index) {
                    columns[column_index][row_index] = values[column_index * RowCount + row_index];
                }
            }
        }

        template <typename U, std::size_t OtherColumnCount, std::size_t OtherRowCount, std::size_t OtherAlignment>
            requires std::convertible_to<U, T>
        NGP_MATH_MAT_HOST_DEVICE explicit constexpr Mat(const Mat<U, OtherColumnCount, OtherRowCount, OtherAlignment>& other) noexcept {
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                for (std::size_t row_index = 0; row_index < RowCount; ++row_index) {
                    if (column_index < OtherColumnCount && row_index < OtherRowCount) {
                        columns[column_index][row_index] = static_cast<T>(other[column_index][row_index]);
                    } else {
                        columns[column_index][row_index] = column_index == row_index ? T{1} : T{};
                    }
                }
            }
        }

        [[nodiscard]] static constexpr std::size_t column_count() noexcept {
            return ColumnCount;
        }

        [[nodiscard]] static constexpr std::size_t row_count() noexcept {
            return RowCount;
        }

        [[nodiscard]] static constexpr std::size_t scalar_count() noexcept {
            return ColumnCount * RowCount;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr T* data() noexcept {
            return columns[0].data();
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr const T* data() const noexcept {
            return columns[0].data();
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr Vec<T, RowCount>& operator[](std::size_t column_index) noexcept {
            return columns[column_index];
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr const Vec<T, RowCount>& operator[](std::size_t column_index) const noexcept {
            return columns[column_index];
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr Vec<T, ColumnCount> row(std::size_t row_index) const noexcept {
            Vec<T, ColumnCount> result{};
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                result[column_index] = columns[column_index][row_index];
            }
            return result;
        }

        template <std::size_t VectorAlignment>
        NGP_MATH_MAT_HOST_DEVICE constexpr void set_row(std::size_t row_index, const Vec<T, ColumnCount, VectorAlignment>& values) noexcept {
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                columns[column_index][row_index] = values[column_index];
            }
        }

        template <std::size_t VectorAlignment>
        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr Vec<T, RowCount> operator*(const Vec<T, ColumnCount, VectorAlignment>& vector) const noexcept {
            Vec<T, RowCount> result{};
            for (std::size_t column_index = 0; column_index < ColumnCount; ++column_index) {
                for (std::size_t row_index = 0; row_index < RowCount; ++row_index) {
                    result[row_index] += columns[column_index][row_index] * vector[column_index];
                }
            }
            return result;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr Vec<T, RowCount>* begin() noexcept {
            return columns;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr const Vec<T, RowCount>* begin() const noexcept {
            return columns;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr Vec<T, RowCount>* end() noexcept {
            return columns + ColumnCount;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr const Vec<T, RowCount>* end() const noexcept {
            return columns + ColumnCount;
        }

        [[nodiscard]] NGP_MATH_MAT_HOST_DEVICE constexpr bool operator==(const Mat&) const noexcept = default;
    };

} // namespace ngp

#undef NGP_MATH_MAT_HOST_DEVICE

#endif // NGP_MATH_MAT_H
