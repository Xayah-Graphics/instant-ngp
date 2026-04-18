#ifndef NGP_LEGACY_CUH
#define NGP_LEGACY_CUH

#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace ngp::legacy {

    struct Int2 final {
        int x = 0;
        int y = 0;
    };

    struct Mat4x3 final {
        float columns[4][3] = {};
    };

    inline constexpr float nerf_scale = 0.33f;

    [[nodiscard]] inline Mat4x3 nerf_matrix_to_ngp(const std::array<float, 16>& transform_matrix_4x4, bool scale_columns = false) noexcept {
        Mat4x3 result{};

        for (std::size_t row_index = 0; row_index < 3; ++row_index) {
            for (std::size_t column_index = 0; column_index < 4; ++column_index) {
                result.columns[column_index][row_index] = transform_matrix_4x4[row_index * 4 + column_index];
            }
        }

        const float column_scale_x = scale_columns ? nerf_scale : 1.0f;
        const float column_scale_y = scale_columns ? -nerf_scale : -1.0f;
        const float column_scale_z = scale_columns ? -nerf_scale : -1.0f;

        for (std::size_t row_index = 0; row_index < 3; ++row_index) {
            result.columns[0][row_index] *= column_scale_x;
            result.columns[1][row_index] *= column_scale_y;
            result.columns[2][row_index] *= column_scale_z;
            result.columns[3][row_index] = result.columns[3][row_index] * nerf_scale + 0.5f;
        }

        for (std::size_t column_index = 0; column_index < 4; ++column_index) {
            const float previous_row_0 = result.columns[column_index][0];
            result.columns[column_index][0] = result.columns[column_index][1];
            result.columns[column_index][1] = result.columns[column_index][2];
            result.columns[column_index][2] = previous_row_0;
        }

        return result;
    }

    template <typename T>
    class GpuBuffer final {
    public:
        static_assert(std::is_trivially_copyable_v<T>, "GpuBuffer requires trivially copyable element types.");

        GpuBuffer() = default;
        ~GpuBuffer() {
            if (data_ != nullptr) {
                cudaFree(data_);
                data_ = nullptr;
                size_ = 0;
            }
        }

        GpuBuffer(const GpuBuffer&) = delete;
        GpuBuffer& operator=(const GpuBuffer&) = delete;

        GpuBuffer(GpuBuffer&& other) noexcept : data_{other.data_}, size_{other.size_} {
            other.data_ = nullptr;
            other.size_ = 0;
        }

        GpuBuffer& operator=(GpuBuffer&& other) noexcept {
            if (this == &other) {
                return *this;
            }

            if (data_ != nullptr) {
                cudaFree(data_);
            }

            data_ = other.data_;
            size_ = other.size_;

            other.data_ = nullptr;
            other.size_ = 0;

            return *this;
        }

        [[nodiscard]] T* data() noexcept {
            return data_;
        }

        [[nodiscard]] const T* data() const noexcept {
            return data_;
        }

        [[nodiscard]] std::size_t size() const noexcept {
            return size_;
        }

        [[nodiscard]] std::size_t bytes() const noexcept {
            return size_ * sizeof(T);
        }

        void resize(std::size_t new_size) {
            if (new_size == size_) {
                return;
            }

            if (data_ != nullptr) {
                const cudaError_t free_status = cudaFree(data_);
                if (free_status != cudaSuccess) {
                    throw std::runtime_error{std::string{"GpuBuffer failed to release memory: "} + cudaGetErrorString(free_status)};
                }
                data_ = nullptr;
                size_ = 0;
            }

            if (new_size == 0) {
                return;
            }

            const cudaError_t allocation_status = cudaMalloc(reinterpret_cast<void**>(&data_), new_size * sizeof(T));
            if (allocation_status != cudaSuccess) {
                throw std::runtime_error{std::string{"GpuBuffer failed to allocate memory: "} + cudaGetErrorString(allocation_status)};
            }

            size_ = new_size;
        }

        void copy_from_host(const T* host_data) {
            if (size_ == 0) {
                return;
            }
            if (host_data == nullptr) {
                throw std::runtime_error{"GpuBuffer::copy_from_host requires a non-null host pointer when size() > 0."};
            }

            const cudaError_t copy_status = cudaMemcpy(data_, host_data, bytes(), cudaMemcpyHostToDevice);
            if (copy_status != cudaSuccess) {
                throw std::runtime_error{std::string{"GpuBuffer failed to copy from host: "} + cudaGetErrorString(copy_status)};
            }
        }

        void copy_from_host_async(const T* host_data, cudaStream_t stream) {
            if (size_ == 0) {
                return;
            }
            if (host_data == nullptr) {
                throw std::runtime_error{"GpuBuffer::copy_from_host_async requires a non-null host pointer when size() > 0."};
            }

            const cudaError_t copy_status = cudaMemcpyAsync(data_, host_data, bytes(), cudaMemcpyHostToDevice, stream);
            if (copy_status != cudaSuccess) {
                throw std::runtime_error{std::string{"GpuBuffer failed to copy from host asynchronously: "} + cudaGetErrorString(copy_status)};
            }
        }

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;
    };

} // namespace ngp::legacy

#endif // NGP_LEGACY_CUH
