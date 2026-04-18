#ifndef NGP_ENCODING_CUH
#define NGP_ENCODING_CUH

#include "legacy.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <span>
#include <vector>

namespace ngp {

    class HashGridEncoding final {
    public:
        enum class Storage : std::uint32_t {
            Hash,
            Dense,
            Tiled,
        };

        struct Config final {
            std::uint32_t input_dimensions = 0;
            std::uint32_t level_count = 0;
            std::uint32_t features_per_level = 0;
            std::uint32_t log2_hashmap_size = 0;
            std::uint32_t base_resolution = 0;
            float per_level_scale = 0.0f;
            Storage storage = Storage::Hash;
        };

        explicit HashGridEncoding(const Config& config);
        ~HashGridEncoding() = default;

        HashGridEncoding(const HashGridEncoding&) = delete;
        HashGridEncoding& operator=(const HashGridEncoding&) = delete;
        HashGridEncoding(HashGridEncoding&&) noexcept = default;
        HashGridEncoding& operator=(HashGridEncoding&&) noexcept = default;

        [[nodiscard]] static constexpr std::uint32_t required_batch_granularity() noexcept {
            return 256;
        }

        [[nodiscard]] const Config& config() const noexcept {
            return config_;
        }

        [[nodiscard]] std::uint32_t output_width() const noexcept {
            return output_width_;
        }

        [[nodiscard]] std::size_t parameter_count() const noexcept {
            return parameter_count_;
        }

        [[nodiscard]] const __half* device_parameters() const noexcept {
            return parameters_.data();
        }

        void upload_parameters(std::span<const float> host_parameters);
        void initialize_parameters(std::uint64_t seed, float scale = 1.0f);

        // `input` uses contiguous SoA layout: `input[dimension * batch_size + sample_index]`.
        // `output` uses contiguous SoA layout: `output[feature_index * batch_size + sample_index]`.
        // Input positions are expected to already be normalized into `[0, 1]`.
        void encode(cudaStream_t stream, const float* input, std::uint32_t batch_size, __half* output) const;

    private:
        Config config_ = {};
        std::uint32_t output_width_ = 0;
        float log2_per_level_scale_ = 0.0f;
        std::size_t parameter_count_ = 0;
        legacy::GpuBuffer<__half> parameters_ = {};
        legacy::GpuBuffer<std::uint32_t> level_offsets_gpu_ = {};
        std::vector<std::uint32_t> level_offsets_cpu_ = {};
        bool has_parameters_ = false;
    };

    class SphericalHarmonicsEncoding final {
    public:
        struct Config final {
            std::uint32_t input_dimensions = 0;
            std::uint32_t degree = 0;
        };

        explicit SphericalHarmonicsEncoding(const Config& config);
        ~SphericalHarmonicsEncoding() = default;

        SphericalHarmonicsEncoding(const SphericalHarmonicsEncoding&) = delete;
        SphericalHarmonicsEncoding& operator=(const SphericalHarmonicsEncoding&) = delete;
        SphericalHarmonicsEncoding(SphericalHarmonicsEncoding&&) noexcept = default;
        SphericalHarmonicsEncoding& operator=(SphericalHarmonicsEncoding&&) noexcept = default;

        [[nodiscard]] static constexpr std::uint32_t required_batch_granularity() noexcept {
            return 256;
        }

        [[nodiscard]] const Config& config() const noexcept {
            return config_;
        }

        [[nodiscard]] std::uint32_t output_width() const noexcept {
            return output_width_;
        }

        // `input` uses contiguous SoA layout: `input[dimension * batch_size + sample_index]`.
        // `output` uses contiguous SoA layout: `output[feature_index * batch_size + sample_index]`.
        // Input directions are expected to already be normalized into `[-1, 1]`.
        void encode(cudaStream_t stream, const float* input, std::uint32_t batch_size, __half* output) const;

    private:
        Config config_ = {};
        std::uint32_t output_width_ = 0;
    };

} // namespace ngp

#endif // NGP_ENCODING_CUH
