export module ngp.inspector;
import std;
import ngp.train;

namespace ngp::inspector {
    export struct EvaluationPreviewRequest final {
        std::string_view frame_set;
        std::uint32_t image_index = 0u;
    };

    export struct EvaluationPreviewResult final {
        std::string frame_set;
        std::uint32_t image_index = 0u;
        std::uint32_t step        = 0u;
        std::uint32_t width       = 0u;
        std::uint32_t height      = 0u;
        float mse                 = 0.0f;
        float psnr                = 0.0f;
        float elapsed_ms          = 0.0f;
    };

    export enum class DensityGridEncoding : std::uint32_t {
        MortonFloat32 = 0u,
    };

    export struct DensityGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t cell_count     = 0u;
        const float* values          = nullptr;
        std::uint64_t byte_size      = 0u;
        std::uint64_t revision       = 0u;
        float optical_thickness_step = 0.0f;
        DensityGridEncoding encoding{DensityGridEncoding::MortonFloat32};
        bool initialized = false;
    };

    export enum class OccupancyGridEncoding : std::uint32_t {
        MortonBitfield = 0u,
    };

    export struct OccupancyGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t cell_count     = 0u;
        const std::uint8_t* bitfield = nullptr;
        std::uint64_t bitfield_bytes = 0u;
        std::uint32_t occupied_cells = 0u;
        std::uint64_t revision       = 0u;
        OccupancyGridEncoding encoding{OccupancyGridEncoding::MortonBitfield};
        bool initialized = false;
    };

    export enum class ColorGridEncoding : std::uint32_t {
        MortonFloat32x3 = 0u,
    };

    export struct ColorGridSampleRequest final {
        std::array<std::uint32_t, 3u> dimensions{};
        float* output_rgb       = nullptr;
        std::uint64_t byte_size = 0u;
        std::array<float, 3u> reference_direction{};
        ColorGridEncoding encoding{ColorGridEncoding::MortonFloat32x3};
    };

    export struct ColorGridSampleStats final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t byte_size = 0u;
        std::uint64_t revision  = 0u;
        ColorGridEncoding encoding{ColorGridEncoding::MortonFloat32x3};
    };

    export inline constexpr std::uint64_t SamplerPointInstanceBytes   = static_cast<std::uint64_t>(8u * sizeof(float));
    export inline constexpr std::uint64_t SamplerSegmentInstanceBytes = static_cast<std::uint64_t>(12u * sizeof(float));

    export struct SamplerBatchDeviceView final {
        std::uint32_t current_step{};
        std::uint32_t ray_count{};
        std::uint32_t compacted_sample_count{};
        const float* rays{};
        const std::uint32_t* numsteps{};
        const float* compacted_sample_coords{};
        const float* loss_values{};
        std::uint64_t revision{};
        bool initialized{};
    };

    export struct SamplerVisualizationRequest final {
        std::byte* point_instances{};
        std::uint64_t point_byte_size{};
        std::byte* segment_instances{};
        std::uint64_t segment_byte_size{};
        float point_radius{0.002f};
        float ray_width{1.5f};
        std::uint32_t width_mode{};
    };

    export struct SamplerVisualizationStats final {
        std::uint32_t ray_count{};
        std::uint32_t point_count{};
        std::uint64_t point_byte_size{};
        std::uint64_t segment_byte_size{};
        std::uint64_t revision{};
    };

    export struct Inspector final {
        explicit Inspector(const train::InstantNGP& trainer);

        [[nodiscard]] std::expected<EvaluationPreviewResult, std::string> evaluate_preview(EvaluationPreviewRequest request) const;
        [[nodiscard]] DensityGridDeviceView density_grid_device_view() const;
        [[nodiscard]] OccupancyGridDeviceView occupancy_grid_device_view() const;
        [[nodiscard]] std::expected<ColorGridSampleStats, std::string> sample_color_grid(ColorGridSampleRequest request) const;
        [[nodiscard]] SamplerBatchDeviceView sampler_batch_device_view() const;
        [[nodiscard]] std::expected<SamplerVisualizationStats, std::string> write_sampler_visualization(SamplerVisualizationRequest request) const;

        const train::InstantNGP* trainer = nullptr;
    };
} // namespace ngp::inspector
