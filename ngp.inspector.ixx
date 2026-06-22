export module ngp.inspector;
import std;
import ngp.train;

namespace ngp::inspector {
    export struct EvaluationPreviewRequest final {
        std::string_view frame_set;
        std::uint32_t image_index = 0u;
        bool refresh_acceleration = false;
    };

    export struct EvaluationPreviewResult final {
        std::string frame_set;
        std::uint32_t image_index = 0u;
        std::uint32_t step = 0u;
        std::uint32_t width = 0u;
        std::uint32_t height = 0u;
        float mse = 0.0f;
        float psnr = 0.0f;
        float elapsed_ms = 0.0f;
    };

    export enum class DensityGridEncoding : std::uint32_t {
        MortonFloat32 = 0u,
    };

    export struct DensityGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t cell_count = 0u;
        const float* values = nullptr;
        std::uint64_t byte_size = 0u;
        std::uint64_t revision = 0u;
        float optical_thickness_step = 0.0f;
        DensityGridEncoding encoding{DensityGridEncoding::MortonFloat32};
        bool initialized = false;
    };

    export enum class OccupancyGridEncoding : std::uint32_t {
        MortonBitfield = 0u,
    };

    export struct OccupancyGridDeviceView final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t cell_count = 0u;
        const std::uint8_t* bitfield = nullptr;
        std::uint64_t bitfield_bytes = 0u;
        std::uint32_t occupied_cells = 0u;
        std::uint64_t revision = 0u;
        OccupancyGridEncoding encoding{OccupancyGridEncoding::MortonBitfield};
        bool initialized = false;
    };

    export enum class ColorGridEncoding : std::uint32_t {
        MortonFloat32x3 = 0u,
    };

    export struct ColorGridSampleRequest final {
        std::array<std::uint32_t, 3u> dimensions{};
        float* output_rgb = nullptr;
        std::uint64_t byte_size = 0u;
        std::array<float, 3u> reference_direction{};
        ColorGridEncoding encoding{ColorGridEncoding::MortonFloat32x3};
    };

    export struct ColorGridSampleStats final {
        std::array<std::uint32_t, 3u> dimensions{};
        std::uint64_t byte_size = 0u;
        std::uint64_t revision = 0u;
        ColorGridEncoding encoding{ColorGridEncoding::MortonFloat32x3};
    };

    export struct Inspector final {
        explicit Inspector(const train::InstantNGP& trainer);

        [[nodiscard]] std::expected<EvaluationPreviewResult, std::string> evaluate_preview(EvaluationPreviewRequest request) const;
        [[nodiscard]] DensityGridDeviceView density_grid_device_view() const;
        [[nodiscard]] OccupancyGridDeviceView occupancy_grid_device_view() const;
        [[nodiscard]] std::expected<ColorGridSampleStats, std::string> sample_color_grid(ColorGridSampleRequest request) const;

        const train::InstantNGP* trainer = nullptr;
    };
}
