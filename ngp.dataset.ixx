export module ngp.dataset;
import std;

namespace ngp::dataset {
    export inline constexpr float DEFAULT_SCENE_SCALE = 0.33f;

    export struct Frame final {
        std::vector<std::uint8_t> rgba = {};
        std::array<float, 12> camera   = {};
        std::uint32_t width            = 0;
        std::uint32_t height           = 0;
        float focal_x                  = 0.0f;
        float focal_y                  = 0.0f;
        float principal_x              = 0.0f;
        float principal_y              = 0.0f;
    };

    export struct FrameSet final {
        std::string name;
        std::vector<Frame> frames = {};
    };

    export struct NGPDataset final {
        std::vector<FrameSet> frame_sets = {};
        float scene_scale                = 0.0f;
    };

    export struct DatasetLoadPlan final {
        std::vector<std::string> frame_sets = {"train", "validation"};
    };

    export std::expected<NGPDataset, std::string> load_nerf_synthetic(const std::filesystem::path& path, float scene_scale = DEFAULT_SCENE_SCALE, DatasetLoadPlan plan = {});
    export std::expected<NGPDataset, std::string> load_dd_nerf_dataset(const std::filesystem::path& path, float scene_scale = DEFAULT_SCENE_SCALE, DatasetLoadPlan plan = {});
} // namespace ngp::dataset
