export module ngp.dataset;
import std;

namespace ngp::dataset {
    export struct NeRFSynthetic final {
        struct Frame {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0;
            std::uint32_t height           = 0;
            float focal_x                  = 0.0f;
            float focal_y                  = 0.0f;
            float principal_x              = 0.0f;
            float principal_y              = 0.0f;
        };
        std::vector<Frame> train      = {};
        std::vector<Frame> validation = {};
        std::vector<Frame> test       = {};
    };

    export struct DdNerfDataset final {
        struct Frame {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0;
            std::uint32_t height           = 0;
            float focal_x                  = 0.0f;
            float focal_y                  = 0.0f;
            float principal_x              = 0.0f;
            float principal_y              = 0.0f;
        };
        std::vector<Frame> train      = {};
        std::vector<Frame> validation = {};
        std::vector<Frame> test       = {};
    };

    export std::expected<NeRFSynthetic, std::string> load_nerf_synthetic(const std::filesystem::path& path);
    export std::expected<DdNerfDataset, std::string> load_dd_nerf_dataset(const std::filesystem::path& path);
} // namespace ngp::dataset
