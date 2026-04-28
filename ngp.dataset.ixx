export module ngp.dataset;
import std;

namespace ngp::dataset {
    export struct NeRFSynthetic final {
        struct Frame {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0;
            std::uint32_t height           = 0;
            float focal_length             = 0.0f;
        };
        std::vector<Frame> train      = {};
        std::vector<Frame> validation = {};
        std::vector<Frame> test       = {};
    };

    export struct NeRFLlffData final {
        struct Frame {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0;
            std::uint32_t height           = 0;
            float focal_length             = 0.0f;
            std::filesystem::path image_path;
            std::size_t source_index = 0uz;
            float near_bound         = 0.0f;
            float far_bound          = 0.0f;
        };
        std::vector<Frame> train      = {};
        std::vector<Frame> validation = {};
        std::vector<Frame> test       = {};
    };

    export struct NeRFReal360 final {
        struct Frame {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0;
            std::uint32_t height           = 0;
            float focal_length             = 0.0f;
            std::filesystem::path image_path;
            std::size_t source_index = 0uz;
            float near_bound         = 0.0f;
            float far_bound          = 0.0f;
        };
        std::vector<Frame> train      = {};
        std::vector<Frame> validation = {};
        std::vector<Frame> test       = {};
    };

    export std::expected<NeRFSynthetic, std::string> load_nerf_synthetic(const std::filesystem::path& path);
    export std::expected<NeRFLlffData, std::string> load_nerf_llff_data(const std::filesystem::path& path);
    export std::expected<NeRFReal360, std::string> load_nerf_real_360(const std::filesystem::path& path);
} // namespace ngp::dataset
