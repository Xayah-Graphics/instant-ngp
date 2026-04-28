module;
#include "json/json.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
module ngp.dataset;
import std;

namespace ngp::dataset {
    namespace {
        inline constexpr float NERF_SYNTHETIC_SCENE_SCALE  = 0.33f;
        inline constexpr float NERF_SYNTHETIC_SCENE_OFFSET = 0.5f;

        std::vector<NeRFSynthetic::Frame> load_nerf_synthetic_split(const std::filesystem::path& dataset_path, const std::string_view file_name) {
            const std::filesystem::path json_path  = dataset_path / file_name;
            const std::filesystem::path split_root = json_path.parent_path();
            const nlohmann::json json              = nlohmann::json::parse(std::ifstream{json_path, std::ios::binary}, nullptr, true, true);

            const float camera_angle_x        = json.at("camera_angle_x").get<float>();
            const nlohmann::json& frames_json = json.at("frames");
            const std::size_t frame_count     = frames_json.size();

            std::vector<NeRFSynthetic::Frame> frames(frame_count);
            const std::vector<std::size_t> indices = std::views::iota(0uz, frame_count) | std::ranges::to<std::vector<std::size_t>>();

            std::for_each(std::execution::par, indices.begin(), indices.end(), [&](const std::size_t frame_index) {
                const nlohmann::json& frame_json = frames_json.at(frame_index);

                const std::filesystem::path image_path = [&] {
                    std::filesystem::path path = frame_json.at("file_path").get<std::string>();
                    path.make_preferred();
                    if (path.extension().empty()) path.replace_extension(".png");
                    if (path.is_relative()) path = split_root / path;
                    return path.lexically_normal();
                }();

                int width           = 0;
                int height          = 0;
                int component_count = 0;

                const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{stbi_load(image_path.string().c_str(), &width, &height, &component_count, 4), stbi_image_free};

                const auto width_u          = static_cast<std::uint32_t>(width);
                const auto height_u         = static_cast<std::uint32_t>(height);
                const std::size_t rgba_size = static_cast<std::size_t>(width_u) * static_cast<std::size_t>(height_u) * 4uz;

                std::array<std::array<float, 4>, 4> camera{};

                const nlohmann::json& transform_matrix = frame_json.at("transform_matrix");

                for (const std::size_t row : std::views::iota(0uz, 4uz))
                    for (const std::size_t column : std::views::iota(0uz, 4uz)) camera[column][row] = transform_matrix.at(row).at(column).get<float>();


                std::ranges::for_each(camera[1], [](float& value) { value = -value; });
                std::ranges::for_each(camera[2], [](float& value) { value = -value; });

                camera[3][0] = camera[3][0] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;
                camera[3][1] = camera[3][1] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;
                camera[3][2] = camera[3][2] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;

                const std::array camera_row0{camera[0][0], camera[1][0], camera[2][0], camera[3][0]};
                const std::array camera_row1{camera[0][1], camera[1][1], camera[2][1], camera[3][1]};
                const std::array camera_row2{camera[0][2], camera[1][2], camera[2][2], camera[3][2]};
                const std::array ngp_camera{camera_row1[0], camera_row2[0], camera_row0[0], camera_row1[1], camera_row2[1], camera_row0[1], camera_row1[2], camera_row2[2], camera_row0[2], camera_row1[3], camera_row2[3], camera_row0[3]};
                const float focal_length = 0.5f * static_cast<float>(width_u) / std::tan(camera_angle_x * 0.5f);

                frames[frame_index] = NeRFSynthetic::Frame{
                    .rgba         = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size},
                    .camera       = ngp_camera,
                    .width        = width_u,
                    .height       = height_u,
                    .focal_length = focal_length,
                };
            });

            return frames;
        }
    } // namespace
    std::expected<NeRFSynthetic, std::string> load_nerf_synthetic(const std::filesystem::path& path) {
        return NeRFSynthetic{
            .train      = load_nerf_synthetic_split(path, "transforms_train.json"),
            .validation = load_nerf_synthetic_split(path, "transforms_val.json"),
            .test       = load_nerf_synthetic_split(path, "transforms_test.json"),
        };
    }
} // namespace ngp::dataset
