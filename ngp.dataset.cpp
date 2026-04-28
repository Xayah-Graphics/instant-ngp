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
                if (raw_pixels == nullptr) throw std::runtime_error{std::format("failed to load image '{}'.", image_path.string())};
                if (width <= 0 || height <= 0) throw std::runtime_error{std::format("image '{}' has invalid dimensions.", image_path.string())};

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
                    .rgba        = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size},
                    .camera      = ngp_camera,
                    .width       = width_u,
                    .height      = height_u,
                    .focal_x     = focal_length,
                    .focal_y     = focal_length,
                    .principal_x = static_cast<float>(width_u) * 0.5f,
                    .principal_y = static_cast<float>(height_u) * 0.5f,
                };
            });

            return frames;
        }
    } // namespace
    std::expected<NeRFSynthetic, std::string> load_nerf_synthetic(const std::filesystem::path& path) {
        try {
            return NeRFSynthetic{
                .train      = load_nerf_synthetic_split(path, "transforms_train.json"),
                .validation = load_nerf_synthetic_split(path, "transforms_val.json"),
                .test       = load_nerf_synthetic_split(path, "transforms_test.json"),
            };
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<DdNerfDataset, std::string> load_dd_nerf_dataset(const std::filesystem::path& path) {
        try {
            const std::filesystem::path json_path = path / "cameras.json";
            if (!std::filesystem::is_regular_file(json_path)) throw std::runtime_error{std::format("DD-NeRF dataset '{}' is missing cameras.json.", path.string())};
            if (!std::filesystem::is_directory(path / "images")) throw std::runtime_error{std::format("DD-NeRF dataset '{}' is missing an images directory.", path.string())};

            const nlohmann::json json         = nlohmann::json::parse(std::ifstream{json_path, std::ios::binary}, nullptr, true, true);
            const nlohmann::json& frames_json = json.at("frames");
            const std::uint32_t width         = json.at("w").get<std::uint32_t>();
            const std::uint32_t height        = json.at("h").get<std::uint32_t>();
            const float focal_x               = json.at("fl_x").get<float>();
            const float focal_y               = json.at("fl_y").get<float>();
            const float principal_x           = json.at("cx").get<float>();
            const float principal_y           = json.at("cy").get<float>();
            if (!frames_json.is_array() || frames_json.empty()) throw std::runtime_error{"DD-NeRF cameras.json must contain a non-empty frames array."};
            if (width == 0u || height == 0u) throw std::runtime_error{"DD-NeRF cameras.json declares invalid image dimensions."};
            if (!std::isfinite(focal_x) || !std::isfinite(focal_y) || focal_x <= 0.0f || focal_y <= 0.0f) throw std::runtime_error{"DD-NeRF cameras.json declares invalid focal lengths."};
            if (!std::isfinite(principal_x) || !std::isfinite(principal_y) || principal_x < 0.0f || principal_y < 0.0f || principal_x >= static_cast<float>(width) || principal_y >= static_cast<float>(height)) throw std::runtime_error{"DD-NeRF cameras.json declares an invalid principal point."};

            DdNerfDataset dataset = {};
            dataset.train.reserve(frames_json.size());
            dataset.validation.reserve(frames_json.size() / 10uz + 1uz);
            dataset.test.reserve(frames_json.size() / 10uz + 1uz);

            for (const nlohmann::json& frame_json : frames_json) {
                std::filesystem::path hash_path = frame_json.at("file_path").get<std::string>();
                const std::string hash_text     = hash_path.filename().string();
                std::uint64_t hash              = 14695981039346656037ull;
                for (const unsigned char byte : hash_text) {
                    hash ^= static_cast<std::uint64_t>(byte);
                    hash *= 1099511628211ull;
                }

                std::filesystem::path image_path = frame_json.at("file_path").get<std::string>();
                image_path.make_preferred();
                if (image_path.is_relative()) image_path = path / image_path;
                image_path = image_path.lexically_normal();
                if (!std::filesystem::is_regular_file(image_path)) throw std::runtime_error{std::format("DD-NeRF image '{}' does not exist.", image_path.string())};

                int image_width     = 0;
                int image_height    = 0;
                int component_count = 0;
                const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{stbi_load(image_path.string().c_str(), &image_width, &image_height, &component_count, 4), stbi_image_free};
                if (raw_pixels == nullptr) throw std::runtime_error{std::format("failed to load DD-NeRF image '{}'.", image_path.string())};
                if (image_width <= 0 || image_height <= 0) throw std::runtime_error{std::format("DD-NeRF image '{}' has invalid dimensions.", image_path.string())};
                if (static_cast<std::uint32_t>(image_width) != width || static_cast<std::uint32_t>(image_height) != height) throw std::runtime_error{std::format("DD-NeRF image '{}' is {}x{} but cameras.json declares {}x{}.", image_path.string(), image_width, image_height, width, height)};

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
                const std::size_t rgba_size = static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4uz;

                DdNerfDataset::Frame frame{
                    .rgba        = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size},
                    .camera      = ngp_camera,
                    .width       = width,
                    .height      = height,
                    .focal_x     = focal_x,
                    .focal_y     = focal_y,
                    .principal_x = principal_x,
                    .principal_y = principal_y,
                };
                if (hash % 10ull == 0ull)
                    dataset.test.push_back(std::move(frame));
                else if (hash % 10ull == 1ull)
                    dataset.validation.push_back(std::move(frame));
                else
                    dataset.train.push_back(std::move(frame));
            }

            if (dataset.train.empty()) throw std::runtime_error{"DD-NeRF split produced no training frames."};
            if (dataset.validation.empty()) throw std::runtime_error{"DD-NeRF split produced no validation frames."};
            if (dataset.test.empty()) throw std::runtime_error{"DD-NeRF split produced no test frames."};
            return dataset;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace ngp::dataset
