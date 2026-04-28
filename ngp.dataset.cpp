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

        struct PosesBounds final {
            std::vector<double> values = {};
            std::size_t row_count      = 0uz;
        };

        struct PoseBoundFrame final {
            std::vector<std::uint8_t> rgba = {};
            std::array<float, 12> camera   = {};
            std::uint32_t width            = 0u;
            std::uint32_t height           = 0u;
            float focal_length             = 0.0f;
            std::filesystem::path image_path;
            std::size_t source_index = 0uz;
            float near_bound         = 0.0f;
            float far_bound          = 0.0f;
        };

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

        std::string_view trim_header_number(std::string_view text) {
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front())) != 0) text.remove_prefix(1uz);
            while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back())) != 0) text.remove_suffix(1uz);
            return text;
        }

        std::size_t parse_header_size(const std::string_view text, const std::string_view field_name) {
            std::size_t parsed    = 0uz;
            const auto clean_text = trim_header_number(text);
            const auto result     = std::from_chars(clean_text.data(), clean_text.data() + clean_text.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != clean_text.data() + clean_text.size()) throw std::runtime_error{std::format("invalid {} in poses_bounds.npy header.", field_name)};
            return parsed;
        }

        PosesBounds load_poses_bounds(const std::filesystem::path& dataset_path) {
            if (std::endian::native != std::endian::little) throw std::runtime_error{"poses_bounds.npy requires a little-endian host."};

            const std::filesystem::path path = dataset_path / "poses_bounds.npy";
            std::ifstream input{path, std::ios::binary};
            if (!input) throw std::runtime_error{std::format("failed to open '{}'.", path.string())};

            std::array<unsigned char, 6> magic = {};
            input.read(reinterpret_cast<char*>(magic.data()), static_cast<std::streamsize>(magic.size()));
            if (!input || magic != std::array<unsigned char, 6>{0x93u, 'N', 'U', 'M', 'P', 'Y'}) throw std::runtime_error{std::format("'{}' is not a NumPy .npy file.", path.string())};

            std::array<unsigned char, 2> version = {};
            input.read(reinterpret_cast<char*>(version.data()), static_cast<std::streamsize>(version.size()));
            if (!input) throw std::runtime_error{std::format("'{}' has an incomplete NumPy header.", path.string())};

            std::uint32_t header_length = 0u;
            if (version[0] == 1u) {
                std::array<unsigned char, 2> bytes = {};
                input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!input) throw std::runtime_error{std::format("'{}' has an incomplete NumPy v1 header length.", path.string())};
                header_length = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8u);
            } else if (version[0] == 2u) {
                std::array<unsigned char, 4> bytes = {};
                input.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
                if (!input) throw std::runtime_error{std::format("'{}' has an incomplete NumPy v2 header length.", path.string())};
                header_length = static_cast<std::uint32_t>(bytes[0]) | (static_cast<std::uint32_t>(bytes[1]) << 8u) | (static_cast<std::uint32_t>(bytes[2]) << 16u) | (static_cast<std::uint32_t>(bytes[3]) << 24u);
            } else {
                throw std::runtime_error{std::format("'{}' uses unsupported NumPy format version {}.{}.", path.string(), version[0], version[1])};
            }

            std::string header(header_length, '\0');
            input.read(header.data(), static_cast<std::streamsize>(header.size()));
            if (!input) throw std::runtime_error{std::format("'{}' has an incomplete NumPy header.", path.string())};
            if (!header.contains("'descr': '<f8'") && !header.contains("\"descr\": \"<f8\"")) throw std::runtime_error{std::format("'{}' must be little-endian float64.", path.string())};
            if (!header.contains("'fortran_order': False") && !header.contains("\"fortran_order\": False")) throw std::runtime_error{std::format("'{}' must be C-order.", path.string())};

            const std::size_t shape_position = header.find("shape");
            if (shape_position == std::string::npos) throw std::runtime_error{std::format("'{}' is missing a shape field.", path.string())};
            const std::size_t shape_begin = header.find('(', shape_position);
            const std::size_t comma       = header.find(',', shape_begin);
            const std::size_t shape_end   = header.find(')', comma);
            if (shape_begin == std::string::npos || comma == std::string::npos || shape_end == std::string::npos) throw std::runtime_error{std::format("'{}' has an invalid shape field.", path.string())};

            const std::size_t row_count    = parse_header_size(std::string_view{header}.substr(shape_begin + 1uz, comma - shape_begin - 1uz), "row count");
            const std::size_t column_count = parse_header_size(std::string_view{header}.substr(comma + 1uz, shape_end - comma - 1uz), "column count");
            if (row_count == 0uz || column_count != 17uz) throw std::runtime_error{std::format("'{}' must have shape (N, 17).", path.string())};

            std::vector<double> values(row_count * column_count);
            input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(double)));
            if (!input) throw std::runtime_error{std::format("'{}' has incomplete pose/bounds data.", path.string())};

            return PosesBounds{.values = std::move(values), .row_count = row_count};
        }

        std::vector<std::filesystem::path> load_pose_bound_image_paths(const std::filesystem::path& dataset_path, const std::size_t expected_count) {
            const std::filesystem::path images_path = dataset_path / "images";
            if (!std::filesystem::is_directory(images_path)) throw std::runtime_error{std::format("'{}' is missing a full-resolution images directory.", dataset_path.string())};

            std::vector<std::filesystem::path> image_paths;
            for (const std::filesystem::directory_entry& entry : std::filesystem::directory_iterator{images_path}) {
                if (!entry.is_regular_file()) continue;
                std::string extension = entry.path().extension().string();
                std::ranges::transform(extension, extension.begin(), [](const unsigned char value) { return static_cast<char>(std::tolower(value)); });
                if (extension == ".jpg" || extension == ".jpeg" || extension == ".png") image_paths.push_back(entry.path());
            }

            std::ranges::sort(image_paths, [](const std::filesystem::path& lhs, const std::filesystem::path& rhs) { return lhs.filename().string() < rhs.filename().string(); });
            if (image_paths.size() != expected_count) throw std::runtime_error{std::format("'{}' contains {} images but poses_bounds.npy contains {} poses.", images_path.string(), image_paths.size(), expected_count)};
            return image_paths;
        }

        std::array<double, 3> cross(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
            return {
                lhs[1] * rhs[2] - lhs[2] * rhs[1],
                lhs[2] * rhs[0] - lhs[0] * rhs[2],
                lhs[0] * rhs[1] - lhs[1] * rhs[0],
            };
        }

        double dot(const std::array<double, 3>& lhs, const std::array<double, 3>& rhs) {
            return lhs[0] * rhs[0] + lhs[1] * rhs[1] + lhs[2] * rhs[2];
        }

        std::array<double, 3> normalize(const std::array<double, 3>& value) {
            const double length = std::sqrt(dot(value, value));
            if (!std::isfinite(length) || length <= 0.0) throw std::runtime_error{"poses_bounds.npy contains an invalid camera direction."};
            return {value[0] / length, value[1] / length, value[2] / length};
        }

        std::array<float, 12> nerf_matrix_to_ngp_camera(const std::array<std::array<float, 4>, 3>& nerf_matrix) {
            std::array<std::array<float, 4>, 4> camera = {};
            for (const std::size_t row : std::views::iota(0uz, 3uz))
                for (const std::size_t column : std::views::iota(0uz, 4uz)) camera[column][row] = nerf_matrix[row][column];

            std::ranges::for_each(camera[1], [](float& value) { value = -value; });
            std::ranges::for_each(camera[2], [](float& value) { value = -value; });

            camera[3][0] = camera[3][0] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;
            camera[3][1] = camera[3][1] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;
            camera[3][2] = camera[3][2] * NERF_SYNTHETIC_SCENE_SCALE + NERF_SYNTHETIC_SCENE_OFFSET;

            const std::array camera_row0{camera[0][0], camera[1][0], camera[2][0], camera[3][0]};
            const std::array camera_row1{camera[0][1], camera[1][1], camera[2][1], camera[3][1]};
            const std::array camera_row2{camera[0][2], camera[1][2], camera[2][2], camera[3][2]};
            return {camera_row1[0], camera_row2[0], camera_row0[0], camera_row1[1], camera_row2[1], camera_row0[1], camera_row1[2], camera_row2[2], camera_row0[2], camera_row1[3], camera_row2[3], camera_row0[3]};
        }

        std::vector<PoseBoundFrame> load_pose_bound_frames(const std::filesystem::path& dataset_path) {
            const PosesBounds poses_bounds                         = load_poses_bounds(dataset_path);
            const std::vector<std::filesystem::path> image_paths   = load_pose_bound_image_paths(dataset_path, poses_bounds.row_count);
            std::vector<std::array<std::array<float, 4>, 3>> poses = {};
            poses.reserve(poses_bounds.row_count);

            for (const std::size_t frame_index : std::views::iota(0uz, poses_bounds.row_count)) {
                const double* row                        = poses_bounds.values.data() + frame_index * 17uz;
                std::array<std::array<float, 4>, 3> pose = {};
                for (const std::size_t matrix_row : std::views::iota(0uz, 3uz)) {
                    pose[matrix_row][0] = static_cast<float>(row[matrix_row * 5uz + 1uz]);
                    pose[matrix_row][1] = static_cast<float>(-row[matrix_row * 5uz + 0uz]);
                    pose[matrix_row][2] = static_cast<float>(row[matrix_row * 5uz + 2uz]);
                    pose[matrix_row][3] = static_cast<float>(row[matrix_row * 5uz + 3uz]);
                }
                poses.push_back(pose);
            }

            std::array<double, 3> center = {};
            double total_weight          = 0.0;
            for (const std::array<std::array<float, 4>, 3>& lhs : poses) {
                const std::array<double, 3> lhs_origin{lhs[0][3], lhs[1][3], lhs[2][3]};
                const std::array<double, 3> lhs_direction = normalize({lhs[0][2], lhs[1][2], lhs[2][2]});
                for (const std::array<std::array<float, 4>, 3>& rhs : poses) {
                    const std::array<double, 3> rhs_origin{rhs[0][3], rhs[1][3], rhs[2][3]};
                    const std::array<double, 3> rhs_direction = normalize({rhs[0][2], rhs[1][2], rhs[2][2]});
                    const std::array<double, 3> crossed       = cross(lhs_direction, rhs_direction);
                    const double weight                       = dot(crossed, crossed);
                    if (weight <= 0.00001) continue;

                    const std::array<double, 3> origin_delta{rhs_origin[0] - lhs_origin[0], rhs_origin[1] - lhs_origin[1], rhs_origin[2] - lhs_origin[2]};
                    double lhs_t = dot(origin_delta, cross(rhs_direction, crossed)) / (weight + 1e-10);
                    double rhs_t = dot(origin_delta, cross(lhs_direction, crossed)) / (weight + 1e-10);
                    if (lhs_t > 0.0) lhs_t = 0.0;
                    if (rhs_t > 0.0) rhs_t = 0.0;

                    const std::array<double, 3> point{
                        (lhs_origin[0] + lhs_t * lhs_direction[0] + rhs_origin[0] + rhs_t * rhs_direction[0]) * 0.5,
                        (lhs_origin[1] + lhs_t * lhs_direction[1] + rhs_origin[1] + rhs_t * rhs_direction[1]) * 0.5,
                        (lhs_origin[2] + lhs_t * lhs_direction[2] + rhs_origin[2] + rhs_t * rhs_direction[2]) * 0.5,
                    };
                    center[0] += point[0] * weight;
                    center[1] += point[1] * weight;
                    center[2] += point[2] * weight;
                    total_weight += weight;
                }
            }
            if (total_weight > 0.0) {
                center[0] /= total_weight;
                center[1] /= total_weight;
                center[2] /= total_weight;
            }

            double average_distance = 0.0;
            for (std::array<std::array<float, 4>, 3>& pose : poses) {
                pose[0][3] = static_cast<float>(static_cast<double>(pose[0][3]) - center[0]);
                pose[1][3] = static_cast<float>(static_cast<double>(pose[1][3]) - center[1]);
                pose[2][3] = static_cast<float>(static_cast<double>(pose[2][3]) - center[2]);
                average_distance += std::sqrt(static_cast<double>(pose[0][3]) * pose[0][3] + static_cast<double>(pose[1][3]) * pose[1][3] + static_cast<double>(pose[2][3]) * pose[2][3]);
            }
            average_distance /= static_cast<double>(poses.size());
            if (!std::isfinite(average_distance) || average_distance <= 0.0) throw std::runtime_error{"poses_bounds.npy camera normalization failed."};

            const double camera_scale = 4.0 / average_distance;
            for (std::array<std::array<float, 4>, 3>& pose : poses) {
                pose[0][3] = static_cast<float>(static_cast<double>(pose[0][3]) * camera_scale);
                pose[1][3] = static_cast<float>(static_cast<double>(pose[1][3]) * camera_scale);
                pose[2][3] = static_cast<float>(static_cast<double>(pose[2][3]) * camera_scale);
            }

            std::vector<PoseBoundFrame> frames;
            frames.reserve(poses_bounds.row_count);

            for (const std::size_t frame_index : std::views::iota(0uz, poses_bounds.row_count)) {
                const double* pose_row = poses_bounds.values.data() + frame_index * 17uz;
                const auto pose_height = static_cast<std::uint32_t>(std::llround(pose_row[4]));
                const auto pose_width  = static_cast<std::uint32_t>(std::llround(pose_row[9]));
                const float focal      = static_cast<float>(pose_row[14]);
                const float near_bound = static_cast<float>(pose_row[15]);
                const float far_bound  = static_cast<float>(pose_row[16]);
                if (pose_width == 0u || pose_height == 0u || !std::isfinite(focal) || focal <= 0.0f) throw std::runtime_error{std::format("invalid camera intrinsics at poses_bounds.npy row {}.", frame_index)};
                if (!std::isfinite(near_bound) || !std::isfinite(far_bound) || near_bound <= 0.0f || far_bound <= near_bound) throw std::runtime_error{std::format("invalid near/far bounds at poses_bounds.npy row {}.", frame_index)};

                int image_width     = 0;
                int image_height    = 0;
                int component_count = 0;
                const std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{stbi_load(image_paths[frame_index].string().c_str(), &image_width, &image_height, &component_count, 4), stbi_image_free};
                if (raw_pixels == nullptr) throw std::runtime_error{std::format("failed to load image '{}'.", image_paths[frame_index].string())};
                if (image_width <= 0 || image_height <= 0) throw std::runtime_error{std::format("image '{}' has invalid dimensions.", image_paths[frame_index].string())};
                if (static_cast<std::uint32_t>(image_width) != pose_width || static_cast<std::uint32_t>(image_height) != pose_height) throw std::runtime_error{std::format("image '{}' is {}x{} but poses_bounds.npy row {} declares {}x{}.", image_paths[frame_index].string(), image_width, image_height, frame_index, pose_width, pose_height)};

                const std::size_t rgba_size = static_cast<std::size_t>(pose_width) * static_cast<std::size_t>(pose_height) * 4uz;
                frames.push_back(PoseBoundFrame{
                    .rgba         = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size},
                    .camera       = nerf_matrix_to_ngp_camera(poses[frame_index]),
                    .width        = pose_width,
                    .height       = pose_height,
                    .focal_length = focal,
                    .image_path   = image_paths[frame_index],
                    .source_index = frame_index,
                    .near_bound   = near_bound,
                    .far_bound    = far_bound,
                });
            }

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

    std::expected<NeRFLlffData, std::string> load_nerf_llff_data(const std::filesystem::path& path) {
        try {
            NeRFLlffData dataset               = {};
            std::vector<PoseBoundFrame> frames = load_pose_bound_frames(path);
            for (PoseBoundFrame& source : frames) {
                NeRFLlffData::Frame frame{
                    .rgba         = std::move(source.rgba),
                    .camera       = source.camera,
                    .width        = source.width,
                    .height       = source.height,
                    .focal_length = source.focal_length,
                    .image_path   = std::move(source.image_path),
                    .source_index = source.source_index,
                    .near_bound   = source.near_bound,
                    .far_bound    = source.far_bound,
                };
                if (source.source_index % 8uz == 0uz)
                    dataset.test.push_back(std::move(frame));
                else if (source.source_index % 8uz == 4uz)
                    dataset.validation.push_back(std::move(frame));
                else
                    dataset.train.push_back(std::move(frame));
            }
            if (dataset.train.empty()) throw std::runtime_error{"nerf_llff_data split produced no training frames."};
            if (dataset.validation.empty()) throw std::runtime_error{"nerf_llff_data split produced no validation frames."};
            if (dataset.test.empty()) throw std::runtime_error{"nerf_llff_data split produced no test frames."};
            return dataset;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }

    std::expected<NeRFReal360, std::string> load_nerf_real_360(const std::filesystem::path& path) {
        try {
            NeRFReal360 dataset                = {};
            std::vector<PoseBoundFrame> frames = load_pose_bound_frames(path);
            for (PoseBoundFrame& source : frames) {
                NeRFReal360::Frame frame{
                    .rgba         = std::move(source.rgba),
                    .camera       = source.camera,
                    .width        = source.width,
                    .height       = source.height,
                    .focal_length = source.focal_length,
                    .image_path   = std::move(source.image_path),
                    .source_index = source.source_index,
                    .near_bound   = source.near_bound,
                    .far_bound    = source.far_bound,
                };
                if (source.source_index % 8uz == 0uz)
                    dataset.test.push_back(std::move(frame));
                else if (source.source_index % 8uz == 4uz)
                    dataset.validation.push_back(std::move(frame));
                else
                    dataset.train.push_back(std::move(frame));
            }
            if (dataset.train.empty()) throw std::runtime_error{"nerf_real_360 split produced no training frames."};
            if (dataset.validation.empty()) throw std::runtime_error{"nerf_real_360 split produced no validation frames."};
            if (dataset.test.empty()) throw std::runtime_error{"nerf_real_360 split produced no test frames."};
            return dataset;
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    }
} // namespace ngp::dataset
