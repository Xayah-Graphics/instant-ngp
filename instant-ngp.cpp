#include "instant-ngp.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "json/json.hpp"

import std;

namespace ngp {

    void InstantNGP::load_dataset(const std::filesystem::path& dataset_path) {
        if (dataset_path.empty()) throw std::invalid_argument{"dataset_path must not be empty."};
        if (!std::filesystem::exists(dataset_path)) throw std::runtime_error{std::format("Dataset path does not exist: '{}'.", dataset_path.string())};
        if (!std::filesystem::is_directory(dataset_path)) throw std::runtime_error{std::format("Dataset path must be a directory: '{}'.", dataset_path.string())};

        DatasetState loaded_dataset{};
        const std::array<std::pair<std::string_view, std::vector<DatasetState::HostData::Frame>*>, 3> splits{
            std::pair{"transforms_train.json", &loaded_dataset.host.train},
            std::pair{"transforms_val.json", &loaded_dataset.host.validation},
            std::pair{"transforms_test.json", &loaded_dataset.host.test},
        };

        for (const auto& [file_name, frames] : splits) {
            const std::filesystem::path json_path = dataset_path / file_name;
            if (!std::filesystem::is_regular_file(json_path)) throw std::runtime_error{std::format("Dataset transform file does not exist: '{}'.", json_path.string())};

            std::ifstream json_stream{json_path, std::ios::binary};
            if (!json_stream.is_open()) throw std::runtime_error{std::format("Failed to open dataset transform file '{}'.", json_path.string())};

            const nlohmann::json json        = nlohmann::json::parse(json_stream, nullptr, true, true);
            const float camera_angle_x       = json.at("camera_angle_x").get<float>();
            const nlohmann::json& split_json = json.at("frames");
            if (!std::isfinite(camera_angle_x) || camera_angle_x <= 0.0f || camera_angle_x >= std::numbers::pi_v<float>) throw std::runtime_error{std::format("Invalid 'camera_angle_x' in '{}'.", json_path.string())};
            if (!split_json.is_array()) throw std::runtime_error{std::format("'frames' must be an array in '{}'.", json_path.string())};

            frames->clear();
            frames->resize(split_json.size());
            for (std::size_t frame_index = 0; frame_index < split_json.size(); ++frame_index) {
                const nlohmann::json& frame_json = split_json.at(frame_index);
                auto image_path_string           = frame_json.at("file_path").get<std::string>();
                if (image_path_string.empty()) throw std::runtime_error{std::format("Empty 'file_path' in '{}', frame {}.", json_path.string(), frame_index)};
                std::ranges::replace(image_path_string, '\\', '/');

                std::filesystem::path image_path = json_path.parent_path() / std::filesystem::path{image_path_string};
                if (image_path.extension().empty()) image_path.replace_extension(".png");
                if (!std::filesystem::is_regular_file(image_path)) throw std::runtime_error{std::format("Dataset image file does not exist: '{}'.", image_path.string())};

                int width           = 0;
                int height          = 0;
                int component_count = 0;
                std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{
                    stbi_load(image_path.string().c_str(), &width, &height, &component_count, 4),
                    stbi_image_free,
                };
                if (!raw_pixels) throw std::runtime_error{std::format("Failed to decode image file '{}': {}.", image_path.string(), stbi_failure_reason())};
                if (width <= 0 || height <= 0) throw std::runtime_error{std::format("Image '{}' has invalid resolution.", image_path.string())};

                const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
                const std::size_t rgba_size   = pixel_count * 4ull;
                if (pixel_count == 0 || rgba_size / 4ull != pixel_count) throw std::runtime_error{std::format("Image '{}' size overflows addressable memory.", image_path.string())};

                const float focal_length_x = 0.5f * static_cast<float>(width) / std::tan(camera_angle_x * 0.5f);
                if (!std::isfinite(focal_length_x) || focal_length_x <= 0.0f) throw std::runtime_error{std::format("Computed focal length is invalid for '{}'.", image_path.string())};

                const nlohmann::json& transform_matrix = frame_json.at("transform_matrix");
                if (!transform_matrix.is_array() || transform_matrix.size() != 4) throw std::runtime_error{std::format("Invalid 'transform_matrix' in '{}', frame {}.", json_path.string(), frame_index)};

                DatasetState::HostData::Frame& frame = (*frames)[frame_index];
                frame.rgba         = std::vector<std::uint8_t>{raw_pixels.get(), raw_pixels.get() + rgba_size};
                frame.resolution   = legacy::math::ivec2{width, height};
                frame.focal_length = focal_length_x;

                for (std::size_t row = 0; row < 4; ++row) {
                    const nlohmann::json& transform_row = transform_matrix.at(row);
                    if (!transform_row.is_array() || transform_row.size() != 4) throw std::runtime_error{std::format("Invalid 'transform_matrix' row in '{}', frame {}, row {}.", json_path.string(), frame_index, row)};

                    for (std::size_t column = 0; column < 4; ++column) {
                        const float transform_element = transform_row.at(column).get<float>();
                        if (!std::isfinite(transform_element)) throw std::runtime_error{std::format("Non-finite 'transform_matrix' element in '{}', frame {}, row {}, column {}.", json_path.string(), frame_index, row, column)};
                        if (row < 3u) frame.camera[column][row] = transform_element;
                    }
                }

                frame.camera[1] *= -1.0f;
                frame.camera[2] *= -1.0f;
                frame.camera[3]               = frame.camera[3] * 0.33f + legacy::math::vec3(0.5f);
                const legacy::math::vec4 camera_row0 = ngp::legacy::math::row(frame.camera, 0);
                frame.camera                  = ngp::legacy::math::row(frame.camera, 0, ngp::legacy::math::row(frame.camera, 1));
                frame.camera                  = ngp::legacy::math::row(frame.camera, 1, ngp::legacy::math::row(frame.camera, 2));
                frame.camera                  = ngp::legacy::math::row(frame.camera, 2, camera_row0);
            }
        }

        if (loaded_dataset.host.train.empty()) throw std::runtime_error{"load_dataset requires at least one training frame after parsing the dataset."};

        loaded_dataset.device.pixels.resize(loaded_dataset.host.train.size());
        std::vector<instant_ngp_detail::GpuFrame> uploaded_frames(loaded_dataset.host.train.size());

        for (std::size_t frame_index = 0; frame_index < loaded_dataset.host.train.size(); ++frame_index) {
            const DatasetState::HostData::Frame& source_frame = loaded_dataset.host.train[frame_index];
            if (source_frame.resolution.x <= 0 || source_frame.resolution.y <= 0) throw std::runtime_error{"load_dataset encountered a training frame with zero resolution during upload staging."};
            if (!std::isfinite(source_frame.focal_length) || source_frame.focal_length <= 0.0f) throw std::runtime_error{"load_dataset encountered a training frame with an invalid focal length during upload staging."};

            const std::size_t expected_rgba_size = static_cast<std::size_t>(source_frame.resolution.x) * static_cast<std::size_t>(source_frame.resolution.y) * 4ull;
            if (source_frame.rgba.size() != expected_rgba_size) throw std::runtime_error{"load_dataset encountered a training frame whose RGBA byte count no longer matches width * height * 4 during upload staging."};

            legacy::GpuBuffer<std::uint8_t>& pixel_buffer = loaded_dataset.device.pixels[frame_index];
            pixel_buffer.resize(source_frame.rgba.size());
            pixel_buffer.copy_from_host(source_frame.rgba);

            instant_ngp_detail::GpuFrame& target_frame = uploaded_frames[frame_index];
            target_frame.pixels       = pixel_buffer.data();
            target_frame.resolution   = source_frame.resolution;
            target_frame.focal_length = source_frame.focal_length;
            target_frame.camera       = source_frame.camera;
        }

        loaded_dataset.device.frames.resize(uploaded_frames.size());
        loaded_dataset.device.frames.copy_from_host(uploaded_frames);
        dataset = std::move(loaded_dataset);
        std::print("Loaded dataset with {} training frames, {} validation frames, and {} test frames.\n", dataset.host.train.size(), dataset.host.validation.size(), dataset.host.test.size());
    }
} // namespace ngp
