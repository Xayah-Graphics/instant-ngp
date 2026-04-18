#include "instant-ngp.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include "json/json.hpp"

import std;

namespace ngp {

    InstantNGP::InstantNGP() {
        std::print("Hello, Instant NGP!\n");
    }

    InstantNGP::~InstantNGP() {
        std::print("Bye!\n");
    }

    void InstantNGP::load_dataset(const std::filesystem::path& dataset_path, Runtime::Dataset::Type dataset_type) {
        if (dataset_path.empty()) {
            throw std::invalid_argument{"dataset_path must not be empty."};
        }
        if (!std::filesystem::exists(dataset_path)) {
            throw std::runtime_error{std::format("Dataset path does not exist: '{}'.", dataset_path.string())};
        }
        if (!std::filesystem::is_directory(dataset_path)) {
            throw std::runtime_error{std::format("Dataset path must be a directory: '{}'.", dataset_path.string())};
        }

        switch (dataset_type) {
        case Runtime::Dataset::Type::NerfSynthetic:
            {
                Runtime::Dataset loaded_dataset{};
                const std::array<std::pair<std::string_view, std::vector<Runtime::Dataset::CPU::Frame>*>, 3> splits{
                    std::pair{"transforms_train.json", &loaded_dataset.cpu.train},
                    std::pair{"transforms_val.json", &loaded_dataset.cpu.validation},
                    std::pair{"transforms_test.json", &loaded_dataset.cpu.test},
                };

                for (const auto& [file_name, frames] : splits) {
                    const std::filesystem::path json_path = dataset_path / file_name;
                    const std::string json_path_string    = json_path.string();

                    if (!std::filesystem::is_regular_file(json_path)) {
                        throw std::runtime_error{std::format("Dataset transform file does not exist: '{}'.", json_path_string)};
                    }

                    std::ifstream json_stream{json_path, std::ios::binary};
                    json_stream.exceptions(std::ios::badbit);

                    const nlohmann::json json                                    = nlohmann::json::parse(json_stream, nullptr, true, true);
                    const nlohmann::json::const_iterator camera_angle_x_iterator = json.find("camera_angle_x");
                    if (camera_angle_x_iterator == json.end()) {
                        throw std::runtime_error{std::format("Missing 'camera_angle_x' in '{}'.", json_path_string)};
                    }
                    const nlohmann::json::const_iterator frames_iterator = json.find("frames");
                    if (frames_iterator == json.end()) {
                        throw std::runtime_error{std::format("Missing 'frames' in '{}'.", json_path_string)};
                    }

                    const float camera_angle_x = camera_angle_x_iterator->get<float>();
                    if (!std::isfinite(camera_angle_x) || camera_angle_x <= 0.0f || camera_angle_x >= std::numbers::pi_v<float>) {
                        throw std::runtime_error{std::format("Invalid 'camera_angle_x' in '{}'.", json_path_string)};
                    }

                    const nlohmann::json& frames_json = *frames_iterator;
                    if (!frames_json.is_array()) {
                        throw std::runtime_error{std::format("'frames' must be an array in '{}'.", json_path_string)};
                    }

                    const std::size_t frame_count = frames_json.size();
                    frames->clear();
                    frames->resize(frame_count);
                    if (frame_count == 0) {
                        continue;
                    }

                    const unsigned int hardware_thread_count = std::thread::hardware_concurrency();
                    if (hardware_thread_count == 0) {
                        throw std::runtime_error{"std::thread::hardware_concurrency returned 0."};
                    }

                    const std::size_t worker_count            = std::min(frame_count, static_cast<std::size_t>(hardware_thread_count));
                    const std::filesystem::path base_path     = json_path.parent_path();
                    std::atomic<std::size_t> next_frame_index = 0;
                    std::exception_ptr first_exception;
                    std::mutex first_exception_mutex;
                    std::stop_source stop_source;
                    const std::stop_token stop_token = stop_source.get_token();
                    std::vector<std::jthread> workers;
                    workers.reserve(worker_count);

                    for (std::size_t worker_index = 0; worker_index < worker_count; ++worker_index) {
                        workers.emplace_back([&, stop_token]() {
                            while (!stop_token.stop_requested()) {
                                const std::size_t frame_index = next_frame_index.fetch_add(1, std::memory_order_relaxed);
                                if (frame_index >= frame_count) {
                                    return;
                                }

                                try {
                                    const nlohmann::json& frame_json                        = frames_json.at(frame_index);
                                    const nlohmann::json::const_iterator file_path_iterator = frame_json.find("file_path");
                                    if (file_path_iterator == frame_json.end()) {
                                        throw std::runtime_error{std::format("Missing 'file_path' in '{}', frame {}.", json_path_string, frame_index)};
                                    }
                                    const nlohmann::json::const_iterator transform_matrix_iterator = frame_json.find("transform_matrix");
                                    if (transform_matrix_iterator == frame_json.end()) {
                                        throw std::runtime_error{std::format("Missing 'transform_matrix' in '{}', frame {}.", json_path_string, frame_index)};
                                    }

                                    std::string image_path_string = file_path_iterator->get<std::string>();
                                    if (image_path_string.empty()) {
                                        throw std::runtime_error{std::format("Empty 'file_path' in '{}', frame {}.", json_path_string, frame_index)};
                                    }
                                    std::ranges::replace(image_path_string, '\\', '/');

                                    std::filesystem::path image_path = base_path / std::filesystem::path{image_path_string};
                                    if (image_path.extension().empty()) {
                                        image_path.replace_extension(".png");
                                    }

                                    const std::string image_path_string_full = image_path.string();
                                    int width                                = 0;
                                    int height                               = 0;
                                    int component_count                      = 0;
                                    std::unique_ptr<stbi_uc, decltype(&stbi_image_free)> raw_pixels{
                                        stbi_load(image_path_string_full.c_str(), &width, &height, &component_count, 4),
                                        stbi_image_free,
                                    };
                                    if (raw_pixels == nullptr) {
                                        throw std::runtime_error{std::format("Failed to decode image file '{}'.", image_path_string_full)};
                                    }

                                    if (width <= 0 || height <= 0) {
                                        throw std::runtime_error{std::format("Image '{}' has invalid resolution.", image_path_string_full)};
                                    }

                                    const std::size_t pixel_count = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
                                    const std::size_t rgba_size   = pixel_count * 4ull;
                                    if (pixel_count == 0 || rgba_size / 4ull != pixel_count) {
                                        throw std::runtime_error{std::format("Image '{}' size overflows addressable memory.", image_path_string_full)};
                                    }

                                    const float focal_length_x = 0.5f * static_cast<float>(width) / std::tan(camera_angle_x * 0.5f);
                                    if (!std::isfinite(focal_length_x) || focal_length_x <= 0.0f) {
                                        throw std::runtime_error{std::format("Computed focal length is invalid for '{}'.", image_path_string_full)};
                                    }

                                    const nlohmann::json& transform_matrix = *transform_matrix_iterator;
                                    if (!transform_matrix.is_array() || transform_matrix.size() != 4) {
                                        throw std::runtime_error{std::format("Invalid 'transform_matrix' in '{}', frame {}.", json_path_string, frame_index)};
                                    }

                                    Runtime::Dataset::CPU::Frame& frame = (*frames)[frame_index];
                                    frame.width                         = static_cast<std::uint32_t>(width);
                                    frame.height                        = static_cast<std::uint32_t>(height);
                                    frame.focal_length_x                = focal_length_x;
                                    frame.focal_length_y                = focal_length_x;
                                    frame.rgba.resize(rgba_size);
                                    std::memcpy(frame.rgba.data(), raw_pixels.get(), rgba_size);

                                    for (std::size_t row = 0; row < transform_matrix.size(); ++row) {
                                        const nlohmann::json& transform_row = transform_matrix.at(row);
                                        if (!transform_row.is_array() || transform_row.size() != 4) {
                                            throw std::runtime_error{std::format("Invalid 'transform_matrix' row in '{}', frame {}, row {}.", json_path_string, frame_index, row)};
                                        }

                                        for (std::size_t column = 0; column < 4; ++column) {
                                            const float transform_element = transform_row.at(column).get<float>();
                                            if (!std::isfinite(transform_element)) {
                                                throw std::runtime_error{
                                                    std::format("Non-finite 'transform_matrix' element in '{}', frame {}, row {}, column {}.", json_path_string, frame_index, row, column),
                                                };
                                            }
                                            frame.transform_matrix_4x4[row * 4 + column] = transform_element;
                                        }
                                    }
                                } catch (...) {
                                    {
                                        std::scoped_lock first_exception_lock{first_exception_mutex};
                                        if (!first_exception) {
                                            first_exception = std::current_exception();
                                        }
                                    }
                                    stop_source.request_stop();
                                    return;
                                }
                            }
                        });
                    }

                    workers.clear();
                    if (first_exception) {
                        std::rethrow_exception(first_exception);
                    }
                }

                if (loaded_dataset.cpu.train.empty()) {
                    throw std::runtime_error{"load_dataset requires at least one training frame after parsing the dataset."};
                }

                loaded_dataset.gpu.train.pixels.resize(loaded_dataset.cpu.train.size());
                std::vector<Runtime::Dataset::GPU::Frame> uploaded_frames(loaded_dataset.cpu.train.size());

                cudaStream_t upload_stream             = nullptr;
                const cudaError_t stream_create_status = cudaStreamCreateWithFlags(&upload_stream, cudaStreamNonBlocking);
                if (stream_create_status != cudaSuccess) {
                    throw std::runtime_error{std::string{"load_dataset failed to create a CUDA upload stream: "} + cudaGetErrorString(stream_create_status)};
                }

                try {
                    for (std::size_t frame_index = 0; frame_index < loaded_dataset.cpu.train.size(); ++frame_index) {
                        const Runtime::Dataset::CPU::Frame& source_frame = loaded_dataset.cpu.train[frame_index];
                        if (source_frame.width == 0 || source_frame.height == 0) {
                            throw std::runtime_error{"load_dataset encountered a training frame with zero resolution during upload staging."};
                        }
                        if (!std::isfinite(source_frame.focal_length_x) || source_frame.focal_length_x <= 0.0f) {
                            throw std::runtime_error{"load_dataset encountered a training frame with an invalid focal_length_x during upload staging."};
                        }
                        if (!std::isfinite(source_frame.focal_length_y) || source_frame.focal_length_y <= 0.0f) {
                            throw std::runtime_error{"load_dataset encountered a training frame with an invalid focal_length_y during upload staging."};
                        }

                        const float focal_length_difference = std::fabs(source_frame.focal_length_x - source_frame.focal_length_y);
                        const float focal_length_scale      = std::max(1.0f, std::max(std::fabs(source_frame.focal_length_x), std::fabs(source_frame.focal_length_y)));
                        if (focal_length_difference > 1e-6f * focal_length_scale) {
                            throw std::runtime_error{"load_dataset currently requires focal_length_x and focal_length_y to match before GPU upload."};
                        }

                        const std::size_t expected_rgba_size = static_cast<std::size_t>(source_frame.width) * static_cast<std::size_t>(source_frame.height) * 4ull;
                        if (source_frame.rgba.size() != expected_rgba_size) {
                            throw std::runtime_error{"load_dataset encountered a training frame whose RGBA byte count no longer matches width * height * 4 during upload staging."};
                        }

                        legacy::GpuBuffer<std::uint8_t>& pixel_buffer = loaded_dataset.gpu.train.pixels[frame_index];
                        pixel_buffer.resize(source_frame.rgba.size());
                        pixel_buffer.copy_from_host_async(source_frame.rgba.data(), upload_stream);

                        Runtime::Dataset::GPU::Frame& target_frame = uploaded_frames[frame_index];
                        target_frame.pixels                        = pixel_buffer.data();
                        target_frame.resolution                    = legacy::Int2{
                            static_cast<int>(source_frame.width),
                            static_cast<int>(source_frame.height),
                        };
                        target_frame.focal_length = source_frame.focal_length_x;
                        target_frame.camera       = legacy::nerf_matrix_to_ngp(source_frame.transform_matrix_4x4);
                    }

                    loaded_dataset.gpu.train.frames.resize(uploaded_frames.size());
                    loaded_dataset.gpu.train.frames.copy_from_host_async(uploaded_frames.data(), upload_stream);

                    const cudaError_t stream_sync_status = cudaStreamSynchronize(upload_stream);
                    if (stream_sync_status != cudaSuccess) {
                        throw std::runtime_error{std::string{"load_dataset failed while synchronizing the CUDA upload stream: "} + cudaGetErrorString(stream_sync_status)};
                    }
                } catch (...) {
                    cudaStreamDestroy(upload_stream);
                    throw;
                }

                const cudaError_t stream_destroy_status = cudaStreamDestroy(upload_stream);
                if (stream_destroy_status != cudaSuccess) {
                    throw std::runtime_error{std::string{"load_dataset failed to destroy the CUDA upload stream: "} + cudaGetErrorString(stream_destroy_status)};
                }

                runtime_.dataset = std::move(loaded_dataset);
                std::print("Loaded dataset with {} training frames, {} validation frames, and {} test frames.\n", runtime_.dataset.cpu.train.size(), runtime_.dataset.cpu.validation.size(), runtime_.dataset.cpu.test.size());
                break;
            }
        default: throw std::runtime_error{std::format("Unsupported dataset type: {}.", std::to_underlying(dataset_type))};
        }
    }

} // namespace ngp
