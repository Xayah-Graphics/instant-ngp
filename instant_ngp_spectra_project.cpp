module;

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#else
#include <unistd.h>
#endif

#include <cuda_runtime_api.h>

module instant_ngp.spectra_project;

import std;
import dataset.nerf_synthetic;
import dataset.dd_nerf;
import ngp.train;
import ngp.inspector;

namespace instant_ngp::spectra_project {
    namespace {
        constexpr std::uint32_t viewport_width_screen = 0u;
        constexpr std::uint32_t viewport_depth_tested = 0u;
        constexpr std::uint32_t viewport_always_visible = 1u;
        constexpr char spectra_y_up[] = "SpectraYUp";
        constexpr char opencv[] = "OpenCV";
        constexpr char action_start_training_id[] = "start_training";
        constexpr char action_pause_training_id[] = "pause_training";
        constexpr char action_reset_training_id[] = "reset_training";
        constexpr char action_render_preview_id[] = "render_preview";
        constexpr char action_option_frame_set_key[] = "frame_set";
        constexpr char action_option_target_steps_key[] = "target_steps";
        constexpr char action_option_steps_per_update_key[] = "steps_per_update";
        constexpr char action_option_log_every_key[] = "log_every";
        constexpr char action_option_image_index_key[] = "image_index";
        constexpr char action_option_refresh_acceleration_key[] = "refresh_acceleration";
        constexpr char setting_show_occupancy_key[] = "show_occupancy";
        constexpr char setting_occupancy_alpha_key[] = "occupancy_alpha";
        constexpr char setting_occupancy_cell_scale_key[] = "occupancy_cell_scale";
        constexpr char density_volume_name[] = "Reconstructed Density";
        constexpr char density_material_name[] = "Reconstructed Density Material";
        constexpr char density_light_name[] = "Reconstructed Density Key Light";

        struct Vector3 {
            float x{};
            float y{};
            float z{};
        };

        struct Quaternion {
            float x{};
            float y{};
            float z{};
            float w{1.0f};
        };

        struct SceneOptions {
            std::filesystem::path dataset_path{};
            std::string format{"auto"};
            std::vector<std::string> frame_sets{"train"};
            float scene_scale{0.33f};
            std::uint64_t frame_stride{1u};
            std::uint64_t max_frames{};
            float visual_far{0.25f};
            float image_alpha{0.35f};
            float frustum_width{1.5f};
            bool show_occupancy{true};
            float occupancy_alpha{0.18f};
            float occupancy_cell_scale{0.75f};
        };

        struct TrainingOptions {
            std::string frame_set{"train"};
            std::uint32_t target_steps{200000u};
            std::uint32_t steps_per_update{1u};
            std::uint32_t log_every{100u};
        };

        struct PreviewOptions {
            std::string frame_set{"train"};
            std::uint32_t image_index{};
            bool refresh_acceleration{};
        };

        struct PreviewState {
            std::string frame_set{};
            std::uint32_t image_index{};
            std::uint32_t step{};
            float mse{};
            float psnr{};
            std::uint64_t revision{};
            std::vector<ProjectImage> images{};
        };

        struct ScalarHistory {
            std::string id{};
            std::string label{};
            std::string description{};
            std::string unit{};
            std::array<float, 4u> color{1.0f, 1.0f, 1.0f, 1.0f};
            std::uint32_t group{ControlActionGroupRun};
            std::int32_t priority{};
            std::uint64_t revision{1u};
            std::vector<ProjectScalarSample> samples{};
        };

        constexpr std::size_t scalar_history_capacity = 4096u;

        [[nodiscard]] OptionChoice choice(std::string value) {
            return OptionChoice{.value = value, .label = std::move(value)};
        }

        [[nodiscard]] OptionSchema option(std::string key, std::string label, std::string description, const OptionKind kind, const bool required, std::string default_value = {}, std::vector<OptionChoice> choices = {}, std::string group = {}, const bool advanced = false, const std::int32_t priority = 0) {
            return OptionSchema{
                .key = std::move(key),
                .label = std::move(label),
                .description = std::move(description),
                .kind = kind,
                .required = required,
                .default_value = std::move(default_value),
                .group = std::move(group),
                .advanced = advanced,
                .priority = priority,
                .choices = std::move(choices),
            };
        }

        [[nodiscard]] ProjectAction action(std::string id, std::string label, std::string description, std::vector<OptionSchema> options = {}, const std::uint32_t group = ControlActionGroupUtility, const std::int32_t priority = 0, const std::uint32_t style = ControlActionStyleSecondary) {
            return ProjectAction{
                .id = std::move(id),
                .label = std::move(label),
                .description = std::move(description),
                .group = group,
                .priority = priority,
                .style = style,
                .options = std::move(options),
            };
        }

        [[nodiscard]] ScalarHistory make_scalar_history(std::string id, std::string label, std::string description, std::string unit, const std::array<float, 4u> color, const std::uint32_t group, const std::int32_t priority) {
            return ScalarHistory{
                .id = std::move(id),
                .label = std::move(label),
                .description = std::move(description),
                .unit = std::move(unit),
                .color = color,
                .group = group,
                .priority = priority,
            };
        }

        [[nodiscard]] Descriptor make_descriptor() {
            return Descriptor{
                .id = "instant-ngp.project",
                .title = "Instant NGP Project",
                .open_action_label = "Open Dataset",
                .open_action_description = "Load the configured dataset and publish scene entities to Spectra.",
                .frames_per_second = 60.0,
                .open_options = {
                    option("dataset", "Dataset", "Dataset root directory.", OptionKind::DirectoryPath, true, {}, {}, "Dataset", false, 0),
                    option("format", "Format", "Dataset provider.", OptionKind::Choice, false, "auto", {choice("auto"), choice("nerf-synthetic"), choice("dd-nerf-dataset")}, "Dataset", false, 10),
                    option("frame_sets", "Frame Sets", "Comma-separated frame sets: train, validation, test.", OptionKind::Text, false, "train", {}, "Dataset", false, 20),
                    option("scene_scale", "Scene Scale", "Dataset scene scale passed to the dataset loader.", OptionKind::Float, false, "0.33", {}, "Dataset", false, 30),
                    option("frame_stride", "Frame Stride", "Only every Nth frame is visualized.", OptionKind::UnsignedInteger, false, "1", {}, "Dataset", false, 40),
                    option("max_frames", "Max Frames", "0 means no frame count limit.", OptionKind::UnsignedInteger, false, "0", {}, "Dataset", false, 50),
                    option("visual_far", "Visual Far", "Camera frustum visualization far plane.", OptionKind::Float, false, "0.25", {}, "Camera Visuals", false, 60),
                    option("image_alpha", "Image Alpha", "Camera image plane opacity in [0, 1].", OptionKind::Float, false, "0.35", {}, "Camera Visuals", false, 70),
                    option("frustum_width", "Frustum Width", "Screen-space frustum line width.", OptionKind::Float, false, "1.5", {}, "Camera Visuals", false, 80),
                },
                .control_actions = {
                    action(
                        action_start_training_id,
                        "Start Training",
                        "Start or resume optimization on the selected frame set.",
                        {
                            option(action_option_frame_set_key, "Frame Set", "Loaded frame set used for optimization.", OptionKind::Choice, false, "train", {choice("train"), choice("validation"), choice("test")}, "Run", false, 0),
                            option(action_option_target_steps_key, "Target Steps", "Optimization stops when this global step is reached.", OptionKind::UnsignedInteger, false, "200000", {}, "Run", false, 10),
                            option(action_option_steps_per_update_key, "Steps Per Update", "Optimization iterations executed during each GUI project update.", OptionKind::UnsignedInteger, false, "1", {}, "Run", false, 20),
                            option(action_option_log_every_key, "Log Every", "Optimization step interval for progress log entries.", OptionKind::UnsignedInteger, false, "100", {}, "Run", false, 30),
                        },
                        ControlActionGroupRun,
                        0,
                        ControlActionStylePrimary),
                    action(action_pause_training_id, "Pause", "Pause optimization after the current GUI update.", {}, ControlActionGroupRun, 10, ControlActionStyleSecondary),
                    action(
                        action_render_preview_id,
                        "Render Preview",
                        "Render one loaded frame through the current model and publish GT, prediction, and error images.",
                        {
                            option(action_option_frame_set_key, "Frame Set", "Loaded frame set used for preview rendering.", OptionKind::Choice, false, "train", {choice("train"), choice("validation"), choice("test")}, "Preview", false, 0),
                            option(action_option_image_index_key, "Image Index", "Zero-based image index in the selected frame set.", OptionKind::UnsignedInteger, false, "0", {}, "Preview", false, 10),
                            option(action_option_refresh_acceleration_key, "Refresh Acceleration", "Rebuild the density-grid acceleration before rendering.", OptionKind::Bool, false, "false", {}, "Preview", false, 20),
                        },
                        ControlActionGroupPreview,
                        0,
                        ControlActionStylePrimary),
                    action(action_reset_training_id, "Reset", "Destroy the current optimizer state and keep the loaded dataset visualization.", {}, ControlActionGroupRun, 90, ControlActionStyleDanger),
                },
                .control_settings = {
                    option(setting_show_occupancy_key, "Show Occupancy", "Show the occupancy voxel grid attached to the reconstructed density volume.", OptionKind::Bool, false, "true", {}, "Debug", false, 0),
                    option(setting_occupancy_alpha_key, "Occupancy Alpha", "Viewport occupancy voxel opacity in [0, 1].", OptionKind::Float, false, "0.18", {}, "Debug", false, 10),
                    option(setting_occupancy_cell_scale_key, "Cell Scale", "Viewport occupancy voxel cube scale in (0, 1].", OptionKind::Float, false, "0.75", {}, "Debug", false, 20),
                },
            };
        }

        [[nodiscard]] float parse_float(const std::string& text, const std::string_view name) {
            float value{};
            const char* const begin = text.data();
            const char* const end = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value)) throw std::runtime_error(std::format("{} must be a finite float", name));
            return value;
        }

        [[nodiscard]] std::uint64_t parse_u64(const std::string& text, const std::string_view name) {
            std::uint64_t value{};
            const char* const begin = text.data();
            const char* const end = text.data() + text.size();
            const std::from_chars_result result = std::from_chars(begin, end, value);
            if (result.ec != std::errc{} || result.ptr != end) throw std::runtime_error(std::format("{} must be an unsigned integer", name));
            return value;
        }

        [[nodiscard]] bool parse_bool(const std::string& text, const std::string_view name) {
            if (text == "true") return true;
            if (text == "false") return false;
            throw std::runtime_error(std::format("{} must be true or false", name));
        }

        [[nodiscard]] std::vector<std::string> parse_frame_sets(const std::string& text) {
            if (text.empty()) throw std::runtime_error("frame_sets must not be empty");
            std::vector<std::string> frame_sets{};
            std::set<std::string> seen{};
            std::size_t offset{};
            while (offset <= text.size()) {
                const std::size_t comma = text.find(',', offset);
                const std::string frame_set = comma == std::string::npos ? text.substr(offset) : text.substr(offset, comma - offset);
                if (frame_set != "train" && frame_set != "validation" && frame_set != "test") throw std::runtime_error(std::format("frame_sets contains unknown frame set '{}'", frame_set));
                if (!seen.insert(frame_set).second) throw std::runtime_error(std::format("frame_sets contains duplicate frame set '{}'", frame_set));
                frame_sets.push_back(frame_set);
                if (comma == std::string::npos) break;
                offset = comma + 1u;
                if (offset == text.size()) throw std::runtime_error("frame_sets contains a trailing comma");
            }
            return frame_sets;
        }

        [[nodiscard]] SceneOptions parse_scene_options(const std::span<const Option> options) {
            SceneOptions parsed{};
            std::optional<std::string> dataset_option{};
            std::set<std::string> seen_options{};
            for (const Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error(std::format("scene plugin open option '{}' is duplicated", option.key));
                if (option.key == "dataset") dataset_option = option.value;
                else if (option.key == "format") parsed.format = option.value;
                else if (option.key == "frame_sets") parsed.frame_sets = parse_frame_sets(option.value);
                else if (option.key == "scene_scale") parsed.scene_scale = parse_float(option.value, "scene_scale");
                else if (option.key == "frame_stride") parsed.frame_stride = parse_u64(option.value, "frame_stride");
                else if (option.key == "max_frames") parsed.max_frames = parse_u64(option.value, "max_frames");
                else if (option.key == "visual_far") parsed.visual_far = parse_float(option.value, "visual_far");
                else if (option.key == "image_alpha") parsed.image_alpha = parse_float(option.value, "image_alpha");
                else if (option.key == "frustum_width") parsed.frustum_width = parse_float(option.value, "frustum_width");
                else throw std::runtime_error(std::format("unknown scene plugin open option '{}'", option.key));
            }

            if (!dataset_option.has_value() || dataset_option->empty()) throw std::runtime_error("dataset option is required");
            parsed.dataset_path = std::filesystem::absolute(std::filesystem::path{*dataset_option}).lexically_normal();
            if (!std::filesystem::is_directory(parsed.dataset_path)) throw std::runtime_error(std::format("{}: dataset option must name an existing directory", parsed.dataset_path.string()));
            if (parsed.format != "auto" && parsed.format != "nerf-synthetic" && parsed.format != "dd-nerf-dataset") throw std::runtime_error(std::format("format must be auto, nerf-synthetic, or dd-nerf-dataset; got '{}'", parsed.format));
            if (!std::isfinite(parsed.scene_scale) || parsed.scene_scale <= 0.0f) throw std::runtime_error("scene_scale must be finite and positive");
            if (parsed.frame_stride == 0u) throw std::runtime_error("frame_stride must be at least 1");
            if (!std::isfinite(parsed.visual_far) || parsed.visual_far <= 0.02f) throw std::runtime_error("visual_far must be finite and greater than visual_near");
            if (!std::isfinite(parsed.image_alpha) || parsed.image_alpha < 0.0f || parsed.image_alpha > 1.0f) throw std::runtime_error("image_alpha must be in [0, 1]");
            if (!std::isfinite(parsed.frustum_width) || parsed.frustum_width <= 0.0f) throw std::runtime_error("frustum_width must be finite and positive");
            return parsed;
        }

        [[nodiscard]] Vector3 add(const Vector3 a, const Vector3 b) {
            return Vector3{a.x + b.x, a.y + b.y, a.z + b.z};
        }

        [[nodiscard]] Vector3 subtract(const Vector3 a, const Vector3 b) {
            return Vector3{a.x - b.x, a.y - b.y, a.z - b.z};
        }

        [[nodiscard]] Vector3 multiply(const Vector3 a, const float scale) {
            return Vector3{a.x * scale, a.y * scale, a.z * scale};
        }

        [[nodiscard]] float dot(const Vector3 a, const Vector3 b) {
            return a.x * b.x + a.y * b.y + a.z * b.z;
        }

        [[nodiscard]] Vector3 cross(const Vector3 a, const Vector3 b) {
            return Vector3{
                a.y * b.z - a.z * b.y,
                a.z * b.x - a.x * b.z,
                a.x * b.y - a.y * b.x,
            };
        }

        [[nodiscard]] Vector3 normalize(const Vector3 value, const std::string_view context) {
            const float length_squared = dot(value, value);
            if (!std::isfinite(length_squared) || length_squared <= 1.0e-12f) throw std::runtime_error(std::format("{} contains an invalid vector", context));
            return multiply(value, 1.0f / std::sqrt(length_squared));
        }

        [[nodiscard]] Quaternion quaternion_from_frame(Vector3 right, Vector3 up, Vector3 forward, const std::string_view context) {
            right = normalize(right, std::format("{} right", context));
            up = normalize(up, std::format("{} up", context));
            forward = normalize(forward, std::format("{} forward", context));
            if (std::abs(dot(right, up)) > 1.0e-3f || std::abs(dot(right, forward)) > 1.0e-3f || std::abs(dot(up, forward)) > 1.0e-3f) throw std::runtime_error(std::format("{} basis is not orthogonal", context));
            if (dot(cross(right, up), forward) <= 0.0f) throw std::runtime_error(std::format("{} basis must be right-handed", context));
            const float m00 = right.x;
            const float m01 = up.x;
            const float m02 = forward.x;
            const float m10 = right.y;
            const float m11 = up.y;
            const float m12 = forward.y;
            const float m20 = right.z;
            const float m21 = up.z;
            const float m22 = forward.z;
            Quaternion result{};
            const float trace = m00 + m11 + m22;
            if (trace > 0.0f) {
                const float s = std::sqrt(trace + 1.0f) * 2.0f;
                result.w = 0.25f * s;
                result.x = (m21 - m12) / s;
                result.y = (m02 - m20) / s;
                result.z = (m10 - m01) / s;
            } else if (m00 > m11 && m00 > m22) {
                const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2.0f;
                result.w = (m21 - m12) / s;
                result.x = 0.25f * s;
                result.y = (m01 + m10) / s;
                result.z = (m02 + m20) / s;
            } else if (m11 > m22) {
                const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2.0f;
                result.w = (m02 - m20) / s;
                result.x = (m01 + m10) / s;
                result.y = 0.25f * s;
                result.z = (m12 + m21) / s;
            } else {
                const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2.0f;
                result.w = (m10 - m01) / s;
                result.x = (m02 + m20) / s;
                result.y = (m12 + m21) / s;
                result.z = 0.25f * s;
            }
            const float quaternion_length_squared = result.x * result.x + result.y * result.y + result.z * result.z + result.w * result.w;
            if (!std::isfinite(quaternion_length_squared) || quaternion_length_squared <= 1.0e-12f) throw std::runtime_error(std::format("{} quaternion is invalid", context));
            const float inverse_length = 1.0f / std::sqrt(quaternion_length_squared);
            return Quaternion{result.x * inverse_length, result.y * inverse_length, result.z * inverse_length, result.w * inverse_length};
        }

        [[nodiscard]] std::string joined_frame_sets(const std::vector<std::string>& frame_sets) {
            std::string text{};
            for (const std::string& frame_set : frame_sets) {
                if (!text.empty()) text += ",";
                text += frame_set;
            }
            return text;
        }

        template <typename Dataset>
        [[nodiscard]] std::array<float, 3u> average_camera_forward(const Dataset& dataset) {
            std::array<double, 3u> sum{};
            std::uint64_t count = 0u;
            for (const auto& frame_set : dataset.frame_sets) {
                for (const auto& frame : frame_set.frames) {
                    if (frame.camera.size() != 12uz) throw std::runtime_error("dataset camera record must contain 12 values");
                    sum[0] += static_cast<double>(frame.camera[6]);
                    sum[1] += static_cast<double>(frame.camera[7]);
                    sum[2] += static_cast<double>(frame.camera[8]);
                    ++count;
                }
            }
            if (count == 0u) throw std::runtime_error("dataset has no frames for color reference direction");
            const double length = std::sqrt(sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]);
            if (!std::isfinite(length) || length <= 0.0) throw std::runtime_error("dataset average camera forward is degenerate");
            return {static_cast<float>(sum[0] / length), static_cast<float>(sum[1] / length), static_cast<float>(sum[2] / length)};
        }
    }

    struct InstantNgpSpectraProject::State {
        ~State() noexcept;

        SceneOptions options{};
        std::shared_ptr<HostServices> host_services{};
        std::variant<std::monostate, dataset::nerf_synthetic::Dataset, dataset::dd_nerf::Dataset> dataset{};
        std::unique_ptr<ngp::train::InstantNGP> ngp{};
        TrainingOptions training{};
        std::optional<ngp::train::OptimizationStats> latest_stats{};
        std::optional<PreviewState> latest_preview{};
        std::uint64_t next_preview_revision{1u};
        double scalar_time_seconds{};
        std::vector<ScalarHistory> scalar_histories{};
        GpuBufferAllocation density_allocation{};
        cudaExternalMemory_t density_external_memory{};
        float* density_values{};
        std::uint64_t density_buffer_byte_size{};
        GpuBufferAllocation color_allocation{};
        cudaExternalMemory_t color_external_memory{};
        float* color_values{};
        std::uint64_t color_buffer_byte_size{};
        std::uint64_t exported_color_revision{};
        std::array<float, 3u> color_reference_direction{};
        std::uint64_t exported_density_revision{};
        std::array<std::uint32_t, 3u> exported_density_dimensions{};
        float exported_density_optical_thickness_step{};
        float exported_volume_density_scale{};
        GpuBufferAllocation occupancy_allocation{};
        cudaExternalMemory_t occupancy_external_memory{};
        std::uint8_t* occupancy_bitfield{};
        std::uint64_t occupancy_buffer_byte_size{};
        std::uint64_t exported_occupancy_revision{};
        std::optional<ViewportVoxelGrid> occupancy_grid{};
        std::vector<Material> materials{};
        std::vector<Light> lights{};
        std::optional<VolumeGrid> density_volume{};
        DebugAttachmentSet debug_attachments{};
        bool training_running{};
        bool training_complete{};
        bool host_timeline_playing{true};
        std::uint32_t host_timeline_mode{ControlTimelineModeLive};
        std::string project_error{};
        std::uint32_t last_logged_step{};
        std::uint64_t next_log_sequence{};
        std::uint64_t scene_revision{1u};
        std::vector<ProjectLogEntry> logs{};
        std::vector<Camera> cameras{};
        std::string overview_camera_name{"Overview"};
    };

    InstantNgpSpectraProject::InstantNgpSpectraProject() = default;
    InstantNgpSpectraProject::InstantNgpSpectraProject(std::unique_ptr<State> state) : state(std::move(state)) {}
    InstantNgpSpectraProject::InstantNgpSpectraProject(InstantNgpSpectraProject&& other) noexcept = default;
    InstantNgpSpectraProject& InstantNgpSpectraProject::operator=(InstantNgpSpectraProject&& other) noexcept = default;
    InstantNgpSpectraProject::~InstantNgpSpectraProject() noexcept = default;

    const Descriptor& InstantNgpSpectraProject::descriptor() {
        static const Descriptor descriptor = make_descriptor();
        return descriptor;
    }

    namespace {
        [[nodiscard]] bool frame_set_loaded(const InstantNgpSpectraProject::State& state, const std::string& frame_set) {
            return std::ranges::any_of(state.options.frame_sets, [&frame_set](const std::string& loaded_frame_set) { return loaded_frame_set == frame_set; });
        }

        [[nodiscard]] std::uint32_t current_training_step(const InstantNgpSpectraProject::State& state) {
            if (!state.latest_stats.has_value()) return 0u;
            return state.latest_stats->step;
        }

        [[nodiscard]] bool host_timeline_advances_scene(const InstantNgpSpectraProject::State& state) {
            return state.host_timeline_playing && state.host_timeline_mode != ControlTimelineModePlayback;
        }

        void push_log(InstantNgpSpectraProject::State& state, std::string level, std::string message) {
            state.logs.push_back(ProjectLogEntry{
                .sequence = state.next_log_sequence++,
                .level = std::move(level),
                .message = std::move(message),
            });
            if (state.logs.size() > 2048u) state.logs.erase(state.logs.begin(), state.logs.begin() + static_cast<std::ptrdiff_t>(state.logs.size() - 2048u));
        }

        [[nodiscard]] ScalarHistory& scalar_history(InstantNgpSpectraProject::State& state, const std::string_view id) {
            const auto history = std::ranges::find_if(state.scalar_histories, [id](const ScalarHistory& candidate) { return candidate.id == id; });
            if (history == state.scalar_histories.end()) throw std::runtime_error(std::format("unknown scalar series '{}'", id));
            return *history;
        }

        void append_scalar_sample(InstantNgpSpectraProject::State& state, const std::string_view id, const std::uint64_t step, const double value) {
            if (!std::isfinite(state.scalar_time_seconds)) throw std::runtime_error("scalar series time must be finite");
            if (!std::isfinite(value)) throw std::runtime_error(std::format("scalar series '{}' value must be finite", id));
            ScalarHistory& history = scalar_history(state, id);
            history.samples.push_back(ProjectScalarSample{
                .step = step,
                .time_seconds = state.scalar_time_seconds,
                .value = value,
            });
            if (history.samples.size() > scalar_history_capacity) history.samples.erase(history.samples.begin(), history.samples.begin() + static_cast<std::ptrdiff_t>(history.samples.size() - scalar_history_capacity));
            ++history.revision;
        }

        void clear_scalar_histories(InstantNgpSpectraProject::State& state) {
            state.scalar_time_seconds = 0.0;
            for (ScalarHistory& history : state.scalar_histories) {
                history.samples.clear();
                ++history.revision;
            }
        }

        [[nodiscard]] bool has_nonzero_bytes(const std::span<const std::uint8_t> bytes) {
            return std::ranges::any_of(bytes, [](const std::uint8_t value) { return value != 0u; });
        }

        void close_imported_handle(GpuBufferAllocation& allocation) noexcept {
#if defined(_WIN32)
            if (allocation.handle_kind == GpuResourceHandleKind::OpaqueWin32 && allocation.handle != 0u) static_cast<void>(CloseHandle(reinterpret_cast<HANDLE>(allocation.handle)));
#else
            if (allocation.handle_kind == GpuResourceHandleKind::OpaqueFileDescriptor && allocation.handle != 0u) static_cast<void>(close(static_cast<int>(allocation.handle)));
#endif
            allocation.handle = 0u;
        }

        void release_density_buffer(InstantNgpSpectraProject::State& state) noexcept {
            state.density_values = nullptr;
            if (state.density_external_memory != nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(state.density_external_memory));
                state.density_external_memory = nullptr;
            }
            if (state.density_allocation.resource_id != 0u && state.host_services != nullptr) {
                try {
                    state.host_services->release_gpu_buffer(state.density_allocation.resource_id);
                } catch (...) {
                }
            }
            state.density_allocation = GpuBufferAllocation{};
            state.density_buffer_byte_size = 0u;
            state.exported_density_revision = 0u;
            state.exported_density_dimensions = {};
            state.exported_density_optical_thickness_step = 0.0f;
            state.exported_volume_density_scale = 0.0f;
            state.density_volume.reset();
        }

        void release_color_buffer(InstantNgpSpectraProject::State& state) noexcept {
            state.color_values = nullptr;
            if (state.color_external_memory != nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(state.color_external_memory));
                state.color_external_memory = nullptr;
            }
            if (state.color_allocation.resource_id != 0u && state.host_services != nullptr) {
                try {
                    state.host_services->release_gpu_buffer(state.color_allocation.resource_id);
                } catch (...) {
                }
            }
            state.color_allocation = GpuBufferAllocation{};
            state.color_buffer_byte_size = 0u;
            state.exported_color_revision = 0u;
        }

        void release_occupancy_buffer(InstantNgpSpectraProject::State& state) noexcept {
            state.occupancy_bitfield = nullptr;
            if (state.occupancy_external_memory != nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(state.occupancy_external_memory));
                state.occupancy_external_memory = nullptr;
            }
            if (state.occupancy_allocation.resource_id != 0u && state.host_services != nullptr) {
                try {
                    state.host_services->release_gpu_buffer(state.occupancy_allocation.resource_id);
                } catch (...) {
                }
            }
            state.occupancy_allocation = GpuBufferAllocation{};
            state.occupancy_buffer_byte_size = 0u;
            state.exported_occupancy_revision = 0u;
            state.occupancy_grid.reset();
        }

        void validate_cuda_device_identity(const GpuDeviceIdentity& identity) {
            int cuda_device = -1;
            if (const cudaError_t status = cudaGetDevice(&cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDevice failed: "} + cudaGetErrorString(status)};
            cudaDeviceProp device_properties{};
            if (const cudaError_t status = cudaGetDeviceProperties(&device_properties, cuda_device); status != cudaSuccess) throw std::runtime_error{std::string{"cudaGetDeviceProperties failed: "} + cudaGetErrorString(status)};

            const bool has_vulkan_uuid = has_nonzero_bytes(std::span<const std::uint8_t>{identity.device_uuid.data(), identity.device_uuid.size()});
            const bool has_vulkan_luid = has_nonzero_bytes(std::span<const std::uint8_t>{identity.device_luid.data(), identity.device_luid.size()});
            if (!has_vulkan_uuid && !has_vulkan_luid) throw std::runtime_error{"Spectra GPU resource device identity did not include UUID or LUID."};
            if (has_vulkan_uuid) {
                const std::uint8_t* const cuda_uuid = reinterpret_cast<const std::uint8_t*>(device_properties.uuid.bytes);
                for (std::size_t index = 0u; index < identity.device_uuid.size(); ++index)
                    if (identity.device_uuid[index] != cuda_uuid[index]) throw std::runtime_error{"Spectra Vulkan device UUID does not match the active CUDA device."};
            }
#if defined(_WIN32)
            if (has_vulkan_luid) {
                const std::uint8_t* const cuda_luid = reinterpret_cast<const std::uint8_t*>(device_properties.luid);
                for (std::size_t index = 0u; index < identity.device_luid.size(); ++index)
                    if (identity.device_luid[index] != cuda_luid[index]) throw std::runtime_error{"Spectra Vulkan device LUID does not match the active CUDA device."};
                if (identity.device_node_mask != device_properties.luidDeviceNodeMask) throw std::runtime_error{"Spectra Vulkan device node mask does not match the active CUDA device."};
            }
#endif
        }

        void ensure_density_buffer(InstantNgpSpectraProject::State& state, const std::uint64_t byte_size) {
            if (state.density_allocation.resource_id != 0u && state.density_buffer_byte_size >= byte_size) return;
            if (state.density_allocation.resource_id != 0u) release_density_buffer(state);
            if (state.host_services == nullptr) throw std::runtime_error{"Spectra host services are required for density volume visualization."};
            if (byte_size == 0u) throw std::runtime_error{"density grid volume byte size is invalid."};
            if (!state.host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
            if (!state.host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
            GpuBufferAllocation allocation = state.host_services->request_gpu_buffer(GpuBufferKindVolumeChannel, byte_size, "instant-ngp direct density grid volume");
            if (allocation.resource_id == 0u) throw std::runtime_error{"Spectra returned an invalid density volume resource id."};
            if (allocation.kind != GpuBufferKindVolumeChannel) throw std::runtime_error{"Spectra returned a non-volume density GPU buffer."};
            if (allocation.byte_size < byte_size) {
                close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned a density volume buffer smaller than requested."};
            }
            if (allocation.handle == 0u) {
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned an empty density volume external memory handle."};
            }

            try {
                validate_cuda_device_identity(allocation.device_identity);
                cudaExternalMemoryHandleDesc memory_desc{};
                memory_desc.size = allocation.byte_size;
                switch (allocation.handle_kind) {
#if defined(_WIN32)
                    case GpuResourceHandleKind::OpaqueWin32:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                        memory_desc.handle.win32.handle = reinterpret_cast<void*>(allocation.handle);
                        break;
#else
                    case GpuResourceHandleKind::OpaqueFileDescriptor:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                        memory_desc.handle.fd = static_cast<int>(allocation.handle);
                        break;
#endif
                    default:
                        throw std::runtime_error{"Spectra returned an unsupported density volume external memory handle kind."};
                }

                cudaExternalMemory_t imported_memory{};
                const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
                close_imported_handle(allocation);
                if (import_status != cudaSuccess) throw std::runtime_error{std::string{"cudaImportExternalMemory for density volume failed: "} + cudaGetErrorString(import_status)};

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.size = allocation.byte_size;
                void* mapped_buffer = nullptr;
                if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{std::string{"cudaExternalMemoryGetMappedBuffer for density volume failed: "} + cudaGetErrorString(status)};
                }
                if (mapped_buffer == nullptr) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{"cudaExternalMemoryGetMappedBuffer returned null for density volume."};
                }
                state.density_allocation = allocation;
                state.density_external_memory = imported_memory;
                state.density_values = static_cast<float*>(mapped_buffer);
                state.density_buffer_byte_size = byte_size;
            } catch (...) {
                if (allocation.handle != 0u) close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw;
            }
        }

        void ensure_color_buffer(InstantNgpSpectraProject::State& state, const std::uint64_t byte_size) {
            if (state.color_allocation.resource_id != 0u && state.color_buffer_byte_size >= byte_size) return;
            if (state.color_allocation.resource_id != 0u) release_color_buffer(state);
            if (state.host_services == nullptr) throw std::runtime_error{"Spectra host services are required for color volume visualization."};
            if (byte_size == 0u) throw std::runtime_error{"color grid volume byte size is invalid."};
            if (!state.host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
            if (!state.host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
            GpuBufferAllocation allocation = state.host_services->request_gpu_buffer(GpuBufferKindVolumeChannel, byte_size, "instant-ngp color grid volume");
            if (allocation.resource_id == 0u) throw std::runtime_error{"Spectra returned an invalid color volume resource id."};
            if (allocation.kind != GpuBufferKindVolumeChannel) throw std::runtime_error{"Spectra returned a non-volume color GPU buffer."};
            if (allocation.byte_size < byte_size) {
                close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned a color volume buffer smaller than requested."};
            }
            if (allocation.handle == 0u) {
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned an empty color volume external memory handle."};
            }

            try {
                validate_cuda_device_identity(allocation.device_identity);
                cudaExternalMemoryHandleDesc memory_desc{};
                memory_desc.size = allocation.byte_size;
                switch (allocation.handle_kind) {
#if defined(_WIN32)
                    case GpuResourceHandleKind::OpaqueWin32:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                        memory_desc.handle.win32.handle = reinterpret_cast<void*>(allocation.handle);
                        break;
#else
                    case GpuResourceHandleKind::OpaqueFileDescriptor:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                        memory_desc.handle.fd = static_cast<int>(allocation.handle);
                        break;
#endif
                    default:
                        throw std::runtime_error{"Spectra returned an unsupported color volume external memory handle kind."};
                }

                cudaExternalMemory_t imported_memory{};
                const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
                close_imported_handle(allocation);
                if (import_status != cudaSuccess) throw std::runtime_error{std::string{"cudaImportExternalMemory for color volume failed: "} + cudaGetErrorString(import_status)};

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.size = allocation.byte_size;
                void* mapped_buffer = nullptr;
                if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{std::string{"cudaExternalMemoryGetMappedBuffer for color volume failed: "} + cudaGetErrorString(status)};
                }
                if (mapped_buffer == nullptr) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{"cudaExternalMemoryGetMappedBuffer returned null for color volume."};
                }
                state.color_allocation = allocation;
                state.color_external_memory = imported_memory;
                state.color_values = static_cast<float*>(mapped_buffer);
                state.color_buffer_byte_size = byte_size;
            } catch (...) {
                if (allocation.handle != 0u) close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw;
            }
        }

        void ensure_occupancy_buffer(InstantNgpSpectraProject::State& state, const ngp::inspector::OccupancyGridDeviceView& view) {
            if (state.occupancy_allocation.resource_id != 0u && state.occupancy_buffer_byte_size >= view.bitfield_bytes) return;
            if (state.occupancy_allocation.resource_id != 0u) release_occupancy_buffer(state);
            if (state.host_services == nullptr) throw std::runtime_error{"Spectra host services are required for occupancy grid visualization."};
            if (view.bitfield_bytes == 0u) throw std::runtime_error{"occupancy grid bitfield byte size is invalid."};
            if (!state.host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
            if (!state.host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
            GpuBufferAllocation allocation = state.host_services->request_gpu_buffer(GpuBufferKindViewportVoxelGrid, view.bitfield_bytes, "instant-ngp occupancy grid bitfield");
            if (allocation.resource_id == 0u) throw std::runtime_error{"Spectra returned an invalid occupancy grid resource id."};
            if (allocation.kind != GpuBufferKindViewportVoxelGrid) throw std::runtime_error{"Spectra returned a non-viewport-voxel occupancy GPU buffer."};
            if (allocation.byte_size < view.bitfield_bytes) {
                close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned an occupancy grid buffer smaller than requested."};
            }
            if (allocation.handle == 0u) {
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw std::runtime_error{"Spectra returned an empty occupancy grid external memory handle."};
            }

            try {
                validate_cuda_device_identity(allocation.device_identity);
                cudaExternalMemoryHandleDesc memory_desc{};
                memory_desc.size = allocation.byte_size;
                switch (allocation.handle_kind) {
#if defined(_WIN32)
                    case GpuResourceHandleKind::OpaqueWin32:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                        memory_desc.handle.win32.handle = reinterpret_cast<void*>(allocation.handle);
                        break;
#else
                    case GpuResourceHandleKind::OpaqueFileDescriptor:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                        memory_desc.handle.fd = static_cast<int>(allocation.handle);
                        break;
#endif
                    default:
                        throw std::runtime_error{"Spectra returned an unsupported occupancy grid external memory handle kind."};
                }

                cudaExternalMemory_t imported_memory{};
                const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
                close_imported_handle(allocation);
                if (import_status != cudaSuccess) throw std::runtime_error{std::string{"cudaImportExternalMemory for occupancy grid failed: "} + cudaGetErrorString(import_status)};

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.size = allocation.byte_size;
                void* mapped_buffer = nullptr;
                if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{std::string{"cudaExternalMemoryGetMappedBuffer for occupancy grid failed: "} + cudaGetErrorString(status)};
                }
                if (mapped_buffer == nullptr) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{"cudaExternalMemoryGetMappedBuffer returned null for occupancy grid."};
                }
                state.occupancy_allocation = allocation;
                state.occupancy_external_memory = imported_memory;
                state.occupancy_bitfield = static_cast<std::uint8_t*>(mapped_buffer);
                state.occupancy_buffer_byte_size = view.bitfield_bytes;
            } catch (...) {
                if (allocation.handle != 0u) close_imported_handle(allocation);
                state.host_services->release_gpu_buffer(allocation.resource_id);
                throw;
            }
        }

        void publish_occupancy_grid_if_ready(InstantNgpSpectraProject::State& state) {
            if (!state.options.show_occupancy || state.ngp == nullptr || !state.density_volume.has_value()) {
                state.occupancy_grid.reset();
                return;
            }
            const ngp::inspector::Inspector inspector{*state.ngp};
            const ngp::inspector::OccupancyGridDeviceView view = inspector.occupancy_grid_device_view();
            if (!view.initialized) {
                state.occupancy_grid.reset();
                return;
            }
            if (view.encoding != ngp::inspector::OccupancyGridEncoding::MortonBitfield) throw std::runtime_error{"Unsupported Instant NGP occupancy grid encoding."};
            if (view.bitfield == nullptr || view.bitfield_bytes == 0u) throw std::runtime_error{"Instant NGP occupancy grid bitfield view is empty."};
            if (view.bitfield_bytes % sizeof(std::uint32_t) != 0u) throw std::runtime_error{"Instant NGP occupancy grid bitfield byte size must be uint32 aligned for Spectra visualization."};
            if (state.exported_occupancy_revision == view.revision && state.occupancy_grid.has_value()) {
                state.occupancy_grid->color = {0.12f, 0.78f, 1.0f, state.options.occupancy_alpha};
                state.occupancy_grid->cell_scale = state.options.occupancy_cell_scale;
                return;
            }
            ensure_occupancy_buffer(state, view);
            if (state.occupancy_bitfield == nullptr) throw std::runtime_error{"Occupancy grid bitfield buffer was not mapped."};
            if (const cudaError_t status = cudaMemcpy(state.occupancy_bitfield, view.bitfield, view.bitfield_bytes, cudaMemcpyDeviceToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy occupancy grid bitfield failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after occupancy grid export failed: "} + cudaGetErrorString(status)};
            const float voxel_size = 1.0f / static_cast<float>(view.dimensions[0]);
            state.exported_occupancy_revision = view.revision;
            state.occupancy_grid = ViewportVoxelGrid{
                .name = "NGP Occupancy Grid",
                .owner = SceneEntityRef{.kind = SceneEntityKind::VolumeGrid, .name = density_volume_name},
                .dimensions = view.dimensions,
                .origin = {0.0f, 0.0f, 0.0f},
                .voxel_size = {voxel_size, voxel_size, voxel_size},
                .transform = Transform{},
                .color = {0.12f, 0.78f, 1.0f, state.options.occupancy_alpha},
                .cell_scale = state.options.occupancy_cell_scale,
                .depth_mode = viewport_depth_tested,
                .source_kind = ViewportVoxelGridSourceKind::Bitfield,
                .index_encoding = ViewportVoxelGridIndexEncoding::Morton3D,
                .buffer_id = state.occupancy_allocation.resource_id,
                .source_byte_size = view.bitfield_bytes,
                .revision = view.revision,
            };
        }

        void create_trainer_if_needed(InstantNgpSpectraProject::State& state) {
            if (state.ngp != nullptr) return;
            std::visit([&](const auto& dataset) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                    throw std::runtime_error("dataset must be loaded before training");
                } else {
                    state.ngp = std::make_unique<ngp::train::InstantNGP>(dataset);
                }
            }, state.dataset);
        }

        void refresh_debug_attachments(InstantNgpSpectraProject::State& state) {
            publish_occupancy_grid_if_ready(state);
            state.debug_attachments.viewport_voxel_grids.clear();
            if (state.occupancy_grid.has_value()) state.debug_attachments.viewport_voxel_grids.push_back(*state.occupancy_grid);
        }

        void publish_density_grid_volume(InstantNgpSpectraProject::State& state) {
            create_trainer_if_needed(state);
            const ngp::inspector::Inspector inspector{*state.ngp};
            const ngp::inspector::DensityGridDeviceView view = inspector.density_grid_device_view();
            if (!view.initialized) throw std::runtime_error("Instant NGP density grid has not been initialized");
            if (view.encoding != ngp::inspector::DensityGridEncoding::MortonFloat32) throw std::runtime_error("Unsupported Instant NGP density grid encoding");
            if (view.values == nullptr || view.byte_size == 0u) throw std::runtime_error("Instant NGP density grid values view is empty");
            if (view.dimensions[0] == 0u || view.dimensions[1] == 0u || view.dimensions[2] == 0u) throw std::runtime_error("Instant NGP density grid dimensions must be non-zero");
            if (static_cast<std::uint64_t>(view.dimensions[0]) > std::numeric_limits<std::uint64_t>::max() / view.dimensions[1] / view.dimensions[2]) throw std::runtime_error("Instant NGP density grid dimensions overflow cell count");
            const std::uint64_t value_count = static_cast<std::uint64_t>(view.dimensions[0]) * static_cast<std::uint64_t>(view.dimensions[1]) * static_cast<std::uint64_t>(view.dimensions[2]);
            if (view.cell_count != value_count) throw std::runtime_error("Instant NGP density grid cell count does not match dimensions");
            if (value_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) throw std::runtime_error("Instant NGP density grid byte size overflows uint64");
            const std::uint64_t byte_size = value_count * sizeof(float);
            if (view.byte_size < byte_size) throw std::runtime_error("Instant NGP density grid byte size is smaller than dimensions");
            if (byte_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error("Instant NGP density grid byte size exceeds host addressable size");
            if (!std::isfinite(view.optical_thickness_step) || view.optical_thickness_step <= 0.0f) throw std::runtime_error("Instant NGP density grid optical thickness step must be finite and positive");
            if (value_count > std::numeric_limits<std::uint64_t>::max() / (3u * sizeof(float))) throw std::runtime_error("Instant NGP color grid byte size overflows uint64");
            const std::uint64_t color_byte_size = value_count * 3u * sizeof(float);
            if (color_byte_size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error("Instant NGP color grid byte size exceeds host addressable size");
            ensure_density_buffer(state, byte_size);
            ensure_color_buffer(state, color_byte_size);
            if (state.density_values == nullptr) throw std::runtime_error("Density volume external buffer was not mapped");
            if (state.color_values == nullptr) throw std::runtime_error("Color volume external buffer was not mapped");
            if (const cudaError_t status = cudaMemcpy(state.density_values, view.values, static_cast<std::size_t>(byte_size), cudaMemcpyDeviceToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy direct density grid failed: "} + cudaGetErrorString(status)};
            const std::expected<ngp::inspector::ColorGridSampleStats, std::string> color_stats = inspector.sample_color_grid(ngp::inspector::ColorGridSampleRequest{
                .dimensions = view.dimensions,
                .output_rgb = state.color_values,
                .byte_size = color_byte_size,
                .reference_direction = state.color_reference_direction,
                .encoding = ngp::inspector::ColorGridEncoding::MortonFloat32x3,
            });
            if (!color_stats) throw std::runtime_error(color_stats.error());
            if (color_stats->dimensions != view.dimensions) throw std::runtime_error("Instant NGP color grid sample dimensions do not match density grid dimensions");
            if (color_stats->byte_size < color_byte_size) throw std::runtime_error("Instant NGP color grid sample byte size is smaller than expected");
            if (color_stats->encoding != ngp::inspector::ColorGridEncoding::MortonFloat32x3) throw std::runtime_error("Instant NGP color grid sample returned unsupported encoding");
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after direct density grid export failed: "} + cudaGetErrorString(status)};
            const float volume_density_scale = 1.0f / view.optical_thickness_step;

            state.materials = {
                Material{
                    .name = density_material_name,
                    .model = "volume",
                    .alpha_mode = "blend",
                    .base_color = {1.0f, 1.0f, 1.0f, 1.0f},
                    .roughness = 0.35f,
                    .volume_density_scale = volume_density_scale,
                    .volume_temperature_scale = 1.0f,
                },
            };
            state.lights = {
                Light{
                    .name = density_light_name,
                    .kind = "directional",
                    .color = {1.0f, 1.0f, 1.0f},
                    .intensity = 3.0f,
                },
            };
            state.density_volume = VolumeGrid{
                .name = density_volume_name,
                .dimensions = view.dimensions,
                .origin = {0.0f, 0.0f, 0.0f},
                .voxel_size = {
                    1.0f / static_cast<float>(view.dimensions[0]),
                    1.0f / static_cast<float>(view.dimensions[1]),
                    1.0f / static_cast<float>(view.dimensions[2]),
                },
                .channels = {
                    VolumeChannel{
                        .name = "density",
                        .dimensions = view.dimensions,
                        .format = VolumeChannelFormat::Float32,
                        .source_kind = VolumeChannelSourceKind::ExternalGpuBuffer,
                        .index_encoding = VolumeChannelIndexEncoding::Morton3D,
                        .buffer_id = state.density_allocation.resource_id,
                        .external_device_pointer = reinterpret_cast<std::uintptr_t>(state.density_values),
                        .source_byte_size = byte_size,
                        .revision = view.revision,
                    },
                    VolumeChannel{
                        .name = "color",
                        .dimensions = view.dimensions,
                        .format = VolumeChannelFormat::Float32x3,
                        .source_kind = VolumeChannelSourceKind::ExternalGpuBuffer,
                        .index_encoding = VolumeChannelIndexEncoding::Morton3D,
                        .buffer_id = state.color_allocation.resource_id,
                        .external_device_pointer = reinterpret_cast<std::uintptr_t>(state.color_values),
                        .source_byte_size = color_byte_size,
                        .revision = view.revision,
                    },
                },
                .material_name = density_material_name,
            };
            state.exported_density_revision = view.revision;
            state.exported_color_revision = view.revision;
            state.exported_density_dimensions = view.dimensions;
            state.exported_density_optical_thickness_step = view.optical_thickness_step;
            state.exported_volume_density_scale = volume_density_scale;
            refresh_debug_attachments(state);
            ++state.scene_revision;
            push_log(state, "DENSITY_GRID", std::format("volume='{}' revision={} color_revision={} dimensions={}x{}x{} encoding=Morton3D density_format=Float32 color_format=Float32x3 optical_thickness_step={} volume_density_scale={}", density_volume_name, view.revision, state.exported_color_revision, view.dimensions[0], view.dimensions[1], view.dimensions[2], view.optical_thickness_step, volume_density_scale));
        }

        void set_project_error(InstantNgpSpectraProject::State& state, std::string message) {
            state.project_error = std::move(message);
            state.training_running = false;
            state.training_complete = false;
            push_log(state, "ERROR", state.project_error);
        }

        [[nodiscard]] TrainingOptions parse_training_options(const InstantNgpSpectraProject::State& state, const std::span<const Option> options) {
            TrainingOptions parsed{};
            std::set<std::string> seen_options{};
            for (const Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error(std::format("project action option '{}' is duplicated", option.key));
                if (option.key == action_option_frame_set_key) parsed.frame_set = option.value;
                else if (option.key == action_option_target_steps_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, action_option_target_steps_key);
                    if (value_u64 == 0u || value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error("target_steps must fit in uint32 and be positive");
                    parsed.target_steps = static_cast<std::uint32_t>(value_u64);
                } else if (option.key == action_option_steps_per_update_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, action_option_steps_per_update_key);
                    if (value_u64 == 0u || value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("steps_per_update must fit in int32 and be positive");
                    parsed.steps_per_update = static_cast<std::uint32_t>(value_u64);
                } else if (option.key == action_option_log_every_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, action_option_log_every_key);
                    if (value_u64 == 0u || value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error("log_every must fit in uint32 and be positive");
                    parsed.log_every = static_cast<std::uint32_t>(value_u64);
                } else {
                    throw std::runtime_error(std::format("unknown project action option '{}'", option.key));
                }
            }
            if (parsed.frame_set != "train" && parsed.frame_set != "validation" && parsed.frame_set != "test") throw std::runtime_error(std::format("frame_set must be train, validation, or test; got '{}'", parsed.frame_set));
            if (!frame_set_loaded(state, parsed.frame_set)) throw std::runtime_error(std::format("frame_set '{}' was not loaded by Open Dataset", parsed.frame_set));
            if (parsed.target_steps <= current_training_step(state)) throw std::runtime_error(std::format("target_steps {} must be greater than current step {}", parsed.target_steps, current_training_step(state)));
            return parsed;
        }

        [[nodiscard]] PreviewOptions parse_preview_options(const InstantNgpSpectraProject::State& state, const std::span<const Option> options) {
            PreviewOptions parsed{};
            std::set<std::string> seen_options{};
            for (const Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error(std::format("project action option '{}' is duplicated", option.key));
                if (option.key == action_option_frame_set_key) parsed.frame_set = option.value;
                else if (option.key == action_option_image_index_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, action_option_image_index_key);
                    if (value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error("image_index must fit in uint32");
                    parsed.image_index = static_cast<std::uint32_t>(value_u64);
                } else if (option.key == action_option_refresh_acceleration_key) {
                    parsed.refresh_acceleration = parse_bool(option.value, action_option_refresh_acceleration_key);
                } else {
                    throw std::runtime_error(std::format("unknown project action option '{}'", option.key));
                }
            }
            if (parsed.frame_set != "train" && parsed.frame_set != "validation" && parsed.frame_set != "test") throw std::runtime_error(std::format("frame_set must be train, validation, or test; got '{}'", parsed.frame_set));
            if (!frame_set_loaded(state, parsed.frame_set)) throw std::runtime_error(std::format("frame_set '{}' was not loaded by Open Dataset", parsed.frame_set));
            return parsed;
        }

        [[nodiscard]] ProjectImage make_project_image(std::string id, std::string label, std::string description, const std::uint32_t width, const std::uint32_t height, const std::uint64_t revision, std::vector<std::uint8_t> rgba8) {
            if (id.empty()) throw std::runtime_error("project image id must not be empty");
            if (label.empty()) throw std::runtime_error(std::format("project image '{}' label must not be empty", id));
            if (width == 0u || height == 0u) throw std::runtime_error(std::format("project image '{}' dimensions must be non-zero", id));
            const std::uint64_t width_64 = width;
            const std::uint64_t height_64 = height;
            if (width_64 > std::numeric_limits<std::uint64_t>::max() / height_64 / 4u) throw std::runtime_error(std::format("project image '{}' dimensions overflow RGBA8 byte count", id));
            const std::uint64_t expected_byte_count = width_64 * height_64 * 4u;
            if (rgba8.size() != expected_byte_count) throw std::runtime_error(std::format("project image '{}' RGBA8 byte count must be width * height * 4", id));
            if (revision == 0u) throw std::runtime_error(std::format("project image '{}' revision must not be zero", id));
            return ProjectImage{
                .id = std::move(id),
                .label = std::move(label),
                .description = std::move(description),
                .width = width,
                .height = height,
                .revision = revision,
                .rgba8 = std::move(rgba8),
            };
        }

        void add_metric(ProjectStatus& status, std::string key, std::string label, std::string value, const std::uint32_t placement_flags = ControlPlacementPanelDetail, const std::int32_t priority = 0, const std::optional<std::array<float, 4u>> color = {}) {
            ProjectMetric metric{
                .key = std::move(key),
                .label = std::move(label),
                .value = std::move(value),
                .placement_flags = placement_flags,
                .priority = priority,
            };
            if (color.has_value()) {
                metric.has_color = true;
                metric.color = *color;
            }
            status.metrics.push_back(std::move(metric));
        }

        void add_action_state(ProjectStatus& status, std::string action_id, const bool enabled, std::string disabled_reason = {}) {
            status.action_states.push_back(ProjectActionState{.action_id = std::move(action_id), .enabled = enabled, .disabled_reason = std::move(disabled_reason)});
        }
    }

    InstantNgpSpectraProject::State::~State() noexcept {
        release_color_buffer(*this);
        release_density_buffer(*this);
        release_occupancy_buffer(*this);
    }

    InstantNgpSpectraProject InstantNgpSpectraProject::open(const std::span<const Option> options, std::shared_ptr<HostServices> host_services) {
        if (host_services == nullptr) throw std::runtime_error("host services are required to open the Instant NGP Spectra project");
        std::unique_ptr<State> created = std::make_unique<State>();
        created->host_services = std::move(host_services);
        created->options = parse_scene_options(options);
        created->scalar_histories = {
            make_scalar_history("training_loss", "Training Loss", "Training loss reported by optimizer updates.", "", {1.0f, 0.38f, 0.25f, 1.0f}, ControlActionGroupRun, 0),
            make_scalar_history("sample_efficiency_percent", "Sample Efficiency", "Useful sample ratio reported by optimization.", "%", {0.25f, 0.75f, 1.0f, 1.0f}, ControlActionGroupRun, 10),
            make_scalar_history("occupancy_percent", "Occupancy", "Density-grid occupied cell ratio.", "%", {0.16f, 0.86f, 0.55f, 1.0f}, ControlActionGroupRun, 20),
            make_scalar_history("step_time_ms", "Step Time", "Wall-clock optimizer update duration.", "ms", {1.0f, 0.75f, 0.25f, 1.0f}, ControlActionGroupRun, 30),
            make_scalar_history("step_rate", "Step Rate", "Optimizer throughput for the last update.", "step/s", {0.65f, 0.5f, 1.0f, 1.0f}, ControlActionGroupRun, 40),
            make_scalar_history("preview_mse", "Preview MSE", "Mean squared error from the most recent preview render.", "", {1.0f, 0.52f, 0.52f, 1.0f}, ControlActionGroupPreview, 0),
            make_scalar_history("preview_psnr", "Preview PSNR", "Peak signal-to-noise ratio from the most recent preview render.", "dB", {0.55f, 0.85f, 1.0f, 1.0f}, ControlActionGroupPreview, 10),
        };

        const bool is_nerf_synthetic = dataset::nerf_synthetic::is_dataset(created->options.dataset_path);
        const bool is_dd_nerf = dataset::dd_nerf::is_dataset(created->options.dataset_path);
        if (created->options.format == "auto") {
            if (is_nerf_synthetic == is_dd_nerf) throw std::runtime_error(std::format("{}: format=auto matched {} dataset providers", created->options.dataset_path.string(), is_nerf_synthetic ? 2 : 0));
            created->options.format = is_nerf_synthetic ? "nerf-synthetic" : "dd-nerf-dataset";
        } else if (created->options.format == "nerf-synthetic" && !is_nerf_synthetic) {
            throw std::runtime_error(std::format("{}: format=nerf-synthetic does not match this dataset", created->options.dataset_path.string()));
        } else if (created->options.format == "dd-nerf-dataset" && !is_dd_nerf) {
            throw std::runtime_error(std::format("{}: format=dd-nerf-dataset does not match this dataset", created->options.dataset_path.string()));
        }

        if (created->options.format == "nerf-synthetic") {
            std::expected<dataset::nerf_synthetic::Dataset, std::string> loaded = dataset::nerf_synthetic::load(created->options.dataset_path, {.frame_sets = created->options.frame_sets, .scene_scale = created->options.scene_scale});
            if (!loaded) throw std::runtime_error(loaded.error());
            created->dataset = std::move(*loaded);
        } else {
            std::expected<dataset::dd_nerf::Dataset, std::string> loaded = dataset::dd_nerf::load(created->options.dataset_path, {.frame_sets = created->options.frame_sets, .scene_scale = created->options.scene_scale});
            if (!loaded) throw std::runtime_error(loaded.error());
            created->dataset = std::move(*loaded);
        }

        std::visit([&](const auto& dataset) {
            if constexpr (std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                throw std::runtime_error("dataset must be loaded before computing color reference direction");
            } else {
                created->color_reference_direction = average_camera_forward(dataset);
            }
        }, created->dataset);

        std::uint64_t selected_camera_count{};
        std::visit([&](const auto& dataset) {
            if constexpr (!std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                bool done = false;
                for (const auto& frame_set : dataset.frame_sets) {
                    if (done) break;
                    for (std::size_t frame_index = 0u; frame_index < frame_set.frames.size(); ++frame_index) {
                        if ((static_cast<std::uint64_t>(frame_index) % created->options.frame_stride) != 0u) continue;
                        if (created->options.max_frames != 0u && selected_camera_count >= created->options.max_frames) {
                            done = true;
                            break;
                        }
                        ++selected_camera_count;
                    }
                }
            }
        }, created->dataset);
        if (selected_camera_count == 0u) throw std::runtime_error("dataset selection produced no camera frames");

        created->cameras.reserve(static_cast<std::size_t>(selected_camera_count + 1u));
        created->debug_attachments.viewport_camera_visuals.reserve(static_cast<std::size_t>(selected_camera_count));
        const Vector3 overview_target{0.5f, 0.5f, 0.5f};
        const Vector3 overview_eye{0.5f, 1.55f, -1.65f};
        const Vector3 overview_up{0.0f, 1.0f, 0.0f};
        const Vector3 overview_forward = normalize(subtract(overview_target, overview_eye), "overview camera forward");
        const Vector3 overview_right = normalize(cross(overview_up, overview_forward), "overview camera right");
        const Vector3 overview_camera_up = cross(overview_forward, overview_right);
        const Quaternion overview_rotation = quaternion_from_frame(overview_right, overview_camera_up, overview_forward, "overview camera");
        created->cameras.push_back(Camera{
            .name = created->overview_camera_name,
            .local_coordinate_system = spectra_y_up,
            .transform = Transform{.position = {overview_eye.x, overview_eye.y, overview_eye.z}, .rotation = {overview_rotation.x, overview_rotation.y, overview_rotation.z, overview_rotation.w}, .scale = {1.0f, 1.0f, 1.0f}},
            .target = {overview_target.x, overview_target.y, overview_target.z},
            .up = {overview_up.x, overview_up.y, overview_up.z},
            .projection = CameraProjection::Perspective,
            .vertical_fov_degrees = 45.0f,
            .near_plane = 0.01f,
            .far_plane = 20.0f,
        });

        std::uint64_t selected_index{};
        std::visit([&](const auto& dataset) {
            if constexpr (!std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                bool done = false;
                for (const auto& frame_set : dataset.frame_sets) {
                    if (done) break;
                    for (std::size_t frame_index = 0u; frame_index < frame_set.frames.size(); ++frame_index) {
                        if ((static_cast<std::uint64_t>(frame_index) % created->options.frame_stride) != 0u) continue;
                        if (created->options.max_frames != 0u && selected_index >= created->options.max_frames) {
                            done = true;
                            break;
                        }
                        const auto& frame = frame_set.frames.at(frame_index);
                        const std::string camera_name = std::format("{}-{:04}", frame_set.name, frame_index);
                        const Vector3 camera_x{frame.camera.at(0u), frame.camera.at(1u), frame.camera.at(2u)};
                        const Vector3 camera_y{frame.camera.at(3u), frame.camera.at(4u), frame.camera.at(5u)};
                        const Vector3 camera_z{frame.camera.at(6u), frame.camera.at(7u), frame.camera.at(8u)};
                        const Vector3 origin{frame.camera.at(9u), frame.camera.at(10u), frame.camera.at(11u)};
                        const Quaternion rotation = quaternion_from_frame(camera_x, camera_y, camera_z, std::format("dataset camera '{}'", camera_name));
                        const Vector3 focus = add(origin, normalize(camera_z, std::format("dataset camera '{}' forward", camera_name)));
                        const Vector3 navigation_up = multiply(normalize(camera_y, std::format("dataset camera '{}' visual up", camera_name)), -1.0f);
                        const std::uint64_t expected_bytes = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 4u;
                        if (frame.width == 0u || frame.height == 0u) throw std::runtime_error(std::format("dataset camera '{}' image dimensions must be non-zero", camera_name));
                        if (frame.rgba.empty()) throw std::runtime_error(std::format("dataset camera '{}' image is empty", camera_name));
                        if (static_cast<std::uint64_t>(frame.rgba.size()) != expected_bytes) throw std::runtime_error(std::format("dataset camera '{}' image byte count does not match width * height * 4", camera_name));
                        const float t = selected_camera_count > 1u ? static_cast<float>(selected_index) / static_cast<float>(selected_camera_count - 1u) : 0.0f;
                        created->cameras.push_back(Camera{
                            .name = camera_name,
                            .local_coordinate_system = opencv,
                            .transform = Transform{.position = {origin.x, origin.y, origin.z}, .rotation = {rotation.x, rotation.y, rotation.z, rotation.w}, .scale = {1.0f, 1.0f, 1.0f}},
                            .target = {focus.x, focus.y, focus.z},
                            .up = {navigation_up.x, navigation_up.y, navigation_up.z},
                            .projection = CameraProjection::Pinhole,
                            .vertical_fov_degrees = 45.0f,
                            .image_width = frame.width,
                            .image_height = frame.height,
                            .fx = frame.focal_x,
                            .fy = frame.focal_y,
                            .cx = frame.principal_x,
                            .cy = frame.principal_y,
                            .near_plane = 0.01f,
                            .far_plane = 10.0f,
                        });
                        created->debug_attachments.viewport_camera_visuals.push_back(ViewportCameraVisual{
                            .name = std::format("{} Visual", camera_name),
                            .owner = SceneEntityRef{.kind = SceneEntityKind::Camera, .name = camera_name},
                            .color = {0.12f + 0.72f * t, 0.82f, 1.0f - 0.55f * t, 0.52f},
                            .width = created->options.frustum_width,
                            .width_mode = viewport_width_screen,
                            .depth_mode = viewport_always_visible,
                            .visual_near = 0.02f,
                            .visual_far = created->options.visual_far,
                            .image = ViewportCameraVisualImage{
                                .rgba8 = frame.rgba.data(),
                                .rgba8_size = expected_bytes,
                                .revision = 1u,
                                .width = frame.width,
                                .height = frame.height,
                                .tint = {1.0f, 1.0f, 1.0f, created->options.image_alpha},
                            },
                        });
                        ++selected_index;
                    }
                }
            }
        }, created->dataset);

        publish_density_grid_volume(*created);
        push_log(*created, "CONFIG", std::format("profile={} dataset={} format={} scene_scale={} frame_sets={} visualized_cameras={} density_grid_revision={} color_grid_revision={} volume_density_scale={}", NGP_TRAIN_PROFILE_NAME, created->options.dataset_path.string(), created->options.format, created->options.scene_scale, joined_frame_sets(created->options.frame_sets), created->cameras.size() - 1u, created->exported_density_revision, created->exported_color_revision, created->exported_volume_density_scale));
        return InstantNgpSpectraProject{std::move(created)};
    }

    void InstantNgpSpectraProject::update(const UpdateInfo& update) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        if (!std::isfinite(update.wall_delta_seconds) || update.wall_delta_seconds < 0.0f) throw std::runtime_error("project update wall delta time is invalid");
        if (!std::isfinite(update.scene_delta_seconds) || update.scene_delta_seconds < 0.0f) throw std::runtime_error("project update scene delta time is invalid");
        if (!std::isfinite(update.time_seconds) || update.time_seconds < 0.0f) throw std::runtime_error("project update timeline time is invalid");
        if (update.timeline_mode > ControlTimelineModePlayback) throw std::runtime_error("project update timeline mode is invalid");
        state.host_timeline_playing = update.timeline_playing;
        state.host_timeline_mode = update.timeline_mode;
        if (!state.training_running) return;
        if (update.scene_delta_seconds == 0.0) return;
        try {
            create_trainer_if_needed(state);
            const std::uint32_t current_step = current_training_step(state);
            if (current_step >= state.training.target_steps) {
                state.training_running = false;
                state.training_complete = true;
                push_log(state, "COMPLETE", std::format("step={}", current_step));
                return;
            }
            const std::uint32_t remaining_steps = state.training.target_steps - current_step;
            const std::uint32_t requested_steps = std::min(state.training.steps_per_update, remaining_steps);
            const std::expected<ngp::train::OptimizationStats, std::string> stats = state.ngp->optimize(ngp::train::OptimizationRequest{
                .frame_set = state.training.frame_set,
                .iterations = static_cast<std::int32_t>(requested_steps),
            });
            if (!stats) {
                set_project_error(state, stats.error());
                return;
            }
            state.latest_stats = *stats;
            const bool reached_target = stats->step >= state.training.target_steps;
            const ngp::inspector::Inspector inspector{*state.ngp};
            const ngp::inspector::DensityGridDeviceView density_view = inspector.density_grid_device_view();
            if (density_view.initialized && density_view.revision != state.exported_density_revision) {
                publish_density_grid_volume(state);
            } else {
                const std::uint64_t previous_occupancy_revision = state.exported_occupancy_revision;
                const bool previous_occupancy_visible = state.occupancy_grid.has_value();
                refresh_debug_attachments(state);
                if (previous_occupancy_revision != state.exported_occupancy_revision || previous_occupancy_visible != state.occupancy_grid.has_value()) ++state.scene_revision;
            }
            state.scalar_time_seconds += static_cast<double>(stats->elapsed_ms) / 1000.0;
            const float step_rate = stats->elapsed_ms > 0.0f ? static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms : 0.0f;
            append_scalar_sample(state, "training_loss", stats->step, static_cast<double>(stats->loss));
            append_scalar_sample(state, "sample_efficiency_percent", stats->step, static_cast<double>(stats->sample_efficiency_ratio) * 100.0);
            append_scalar_sample(state, "occupancy_percent", stats->step, static_cast<double>(stats->density_grid_occupancy_ratio) * 100.0);
            append_scalar_sample(state, "step_time_ms", stats->step, static_cast<double>(stats->elapsed_ms));
            append_scalar_sample(state, "step_rate", stats->step, static_cast<double>(step_rate));
            const bool should_log = state.last_logged_step == 0u || stats->step - state.last_logged_step >= state.training.log_every || reached_target;
            if (should_log) {
                push_log(state, "OPTIMIZE", std::format("frame_set={} step={}/{} loss={:.6f} chunk={:.3f}ms rate={:.2f} step/s sample_eff={:.2f}% occupancy={:.2f}%", state.training.frame_set, stats->step, state.training.target_steps, stats->loss, stats->elapsed_ms, step_rate, stats->sample_efficiency_ratio * 100.0f, stats->density_grid_occupancy_ratio * 100.0f));
                state.last_logged_step = stats->step;
            }
            if (reached_target) {
                state.training_running = false;
                state.training_complete = true;
                push_log(state, "COMPLETE", std::format("step={} target_steps={}", stats->step, state.training.target_steps));
            }
        } catch (const std::exception& error) {
            set_project_error(state, error.what());
        }
    }

    void InstantNgpSpectraProject::execute_action(const std::string_view action_id, const std::span<const Option> options) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        if (action_id == action_start_training_id) {
            const TrainingOptions parsed = parse_training_options(state, options);
            create_trainer_if_needed(state);
            state.training = parsed;
            state.project_error.clear();
            state.training_complete = false;
            state.training_running = true;
            push_log(state, "START", std::format("frame_set={} target_steps={} steps_per_update={} log_every={}", state.training.frame_set, state.training.target_steps, state.training.steps_per_update, state.training.log_every));
        } else if (action_id == action_pause_training_id) {
            if (!state.training_running) throw std::runtime_error("training is not running");
            state.training_running = false;
            push_log(state, "PAUSE", std::format("step={}", current_training_step(state)));
        } else if (action_id == action_render_preview_id) {
            if (state.training_running && host_timeline_advances_scene(state)) throw std::runtime_error("pause the host timeline before rendering a preview");
            const PreviewOptions parsed = parse_preview_options(state, options);
            create_trainer_if_needed(state);
            const ngp::inspector::Inspector inspector{*state.ngp};
            std::expected<ngp::inspector::EvaluationPreviewResult, std::string> preview = inspector.evaluate_preview(ngp::inspector::EvaluationPreviewRequest{
                .frame_set = parsed.frame_set,
                .image_index = parsed.image_index,
                .refresh_acceleration = parsed.refresh_acceleration,
            });
            if (!preview) throw std::runtime_error(preview.error());
            state.scalar_time_seconds += static_cast<double>(preview->elapsed_ms) / 1000.0;
            append_scalar_sample(state, "preview_mse", preview->step, static_cast<double>(preview->mse));
            append_scalar_sample(state, "preview_psnr", preview->step, static_cast<double>(preview->psnr));
            const std::uint64_t revision = state.next_preview_revision++;
            PreviewState next_preview{
                .frame_set = preview->frame_set,
                .image_index = preview->image_index,
                .step = preview->step,
                .mse = preview->mse,
                .psnr = preview->psnr,
                .revision = revision,
                .images = {
                    make_project_image("ground_truth", "Ground Truth", std::format("{} image {}", preview->frame_set, preview->image_index), preview->width, preview->height, revision, std::move(preview->ground_truth_rgba8)),
                    make_project_image("prediction", "Prediction", std::format("Model render at step {}", preview->step), preview->width, preview->height, revision, std::move(preview->prediction_rgba8)),
                    make_project_image("error", "Error", "Per-channel absolute prediction error", preview->width, preview->height, revision, std::move(preview->error_rgba8)),
                },
            };
            state.latest_preview = std::move(next_preview);
            state.project_error.clear();
            push_log(state, "PREVIEW", std::format("frame_set={} image={} step={} mse={:.8f} psnr={:.2f} elapsed={:.3f}ms", preview->frame_set, preview->image_index, preview->step, preview->mse, preview->psnr, preview->elapsed_ms));
        } else if (action_id == action_reset_training_id) {
            release_color_buffer(state);
            release_density_buffer(state);
            release_occupancy_buffer(state);
            state.ngp.reset();
            state.latest_stats.reset();
            state.latest_preview.reset();
            state.materials.clear();
            state.lights.clear();
            state.debug_attachments.viewport_voxel_grids.clear();
            state.training = TrainingOptions{};
            state.training_running = false;
            state.training_complete = false;
            state.project_error.clear();
            state.last_logged_step = 0u;
            clear_scalar_histories(state);
            publish_density_grid_volume(state);
            push_log(state, "RESET", "training state reset and initial density restored");
        } else {
            throw std::runtime_error(std::format("unknown project action '{}'", action_id));
        }
    }

    std::vector<ProjectSettingValue> InstantNgpSpectraProject::settings() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        const State& state = *this->state;
        return {
            ProjectSettingValue{
                .key = setting_show_occupancy_key,
                .value = state.options.show_occupancy ? "true" : "false",
            },
            ProjectSettingValue{
                .key = setting_occupancy_alpha_key,
                .value = std::format("{:.9g}", state.options.occupancy_alpha),
            },
            ProjectSettingValue{
                .key = setting_occupancy_cell_scale_key,
                .value = std::format("{:.9g}", state.options.occupancy_cell_scale),
            },
        };
    }

    void InstantNgpSpectraProject::update_setting(const std::string_view key, const std::string_view value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        bool changed{};
        if (key == setting_show_occupancy_key) {
            const bool parsed = parse_bool(std::string{value}, setting_show_occupancy_key);
            changed = state.options.show_occupancy != parsed;
            state.options.show_occupancy = parsed;
            if (!state.options.show_occupancy) release_occupancy_buffer(state);
        } else if (key == setting_occupancy_alpha_key) {
            const float parsed = parse_float(std::string{value}, setting_occupancy_alpha_key);
            if (parsed < 0.0f || parsed > 1.0f) throw std::runtime_error("occupancy_alpha must be in [0, 1]");
            changed = state.options.occupancy_alpha != parsed;
            state.options.occupancy_alpha = parsed;
        } else if (key == setting_occupancy_cell_scale_key) {
            const float parsed = parse_float(std::string{value}, setting_occupancy_cell_scale_key);
            if (!(parsed > 0.0f) || parsed > 1.0f) throw std::runtime_error("occupancy_cell_scale must be in (0, 1]");
            changed = state.options.occupancy_cell_scale != parsed;
            state.options.occupancy_cell_scale = parsed;
        } else {
            throw std::runtime_error(std::format("unknown project setting '{}'", key));
        }
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
        push_log(state, "DEBUG", std::format("{}={}", key, value));
    }

    std::uint64_t InstantNgpSpectraProject::scene_revision() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        return this->state->scene_revision;
    }

    ProjectStatus InstantNgpSpectraProject::status() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        const State& state = *this->state;
        ProjectStatus status{};
        if (!state.project_error.empty()) {
            status.phase = "Error";
            status.headline = "Project error";
            status.detail = state.project_error;
        } else if (state.training_running && host_timeline_advances_scene(state)) {
            status.phase = "Running";
            status.headline = "Training running";
            status.detail = std::format("Optimizing frame_set={} in GUI project updates.", state.training.frame_set);
        } else if (state.training_running) {
            status.phase = "Paused";
            status.headline = "Timeline paused";
            status.detail = std::format("Training is armed for frame_set={} but the host timeline is not advancing.", state.training.frame_set);
        } else if (state.training_complete) {
            status.phase = "Complete";
            status.headline = "Training complete";
            status.detail = std::format("Reached target step {}.", state.training.target_steps);
        } else if (state.latest_stats.has_value()) {
            status.phase = "Paused";
            status.headline = "Training paused";
            status.detail = std::format("Current step {}.", current_training_step(state));
        } else {
            status.phase = "Ready";
            status.headline = "Dataset loaded";
            status.detail = "Start training from the Scene controls.";
        }

        add_metric(status, "dataset", "Dataset", state.options.dataset_path.filename().string(), ControlPlacementPanelSummary | ControlPlacementPanelDetail, 0);
        add_metric(status, "format", "Format", state.options.format, ControlPlacementPanelDetail, 10);
        add_metric(status, "frame_sets", "Frame Sets", joined_frame_sets(state.options.frame_sets), ControlPlacementPanelSummary | ControlPlacementPanelDetail, 10);
        add_metric(status, "step", "Step", std::format("{}", current_training_step(state)), ControlPlacementViewportOverlay | ControlPlacementPanelSummary | ControlPlacementPanelDetail, 20, std::array<float, 4u>{0.55f, 0.85f, 1.0f, 1.0f});
        add_metric(status, "target_steps", "Target", std::format("{}", state.training.target_steps), ControlPlacementPanelSummary | ControlPlacementPanelDetail, 30);
        add_metric(status, "density_visible", "Density", state.density_volume.has_value() ? "visible" : "hidden", ControlPlacementPanelDetail, 20);
        add_metric(status, "density_grid_revision", "Density Rev", std::format("{}", state.exported_density_revision), ControlPlacementPanelDetail, 30);
        add_metric(status, "color_grid_revision", "Color Rev", std::format("{}", state.exported_color_revision), ControlPlacementPanelDetail, 35);
        add_metric(status, "density_grid_encoding", "Density Grid Encoding", state.density_volume.has_value() ? "Morton3D Float32" : "None");
        add_metric(status, "color_grid_encoding", "Color Grid Encoding", state.density_volume.has_value() ? "Morton3D Float32x3" : "None");
        if (state.density_volume.has_value()) add_metric(status, "density_grid_dimensions", "Density Grid Dimensions", std::format("{}x{}x{}", state.exported_density_dimensions[0], state.exported_density_dimensions[1], state.exported_density_dimensions[2]));
        if (state.exported_density_optical_thickness_step > 0.0f) add_metric(status, "optical_thickness_step", "Optical Thickness Step", std::format("{:.8g}", state.exported_density_optical_thickness_step));
        if (state.exported_volume_density_scale > 0.0f) add_metric(status, "volume_density_scale", "Volume Density Scale", std::format("{:.8g}", state.exported_volume_density_scale));
        add_metric(status, "occupancy_visible", "Occupancy", state.occupancy_grid.has_value() ? "visible" : "hidden", ControlPlacementPanelDetail, 40);
        if (state.latest_stats.has_value()) {
            add_metric(status, "loss", "Loss", std::format("{:.6f}", state.latest_stats->loss), ControlPlacementViewportOverlay | ControlPlacementPanelSummary | ControlPlacementPanelDetail, 40, std::array<float, 4u>{1.0f, 0.38f, 0.25f, 1.0f});
            add_metric(status, "sample_efficiency", "Sample Eff", std::format("{:.2f}%", state.latest_stats->sample_efficiency_ratio * 100.0f), ControlPlacementViewportOverlay | ControlPlacementPanelSummary | ControlPlacementPanelDetail, 50, std::array<float, 4u>{0.25f, 0.75f, 1.0f, 1.0f});
            add_metric(status, "occupancy", "Occupancy", std::format("{:.2f}%", state.latest_stats->density_grid_occupancy_ratio * 100.0f), ControlPlacementViewportOverlay | ControlPlacementPanelSummary | ControlPlacementPanelDetail, 60, std::array<float, 4u>{0.16f, 0.86f, 0.55f, 1.0f});
            add_metric(status, "occupancy_revision", "Occupancy Revision", std::format("{}", state.exported_occupancy_revision));
        }
        if (state.latest_preview.has_value()) {
            add_metric(status, "preview_frame_set", "Preview Frame Set", state.latest_preview->frame_set);
            add_metric(status, "preview_image", "Preview Image", std::format("{}", state.latest_preview->image_index));
            add_metric(status, "preview_step", "Preview Step", std::format("{}", state.latest_preview->step));
            add_metric(status, "preview_mse", "Preview MSE", std::format("{:.8f}", state.latest_preview->mse));
            add_metric(status, "preview_psnr", "Preview PSNR", std::isfinite(state.latest_preview->psnr) ? std::format("{:.2f} dB", state.latest_preview->psnr) : "inf", ControlPlacementViewportOverlay | ControlPlacementPanelSummary | ControlPlacementPanelDetail, 70, std::array<float, 4u>{0.55f, 0.85f, 1.0f, 1.0f});
        }

        if (!state.project_error.empty()) {
            add_action_state(status, action_start_training_id, false, "Resolve or reset the project error before starting training.");
            add_action_state(status, action_pause_training_id, false, "Training is not running.");
            add_action_state(status, action_render_preview_id, false, "Resolve or reset the project error before rendering a preview.");
            add_action_state(status, action_reset_training_id, true);
        } else if (state.training_running && host_timeline_advances_scene(state)) {
            add_action_state(status, action_start_training_id, false, "Training is already running.");
            add_action_state(status, action_pause_training_id, true);
            add_action_state(status, action_render_preview_id, false, "Pause the host timeline before rendering a preview.");
            add_action_state(status, action_reset_training_id, true);
        } else if (state.training_running) {
            add_action_state(status, action_start_training_id, false, "Training is already running; resume with Space or stop it with Pause.");
            add_action_state(status, action_pause_training_id, true);
            add_action_state(status, action_render_preview_id, true);
            add_action_state(status, action_reset_training_id, true);
        } else {
            add_action_state(status, action_start_training_id, true);
            add_action_state(status, action_pause_training_id, false, "Training is not running.");
            add_action_state(status, action_render_preview_id, true);
            add_action_state(status, action_reset_training_id, true);
        }
        return status;
    }

    std::vector<ProjectLogEntry> InstantNgpSpectraProject::logs() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        return this->state->logs;
    }

    std::span<const ProjectImage> InstantNgpSpectraProject::images() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (!this->state->latest_preview.has_value()) return {};
        return this->state->latest_preview->images;
    }

    std::vector<ProjectScalarSeries> InstantNgpSpectraProject::scalar_series() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        std::vector<ProjectScalarSeries> series{};
        series.reserve(this->state->scalar_histories.size());
        for (const ScalarHistory& history : this->state->scalar_histories) {
            series.push_back(ProjectScalarSeries{
                .id = history.id,
                .label = history.label,
                .description = history.description,
                .unit = history.unit,
                .color = history.color,
                .group = history.group,
                .priority = history.priority,
                .revision = history.revision,
                .samples = history.samples,
            });
        }
        return series;
    }

    Document InstantNgpSpectraProject::document() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        return Document{
            .default_coordinate_system = spectra_y_up,
            .active_camera_name = this->state->overview_camera_name,
            .cameras = this->state->cameras,
            .materials = this->state->materials,
            .lights = this->state->lights,
            .volumes = this->state->density_volume.has_value() ? std::vector<VolumeGrid>{*this->state->density_volume} : std::vector<VolumeGrid>{},
            .debug_attachments = this->state->debug_attachments,
        };
    }

}
