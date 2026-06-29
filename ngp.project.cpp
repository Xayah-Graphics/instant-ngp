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

#if defined(_WIN32)
#define SPECTRA_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_SCENE_EXPORT __attribute__((visibility("default")))
#endif

module ngp.project;

import std;
import dataset.nerf_synthetic;
import dataset.dd_nerf;
import ngp.train;
import ngp.inspector;
import ngp.plugin;

namespace ngp::project {
    namespace {
        constexpr std::uint32_t viewport_depth_tested = 0u;
        constexpr char action_render_preview_id[] = "render_preview";
        constexpr char open_option_training_frame_set_key[] = "training_frame_set";
        constexpr char open_option_target_steps_key[] = "target_steps";
        constexpr char open_option_steps_per_update_key[] = "steps_per_update";
        constexpr char preview_option_frame_set_key[] = "frame_set";
        constexpr char preview_option_image_index_key[] = "image_index";
        constexpr char setting_update_occupancy_grid_key[] = "update_occupancy_grid";
        constexpr char setting_show_volume_key[] = "show_volume";
        constexpr char setting_show_occupancy_key[] = "show_occupancy";
        constexpr char setting_occupancy_alpha_key[] = "occupancy_alpha";
        constexpr char setting_occupancy_cell_scale_key[] = "occupancy_cell_scale";
        constexpr char setting_show_sampler_key[] = "show_sampler";
        constexpr char setting_show_sampler_points_key[] = "show_sampler_points";
        constexpr char setting_show_sampler_rays_key[] = "show_sampler_rays";
        constexpr char setting_sampler_point_radius_key[] = "sampler_point_radius";
        constexpr char setting_sampler_ray_width_key[] = "sampler_ray_width";
        constexpr char section_dataset_id[] = "dataset";
        constexpr char section_training_id[] = "training";
        constexpr char section_preview_id[] = "preview";
        constexpr char section_field_id[] = "field";
        constexpr char section_sampler_id[] = "sampler";
        constexpr char section_diagnostics_id[] = "diagnostics";
        constexpr char density_volume_name[] = "Reconstructed Density";
        constexpr char density_material_name[] = "Reconstructed Density Material";
        constexpr char density_light_name[] = "Reconstructed Density Key Light";
        constexpr char sampler_point_cloud_name[] = "NGP Sampler Samples";
        constexpr char sampler_ray_segments_name[] = "NGP Sampler Rays";
        constexpr char sampler_material_name[] = "NGP Sampler Point Material";

        struct Vector3 {
            float x{};
            float y{};
            float z{};
        };

        struct DatasetOptions {
            std::filesystem::path dataset_path{};
            std::string format{"auto"};
            std::vector<std::string> frame_sets{"train"};
            float scene_scale{0.33f};
            std::uint64_t frame_stride{1u};
            std::uint64_t max_frames{};
        };

        struct TrainingSettings {
            std::string frame_set{"train"};
            std::uint32_t target_steps{200000u};
            std::uint32_t steps_per_update{1u};
            bool update_occupancy_grid{true};
        };

        struct DebugOptions {
            bool show_volume{true};
            bool show_occupancy{false};
            float occupancy_alpha{0.18f};
            float occupancy_cell_scale{0.75f};
            bool show_sampler{false};
            bool show_sampler_points{true};
            bool show_sampler_rays{true};
            float sampler_point_radius{0.002f};
            float sampler_ray_width{1.5f};
        };

        struct ProjectOpenOptions {
            DatasetOptions dataset{};
            TrainingSettings training{};
        };

        struct PreviewState {
            std::string frame_set{};
            std::uint32_t image_index{};
            std::uint32_t step{};
            float mse{};
            float psnr{};
            std::uint64_t revision{};
        };

        class ExternalGpuBuffer final {
        public:
            ExternalGpuBuffer() = default;
            ExternalGpuBuffer(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer(ExternalGpuBuffer&&) = delete;
            ExternalGpuBuffer& operator=(const ExternalGpuBuffer&) = delete;
            ExternalGpuBuffer& operator=(ExternalGpuBuffer&&) = delete;
            ~ExternalGpuBuffer() noexcept;

            void ensure(std::shared_ptr<plugin::HostServices> host_services, std::uint32_t kind, std::uint64_t requested_byte_size, std::string_view debug_name, std::string_view label);
            void reset() noexcept;

            [[nodiscard]] bool has_capacity(std::uint64_t requested_byte_size) const noexcept;
            [[nodiscard]] std::uint64_t resource_id() const noexcept;

            template <typename Value>
            [[nodiscard]] Value* mapped_as() const noexcept {
                return static_cast<Value*>(this->mapped_buffer);
            }

        private:
            std::shared_ptr<plugin::HostServices> host_services{};
            plugin::GpuBufferAllocation allocation{};
            cudaExternalMemory_t external_memory{};
            void* mapped_buffer{};
            std::uint64_t byte_size{};
        };

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

        [[nodiscard]] ProjectOpenOptions parse_project_open_options(const std::span<const plugin::Option> options) {
            ProjectOpenOptions parsed{};
            std::optional<std::string> dataset_option{};
            std::set<std::string> seen_options{};
            for (const plugin::Option& option : options) {
                if (!seen_options.insert(option.key).second) throw std::runtime_error(std::format("scene plugin open option '{}' is duplicated", option.key));
                if (option.key == "dataset") dataset_option = option.value;
                else if (option.key == "format") parsed.dataset.format = option.value;
                else if (option.key == "frame_sets") parsed.dataset.frame_sets = parse_frame_sets(option.value);
                else if (option.key == "scene_scale") parsed.dataset.scene_scale = parse_float(option.value, "scene_scale");
                else if (option.key == "frame_stride") parsed.dataset.frame_stride = parse_u64(option.value, "frame_stride");
                else if (option.key == "max_frames") parsed.dataset.max_frames = parse_u64(option.value, "max_frames");
                else if (option.key == open_option_training_frame_set_key) parsed.training.frame_set = option.value;
                else if (option.key == open_option_target_steps_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, open_option_target_steps_key);
                    if (value_u64 == 0u || value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error("target_steps must fit in uint32 and be positive");
                    parsed.training.target_steps = static_cast<std::uint32_t>(value_u64);
                } else if (option.key == open_option_steps_per_update_key) {
                    const std::uint64_t value_u64 = parse_u64(option.value, open_option_steps_per_update_key);
                    if (value_u64 == 0u || value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::int32_t>::max())) throw std::runtime_error("steps_per_update must fit in int32 and be positive");
                    parsed.training.steps_per_update = static_cast<std::uint32_t>(value_u64);
                } else throw std::runtime_error(std::format("unknown scene plugin open option '{}'", option.key));
            }

            if (!dataset_option.has_value() || dataset_option->empty()) throw std::runtime_error("dataset option is required");
            parsed.dataset.dataset_path = std::filesystem::absolute(std::filesystem::path{*dataset_option}).lexically_normal();
            if (!std::filesystem::is_directory(parsed.dataset.dataset_path)) throw std::runtime_error(std::format("{}: dataset option must name an existing directory", parsed.dataset.dataset_path.string()));
            if (parsed.dataset.format != "auto" && parsed.dataset.format != "nerf-synthetic" && parsed.dataset.format != "dd-nerf-dataset") throw std::runtime_error(std::format("format must be auto, nerf-synthetic, or dd-nerf-dataset; got '{}'", parsed.dataset.format));
            if (!std::isfinite(parsed.dataset.scene_scale) || parsed.dataset.scene_scale <= 0.0f) throw std::runtime_error("scene_scale must be finite and positive");
            if (parsed.dataset.frame_stride == 0u) throw std::runtime_error("frame_stride must be at least 1");
            if (parsed.training.frame_set != "train" && parsed.training.frame_set != "validation" && parsed.training.frame_set != "test") throw std::runtime_error(std::format("training_frame_set must be train, validation, or test; got '{}'", parsed.training.frame_set));
            return parsed;
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

        [[nodiscard]] std::array<Vector3, 3u> spectra_image_down_camera_basis(Vector3 right, Vector3 down, Vector3 forward, const std::string_view context) {
            right = normalize(right, std::format("{} right", context));
            down = normalize(down, std::format("{} down", context));
            forward = normalize(forward, std::format("{} forward", context));
            if (std::abs(dot(right, down)) > 1.0e-3f || std::abs(dot(right, forward)) > 1.0e-3f || std::abs(dot(down, forward)) > 1.0e-3f) throw std::runtime_error(std::format("{} basis is not orthogonal", context));
            if (dot(cross(right, down), forward) <= 0.0f) throw std::runtime_error(std::format("{} basis must satisfy cross(right, down) == forward in Spectra image-down camera space", context));
            return {right, down, forward};
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

    struct Project::State {
        DatasetOptions dataset_options{};
        TrainingSettings training{};
        DebugOptions debug{};
        std::shared_ptr<plugin::HostServices> host_services{};
        std::variant<std::monostate, dataset::nerf_synthetic::Dataset, dataset::dd_nerf::Dataset> dataset{};
        std::unique_ptr<train::InstantNGP> ngp{};
        std::optional<train::OptimizationStats> latest_stats{};
        std::optional<PreviewState> latest_preview{};
        std::uint64_t next_preview_revision{1u};
        ExternalGpuBuffer density_buffer{};
        ExternalGpuBuffer color_buffer{};
        std::uint64_t exported_color_revision{};
        std::array<float, 3u> color_reference_direction{};
        std::uint64_t exported_density_revision{};
        std::array<std::uint32_t, 3u> exported_density_dimensions{};
        float exported_density_optical_thickness_step{};
        float exported_volume_density_scale{};
        ExternalGpuBuffer occupancy_buffer{};
        std::uint64_t exported_occupancy_revision{};
        std::optional<plugin::ViewportVoxelGrid> occupancy_grid{};
        ExternalGpuBuffer sampler_points_buffer{};
        ExternalGpuBuffer sampler_segments_buffer{};
        std::uint64_t exported_sampler_revision{};
        std::uint32_t exported_sampler_point_count{};
        std::uint32_t exported_sampler_ray_count{};
        std::optional<plugin::PointCloud> sampler_point_cloud{};
        std::optional<plugin::ViewportSegmentSet> sampler_ray_segments{};
        std::vector<plugin::Material> materials{};
        std::vector<plugin::Light> lights{};
        std::optional<plugin::VolumeGrid> density_volume{};
        plugin::DebugAttachmentSet debug_attachments{};
        bool host_update_running{};
        std::string project_error{};
        std::uint64_t scene_revision{1u};
        std::vector<plugin::Camera> cameras{};
        std::string overview_camera_name{"Overview"};
    };

    namespace {
        [[nodiscard]] bool density_volume_visible(const Project::State& state) {
            return state.debug.show_volume && state.density_volume.has_value();
        }

        struct VisualizationSnapshot final {
            std::uint64_t density_revision{};
            std::uint64_t color_revision{};
            bool density_visible{};
            bool occupancy_visible{};
            std::uint64_t occupancy_revision{};
            bool sampler_points_visible{};
            bool sampler_rays_visible{};
            std::uint64_t sampler_revision{};
            std::uint32_t sampler_point_count{};
            std::uint32_t sampler_ray_count{};

            friend bool operator==(const VisualizationSnapshot&, const VisualizationSnapshot&) = default;
        };

        [[nodiscard]] VisualizationSnapshot visualization_snapshot(const Project::State& state) {
            const bool volume_visible = density_volume_visible(state);
            return VisualizationSnapshot{
                .density_revision = state.exported_density_revision,
                .color_revision = state.exported_color_revision,
                .density_visible = volume_visible,
                .occupancy_visible = volume_visible && state.occupancy_grid.has_value(),
                .occupancy_revision = state.exported_occupancy_revision,
                .sampler_points_visible = state.sampler_point_cloud.has_value(),
                .sampler_rays_visible = state.sampler_ray_segments.has_value(),
                .sampler_revision = state.exported_sampler_revision,
                .sampler_point_count = state.exported_sampler_point_count,
                .sampler_ray_count = state.exported_sampler_ray_count,
            };
        }
    }

    Project::Project() = default;
    Project::Project(std::unique_ptr<State> state) : state(std::move(state)) {}
    Project::Project(Project&& other) noexcept = default;
    Project& Project::operator=(Project&& other) noexcept = default;
    Project::~Project() noexcept = default;

    const plugin::PluginDefinition<Project>& Project::plugin() {
        static const plugin::PluginDefinition<Project> definition{
            .id = "ngp.project",
            .title = "Instant NGP Project",
            .open_action_label = "Open Dataset",
            .sections = {
                plugin::section(section_dataset_id, "Dataset"),
                plugin::section(section_training_id, "Training"),
                plugin::section(section_preview_id, "Preview"),
                plugin::section(section_field_id, "Field"),
                plugin::section(section_sampler_id, "Sampler"),
                plugin::section(section_diagnostics_id, "Diagnostics"),
            },
            .open_options = {
                plugin::directory("dataset", "Dataset").describe("Dataset root directory.").section(section_dataset_id).defaulted(R"(C:\Users\xayah\Desktop\instant-ngp-new\data\nerf-synthetic\lego)").required(),
                plugin::choice("format", "Format", {"auto", "nerf-synthetic", "dd-nerf-dataset"}).describe("Dataset provider.").section(section_dataset_id).defaulted("auto"),
                plugin::text("frame_sets", "Frame Sets").describe("Comma-separated frame sets: train, validation, test.").section(section_dataset_id).defaulted("train"),
                plugin::float_option("scene_scale", "Scene Scale", 0.33f).describe("Dataset scene scale passed to the dataset loader.").section(section_dataset_id),
                plugin::unsigned_integer("frame_stride", "Frame Stride", 1u).describe("Only every Nth frame is visualized.").section(section_dataset_id),
                plugin::unsigned_integer("max_frames", "Max Frames", 0u).describe("0 means no frame count limit.").section(section_dataset_id),
                plugin::choice(open_option_training_frame_set_key, "Training Frame Set", {"train", "validation", "test"}).describe("Loaded frame set used for optimization.").section(section_training_id).defaulted("train"),
                plugin::unsigned_integer(open_option_target_steps_key, "Target Steps", 200000u).describe("Optimization stops when this global step is reached.").section(section_training_id),
                plugin::unsigned_integer(open_option_steps_per_update_key, "Steps Per Update", 1u).describe("Optimization iterations executed during each GUI project update.").section(section_training_id),
            },
            .actions = {
                plugin::action(action_render_preview_id, "Render Preview", &Project::render_preview)
                    .description("Render one loaded frame through the current model and publish preview metrics.")
                    .section(section_preview_id)
                    .option(plugin::choice(preview_option_frame_set_key, "Frame Set", {"train", "validation", "test"}).describe("Loaded frame set used for preview rendering.").section(section_preview_id).defaulted("train"))
                    .option(plugin::unsigned_integer(preview_option_image_index_key, "Image Index", 0u).describe("Zero-based image index in the selected frame set.").section(section_preview_id)),
            },
            .settings = {
                plugin::toggle(setting_update_occupancy_grid_key, "Update Occupancy Grid", true, &Project::set_update_occupancy_grid)
                    .section(section_training_id),
                plugin::toggle(setting_show_volume_key, "Show Volume", true, &Project::set_show_volume)
                    .section(section_field_id),
                plugin::toggle(setting_show_occupancy_key, "Show Occupancy", false, &Project::set_show_occupancy)
                    .section(section_field_id),
                plugin::float_value(setting_occupancy_alpha_key, "Occupancy Alpha", 0.18f, &Project::set_occupancy_alpha)
                    .section(section_field_id)
                    .slider(0.0f, 1.0f, 0.01f),
                plugin::float_value(setting_occupancy_cell_scale_key, "Cell Scale", 0.75f, &Project::set_occupancy_cell_scale)
                    .section(section_field_id)
                    .slider(0.01f, 1.0f, 0.01f),
                plugin::toggle(setting_show_sampler_key, "Show Sampler", false, &Project::set_show_sampler)
                    .section(section_sampler_id),
                plugin::toggle(setting_show_sampler_points_key, "Show Sampler Points", true, &Project::set_show_sampler_points)
                    .section(section_sampler_id),
                plugin::toggle(setting_show_sampler_rays_key, "Show Sampler Rays", true, &Project::set_show_sampler_rays)
                    .section(section_sampler_id),
                plugin::float_value(setting_sampler_point_radius_key, "Point Radius", 0.002f, &Project::set_sampler_point_radius)
                    .section(section_sampler_id)
                    .slider(0.0001f, 0.01f, 0.0001f),
                plugin::float_value(setting_sampler_ray_width_key, "Ray Width", 1.5f, &Project::set_sampler_ray_width)
                    .section(section_sampler_id)
                    .slider(0.5f, 8.0f, 0.1f),
            },
        };
        return definition;
    }

    namespace {
        [[nodiscard]] bool frame_set_loaded(const Project::State& state, const std::string& frame_set) {
            return std::visit([&frame_set](const auto& dataset) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                    return false;
                } else {
                    return std::ranges::any_of(dataset.frame_sets, [&frame_set](const auto& loaded_frame_set) { return loaded_frame_set.name == frame_set; });
                }
            }, state.dataset);
        }

        [[nodiscard]] std::uint32_t current_training_step(const Project::State& state) {
            if (!state.latest_stats.has_value()) return 0u;
            return state.latest_stats->step;
        }

        [[nodiscard]] bool has_nonzero_bytes(const std::span<const std::uint8_t> bytes) {
            return std::ranges::any_of(bytes, [](const std::uint8_t value) { return value != 0u; });
        }

        void close_imported_handle(plugin::GpuBufferAllocation& allocation) noexcept {
#if defined(_WIN32)
            if (allocation.handle_kind == plugin::GpuResourceHandleKind::OpaqueWin32 && allocation.handle != 0u) static_cast<void>(CloseHandle(reinterpret_cast<HANDLE>(allocation.handle)));
#else
            if (allocation.handle_kind == ngp::plugin::GpuResourceHandleKind::OpaqueFileDescriptor && allocation.handle != 0u) static_cast<void>(close(static_cast<int>(allocation.handle)));
#endif
            allocation.handle = 0u;
        }

        ExternalGpuBuffer::~ExternalGpuBuffer() noexcept {
            this->reset();
        }

        void ExternalGpuBuffer::reset() noexcept {
            this->mapped_buffer = nullptr;
            if (this->external_memory != nullptr) {
                static_cast<void>(cudaDestroyExternalMemory(this->external_memory));
                this->external_memory = nullptr;
            }
            if (this->allocation.resource_id != 0u && this->host_services != nullptr) {
                try {
                    this->host_services->release_gpu_buffer(this->allocation.resource_id);
                } catch (...) {
                }
            }
            this->allocation = plugin::GpuBufferAllocation{};
            this->byte_size = 0u;
            this->host_services.reset();
        }

        bool ExternalGpuBuffer::has_capacity(const std::uint64_t requested_byte_size) const noexcept {
            return this->allocation.resource_id != 0u && this->byte_size >= requested_byte_size;
        }

        std::uint64_t ExternalGpuBuffer::resource_id() const noexcept {
            return this->allocation.resource_id;
        }

        void validate_cuda_device_identity(const plugin::GpuDeviceIdentity& identity) {
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

        void ExternalGpuBuffer::ensure(std::shared_ptr<plugin::HostServices> next_host_services, const std::uint32_t kind, const std::uint64_t requested_byte_size, const std::string_view debug_name, const std::string_view label) {
            if (this->has_capacity(requested_byte_size)) return;
            this->reset();
            if (next_host_services == nullptr) throw std::runtime_error{std::format("Spectra host services are required for {} visualization.", label)};
            if (requested_byte_size == 0u) throw std::runtime_error{std::format("{} byte size is invalid.", label)};
            if (!next_host_services->request_gpu_buffer) throw std::runtime_error{"Spectra host services request_gpu_buffer callback is not configured."};
            if (!next_host_services->release_gpu_buffer) throw std::runtime_error{"Spectra host services release_gpu_buffer callback is not configured."};
            plugin::GpuBufferAllocation next_allocation = next_host_services->request_gpu_buffer(kind, requested_byte_size, debug_name);
            if (next_allocation.resource_id == 0u) throw std::runtime_error{std::format("Spectra returned an invalid {} resource id.", label)};
            if (next_allocation.kind != kind) throw std::runtime_error{std::format("Spectra returned an unexpected GPU buffer kind for {}.", label)};
            if (next_allocation.byte_size < requested_byte_size) {
                close_imported_handle(next_allocation);
                next_host_services->release_gpu_buffer(next_allocation.resource_id);
                throw std::runtime_error{std::format("Spectra returned a {} buffer smaller than requested.", label)};
            }
            if (next_allocation.handle == 0u) {
                next_host_services->release_gpu_buffer(next_allocation.resource_id);
                throw std::runtime_error{std::format("Spectra returned an empty {} external memory handle.", label)};
            }

            try {
                validate_cuda_device_identity(next_allocation.device_identity);
                cudaExternalMemoryHandleDesc memory_desc{};
                memory_desc.size = next_allocation.byte_size;
                switch (next_allocation.handle_kind) {
#if defined(_WIN32)
                    case plugin::GpuResourceHandleKind::OpaqueWin32:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
                        memory_desc.handle.win32.handle = reinterpret_cast<void*>(next_allocation.handle);
                        break;
#else
                    case ngp::plugin::GpuResourceHandleKind::OpaqueFileDescriptor:
                        memory_desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
                        memory_desc.handle.fd = static_cast<int>(next_allocation.handle);
                        break;
#endif
                    default:
                        throw std::runtime_error{std::format("Spectra returned an unsupported {} external memory handle kind.", label)};
                }

                cudaExternalMemory_t imported_memory{};
                const cudaError_t import_status = cudaImportExternalMemory(&imported_memory, &memory_desc);
                close_imported_handle(next_allocation);
                if (import_status != cudaSuccess) throw std::runtime_error{std::format("cudaImportExternalMemory for {} failed: {}", label, cudaGetErrorString(import_status))};

                cudaExternalMemoryBufferDesc buffer_desc{};
                buffer_desc.size = next_allocation.byte_size;
                void* next_mapped_buffer = nullptr;
                if (const cudaError_t status = cudaExternalMemoryGetMappedBuffer(&next_mapped_buffer, imported_memory, &buffer_desc); status != cudaSuccess) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer for {} failed: {}", label, cudaGetErrorString(status))};
                }
                if (next_mapped_buffer == nullptr) {
                    static_cast<void>(cudaDestroyExternalMemory(imported_memory));
                    throw std::runtime_error{std::format("cudaExternalMemoryGetMappedBuffer returned null for {}.", label)};
                }
                this->host_services = std::move(next_host_services);
                this->allocation = next_allocation;
                this->external_memory = imported_memory;
                this->mapped_buffer = next_mapped_buffer;
                this->byte_size = requested_byte_size;
            } catch (...) {
                if (next_allocation.handle != 0u) close_imported_handle(next_allocation);
                next_host_services->release_gpu_buffer(next_allocation.resource_id);
                throw;
            }
        }

        void publish_occupancy_grid_if_ready(Project::State& state) {
            if (!state.debug.show_volume || !state.debug.show_occupancy || state.ngp == nullptr || !state.density_volume.has_value()) {
                state.occupancy_grid.reset();
                return;
            }
            const inspector::Inspector inspector{*state.ngp};
            const inspector::OccupancyGridDeviceView view = inspector.occupancy_grid_device_view();
            if (!view.initialized) {
                state.occupancy_grid.reset();
                return;
            }
            if (view.encoding != inspector::OccupancyGridEncoding::MortonBitfield) throw std::runtime_error{"Unsupported Instant NGP occupancy grid encoding."};
            if (view.bitfield == nullptr || view.bitfield_bytes == 0u) throw std::runtime_error{"Instant NGP occupancy grid bitfield view is empty."};
            if (view.bitfield_bytes % sizeof(std::uint32_t) != 0u) throw std::runtime_error{"Instant NGP occupancy grid bitfield byte size must be uint32 aligned for Spectra visualization."};
            if (state.exported_occupancy_revision == view.revision && state.occupancy_grid.has_value()) {
                state.occupancy_grid->color = {0.12f, 0.78f, 1.0f, state.debug.occupancy_alpha};
                state.occupancy_grid->cell_scale = state.debug.occupancy_cell_scale;
                return;
            }
            state.occupancy_buffer.ensure(state.host_services, plugin::GpuBufferKindViewportVoxelGrid, view.bitfield_bytes, "instant-ngp occupancy grid bitfield", "occupancy grid");
            std::uint8_t* const occupancy_bitfield = state.occupancy_buffer.mapped_as<std::uint8_t>();
            if (occupancy_bitfield == nullptr) throw std::runtime_error{"Occupancy grid bitfield buffer was not mapped."};
            if (const cudaError_t status = cudaMemcpy(occupancy_bitfield, view.bitfield, view.bitfield_bytes, cudaMemcpyDeviceToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy occupancy grid bitfield failed: "} + cudaGetErrorString(status)};
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after occupancy grid export failed: "} + cudaGetErrorString(status)};
            const float voxel_size = 1.0f / static_cast<float>(view.dimensions[0]);
            state.exported_occupancy_revision = view.revision;
            state.occupancy_grid = plugin::ViewportVoxelGrid{
                .name = "NGP Occupancy Grid",
                .owner = plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::VolumeGrid, .name = density_volume_name},
                .dimensions = view.dimensions,
                .origin = {0.0f, 0.0f, 0.0f},
                .voxel_size = {voxel_size, voxel_size, voxel_size},
                .color = {0.12f, 0.78f, 1.0f, state.debug.occupancy_alpha},
                .cell_scale = state.debug.occupancy_cell_scale,
                .depth_mode = viewport_depth_tested,
                .source_kind = plugin::ViewportVoxelGridSourceKind::Bitfield,
                .index_encoding = plugin::ViewportVoxelGridIndexEncoding::Morton3D,
                .buffer_id = state.occupancy_buffer.resource_id(),
                .source_byte_size = view.bitfield_bytes,
                .revision = view.revision,
            };
        }

        void reset_sampler_visualization(Project::State& state, const bool release_buffers) noexcept {
            state.sampler_point_cloud.reset();
            state.sampler_ray_segments.reset();
            state.exported_sampler_revision = 0u;
            state.exported_sampler_point_count = 0u;
            state.exported_sampler_ray_count = 0u;
            if (release_buffers) {
                state.sampler_points_buffer.reset();
                state.sampler_segments_buffer.reset();
            }
        }

        void publish_sampler_visualization_if_ready(Project::State& state) {
            if (!state.debug.show_sampler || state.ngp == nullptr || (!state.debug.show_sampler_points && !state.debug.show_sampler_rays)) {
                reset_sampler_visualization(state, true);
                return;
            }

            const inspector::Inspector inspector{*state.ngp};
            const inspector::SamplerBatchDeviceView view = inspector.sampler_batch_device_view();
            if (!view.initialized) {
                reset_sampler_visualization(state, true);
                return;
            }
            if (view.compacted_sample_count == 0u || view.ray_count == 0u) throw std::runtime_error{"Sampler visualization last batch is empty."};
            if (view.compacted_sample_count > std::numeric_limits<std::uint64_t>::max() / inspector::SamplerPointInstanceBytes) throw std::runtime_error{"Sampler point visualization byte size overflows uint64."};
            if (view.ray_count > std::numeric_limits<std::uint64_t>::max() / inspector::SamplerSegmentInstanceBytes) throw std::runtime_error{"Sampler ray visualization byte size overflows uint64."};

            const std::uint64_t point_byte_size = static_cast<std::uint64_t>(view.compacted_sample_count) * inspector::SamplerPointInstanceBytes;
            const std::uint64_t segment_byte_size = static_cast<std::uint64_t>(view.ray_count) * inspector::SamplerSegmentInstanceBytes;
            std::byte* point_instances = nullptr;
            std::byte* segment_instances = nullptr;
            std::uint64_t requested_point_bytes{};
            std::uint64_t requested_segment_bytes{};
            const bool needs_point_cloud = state.debug.show_sampler_points;

            if (needs_point_cloud) {
                state.sampler_points_buffer.ensure(state.host_services, plugin::GpuBufferKindPointCloud, point_byte_size, "instant-ngp sampler compacted samples", "sampler point cloud");
                point_instances = state.sampler_points_buffer.mapped_as<std::byte>();
                if (point_instances == nullptr) throw std::runtime_error{"Sampler point cloud external buffer was not mapped."};
                requested_point_bytes = point_byte_size;
            } else {
                state.sampler_points_buffer.reset();
                state.sampler_point_cloud.reset();
            }

            if (state.debug.show_sampler_rays) {
                state.sampler_segments_buffer.ensure(state.host_services, plugin::GpuBufferKindViewportSegmentSet, segment_byte_size, "instant-ngp sampler ray segments", "sampler ray segments");
                segment_instances = state.sampler_segments_buffer.mapped_as<std::byte>();
                if (segment_instances == nullptr) throw std::runtime_error{"Sampler ray segment external buffer was not mapped."};
                requested_segment_bytes = segment_byte_size;
            } else {
                state.sampler_segments_buffer.reset();
                state.sampler_ray_segments.reset();
            }

            const std::expected<inspector::SamplerVisualizationStats, std::string> stats = inspector.write_sampler_visualization(inspector::SamplerVisualizationRequest{
                .point_instances = point_instances,
                .point_byte_size = requested_point_bytes,
                .segment_instances = segment_instances,
                .segment_byte_size = requested_segment_bytes,
                .point_radius = state.debug.sampler_point_radius,
                .ray_width = state.debug.sampler_ray_width,
                .width_mode = static_cast<std::uint32_t>(plugin::ViewportSegmentWidthMode::Screen),
            });
            if (!stats) throw std::runtime_error(stats.error());

            state.exported_sampler_revision = stats->revision;
            state.exported_sampler_point_count = needs_point_cloud ? stats->point_count : 0u;
            state.exported_sampler_ray_count = state.debug.show_sampler_rays ? stats->ray_count : 0u;
            if (needs_point_cloud) {
                state.sampler_point_cloud = plugin::PointCloud{
                    .name = sampler_point_cloud_name,
                    .source_kind = plugin::PointCloudSourceKind::ExternalGpuBuffer,
                    .point_count = stats->point_count,
                    .buffer_id = state.sampler_points_buffer.resource_id(),
                    .source_byte_size = stats->point_byte_size,
                    .revision = stats->revision,
                    .material_name = sampler_material_name,
                    .transform = {},
                    .bounds = plugin::Bounds{
                        .minimum = {0.0f, 0.0f, 0.0f},
                        .maximum = {1.0f, 1.0f, 1.0f},
                    },
                };
            }
            if (state.debug.show_sampler_rays) {
                state.sampler_ray_segments = plugin::ViewportSegmentSet{
                    .name = sampler_ray_segments_name,
                    .owner = needs_point_cloud ? plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::PointCloud, .name = sampler_point_cloud_name} : plugin::SceneEntityRef{.kind = plugin::SceneEntityKind::Camera, .name = state.overview_camera_name},
                    .source_kind = plugin::ViewportSegmentSourceKind::ExternalGpuBuffer,
                    .segment_count = stats->ray_count,
                    .buffer_id = state.sampler_segments_buffer.resource_id(),
                    .source_byte_size = stats->segment_byte_size,
                    .revision = stats->revision,
                    .width = state.debug.sampler_ray_width,
                    .width_mode = plugin::ViewportSegmentWidthMode::Screen,
                    .depth_mode = plugin::ViewportSegmentDepthMode::DepthTested,
                };
            }
        }

        void create_trainer_if_needed(Project::State& state) {
            if (state.ngp != nullptr) return;
            std::visit([&](const auto& dataset) {
                if constexpr (std::same_as<std::remove_cvref_t<decltype(dataset)>, std::monostate>) {
                    throw std::runtime_error("dataset must be loaded before training");
                } else {
                    state.ngp = std::make_unique<train::InstantNGP>(dataset, train::TrainingStateRequest{.occupancy_grid_update_active = state.training.update_occupancy_grid});
                }
            }, state.dataset);
        }

        void refresh_debug_attachments(Project::State& state) {
            publish_occupancy_grid_if_ready(state);
            publish_sampler_visualization_if_ready(state);
            state.debug_attachments.viewport_voxel_grids.clear();
            if (state.occupancy_grid.has_value()) state.debug_attachments.viewport_voxel_grids.push_back(*state.occupancy_grid);
            state.debug_attachments.viewport_segment_sets.clear();
            if (state.sampler_ray_segments.has_value()) state.debug_attachments.viewport_segment_sets.push_back(*state.sampler_ray_segments);
        }

        void publish_density_grid_volume(Project::State& state) {
            create_trainer_if_needed(state);
            const inspector::Inspector inspector{*state.ngp};
            const inspector::DensityGridDeviceView view = inspector.density_grid_device_view();
            if (!view.initialized) throw std::runtime_error("Instant NGP density grid has not been initialized");
            if (view.encoding != inspector::DensityGridEncoding::MortonFloat32) throw std::runtime_error("Unsupported Instant NGP density grid encoding");
            if (view.values == nullptr || view.byte_size == 0u) throw std::runtime_error("Instant NGP density grid values view is empty");
            if (view.dimensions[0] == 0u || view.dimensions[1] == 0u || view.dimensions[2] == 0u) throw std::runtime_error("Instant NGP density grid dimensions must be non-zero");
            if (static_cast<std::uint64_t>(view.dimensions[0]) > std::numeric_limits<std::uint64_t>::max() / view.dimensions[1] / view.dimensions[2]) throw std::runtime_error("Instant NGP density grid dimensions overflow cell count");
            const std::uint64_t value_count = static_cast<std::uint64_t>(view.dimensions[0]) * static_cast<std::uint64_t>(view.dimensions[1]) * static_cast<std::uint64_t>(view.dimensions[2]);
            if (view.cell_count != value_count) throw std::runtime_error("Instant NGP density grid cell count does not match dimensions");
            if (value_count > std::numeric_limits<std::uint64_t>::max() / sizeof(float)) throw std::runtime_error("Instant NGP density grid byte size overflows uint64");
            const std::uint64_t byte_size = value_count * sizeof(float);
            if (view.byte_size < byte_size) throw std::runtime_error("Instant NGP density grid byte size is smaller than dimensions");
            if (byte_size > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Instant NGP density grid byte size exceeds host addressable size");
            if (!std::isfinite(view.optical_thickness_step) || view.optical_thickness_step <= 0.0f) throw std::runtime_error("Instant NGP density grid optical thickness step must be finite and positive");
            if (value_count > std::numeric_limits<std::uint64_t>::max() / (3u * sizeof(float))) throw std::runtime_error("Instant NGP color grid byte size overflows uint64");
            const std::uint64_t color_byte_size = value_count * 3u * sizeof(float);
            if (color_byte_size > std::numeric_limits<std::size_t>::max()) throw std::runtime_error("Instant NGP color grid byte size exceeds host addressable size");
            state.density_buffer.ensure(state.host_services, plugin::GpuBufferKindVolumeChannel, byte_size, "instant-ngp direct density grid volume", "density volume");
            state.color_buffer.ensure(state.host_services, plugin::GpuBufferKindVolumeChannel, color_byte_size, "instant-ngp color grid volume", "color volume");
            float* const density_values = state.density_buffer.mapped_as<float>();
            float* const color_values = state.color_buffer.mapped_as<float>();
            if (density_values == nullptr) throw std::runtime_error("Density volume external buffer was not mapped");
            if (color_values == nullptr) throw std::runtime_error("Color volume external buffer was not mapped");
            if (const cudaError_t status = cudaMemcpy(density_values, view.values, byte_size, cudaMemcpyDeviceToDevice); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemcpy direct density grid failed: "} + cudaGetErrorString(status)};
            const std::expected<inspector::ColorGridSampleStats, std::string> color_stats = inspector.sample_color_grid(inspector::ColorGridSampleRequest{
                .dimensions = view.dimensions,
                .output_rgb = color_values,
                .byte_size = color_byte_size,
                .reference_direction = state.color_reference_direction,
                .encoding = inspector::ColorGridEncoding::MortonFloat32x3,
            });
            if (!color_stats) throw std::runtime_error(color_stats.error());
            if (color_stats->dimensions != view.dimensions) throw std::runtime_error("Instant NGP color grid sample dimensions do not match density grid dimensions");
            if (color_stats->byte_size < color_byte_size) throw std::runtime_error("Instant NGP color grid sample byte size is smaller than expected");
            if (color_stats->encoding != inspector::ColorGridEncoding::MortonFloat32x3) throw std::runtime_error("Instant NGP color grid sample returned unsupported encoding");
            if (const cudaError_t status = cudaDeviceSynchronize(); status != cudaSuccess) throw std::runtime_error{std::string{"cudaDeviceSynchronize after direct density grid export failed: "} + cudaGetErrorString(status)};
            const float volume_density_scale = 1.0f / view.optical_thickness_step;

            state.materials = {
                plugin::Material{
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
                plugin::Light{
                    .name = density_light_name,
                    .kind = "directional",
                    .color = {1.0f, 1.0f, 1.0f},
                    .intensity = 3.0f,
                },
            };
            state.density_volume = plugin::VolumeGrid{
                .name = density_volume_name,
                .dimensions = view.dimensions,
                .origin = {0.0f, 0.0f, 0.0f},
                .voxel_size = {
                    1.0f / static_cast<float>(view.dimensions[0]),
                    1.0f / static_cast<float>(view.dimensions[1]),
                    1.0f / static_cast<float>(view.dimensions[2]),
                },
                .channels = {
                    plugin::VolumeChannel{
                        .name = "density",
                        .dimensions = view.dimensions,
                        .format = plugin::VolumeChannelFormat::Float32,
                        .source_kind = plugin::VolumeChannelSourceKind::ExternalGpuBuffer,
                        .index_encoding = plugin::VolumeChannelIndexEncoding::Morton3D,
                        .buffer_id = state.density_buffer.resource_id(),
                        .external_device_pointer = reinterpret_cast<std::uintptr_t>(density_values),
                        .source_byte_size = byte_size,
                        .revision = view.revision,
                    },
                    plugin::VolumeChannel{
                        .name = "color",
                        .dimensions = view.dimensions,
                        .format = plugin::VolumeChannelFormat::Float32x3,
                        .source_kind = plugin::VolumeChannelSourceKind::ExternalGpuBuffer,
                        .index_encoding = plugin::VolumeChannelIndexEncoding::Morton3D,
                        .buffer_id = state.color_buffer.resource_id(),
                        .external_device_pointer = reinterpret_cast<std::uintptr_t>(color_values),
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
        }

        void set_project_error(Project::State& state, std::string message) {
            state.project_error = std::move(message);
        }

        void optimize_training_batch(Project::State& state) {
            create_trainer_if_needed(state);
            const std::uint32_t current_step = current_training_step(state);
            if (current_step >= state.training.target_steps) return;
            const std::uint32_t remaining_steps = state.training.target_steps - current_step;
            const std::uint32_t requested_steps = std::min(state.training.steps_per_update, remaining_steps);
            const std::expected<train::OptimizationStats, std::string> stats = state.ngp->optimize(train::OptimizationRequest{
                .frame_set = state.training.frame_set,
                .iterations = static_cast<std::int32_t>(requested_steps),
            });
            if (!stats) throw std::runtime_error(stats.error());
            state.latest_stats = *stats;
            const VisualizationSnapshot previous_visualization = visualization_snapshot(state);
            const inspector::Inspector inspector{*state.ngp};
            const inspector::DensityGridDeviceView density_view = inspector.density_grid_device_view();
            if (state.debug.show_volume && density_view.initialized && density_view.revision != state.exported_density_revision) publish_density_grid_volume(state);
            refresh_debug_attachments(state);
            if (!(previous_visualization == visualization_snapshot(state))) ++state.scene_revision;
            state.project_error.clear();
        }

    }

    Project Project::open(plugin::OpenContext context) {
        if (context.host_services == nullptr) throw std::runtime_error("host services are required to open the Instant NGP Spectra project");
        std::unique_ptr<State> created = std::make_unique<State>();
        created->host_services = std::move(context.host_services);
        ProjectOpenOptions open_options = parse_project_open_options(context.options);
        created->dataset_options = std::move(open_options.dataset);
        created->training = std::move(open_options.training);

        const bool is_nerf_synthetic = dataset::nerf_synthetic::is_dataset(created->dataset_options.dataset_path);
        const bool is_dd_nerf = dataset::dd_nerf::is_dataset(created->dataset_options.dataset_path);
        if (created->dataset_options.format == "auto") {
            if (is_nerf_synthetic == is_dd_nerf) throw std::runtime_error(std::format("{}: format=auto matched {} dataset providers", created->dataset_options.dataset_path.string(), is_nerf_synthetic ? 2 : 0));
            created->dataset_options.format = is_nerf_synthetic ? "nerf-synthetic" : "dd-nerf-dataset";
        } else if (created->dataset_options.format == "nerf-synthetic" && !is_nerf_synthetic) {
            throw std::runtime_error(std::format("{}: format=nerf-synthetic does not match this dataset", created->dataset_options.dataset_path.string()));
        } else if (created->dataset_options.format == "dd-nerf-dataset" && !is_dd_nerf) {
            throw std::runtime_error(std::format("{}: format=dd-nerf-dataset does not match this dataset", created->dataset_options.dataset_path.string()));
        }

        if (created->dataset_options.format == "nerf-synthetic") {
            std::expected<dataset::nerf_synthetic::Dataset, std::string> loaded = dataset::nerf_synthetic::load(created->dataset_options.dataset_path, {.frame_sets = created->dataset_options.frame_sets, .scene_scale = created->dataset_options.scene_scale});
            if (!loaded) throw std::runtime_error(loaded.error());
            created->dataset = std::move(*loaded);
        } else {
            std::expected<dataset::dd_nerf::Dataset, std::string> loaded = dataset::dd_nerf::load(created->dataset_options.dataset_path, {.frame_sets = created->dataset_options.frame_sets, .scene_scale = created->dataset_options.scene_scale});
            if (!loaded) throw std::runtime_error(loaded.error());
            created->dataset = std::move(*loaded);
        }
        if (!frame_set_loaded(*created, created->training.frame_set)) throw std::runtime_error(std::format("training_frame_set '{}' was not loaded by Open Dataset", created->training.frame_set));

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
                        if ((frame_index % created->dataset_options.frame_stride) != 0u) continue;
                        if (created->dataset_options.max_frames != 0u && selected_camera_count >= created->dataset_options.max_frames) {
                            done = true;
                            break;
                        }
                        ++selected_camera_count;
                    }
                }
            }
        }, created->dataset);
        if (selected_camera_count == 0u) throw std::runtime_error("dataset selection produced no camera frames");

        created->cameras.reserve(selected_camera_count + 1u);
        constexpr Vector3 overview_target{0.5f, 0.5f, 0.5f};
        constexpr Vector3 overview_eye{0.5f, 1.55f, 2.65f};
        constexpr Vector3 overview_up{0.0f, 1.0f, 0.0f};
        const Vector3 overview_forward = normalize(subtract(overview_target, overview_eye), "overview camera forward");
        const Vector3 overview_down = multiply(overview_up, -1.0f);
        const Vector3 overview_right = normalize(cross(overview_down, overview_forward), "overview camera right");
        const Vector3 overview_camera_down = cross(overview_forward, overview_right);
        const std::array<Vector3, 3u> overview_basis = spectra_image_down_camera_basis(overview_right, overview_camera_down, overview_forward, "overview camera");
        created->cameras.push_back(plugin::Camera{
            .name = created->overview_camera_name,
            .position = {overview_eye.x, overview_eye.y, overview_eye.z},
            .right = {overview_basis[0].x, overview_basis[0].y, overview_basis[0].z},
            .down = {overview_basis[1].x, overview_basis[1].y, overview_basis[1].z},
            .forward = {overview_basis[2].x, overview_basis[2].y, overview_basis[2].z},
            .projection = plugin::CameraProjection::Perspective,
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
                        if ((frame_index % created->dataset_options.frame_stride) != 0u) continue;
                        if (created->dataset_options.max_frames != 0u && selected_index >= created->dataset_options.max_frames) {
                            done = true;
                            break;
                        }
                        const auto& frame = frame_set.frames.at(frame_index);
                        const std::string camera_name = std::format("{}-{:04}", frame_set.name, frame_index);
                        const Vector3 camera_x{frame.camera.at(0u), frame.camera.at(1u), frame.camera.at(2u)};
                        const Vector3 camera_y{frame.camera.at(3u), frame.camera.at(4u), frame.camera.at(5u)};
                        const Vector3 camera_z{frame.camera.at(6u), frame.camera.at(7u), frame.camera.at(8u)};
                        const Vector3 origin{frame.camera.at(9u), frame.camera.at(10u), frame.camera.at(11u)};
                        const std::array<Vector3, 3u> basis = spectra_image_down_camera_basis(camera_x, camera_y, camera_z, std::format("dataset camera '{}'", camera_name));
                        const std::uint64_t expected_bytes = static_cast<std::uint64_t>(frame.width) * static_cast<std::uint64_t>(frame.height) * 4u;
                        if (frame.width == 0u || frame.height == 0u) throw std::runtime_error(std::format("dataset camera '{}' image dimensions must be non-zero", camera_name));
                        if (frame.rgba.empty()) throw std::runtime_error(std::format("dataset camera '{}' image is empty", camera_name));
                        if (static_cast<std::uint64_t>(frame.rgba.size()) != expected_bytes) throw std::runtime_error(std::format("dataset camera '{}' image byte count does not match width * height * 4", camera_name));
                        created->cameras.push_back(plugin::Camera{
                            .name = camera_name,
                            .position = {origin.x, origin.y, origin.z},
                            .right = {basis[0].x, basis[0].y, basis[0].z},
                            .down = {basis[1].x, basis[1].y, basis[1].z},
                            .forward = {basis[2].x, basis[2].y, basis[2].z},
                            .projection = plugin::CameraProjection::Pinhole,
                            .vertical_fov_degrees = 45.0f,
                            .image_width = frame.width,
                            .image_height = frame.height,
                            .fx = frame.focal_x,
                            .fy = frame.focal_y,
                            .cx = frame.principal_x,
                            .cy = frame.principal_y,
                            .near_plane = 0.01f,
                            .far_plane = 10.0f,
                            .image = plugin::CameraImage{
                                .rgba8 = frame.rgba.data(),
                                .rgba8_size = expected_bytes,
                                .revision = 1u,
                                .width = frame.width,
                                .height = frame.height,
                            },
                        });
                        ++selected_index;
                    }
                }
            }
        }, created->dataset);

        refresh_debug_attachments(*created);
        return Project{std::move(created)};
    }

    void Project::update(const plugin::UpdateInfo& update) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        if (!std::isfinite(update.wall_delta_seconds) || update.wall_delta_seconds < 0.0f) throw std::runtime_error("project update wall delta time is invalid");
        if (!std::isfinite(update.update_delta_seconds) || update.update_delta_seconds < 0.0f) throw std::runtime_error("project update delta time is invalid");
        if (!std::isfinite(update.timeline_time_seconds) || update.timeline_time_seconds < 0.0f) throw std::runtime_error("project timeline time is invalid");
        if (update.timeline_frame_index != 0u) throw std::runtime_error("project static timeline frame index must remain zero");
        state.host_update_running = update.update_running;
        if (update.update_delta_seconds == 0.0) return;
        try {
            optimize_training_batch(state);
        } catch (const std::exception& error) {
            set_project_error(state, error.what());
        }
    }

    void Project::render_preview(plugin::ActionContext context) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        if (state.host_update_running) throw std::runtime_error("pause the host update clock before rendering a preview");
        std::string frame_set{"train"};
        std::uint32_t image_index{};
        std::set<std::string> seen_options{};
        for (const plugin::Option& option : context.options) {
            if (!seen_options.insert(option.key).second) throw std::runtime_error(std::format("project action option '{}' is duplicated", option.key));
            if (option.key == preview_option_frame_set_key) frame_set = option.value;
            else if (option.key == preview_option_image_index_key) {
                const std::uint64_t value_u64 = parse_u64(option.value, preview_option_image_index_key);
                if (value_u64 > static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max())) throw std::runtime_error("image_index must fit in uint32");
                image_index = static_cast<std::uint32_t>(value_u64);
            } else {
                throw std::runtime_error(std::format("unknown project action option '{}'", option.key));
            }
        }
        if (frame_set != "train" && frame_set != "validation" && frame_set != "test") throw std::runtime_error(std::format("frame_set must be train, validation, or test; got '{}'", frame_set));
        if (!frame_set_loaded(state, frame_set)) throw std::runtime_error(std::format("frame_set '{}' was not loaded by Open Dataset", frame_set));
        create_trainer_if_needed(state);
        const inspector::Inspector inspector{*state.ngp};
        std::expected<inspector::EvaluationPreviewResult, std::string> preview = inspector.evaluate_preview(inspector::EvaluationPreviewRequest{
            .frame_set = frame_set,
            .image_index = image_index,
        });
        if (!preview) throw std::runtime_error(preview.error());
        const std::uint64_t revision = state.next_preview_revision++;
        PreviewState next_preview{
            .frame_set = preview->frame_set,
            .image_index = preview->image_index,
            .step = preview->step,
            .mse = preview->mse,
            .psnr = preview->psnr,
            .revision = revision,
        };
        state.latest_preview = std::move(next_preview);
        state.project_error.clear();
    }

    void Project::set_update_occupancy_grid(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.training.update_occupancy_grid != value;
        if (!changed) return;
        if (state.ngp != nullptr) {
            const std::expected<void, std::string> updated = state.ngp->update_training_state(train::TrainingStateRequest{
                .occupancy_grid_update_active = value,
            });
            if (!updated) throw std::runtime_error(updated.error());
        }
        state.occupancy_buffer.reset();
        state.exported_occupancy_revision = 0u;
        state.occupancy_grid.reset();
        state.training.update_occupancy_grid = value;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_show_volume(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.debug.show_volume != value;
        state.debug.show_volume = value;
        if (!changed) return;
        if (state.debug.show_volume && state.ngp != nullptr) {
            const inspector::Inspector inspector{*state.ngp};
            const inspector::DensityGridDeviceView density_view = inspector.density_grid_device_view();
            if (density_view.initialized && density_view.revision != state.exported_density_revision) publish_density_grid_volume(state);
        }
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_show_occupancy(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.debug.show_occupancy != value;
        state.debug.show_occupancy = value;
        if (!state.debug.show_occupancy) {
            state.occupancy_buffer.reset();
            state.exported_occupancy_revision = 0u;
            state.occupancy_grid.reset();
        }
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_occupancy_alpha(const float value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (value < 0.0f || value > 1.0f) throw std::runtime_error("occupancy_alpha must be in [0, 1]");
        State& state = *this->state;
        const bool changed = state.debug.occupancy_alpha != value;
        state.debug.occupancy_alpha = value;
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_occupancy_cell_scale(const float value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (!(value > 0.0f) || value > 1.0f) throw std::runtime_error("occupancy_cell_scale must be in (0, 1]");
        State& state = *this->state;
        const bool changed = state.debug.occupancy_cell_scale != value;
        state.debug.occupancy_cell_scale = value;
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_show_sampler(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.debug.show_sampler != value;
        state.debug.show_sampler = value;
        if (!state.debug.show_sampler) reset_sampler_visualization(state, true);
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_show_sampler_points(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.debug.show_sampler_points != value;
        state.debug.show_sampler_points = value;
        if (!state.debug.show_sampler_points) {
            state.sampler_points_buffer.reset();
            state.sampler_point_cloud.reset();
            state.exported_sampler_revision = 0u;
        }
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_show_sampler_rays(const bool value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        State& state = *this->state;
        const bool changed = state.debug.show_sampler_rays != value;
        state.debug.show_sampler_rays = value;
        if (!state.debug.show_sampler_rays) {
            state.sampler_segments_buffer.reset();
            state.sampler_ray_segments.reset();
            state.exported_sampler_revision = 0u;
        }
        if (!changed) return;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_sampler_point_radius(const float value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error("sampler_point_radius must be finite and positive");
        State& state = *this->state;
        const bool changed = state.debug.sampler_point_radius != value;
        state.debug.sampler_point_radius = value;
        if (!changed) return;
        state.exported_sampler_revision = 0u;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    void Project::set_sampler_ray_width(const float value) {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (!std::isfinite(value) || value <= 0.0f) throw std::runtime_error("sampler_ray_width must be finite and positive");
        State& state = *this->state;
        const bool changed = state.debug.sampler_ray_width != value;
        state.debug.sampler_ray_width = value;
        if (!changed) return;
        state.exported_sampler_revision = 0u;
        refresh_debug_attachments(state);
        ++state.scene_revision;
        state.project_error.clear();
    }

    std::uint64_t Project::revision() const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        return this->state->scene_revision;
    }

    void Project::write_controls(plugin::ControlBuilder& controls) const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        const State& state = *this->state;
        const std::uint32_t training_step = current_training_step(state);
        const bool target_reached = training_step >= state.training.target_steps;
        if (!state.project_error.empty()) {
            controls.phase("Error").headline("Project error").message(state.project_error);
        } else if (target_reached) {
            controls.phase("Complete").headline("Training complete").message(std::format("Reached target step {}.", state.training.target_steps));
        } else if (state.host_update_running) {
            controls.phase("Running").headline("Training running").message(std::format("Optimizing frame_set={} from Spectra update ticks.", state.training.frame_set));
        } else if (state.latest_stats.has_value()) {
            controls.phase("Paused").headline("Training paused").message(std::format("Current step {}.", training_step));
        } else {
            controls.phase("Paused").headline("Training paused").message("Training is paused at step 0.");
        }

        controls.metric("dataset", "Dataset", state.dataset_options.dataset_path.filename().string()).section(section_training_id);
        controls.metric("format", "Format", state.dataset_options.format).section(section_diagnostics_id);
        controls.metric("frame_sets", "Frame Sets", joined_frame_sets(state.dataset_options.frame_sets)).section(section_training_id);
        controls.metric("training_frame_set", "Training Set", state.training.frame_set).section(section_training_id);
        controls.metric("step", "Step", training_step).section(section_training_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        controls.metric("target_steps", "Target", state.training.target_steps).section(section_training_id);
        controls.metric("steps_per_update", "Steps/Update", state.training.steps_per_update).section(section_training_id);
        const std::string occupancy_update_mode = state.training.update_occupancy_grid ? "active" : state.ngp != nullptr && state.ngp->host.occupancy_grid_from_density ? "frozen" : "full";
        controls.metric("occupancy_update_mode", "Occupancy Updates", occupancy_update_mode).section(section_training_id);
        const bool volume_visible = density_volume_visible(state);
        controls.metric("density_visible", "Density", volume_visible ? "visible" : "hidden").section(section_field_id);
        controls.metric("density_grid_revision", "Density Rev", state.exported_density_revision).section(section_diagnostics_id);
        controls.metric("color_grid_revision", "Color Rev", state.exported_color_revision).section(section_diagnostics_id);
        controls.metric("density_grid_encoding", "Density Grid Encoding", state.density_volume.has_value() ? "Morton3D Float32" : "None").section(section_diagnostics_id);
        controls.metric("color_grid_encoding", "Color Grid Encoding", state.density_volume.has_value() ? "Morton3D Float32x3" : "None").section(section_diagnostics_id);
        if (state.density_volume.has_value()) controls.metric("density_grid_dimensions", "Density Grid Dimensions", std::format("{}x{}x{}", state.exported_density_dimensions[0], state.exported_density_dimensions[1], state.exported_density_dimensions[2])).section(section_diagnostics_id);
        if (state.exported_density_optical_thickness_step > 0.0f) controls.metric("optical_thickness_step", "Optical Thickness Step", std::format("{:.8g}", state.exported_density_optical_thickness_step)).section(section_diagnostics_id);
        if (state.exported_volume_density_scale > 0.0f) controls.metric("volume_density_scale", "Volume Density Scale", std::format("{:.8g}", state.exported_volume_density_scale)).section(section_diagnostics_id);
        controls.metric("occupancy_visible", "Occupancy", volume_visible && state.occupancy_grid.has_value() ? "visible" : "hidden").section(section_field_id);
        controls.metric("sampler_points_visible", "Points", state.debug.show_sampler_points && state.sampler_point_cloud.has_value() ? "visible" : "hidden").section(section_sampler_id);
        controls.metric("sampler_rays_visible", "Rays", state.sampler_ray_segments.has_value() ? "visible" : "hidden").section(section_sampler_id);
        controls.metric("sampler_revision", "Sampler Rev", state.exported_sampler_revision).section(section_sampler_id);
        controls.metric("sampler_point_count", "Sampler Points", state.exported_sampler_point_count).section(section_sampler_id);
        controls.metric("sampler_ray_count", "Sampler Rays", state.exported_sampler_ray_count).section(section_sampler_id);
        if (state.latest_stats.has_value()) {
            controls.metric("loss", "Loss", std::format("{:.6f}", state.latest_stats->loss)).section(section_training_id).display_primary().color({1.0f, 0.38f, 0.25f, 1.0f});
            controls.metric("sample_efficiency", "Sample Eff", std::format("{:.2f}%", state.latest_stats->sample_efficiency_ratio * 100.0f)).section(section_training_id).display_primary().color({0.25f, 0.75f, 1.0f, 1.0f});
            controls.metric("occupancy_grid_ratio", "Occupancy Grid", std::format("{:.2f}%", state.latest_stats->occupancy_grid_ratio * 100.0f)).section(section_training_id).display_primary().color({0.16f, 0.86f, 0.55f, 1.0f});
            controls.metric("occupancy_revision", "Occupancy Revision", state.exported_occupancy_revision).section(section_diagnostics_id);
        }
        if (state.latest_preview.has_value()) {
            controls.metric("preview_frame_set", "Preview Frame Set", state.latest_preview->frame_set).section(section_preview_id);
            controls.metric("preview_image", "Preview Image", state.latest_preview->image_index).section(section_preview_id);
            controls.metric("preview_step", "Preview Step", state.latest_preview->step).section(section_preview_id);
            controls.metric("preview_mse", "Preview MSE", std::format("{:.8f}", state.latest_preview->mse)).section(section_preview_id);
            controls.metric("preview_psnr", "Preview PSNR", std::isfinite(state.latest_preview->psnr) ? std::format("{:.2f} dB", state.latest_preview->psnr) : "inf").section(section_preview_id).display_primary().color({0.55f, 0.85f, 1.0f, 1.0f});
        }

        if (!state.project_error.empty()) {
            controls.disable(action_render_preview_id, "Close and reopen the dataset before rendering a preview.");
        } else if (state.host_update_running) {
            controls.disable(action_render_preview_id, "Pause the host update clock before rendering a preview.");
        } else {
            controls.enable(action_render_preview_id);
        }
    }

    void Project::write_document(plugin::SceneBuilder& scene) const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        const bool volume_visible = density_volume_visible(*this->state);
        std::vector<plugin::Material> materials = volume_visible ? this->state->materials : std::vector<plugin::Material>{};
        if (this->state->debug.show_sampler && this->state->debug.show_sampler_points) {
            materials.push_back(plugin::Material{
                .name = sampler_material_name,
                .model = "point_sprite",
                .alpha_mode = "blend",
                .base_color = {1.0f, 0.58f, 0.16f, 1.0f},
                .roughness = 0.2f,
            });
        }
        scene.set_document(plugin::Document{
            .timeline = plugin::TimelineDescriptor{
                .kind = plugin::TimelineKind::Static,
            },
            .update = plugin::UpdateDescriptor{
                .enabled = true,
                .initial_running = false,
                .step_delta_seconds = 1.0 / 60.0,
            },
            .active_camera_name = this->state->overview_camera_name,
            .materials = std::move(materials),
            .lights = volume_visible ? this->state->lights : std::vector<plugin::Light>{},
        });
    }

    void Project::write_frame(plugin::SceneBuilder& scene, const plugin::FrameInfo frame) const {
        if (this->state == nullptr) throw std::runtime_error("project is not open");
        if (!std::isfinite(frame.delta_seconds) || frame.delta_seconds < 0.0) throw std::runtime_error("project frame delta time is invalid");
        if (!std::isfinite(frame.time_seconds) || frame.time_seconds < 0.0) throw std::runtime_error("project frame time is invalid");
        if (frame.frame_index != 0u) throw std::runtime_error("project static timeline frame index must remain zero");
        const bool volume_visible = density_volume_visible(*this->state);
        std::vector<plugin::PointCloud> point_clouds{};
        if (this->state->debug.show_sampler_points && this->state->sampler_point_cloud.has_value()) point_clouds.push_back(*this->state->sampler_point_cloud);
        plugin::DebugAttachmentSet debug_attachments = this->state->debug_attachments;
        if (!volume_visible) debug_attachments.viewport_voxel_grids.clear();
        scene.set_document(plugin::Document{
            .cameras = this->state->cameras,
            .point_clouds = std::move(point_clouds),
            .volumes = volume_visible ? std::vector<plugin::VolumeGrid>{*this->state->density_volume} : std::vector<plugin::VolumeGrid>{},
            .debug_attachments = std::move(debug_attachments),
        });
    }

}

extern "C" SPECTRA_SCENE_EXPORT auto spectra_scene_plugin_v16(void) -> decltype(ngp::plugin::export_plugin<ngp::project::Project>()) {
    return ngp::plugin::export_plugin<ngp::project::Project>();
}
