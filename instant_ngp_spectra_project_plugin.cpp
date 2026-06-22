#if defined(_WIN32)
#define SPECTRA_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_SCENE_EXPORT __attribute__((visibility("default")))
#endif

import instant_ngp.spectra_project;
import std;

constexpr std::uint32_t plugin_abi_version = 1u;
typedef void SpectraSceneInstance;

typedef std::uint32_t SpectraSceneResult;
constexpr std::uint32_t SPECTRA_SCENE_RESULT_OK = 0u;
constexpr std::uint32_t SPECTRA_SCENE_RESULT_ERROR = 1u;
constexpr std::uint32_t SPECTRA_SCENE_GPU_BUFFER_VOLUME_CHANNEL = 0u;
constexpr std::uint32_t SPECTRA_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID = 1u;

struct SpectraSceneOption {
    const char* key{};
    const char* value{};
};

struct SpectraSceneOptionSpan {
    const SpectraSceneOption* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlOptionChoice {
    const char* value{};
    const char* label{};
};

struct SpectraSceneControlOptionChoiceSpan {
    const SpectraSceneControlOptionChoice* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlOptionSchema {
    const char* key{};
    const char* label{};
    const char* description{};
    std::uint32_t kind{};
    std::uint32_t required{};
    const char* default_value{};
    const char* group{};
    std::uint32_t advanced{};
    std::int32_t priority{};
    SpectraSceneControlOptionChoiceSpan choices{};
};

struct SpectraSceneControlOptionSchemaSpan {
    const SpectraSceneControlOptionSchema* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlAction {
    const char* id{};
    const char* label{};
    const char* description{};
    std::uint32_t group{};
    std::int32_t priority{};
    std::uint32_t style{};
    SpectraSceneControlOptionSchemaSpan options{};
};

struct SpectraSceneControlActionSpan {
    const SpectraSceneControlAction* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlSettingValue {
    const char* key{};
    const char* value{};
};

struct SpectraSceneControlSettingValueSpan {
    const SpectraSceneControlSettingValue* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlMetric {
    const char* key{};
    const char* label{};
    const char* value{};
    std::uint32_t placement_flags{};
    std::int32_t priority{};
    std::uint32_t has_color{};
    float color[4]{};
};

struct SpectraSceneControlMetricSpan {
    const SpectraSceneControlMetric* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlActionState {
    const char* action_id{};
    std::uint32_t enabled{};
    const char* disabled_reason{};
};

struct SpectraSceneControlActionStateSpan {
    const SpectraSceneControlActionState* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlStatusView {
    std::uint64_t struct_size{};
    const char* phase{};
    const char* headline{};
    const char* detail{};
    SpectraSceneControlMetricSpan metrics{};
    SpectraSceneControlActionStateSpan action_states{};
};

struct SpectraSceneControlLogEntry {
    std::uint64_t sequence{};
    const char* level{};
    const char* message{};
};

struct SpectraSceneControlLogEntrySpan {
    const SpectraSceneControlLogEntry* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlImage {
    const char* id{};
    const char* label{};
    const char* description{};
    const std::uint8_t* rgba8{};
    std::uint64_t rgba8_size{};
    std::uint64_t revision{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct SpectraSceneControlImageSpan {
    const SpectraSceneControlImage* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlScalarSample {
    std::uint64_t step{};
    double time_seconds{};
    double value{};
};

struct SpectraSceneControlScalarSampleSpan {
    const SpectraSceneControlScalarSample* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlScalarSeries {
    const char* id{};
    const char* label{};
    const char* description{};
    const char* unit{};
    float color[4]{};
    std::uint32_t group{};
    std::int32_t priority{};
    std::uint64_t revision{};
    SpectraSceneControlScalarSampleSpan samples{};
};

struct SpectraSceneControlScalarSeriesSpan {
    const SpectraSceneControlScalarSeries* data{};
    std::uint64_t count{};
};

struct SpectraSceneControlSnapshotView {
    std::uint64_t struct_size{};
    SpectraSceneControlSettingValueSpan settings{};
    SpectraSceneControlStatusView status{};
    SpectraSceneControlLogEntrySpan logs{};
    SpectraSceneControlImageSpan images{};
    SpectraSceneControlScalarSeriesSpan scalar_series{};
};

struct SpectraSceneUpdateInfo {
    std::uint64_t struct_size{};
    double wall_delta_seconds{};
    double scene_delta_seconds{};
    double time_seconds{};
    std::uint64_t frame_index{};
    std::uint32_t timeline_mode{};
    std::uint32_t timeline_playing{};
};

struct SpectraSceneGpuDeviceIdentity {
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint8_t device_uuid[16]{};
    std::uint8_t device_luid[8]{};
    std::uint32_t device_node_mask{};
};

struct SpectraSceneGpuBufferRequest {
    std::uint64_t struct_size{};
    std::uint32_t kind{};
    std::uint64_t byte_size{};
    const char* debug_name{};
};

struct SpectraSceneGpuBufferAllocation {
    std::uint64_t struct_size{};
    std::uint64_t resource_id{};
    std::uint64_t byte_size{};
    std::uint32_t kind{};
    std::uint32_t handle_kind{};
    std::uintptr_t handle{};
    SpectraSceneGpuDeviceIdentity device_identity{};
};

typedef SpectraSceneResult (*SpectraSceneRequestGpuBufferFn)(void* user_data, const SpectraSceneGpuBufferRequest* request, SpectraSceneGpuBufferAllocation* allocation);
typedef SpectraSceneResult (*SpectraSceneReleaseGpuBufferFn)(void* user_data, std::uint64_t resource_id);
typedef const char* (*SpectraSceneHostLastErrorFn)(void* user_data);

struct SpectraSceneHostServices {
    std::uint64_t struct_size{};
    void* user_data{};
    SpectraSceneRequestGpuBufferFn request_gpu_buffer{};
    SpectraSceneReleaseGpuBufferFn release_gpu_buffer{};
    SpectraSceneHostLastErrorFn last_error{};
};

struct SpectraSceneOpenInfo {
    std::uint64_t struct_size{};
    const char* plugin_path{};
    SpectraSceneOptionSpan options{};
    const SpectraSceneHostServices* host_services{};
};

struct SpectraSceneTransform {
    float position[3]{};
    float rotation[4]{};
    float scale[3]{};
};

struct SpectraSceneMaterial {
    const char* name{};
    const char* model{};
    const char* alpha_mode{};
    float base_color[4]{};
    float emission_color[3]{};
    float emission_strength{};
    float roughness{};
    float metallic{};
    float alpha_cutoff{};
    float volume_density_scale{};
    float volume_temperature_scale{};
};

struct SpectraSceneMaterialSpan {
    const SpectraSceneMaterial* data{};
    std::uint64_t count{};
};

struct SpectraSceneLight {
    const char* name{};
    const char* kind{};
    SpectraSceneTransform transform{};
    float color[3]{};
    float intensity{};
    float cone_angle_degrees{};
};

struct SpectraSceneLightSpan {
    const SpectraSceneLight* data{};
    std::uint64_t count{};
};

struct SpectraSceneCamera {
    const char* name{};
    const char* local_coordinate_system{};
    SpectraSceneTransform transform{};
    float target[3]{};
    float up[3]{};
    std::uint32_t projection{};
    float vertical_fov_degrees{};
    std::uint32_t image_width{};
    std::uint32_t image_height{};
    float fx{};
    float fy{};
    float cx{};
    float cy{};
    float near_plane{};
    float far_plane{};
};

struct SpectraSceneCameraSpan {
    const SpectraSceneCamera* data{};
    std::uint64_t count{};
};

struct SpectraSceneMeshVertex {
    float position[3]{};
    float normal[3]{};
};

struct SpectraSceneMeshVertexSpan {
    const SpectraSceneMeshVertex* data{};
    std::uint64_t count{};
};

struct SpectraSceneUInt32Span {
    const std::uint32_t* data{};
    std::uint64_t count{};
};

struct SpectraSceneMesh {
    const char* name{};
    SpectraSceneMeshVertexSpan vertices{};
    SpectraSceneUInt32Span indices{};
    const char* material_name{};
    SpectraSceneTransform transform{};
};

struct SpectraSceneMeshSpan {
    const SpectraSceneMesh* data{};
    std::uint64_t count{};
};

struct SpectraSceneSphere {
    const char* name{};
    float radius{};
    const char* material_name{};
    SpectraSceneTransform transform{};
};

struct SpectraSceneSphereSpan {
    const SpectraSceneSphere* data{};
    std::uint64_t count{};
};

struct SpectraScenePoint {
    float position[3]{};
    float normal[3]{};
    float color[4]{};
    float radius{};
};

struct SpectraScenePointSpan {
    const SpectraScenePoint* data{};
    std::uint64_t count{};
};

struct SpectraScenePointCloud {
    const char* name{};
    SpectraScenePointSpan points{};
    const char* material_name{};
    SpectraSceneTransform transform{};
};

struct SpectraScenePointCloudSpan {
    const SpectraScenePointCloud* data{};
    std::uint64_t count{};
};

struct SpectraSceneFloatSpan {
    const float* data{};
    std::uint64_t count{};
};

struct SpectraSceneVolumeChannel {
    const char* name{};
    std::uint32_t dimensions[3]{};
    SpectraSceneFloatSpan values{};
    std::uint32_t format{};
    std::uint32_t source_kind{};
    std::uint32_t index_encoding{};
    std::uint64_t buffer_id{};
    std::uintptr_t external_device_pointer{};
    std::uint64_t source_byte_size{};
    std::uint64_t revision{};
};

struct SpectraSceneVolumeChannelSpan {
    const SpectraSceneVolumeChannel* data{};
    std::uint64_t count{};
};

struct SpectraSceneVolume {
    const char* name{};
    std::uint32_t dimensions[3]{};
    float origin[3]{};
    float voxel_size[3]{};
    SpectraSceneVolumeChannelSpan channels{};
    const char* material_name{};
};

struct SpectraSceneVolumeSpan {
    const SpectraSceneVolume* data{};
    std::uint64_t count{};
};

struct SpectraSceneEntityRef {
    std::uint32_t kind{};
    const char* name{};
};

struct SpectraSceneViewportSegment {
    float start[3]{};
    float end[3]{};
};

struct SpectraSceneViewportSegmentSpan {
    const SpectraSceneViewportSegment* data{};
    std::uint64_t count{};
};

struct SpectraSceneColor {
    float value[4]{};
};

struct SpectraSceneColorSpan {
    const SpectraSceneColor* data{};
    std::uint64_t count{};
};

struct SpectraSceneViewportSegmentSet {
    const char* name{};
    SpectraSceneEntityRef owner{};
    SpectraSceneViewportSegmentSpan segments{};
    SpectraSceneColorSpan colors{};
    SpectraSceneFloatSpan widths{};
    float width{};
    std::uint32_t width_mode{};
    std::uint32_t depth_mode{};
    SpectraSceneTransform transform{};
};

struct SpectraSceneViewportSegmentSetSpan {
    const SpectraSceneViewportSegmentSet* data{};
    std::uint64_t count{};
};

struct SpectraSceneViewportVoxelGrid {
    const char* name{};
    SpectraSceneEntityRef owner{};
    std::uint32_t dimensions[3]{};
    float origin[3]{};
    float voxel_size[3]{};
    SpectraSceneTransform transform{};
    float color[4]{};
    float cell_scale{};
    std::uint32_t depth_mode{};
    std::uint32_t source_kind{};
    std::uint32_t index_encoding{};
    std::uint64_t buffer_id{};
    std::uint64_t source_byte_size{};
    std::uint64_t index_count{};
    std::uint64_t revision{};
};

struct SpectraSceneViewportVoxelGridSpan {
    const SpectraSceneViewportVoxelGrid* data{};
    std::uint64_t count{};
};

struct SpectraSceneViewportCameraVisualImage {
    const std::uint8_t* rgba8{};
    std::uint64_t rgba8_size{};
    std::uint64_t revision{};
    std::uint32_t width{};
    std::uint32_t height{};
    float tint[4]{};
};

struct SpectraSceneViewportCameraVisual {
    const char* name{};
    SpectraSceneEntityRef owner{};
    float color[4]{};
    float width{};
    std::uint32_t width_mode{};
    std::uint32_t depth_mode{};
    float visual_near{};
    float visual_far{};
    std::uint32_t has_image{};
    SpectraSceneViewportCameraVisualImage image{};
};

struct SpectraSceneViewportCameraVisualSpan {
    const SpectraSceneViewportCameraVisual* data{};
    std::uint64_t count{};
};

struct SpectraSceneItems {
    SpectraSceneMaterialSpan materials{};
    SpectraSceneLightSpan lights{};
    SpectraSceneCameraSpan cameras{};
    SpectraSceneMeshSpan meshes{};
    SpectraSceneSphereSpan spheres{};
    SpectraScenePointCloudSpan point_clouds{};
    SpectraSceneVolumeSpan volumes{};
    SpectraSceneViewportSegmentSetSpan viewport_segment_sets{};
    SpectraSceneViewportVoxelGridSpan viewport_voxel_grids{};
    SpectraSceneViewportCameraVisualSpan viewport_camera_visuals{};
};

struct SpectraSceneDocumentView {
    std::uint64_t struct_size{};
    const char* default_coordinate_system{};
    const char* active_camera_name{};
    SpectraSceneItems items{};
};

struct SpectraSceneFrameInfo {
    double delta_seconds{};
    double time_seconds{};
    std::uint64_t frame_index{};
};

struct SpectraSceneFrameView {
    std::uint64_t struct_size{};
    SpectraSceneItems items{};
};

typedef SpectraSceneResult (*SpectraSceneCreateFn)(const SpectraSceneOpenInfo* open_info, SpectraSceneInstance** instance);
typedef void (*SpectraSceneDestroyFn)(SpectraSceneInstance* instance);
typedef SpectraSceneResult (*SpectraSceneResetFn)(SpectraSceneInstance* instance);
typedef SpectraSceneResult (*SpectraSceneUpdateFn)(SpectraSceneInstance* instance, const SpectraSceneUpdateInfo* update_info);
typedef SpectraSceneResult (*SpectraSceneDocumentFn)(SpectraSceneInstance* instance, SpectraSceneDocumentView* document);
typedef SpectraSceneResult (*SpectraSceneFrameFn)(SpectraSceneInstance* instance, SpectraSceneFrameInfo frame, SpectraSceneFrameView* snapshot);
typedef SpectraSceneResult (*SpectraSceneRevisionFn)(SpectraSceneInstance* instance, std::uint64_t* revision);
typedef SpectraSceneResult (*SpectraSceneControlActionFn)(SpectraSceneInstance* instance, const char* action_id, SpectraSceneOptionSpan options);
typedef SpectraSceneResult (*SpectraSceneControlSettingUpdateFn)(SpectraSceneInstance* instance, const char* key, const char* value);
typedef SpectraSceneResult (*SpectraSceneControlSnapshotFn)(SpectraSceneInstance* instance, SpectraSceneControlSnapshotView* snapshot);
typedef const char* (*SpectraSceneLastErrorFn)(SpectraSceneInstance* instance);

struct SpectraScenePlugin {
    std::uint32_t abi_version{};
    std::uint64_t struct_size{};
    const char* id{};
    const char* title{};
    const char* controls_panel_title{};
    const char* open_action_label{};
    const char* open_action_description{};
    const char* base_pbrt_path{};
    double frames_per_second{};
    SpectraSceneControlOptionSchemaSpan open_options{};
    SpectraSceneControlActionSpan control_actions{};
    SpectraSceneControlOptionSchemaSpan control_settings{};
    SpectraSceneCreateFn create{};
    SpectraSceneDestroyFn destroy{};
    SpectraSceneResetFn reset{};
    SpectraSceneUpdateFn update{};
    SpectraSceneDocumentFn document{};
    SpectraSceneFrameFn frame{};
    SpectraSceneRevisionFn scene_revision{};
    SpectraSceneControlActionFn control_action{};
    SpectraSceneControlSettingUpdateFn control_setting_update{};
    SpectraSceneControlSnapshotFn control_snapshot{};
    SpectraSceneLastErrorFn last_error{};
};

namespace {
    struct OptionSchemaViews {
        std::vector<std::vector<SpectraSceneControlOptionChoice>> choices{};
        std::vector<SpectraSceneControlOptionSchema> schemas{};
    };

    struct DescriptorViews {
        OptionSchemaViews open_options{};
        std::vector<OptionSchemaViews> action_options{};
        std::vector<SpectraSceneControlAction> control_actions{};
        OptionSchemaViews control_settings{};
    };

    struct SceneViewCache {
        instant_ngp::spectra_project::Document document{};
        std::vector<SpectraSceneMaterial> material_views{};
        std::vector<SpectraSceneLight> light_views{};
        std::vector<std::vector<SpectraSceneVolumeChannel>> volume_channel_storage{};
        std::vector<SpectraSceneVolume> volume_views{};
        std::vector<SpectraSceneCamera> camera_views{};
        std::vector<SpectraSceneViewportVoxelGrid> voxel_grid_views{};
        std::vector<SpectraSceneViewportCameraVisual> camera_visual_views{};
    };

    struct ProjectStatusCache {
        instant_ngp::spectra_project::ProjectStatus status{};
        std::vector<SpectraSceneControlMetric> metric_views{};
        std::vector<SpectraSceneControlActionState> action_state_views{};
    };

    struct ProjectSettingCache {
        std::vector<instant_ngp::spectra_project::ProjectSettingValue> settings{};
        std::vector<SpectraSceneControlSettingValue> setting_views{};
    };

    struct ProjectLogCache {
        std::vector<instant_ngp::spectra_project::ProjectLogEntry> logs{};
        std::vector<SpectraSceneControlLogEntry> log_views{};
    };

    struct ProjectImageCache {
        std::span<const instant_ngp::spectra_project::ProjectImage> images{};
        std::vector<SpectraSceneControlImage> image_views{};
    };

    struct ProjectScalarSeriesCache {
        std::vector<instant_ngp::spectra_project::ProjectScalarSeries> series{};
        std::vector<std::vector<SpectraSceneControlScalarSample>> sample_views{};
        std::vector<SpectraSceneControlScalarSeries> series_views{};
    };

    struct PluginInstance {
        instant_ngp::spectra_project::InstantNgpSpectraProject project{};
        std::string last_error{};
        SceneViewCache scene_cache{};
        ProjectStatusCache status_cache{};
        ProjectSettingCache setting_cache{};
        ProjectLogCache log_cache{};
        ProjectImageCache image_cache{};
        ProjectScalarSeriesCache scalar_series_cache{};
    };

    std::string global_error{};

    [[nodiscard]] std::string string_from_abi(const char* value, const std::string_view context, const bool allow_empty) {
        const std::string_view view = value == nullptr ? std::string_view{} : std::string_view{value};
        if (!allow_empty && view.empty()) throw std::runtime_error(std::format("{} must not be empty", context));
        return std::string{view};
    }

    [[nodiscard]] std::vector<instant_ngp::spectra_project::Option> options_from_abi(const SpectraSceneOptionSpan options, const std::string_view context) {
        if (options.count != 0u && options.data == nullptr) throw std::runtime_error(std::format("{} pointer is null", context));
        if (options.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::format("{} count is too large", context));
        std::vector<instant_ngp::spectra_project::Option> converted{};
        converted.reserve(static_cast<std::size_t>(options.count));
        const std::span<const SpectraSceneOption> option_span{options.data, static_cast<std::size_t>(options.count)};
        for (const SpectraSceneOption& option : option_span) {
            converted.push_back(instant_ngp::spectra_project::Option{
                .key = string_from_abi(option.key, std::format("{} key", context), false),
                .value = string_from_abi(option.value, std::format("{} value", context), true),
            });
        }
        return converted;
    }

    [[nodiscard]] std::string host_services_error(const SpectraSceneHostServices& host_services) {
        if (host_services.last_error == nullptr) return "unknown host service error";
        std::string message = string_from_abi(host_services.last_error(host_services.user_data), "scene plugin host services error", true);
        if (message.empty()) message = "unknown host service error";
        return message;
    }

    [[nodiscard]] instant_ngp::spectra_project::GpuResourceHandleKind gpu_handle_kind_from_abi(const std::uint32_t kind) {
        switch (kind) {
            case 1u: return instant_ngp::spectra_project::GpuResourceHandleKind::OpaqueWin32;
            case 2u: return instant_ngp::spectra_project::GpuResourceHandleKind::OpaqueFileDescriptor;
            default: throw std::runtime_error(std::format("unknown scene plugin GPU resource handle kind {}", kind));
        }
    }

    [[nodiscard]] instant_ngp::spectra_project::GpuDeviceIdentity device_identity_from_abi(const SpectraSceneGpuDeviceIdentity& identity) {
        instant_ngp::spectra_project::GpuDeviceIdentity converted{
            .vendor_id = identity.vendor_id,
            .device_id = identity.device_id,
            .device_node_mask = identity.device_node_mask,
        };
        for (std::size_t index = 0u; index < converted.device_uuid.size(); ++index) converted.device_uuid[index] = identity.device_uuid[index];
        for (std::size_t index = 0u; index < converted.device_luid.size(); ++index) converted.device_luid[index] = identity.device_luid[index];
        return converted;
    }

    [[nodiscard]] OptionSchemaViews make_option_schema_views(const std::vector<instant_ngp::spectra_project::OptionSchema>& schemas) {
        OptionSchemaViews views{};
        views.choices.resize(schemas.size());
        views.schemas.reserve(schemas.size());
        for (std::size_t index = 0u; index < schemas.size(); ++index) {
            const instant_ngp::spectra_project::OptionSchema& schema = schemas[index];
            views.choices[index].reserve(schema.choices.size());
            for (const instant_ngp::spectra_project::OptionChoice& choice : schema.choices) views.choices[index].push_back(SpectraSceneControlOptionChoice{.value = choice.value.c_str(), .label = choice.label.c_str()});
            views.schemas.push_back(SpectraSceneControlOptionSchema{
                .key = schema.key.c_str(),
                .label = schema.label.c_str(),
                .description = schema.description.c_str(),
                .kind = static_cast<std::uint32_t>(schema.kind),
                .required = schema.required ? 1u : 0u,
                .default_value = schema.default_value.c_str(),
                .group = schema.group.c_str(),
                .advanced = schema.advanced ? 1u : 0u,
                .priority = schema.priority,
                .choices = SpectraSceneControlOptionChoiceSpan{.data = views.choices[index].empty() ? nullptr : views.choices[index].data(), .count = static_cast<std::uint64_t>(views.choices[index].size())},
            });
        }
        return views;
    }

    [[nodiscard]] DescriptorViews make_descriptor_views() {
        const instant_ngp::spectra_project::Descriptor& descriptor = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor();
        DescriptorViews views{};
        views.open_options = make_option_schema_views(descriptor.open_options);
        views.action_options.reserve(descriptor.control_actions.size());
        views.control_actions.reserve(descriptor.control_actions.size());
        for (const instant_ngp::spectra_project::ProjectAction& action : descriptor.control_actions) {
            views.action_options.push_back(make_option_schema_views(action.options));
            const OptionSchemaViews& action_options = views.action_options.back();
            views.control_actions.push_back(SpectraSceneControlAction{
                .id = action.id.c_str(),
                .label = action.label.c_str(),
                .description = action.description.c_str(),
                .group = action.group,
                .priority = action.priority,
                .style = action.style,
                .options = SpectraSceneControlOptionSchemaSpan{.data = action_options.schemas.empty() ? nullptr : action_options.schemas.data(), .count = static_cast<std::uint64_t>(action_options.schemas.size())},
            });
        }
        views.control_settings = make_option_schema_views(descriptor.control_settings);
        return views;
    }

    [[nodiscard]] const DescriptorViews& descriptor_views() {
        static const DescriptorViews views = make_descriptor_views();
        return views;
    }

    template <std::size_t Count>
    void copy_array(float (&output)[Count], const std::array<float, Count>& input) {
        for (std::size_t index = 0u; index < Count; ++index) output[index] = input[index];
    }

    [[nodiscard]] SpectraSceneTransform make_transform_view(const instant_ngp::spectra_project::Transform& transform) {
        SpectraSceneTransform view{};
        copy_array(view.position, transform.position);
        copy_array(view.rotation, transform.rotation);
        copy_array(view.scale, transform.scale);
        return view;
    }

    [[nodiscard]] SpectraSceneEntityRef make_entity_ref_view(const instant_ngp::spectra_project::SceneEntityRef& ref) {
        return SpectraSceneEntityRef{
            .kind = static_cast<std::uint32_t>(ref.kind),
            .name = ref.name.c_str(),
        };
    }

    [[nodiscard]] SpectraSceneMaterial make_material_view(const instant_ngp::spectra_project::Material& material) {
        SpectraSceneMaterial view{
            .name = material.name.c_str(),
            .model = material.model.c_str(),
            .alpha_mode = material.alpha_mode.c_str(),
            .roughness = material.roughness,
            .alpha_cutoff = 0.5f,
            .volume_density_scale = material.volume_density_scale,
            .volume_temperature_scale = material.volume_temperature_scale,
        };
        copy_array(view.base_color, material.base_color);
        return view;
    }

    [[nodiscard]] SpectraSceneLight make_light_view(const instant_ngp::spectra_project::Light& light) {
        SpectraSceneLight view{
            .name = light.name.c_str(),
            .kind = light.kind.c_str(),
            .transform = make_transform_view(light.transform),
            .intensity = light.intensity,
            .cone_angle_degrees = 30.0f,
        };
        copy_array(view.color, light.color);
        return view;
    }

    void make_volume_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::VolumeGrid>& volumes) {
        cache.volume_channel_storage.clear();
        cache.volume_views.clear();
        cache.volume_channel_storage.resize(volumes.size());
        cache.volume_views.reserve(volumes.size());
        for (std::size_t volume_index = 0u; volume_index < volumes.size(); ++volume_index) {
            const instant_ngp::spectra_project::VolumeGrid& volume = volumes[volume_index];
            cache.volume_channel_storage[volume_index].reserve(volume.channels.size());
            for (const instant_ngp::spectra_project::VolumeChannel& channel : volume.channels) {
                SpectraSceneVolumeChannel channel_view{
                    .name = channel.name.c_str(),
                    .format = static_cast<std::uint32_t>(channel.format),
                    .source_kind = static_cast<std::uint32_t>(channel.source_kind),
                    .index_encoding = static_cast<std::uint32_t>(channel.index_encoding),
                    .buffer_id = channel.buffer_id,
                    .external_device_pointer = channel.external_device_pointer,
                    .source_byte_size = channel.source_byte_size,
                    .revision = channel.revision,
                };
                channel_view.dimensions[0] = channel.dimensions[0];
                channel_view.dimensions[1] = channel.dimensions[1];
                channel_view.dimensions[2] = channel.dimensions[2];
                cache.volume_channel_storage[volume_index].push_back(channel_view);
            }

            SpectraSceneVolume volume_view{
                .name = volume.name.c_str(),
                .channels = SpectraSceneVolumeChannelSpan{.data = cache.volume_channel_storage[volume_index].empty() ? nullptr : cache.volume_channel_storage[volume_index].data(), .count = static_cast<std::uint64_t>(cache.volume_channel_storage[volume_index].size())},
                .material_name = volume.material_name.c_str(),
            };
            volume_view.dimensions[0] = volume.dimensions[0];
            volume_view.dimensions[1] = volume.dimensions[1];
            volume_view.dimensions[2] = volume.dimensions[2];
            copy_array(volume_view.origin, volume.origin);
            copy_array(volume_view.voxel_size, volume.voxel_size);
            cache.volume_views.push_back(volume_view);
        }
    }

    [[nodiscard]] SpectraSceneCamera make_camera_view(const instant_ngp::spectra_project::Camera& camera) {
        SpectraSceneCamera view{
            .name = camera.name.c_str(),
            .local_coordinate_system = camera.local_coordinate_system.c_str(),
            .transform = make_transform_view(camera.transform),
            .projection = static_cast<std::uint32_t>(camera.projection),
            .vertical_fov_degrees = camera.vertical_fov_degrees,
            .image_width = camera.image_width,
            .image_height = camera.image_height,
            .fx = camera.fx,
            .fy = camera.fy,
            .cx = camera.cx,
            .cy = camera.cy,
            .near_plane = camera.near_plane,
            .far_plane = camera.far_plane,
        };
        copy_array(view.target, camera.target);
        copy_array(view.up, camera.up);
        return view;
    }

    [[nodiscard]] SpectraSceneViewportCameraVisual make_camera_visual_view(const instant_ngp::spectra_project::ViewportCameraVisual& visual) {
        SpectraSceneViewportCameraVisual view{
            .name = visual.name.c_str(),
            .owner = make_entity_ref_view(visual.owner),
            .width = visual.width,
            .width_mode = visual.width_mode,
            .depth_mode = visual.depth_mode,
            .visual_near = visual.visual_near,
            .visual_far = visual.visual_far,
            .has_image = visual.image.has_value() ? 1u : 0u,
        };
        copy_array(view.color, visual.color);
        if (visual.image.has_value()) {
            const instant_ngp::spectra_project::ViewportCameraVisualImage& image = *visual.image;
            view.image = SpectraSceneViewportCameraVisualImage{
                .rgba8 = image.rgba8,
                .rgba8_size = image.rgba8_size,
                .revision = image.revision,
                .width = image.width,
                .height = image.height,
            };
            copy_array(view.image.tint, image.tint);
        }
        return view;
    }

    [[nodiscard]] SpectraSceneViewportVoxelGrid make_voxel_grid_view(const instant_ngp::spectra_project::ViewportVoxelGrid& grid) {
        SpectraSceneViewportVoxelGrid view{
            .name = grid.name.c_str(),
            .owner = make_entity_ref_view(grid.owner),
            .transform = make_transform_view(grid.transform),
            .cell_scale = grid.cell_scale,
            .depth_mode = grid.depth_mode,
            .source_kind = static_cast<std::uint32_t>(grid.source_kind),
            .index_encoding = static_cast<std::uint32_t>(grid.index_encoding),
            .buffer_id = grid.buffer_id,
            .source_byte_size = grid.source_byte_size,
            .revision = grid.revision,
        };
        view.dimensions[0] = grid.dimensions[0];
        view.dimensions[1] = grid.dimensions[1];
        view.dimensions[2] = grid.dimensions[2];
        copy_array(view.origin, grid.origin);
        copy_array(view.voxel_size, grid.voxel_size);
        copy_array(view.color, grid.color);
        return view;
    }

    [[nodiscard]] SpectraSceneDocumentView make_document_view(SceneViewCache& cache) {
        cache.material_views.clear();
        cache.material_views.reserve(cache.document.materials.size());
        for (const instant_ngp::spectra_project::Material& material : cache.document.materials) cache.material_views.push_back(make_material_view(material));
        cache.light_views.clear();
        cache.light_views.reserve(cache.document.lights.size());
        for (const instant_ngp::spectra_project::Light& light : cache.document.lights) cache.light_views.push_back(make_light_view(light));
        make_volume_views(cache, cache.document.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.document.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.document.cameras) cache.camera_views.push_back(make_camera_view(camera));
        cache.voxel_grid_views.clear();
        cache.voxel_grid_views.reserve(cache.document.debug_attachments.viewport_voxel_grids.size());
        for (const instant_ngp::spectra_project::ViewportVoxelGrid& grid : cache.document.debug_attachments.viewport_voxel_grids) cache.voxel_grid_views.push_back(make_voxel_grid_view(grid));
        cache.camera_visual_views.clear();
        cache.camera_visual_views.reserve(cache.document.debug_attachments.viewport_camera_visuals.size());
        for (const instant_ngp::spectra_project::ViewportCameraVisual& visual : cache.document.debug_attachments.viewport_camera_visuals) cache.camera_visual_views.push_back(make_camera_visual_view(visual));
        return SpectraSceneDocumentView{
            .struct_size = sizeof(SpectraSceneDocumentView),
            .default_coordinate_system = cache.document.default_coordinate_system.c_str(),
            .active_camera_name = cache.document.active_camera_name.c_str(),
            .items = SpectraSceneItems{
                .materials = SpectraSceneMaterialSpan{.data = cache.material_views.empty() ? nullptr : cache.material_views.data(), .count = static_cast<std::uint64_t>(cache.material_views.size())},
                .lights = SpectraSceneLightSpan{.data = cache.light_views.empty() ? nullptr : cache.light_views.data(), .count = static_cast<std::uint64_t>(cache.light_views.size())},
                .cameras = SpectraSceneCameraSpan{.data = cache.camera_views.empty() ? nullptr : cache.camera_views.data(), .count = static_cast<std::uint64_t>(cache.camera_views.size())},
                .volumes = SpectraSceneVolumeSpan{.data = cache.volume_views.empty() ? nullptr : cache.volume_views.data(), .count = static_cast<std::uint64_t>(cache.volume_views.size())},
                .viewport_voxel_grids = SpectraSceneViewportVoxelGridSpan{.data = cache.voxel_grid_views.empty() ? nullptr : cache.voxel_grid_views.data(), .count = static_cast<std::uint64_t>(cache.voxel_grid_views.size())},
                .viewport_camera_visuals = SpectraSceneViewportCameraVisualSpan{.data = cache.camera_visual_views.empty() ? nullptr : cache.camera_visual_views.data(), .count = static_cast<std::uint64_t>(cache.camera_visual_views.size())},
            },
        };
    }

    [[nodiscard]] SpectraSceneControlStatusView make_status_view(ProjectStatusCache& cache) {
        cache.metric_views.clear();
        cache.action_state_views.clear();
        cache.metric_views.reserve(cache.status.metrics.size());
        cache.action_state_views.reserve(cache.status.action_states.size());
        for (const instant_ngp::spectra_project::ProjectMetric& metric : cache.status.metrics) {
            cache.metric_views.push_back(SpectraSceneControlMetric{
                .key = metric.key.c_str(),
                .label = metric.label.c_str(),
                .value = metric.value.c_str(),
                .placement_flags = metric.placement_flags,
                .priority = metric.priority,
                .has_color = metric.has_color ? 1u : 0u,
                .color = {},
            });
            copy_array(cache.metric_views.back().color, metric.color);
        }
        for (const instant_ngp::spectra_project::ProjectActionState& action_state : cache.status.action_states) {
            cache.action_state_views.push_back(SpectraSceneControlActionState{
                .action_id = action_state.action_id.c_str(),
                .enabled = action_state.enabled ? 1u : 0u,
                .disabled_reason = action_state.disabled_reason.c_str(),
            });
        }
        return SpectraSceneControlStatusView{
            .struct_size = sizeof(SpectraSceneControlStatusView),
            .phase = cache.status.phase.c_str(),
            .headline = cache.status.headline.c_str(),
            .detail = cache.status.detail.c_str(),
            .metrics = SpectraSceneControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
            .action_states = SpectraSceneControlActionStateSpan{.data = cache.action_state_views.empty() ? nullptr : cache.action_state_views.data(), .count = static_cast<std::uint64_t>(cache.action_state_views.size())},
        };
    }

    [[nodiscard]] std::span<const SpectraSceneControlSettingValue> make_setting_view(ProjectSettingCache& cache) {
        cache.setting_views.clear();
        cache.setting_views.reserve(cache.settings.size());
        for (const instant_ngp::spectra_project::ProjectSettingValue& setting : cache.settings) {
            cache.setting_views.push_back(SpectraSceneControlSettingValue{
                .key = setting.key.c_str(),
                .value = setting.value.c_str(),
            });
        }
        return cache.setting_views;
    }

    [[nodiscard]] std::span<const SpectraSceneControlLogEntry> make_log_view(ProjectLogCache& cache) {
        cache.log_views.clear();
        cache.log_views.reserve(cache.logs.size());
        for (const instant_ngp::spectra_project::ProjectLogEntry& log : cache.logs) {
            cache.log_views.push_back(SpectraSceneControlLogEntry{
                .sequence = log.sequence,
                .level = log.level.c_str(),
                .message = log.message.c_str(),
            });
        }
        return cache.log_views;
    }

    [[nodiscard]] std::span<const SpectraSceneControlImage> make_image_view(ProjectImageCache& cache) {
        cache.image_views.clear();
        cache.image_views.reserve(cache.images.size());
        for (const instant_ngp::spectra_project::ProjectImage& image : cache.images) {
            cache.image_views.push_back(SpectraSceneControlImage{
                .id = image.id.c_str(),
                .label = image.label.c_str(),
                .description = image.description.c_str(),
                .rgba8 = image.rgba8.empty() ? nullptr : image.rgba8.data(),
                .rgba8_size = static_cast<std::uint64_t>(image.rgba8.size()),
                .revision = image.revision,
                .width = image.width,
                .height = image.height,
            });
        }
        return cache.image_views;
    }

    [[nodiscard]] std::span<const SpectraSceneControlScalarSeries> make_scalar_series_view(ProjectScalarSeriesCache& cache) {
        cache.sample_views.clear();
        cache.series_views.clear();
        cache.sample_views.resize(cache.series.size());
        cache.series_views.reserve(cache.series.size());
        for (std::size_t series_index = 0u; series_index < cache.series.size(); ++series_index) {
            const instant_ngp::spectra_project::ProjectScalarSeries& series = cache.series[series_index];
            std::vector<SpectraSceneControlScalarSample>& samples = cache.sample_views[series_index];
            samples.reserve(series.samples.size());
            for (const instant_ngp::spectra_project::ProjectScalarSample& sample : series.samples) {
                samples.push_back(SpectraSceneControlScalarSample{
                    .step = sample.step,
                    .time_seconds = sample.time_seconds,
                    .value = sample.value,
                });
            }
            SpectraSceneControlScalarSeries view{
                .id = series.id.c_str(),
                .label = series.label.c_str(),
                .description = series.description.c_str(),
                .unit = series.unit.c_str(),
                .color = {},
                .group = series.group,
                .priority = series.priority,
                .revision = series.revision,
                .samples = SpectraSceneControlScalarSampleSpan{.data = samples.empty() ? nullptr : samples.data(), .count = static_cast<std::uint64_t>(samples.size())},
            };
            copy_array(view.color, series.color);
            cache.series_views.push_back(view);
        }
        return cache.series_views;
    }

    [[nodiscard]] PluginInstance& checked_instance(SpectraSceneInstance* instance, const std::string_view action) {
        if (instance == nullptr) throw std::runtime_error(std::format("{} instance pointer is null", action));
        return *reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraSceneResult scene_create(const SpectraSceneOpenInfo* open_info, SpectraSceneInstance** instance) noexcept {
        try {
            if (open_info == nullptr) {
                global_error = "create open info pointer is null";
                return SPECTRA_SCENE_RESULT_ERROR;
            }
            if (instance == nullptr) {
                global_error = "create instance output pointer is null";
                return SPECTRA_SCENE_RESULT_ERROR;
            }
            *instance = nullptr;
            if (open_info->struct_size != sizeof(SpectraSceneOpenInfo)) throw std::runtime_error("scene plugin open info ABI size mismatch");
            std::vector<instant_ngp::spectra_project::Option> options = options_from_abi(open_info->options, "scene plugin open options");
            if (open_info->host_services == nullptr) throw std::runtime_error("scene plugin open info host services pointer is null");
            if (open_info->host_services->struct_size != sizeof(SpectraSceneHostServices)) throw std::runtime_error("scene plugin host services ABI size mismatch");
            if (open_info->host_services->request_gpu_buffer == nullptr) throw std::runtime_error("scene plugin host services request_gpu_buffer function is null");
            if (open_info->host_services->release_gpu_buffer == nullptr) throw std::runtime_error("scene plugin host services release_gpu_buffer function is null");
            if (open_info->host_services->last_error == nullptr) throw std::runtime_error("scene plugin host services last_error function is null");
            const SpectraSceneHostServices* host_services_view = open_info->host_services;
            std::shared_ptr<instant_ngp::spectra_project::HostServices> host_services = std::make_shared<instant_ngp::spectra_project::HostServices>();
            host_services->request_gpu_buffer = [host_services_view](const std::uint32_t kind, const std::uint64_t byte_size, const std::string_view debug_name) {
                const std::string debug_name_text{debug_name};
                std::uint32_t abi_kind{};
                switch (kind) {
                    case instant_ngp::spectra_project::GpuBufferKindVolumeChannel:
                        abi_kind = SPECTRA_SCENE_GPU_BUFFER_VOLUME_CHANNEL;
                        break;
                    case instant_ngp::spectra_project::GpuBufferKindViewportVoxelGrid:
                        abi_kind = SPECTRA_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID;
                        break;
                    default:
                        throw std::runtime_error(std::format("unknown scene plugin GPU buffer kind {}", kind));
                }
                SpectraSceneGpuBufferRequest request{
                    .struct_size = sizeof(SpectraSceneGpuBufferRequest),
                    .kind = abi_kind,
                    .byte_size = byte_size,
                    .debug_name = debug_name_text.c_str(),
                };
                SpectraSceneGpuBufferAllocation allocation{};
                const SpectraSceneResult result = host_services_view->request_gpu_buffer(host_services_view->user_data, &request, &allocation);
                if (result != SPECTRA_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
                if (allocation.struct_size != sizeof(SpectraSceneGpuBufferAllocation)) throw std::runtime_error("scene plugin GPU buffer allocation ABI size mismatch");
                if (allocation.kind != abi_kind) throw std::runtime_error(std::format("scene plugin GPU buffer allocation kind {} does not match request kind {}", allocation.kind, abi_kind));
                return instant_ngp::spectra_project::GpuBufferAllocation{
                    .resource_id = allocation.resource_id,
                    .byte_size = allocation.byte_size,
                    .kind = kind,
                    .handle_kind = gpu_handle_kind_from_abi(allocation.handle_kind),
                    .handle = allocation.handle,
                    .device_identity = device_identity_from_abi(allocation.device_identity),
                };
            };
            host_services->release_gpu_buffer = [host_services_view](const std::uint64_t resource_id) {
                const SpectraSceneResult result = host_services_view->release_gpu_buffer(host_services_view->user_data, resource_id);
                if (result != SPECTRA_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
            };
            std::unique_ptr<PluginInstance> created = std::make_unique<PluginInstance>();
            created->project = instant_ngp::spectra_project::InstantNgpSpectraProject::open(options, std::move(host_services));
            *instance = reinterpret_cast<SpectraSceneInstance*>(created.release());
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    void scene_destroy(SpectraSceneInstance* instance) noexcept {
        delete reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraSceneResult scene_reset(SpectraSceneInstance* instance) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "reset");
            plugin_instance.last_error.clear();
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult scene_document(SpectraSceneInstance* instance, SpectraSceneDocumentView* document) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "document");
            if (document == nullptr) throw std::runtime_error("document output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.scene_cache.document = plugin_instance.project.document();
            *document = make_document_view(plugin_instance.scene_cache);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult scene_frame(SpectraSceneInstance* instance, const SpectraSceneFrameInfo frame, SpectraSceneFrameView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "frame");
            static_cast<void>(frame);
            if (snapshot == nullptr) throw std::runtime_error("frame output pointer is null");
            plugin_instance.last_error.clear();
            *snapshot = SpectraSceneFrameView{
                .struct_size = sizeof(SpectraSceneFrameView),
            };
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const char* last_error(SpectraSceneInstance* instance) noexcept {
        if (instance == nullptr) return global_error.c_str();
        return reinterpret_cast<PluginInstance*>(instance)->last_error.c_str();
    }

    [[nodiscard]] SpectraSceneResult scene_update(SpectraSceneInstance* instance, const SpectraSceneUpdateInfo* update_info) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "scene_update");
            if (update_info == nullptr) throw std::runtime_error("scene_update info pointer is null");
            if (update_info->struct_size != sizeof(SpectraSceneUpdateInfo)) throw std::runtime_error("scene_update info ABI size mismatch");
            plugin_instance.last_error.clear();
            plugin_instance.project.update(instant_ngp::spectra_project::UpdateInfo{
                .wall_delta_seconds = update_info->wall_delta_seconds,
                .scene_delta_seconds = update_info->scene_delta_seconds,
                .time_seconds = update_info->time_seconds,
                .frame_index = update_info->frame_index,
                .timeline_mode = update_info->timeline_mode,
                .timeline_playing = update_info->timeline_playing != 0u,
            });
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult scene_revision(SpectraSceneInstance* instance, std::uint64_t* revision) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "scene_revision");
            if (revision == nullptr) throw std::runtime_error("scene_revision output pointer is null");
            plugin_instance.last_error.clear();
            *revision = plugin_instance.project.scene_revision();
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult control_action(SpectraSceneInstance* instance, const char* action_id, const SpectraSceneOptionSpan options) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_action");
            plugin_instance.last_error.clear();
            const std::string action = string_from_abi(action_id, "control action id", false);
            const std::vector<instant_ngp::spectra_project::Option> converted_options = options_from_abi(options, "control action options");
            plugin_instance.project.execute_action(action, converted_options);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult control_setting_update(SpectraSceneInstance* instance, const char* key, const char* value) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_setting_update");
            plugin_instance.last_error.clear();
            const std::string setting_key = string_from_abi(key, "control setting key", false);
            const std::string setting_value = string_from_abi(value, "control setting value", true);
            plugin_instance.project.update_setting(setting_key, setting_value);
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraSceneResult control_snapshot(SpectraSceneInstance* instance, SpectraSceneControlSnapshotView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_snapshot");
            if (snapshot == nullptr) throw std::runtime_error("control_snapshot output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.setting_cache.settings = plugin_instance.project.settings();
            plugin_instance.status_cache.status = plugin_instance.project.status();
            plugin_instance.log_cache.logs = plugin_instance.project.logs();
            plugin_instance.image_cache.images = plugin_instance.project.images();
            plugin_instance.scalar_series_cache.series = plugin_instance.project.scalar_series();
            const std::span<const SpectraSceneControlSettingValue> settings = make_setting_view(plugin_instance.setting_cache);
            const SpectraSceneControlStatusView status = make_status_view(plugin_instance.status_cache);
            const std::span<const SpectraSceneControlLogEntry> logs = make_log_view(plugin_instance.log_cache);
            const std::span<const SpectraSceneControlImage> images = make_image_view(plugin_instance.image_cache);
            const std::span<const SpectraSceneControlScalarSeries> scalar_series = make_scalar_series_view(plugin_instance.scalar_series_cache);
            *snapshot = SpectraSceneControlSnapshotView{
                .struct_size = sizeof(SpectraSceneControlSnapshotView),
                .settings = SpectraSceneControlSettingValueSpan{.data = settings.empty() ? nullptr : settings.data(), .count = static_cast<std::uint64_t>(settings.size())},
                .status = status,
                .logs = SpectraSceneControlLogEntrySpan{.data = logs.empty() ? nullptr : logs.data(), .count = static_cast<std::uint64_t>(logs.size())},
                .images = SpectraSceneControlImageSpan{.data = images.empty() ? nullptr : images.data(), .count = static_cast<std::uint64_t>(images.size())},
                .scalar_series = SpectraSceneControlScalarSeriesSpan{.data = scalar_series.empty() ? nullptr : scalar_series.data(), .count = static_cast<std::uint64_t>(scalar_series.size())},
            };
            return SPECTRA_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const SpectraScenePlugin& plugin() {
        const instant_ngp::spectra_project::Descriptor& descriptor = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor();
        const DescriptorViews& views = descriptor_views();
        static const SpectraScenePlugin value{
            .abi_version = plugin_abi_version,
            .struct_size = sizeof(SpectraScenePlugin),
            .id = descriptor.id.c_str(),
            .title = descriptor.title.c_str(),
            .controls_panel_title = descriptor.title.c_str(),
            .open_action_label = descriptor.open_action_label.c_str(),
            .open_action_description = descriptor.open_action_description.c_str(),
            .base_pbrt_path = "",
            .frames_per_second = descriptor.frames_per_second,
            .open_options = SpectraSceneControlOptionSchemaSpan{.data = views.open_options.schemas.empty() ? nullptr : views.open_options.schemas.data(), .count = static_cast<std::uint64_t>(views.open_options.schemas.size())},
            .control_actions = SpectraSceneControlActionSpan{.data = views.control_actions.empty() ? nullptr : views.control_actions.data(), .count = static_cast<std::uint64_t>(views.control_actions.size())},
            .control_settings = SpectraSceneControlOptionSchemaSpan{.data = views.control_settings.schemas.empty() ? nullptr : views.control_settings.schemas.data(), .count = static_cast<std::uint64_t>(views.control_settings.schemas.size())},
            .create = scene_create,
            .destroy = scene_destroy,
            .reset = scene_reset,
            .update = scene_update,
            .document = scene_document,
            .frame = scene_frame,
            .scene_revision = scene_revision,
            .control_action = control_action,
            .control_setting_update = control_setting_update,
            .control_snapshot = control_snapshot,
            .last_error = last_error,
        };
        return value;
    }
} // namespace

extern "C" SPECTRA_SCENE_EXPORT const SpectraScenePlugin* spectra_scene_plugin_v1(void) {
    return &plugin();
}
