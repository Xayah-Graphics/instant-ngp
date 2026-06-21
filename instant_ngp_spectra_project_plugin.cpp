#if defined(_WIN32)
#define SPECTRA_DYNAMIC_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_DYNAMIC_SCENE_EXPORT __attribute__((visibility("default")))
#endif

import instant_ngp.spectra_project;
import std;

constexpr std::uint32_t plugin_abi_version = 29u;
typedef void SpectraInstance;

typedef std::uint32_t SpectraResult;
constexpr std::uint32_t SPECTRA_DYNAMIC_SCENE_RESULT_OK = 0u;
constexpr std::uint32_t SPECTRA_DYNAMIC_SCENE_RESULT_ERROR = 1u;
constexpr std::uint32_t SPECTRA_DYNAMIC_SCENE_GPU_BUFFER_VOLUME_CHANNEL = 0u;
constexpr std::uint32_t SPECTRA_DYNAMIC_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID = 1u;

struct SpectraOption {
    const char* key{};
    const char* value{};
};

struct SpectraOptionSpan {
    const SpectraOption* data{};
    std::uint64_t count{};
};

struct SpectraOptionChoice {
    const char* value{};
    const char* label{};
};

struct SpectraOptionChoiceSpan {
    const SpectraOptionChoice* data{};
    std::uint64_t count{};
};

struct SpectraOptionSchema {
    const char* key{};
    const char* label{};
    const char* description{};
    std::uint32_t kind{};
    std::uint32_t required{};
    const char* default_value{};
    const char* group{};
    std::uint32_t advanced{};
    std::int32_t priority{};
    SpectraOptionChoiceSpan choices{};
};

struct SpectraOptionSchemaSpan {
    const SpectraOptionSchema* data{};
    std::uint64_t count{};
};

struct SpectraControlAction {
    const char* id{};
    const char* label{};
    const char* description{};
    std::uint32_t group{};
    std::int32_t priority{};
    std::uint32_t style{};
    SpectraOptionSchemaSpan options{};
};

struct SpectraControlActionSpan {
    const SpectraControlAction* data{};
    std::uint64_t count{};
};

struct SpectraControlSettingValue {
    const char* key{};
    const char* value{};
};

struct SpectraControlSettingValueSpan {
    const SpectraControlSettingValue* data{};
    std::uint64_t count{};
};

struct SpectraControlMetric {
    const char* key{};
    const char* label{};
    const char* value{};
    std::uint32_t placement_flags{};
    std::int32_t priority{};
    std::uint32_t has_color{};
    float color[4]{};
};

struct SpectraControlMetricSpan {
    const SpectraControlMetric* data{};
    std::uint64_t count{};
};

struct SpectraControlActionState {
    const char* action_id{};
    std::uint32_t enabled{};
    const char* disabled_reason{};
};

struct SpectraControlActionStateSpan {
    const SpectraControlActionState* data{};
    std::uint64_t count{};
};

struct SpectraControlStatusView {
    std::uint64_t struct_size{};
    const char* phase{};
    const char* headline{};
    const char* detail{};
    SpectraControlMetricSpan metrics{};
    SpectraControlActionStateSpan action_states{};
};

struct SpectraControlLogEntry {
    std::uint64_t sequence{};
    const char* level{};
    const char* message{};
};

struct SpectraControlLogEntrySpan {
    const SpectraControlLogEntry* data{};
    std::uint64_t count{};
};

struct SpectraControlImage {
    const char* id{};
    const char* label{};
    const char* description{};
    const std::uint8_t* rgba8{};
    std::uint64_t rgba8_size{};
    std::uint64_t revision{};
    std::uint32_t width{};
    std::uint32_t height{};
};

struct SpectraControlImageSpan {
    const SpectraControlImage* data{};
    std::uint64_t count{};
};

struct SpectraControlScalarSample {
    std::uint64_t step{};
    double time_seconds{};
    double value{};
};

struct SpectraControlScalarSampleSpan {
    const SpectraControlScalarSample* data{};
    std::uint64_t count{};
};

struct SpectraControlScalarSeries {
    const char* id{};
    const char* label{};
    const char* description{};
    const char* unit{};
    float color[4]{};
    std::uint32_t group{};
    std::int32_t priority{};
    std::uint64_t revision{};
    SpectraControlScalarSampleSpan samples{};
};

struct SpectraControlScalarSeriesSpan {
    const SpectraControlScalarSeries* data{};
    std::uint64_t count{};
};

struct SpectraControlSnapshotView {
    std::uint64_t struct_size{};
    SpectraControlSettingValueSpan settings{};
    SpectraControlStatusView status{};
    SpectraControlLogEntrySpan logs{};
    SpectraControlImageSpan images{};
    SpectraControlScalarSeriesSpan scalar_series{};
};

struct SpectraUpdateInfo {
    std::uint64_t struct_size{};
    double wall_delta_seconds{};
    double scene_delta_seconds{};
    double time_seconds{};
    std::uint64_t frame_index{};
    std::uint32_t timeline_mode{};
    std::uint32_t timeline_playing{};
};

struct SpectraGpuDeviceIdentity {
    std::uint32_t vendor_id{};
    std::uint32_t device_id{};
    std::uint8_t device_uuid[16]{};
    std::uint8_t device_luid[8]{};
    std::uint32_t device_node_mask{};
};

struct SpectraGpuBufferRequest {
    std::uint64_t struct_size{};
    std::uint32_t kind{};
    std::uint64_t byte_size{};
    const char* debug_name{};
};

struct SpectraGpuBufferAllocation {
    std::uint64_t struct_size{};
    std::uint64_t resource_id{};
    std::uint64_t byte_size{};
    std::uint32_t kind{};
    std::uint32_t handle_kind{};
    std::uintptr_t handle{};
    SpectraGpuDeviceIdentity device_identity{};
};

typedef SpectraResult (*SpectraRequestGpuBufferFn)(void* user_data, const SpectraGpuBufferRequest* request, SpectraGpuBufferAllocation* allocation);
typedef SpectraResult (*SpectraReleaseGpuBufferFn)(void* user_data, std::uint64_t resource_id);
typedef const char* (*SpectraHostLastErrorFn)(void* user_data);

struct SpectraHostServices {
    std::uint64_t struct_size{};
    void* user_data{};
    SpectraRequestGpuBufferFn request_gpu_buffer{};
    SpectraReleaseGpuBufferFn release_gpu_buffer{};
    SpectraHostLastErrorFn last_error{};
};

struct SpectraOpenInfo {
    std::uint64_t struct_size{};
    const char* plugin_path{};
    SpectraOptionSpan options{};
    const SpectraHostServices* host_services{};
};

struct SpectraTransform {
    float position[3]{};
    float rotation[4]{};
    float scale[3]{};
};

struct SpectraMaterial {
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

struct SpectraMaterialSpan {
    const SpectraMaterial* data{};
    std::uint64_t count{};
};

struct SpectraLight {
    const char* name{};
    const char* kind{};
    SpectraTransform transform{};
    float color[3]{};
    float intensity{};
    float cone_angle_degrees{};
};

struct SpectraLightSpan {
    const SpectraLight* data{};
    std::uint64_t count{};
};

struct SpectraCamera {
    const char* name{};
    const char* local_coordinate_system{};
    SpectraTransform transform{};
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

struct SpectraCameraSpan {
    const SpectraCamera* data{};
    std::uint64_t count{};
};

struct SpectraMeshVertex {
    float position[3]{};
    float normal[3]{};
};

struct SpectraMeshVertexSpan {
    const SpectraMeshVertex* data{};
    std::uint64_t count{};
};

struct SpectraUInt32Span {
    const std::uint32_t* data{};
    std::uint64_t count{};
};

struct SpectraMesh {
    const char* name{};
    SpectraMeshVertexSpan vertices{};
    SpectraUInt32Span indices{};
    const char* material_name{};
    SpectraTransform transform{};
};

struct SpectraMeshSpan {
    const SpectraMesh* data{};
    std::uint64_t count{};
};

struct SpectraSphere {
    const char* name{};
    float radius{};
    const char* material_name{};
    SpectraTransform transform{};
};

struct SpectraSphereSpan {
    const SpectraSphere* data{};
    std::uint64_t count{};
};

struct SpectraPoint {
    float position[3]{};
    float normal[3]{};
    float color[4]{};
    float radius{};
};

struct SpectraPointSpan {
    const SpectraPoint* data{};
    std::uint64_t count{};
};

struct SpectraPointCloud {
    const char* name{};
    SpectraPointSpan points{};
    const char* material_name{};
    SpectraTransform transform{};
};

struct SpectraPointCloudSpan {
    const SpectraPointCloud* data{};
    std::uint64_t count{};
};

struct SpectraFloatSpan {
    const float* data{};
    std::uint64_t count{};
};

struct SpectraVolumeChannel {
    const char* name{};
    std::uint32_t dimensions[3]{};
    SpectraFloatSpan values{};
    std::uint32_t format{};
    std::uint32_t source_kind{};
    std::uint32_t index_encoding{};
    std::uint64_t buffer_id{};
    std::uintptr_t external_device_pointer{};
    std::uint64_t source_byte_size{};
    std::uint64_t revision{};
};

struct SpectraVolumeChannelSpan {
    const SpectraVolumeChannel* data{};
    std::uint64_t count{};
};

struct SpectraVolume {
    const char* name{};
    std::uint32_t dimensions[3]{};
    float origin[3]{};
    float voxel_size[3]{};
    SpectraVolumeChannelSpan channels{};
    const char* material_name{};
};

struct SpectraVolumeSpan {
    const SpectraVolume* data{};
    std::uint64_t count{};
};

struct SpectraEntityRef {
    std::uint32_t kind{};
    const char* name{};
};

struct SpectraViewportSegment {
    float start[3]{};
    float end[3]{};
};

struct SpectraViewportSegmentSpan {
    const SpectraViewportSegment* data{};
    std::uint64_t count{};
};

struct SpectraColor {
    float value[4]{};
};

struct SpectraColorSpan {
    const SpectraColor* data{};
    std::uint64_t count{};
};

struct SpectraViewportSegmentSet {
    const char* name{};
    SpectraEntityRef owner{};
    SpectraViewportSegmentSpan segments{};
    SpectraColorSpan colors{};
    SpectraFloatSpan widths{};
    float width{};
    std::uint32_t width_mode{};
    std::uint32_t depth_mode{};
    SpectraTransform transform{};
};

struct SpectraViewportSegmentSetSpan {
    const SpectraViewportSegmentSet* data{};
    std::uint64_t count{};
};

struct SpectraViewportVoxelGrid {
    const char* name{};
    SpectraEntityRef owner{};
    std::uint32_t dimensions[3]{};
    float origin[3]{};
    float voxel_size[3]{};
    SpectraTransform transform{};
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

struct SpectraViewportVoxelGridSpan {
    const SpectraViewportVoxelGrid* data{};
    std::uint64_t count{};
};

struct SpectraViewportCameraVisualImage {
    const std::uint8_t* rgba8{};
    std::uint64_t rgba8_size{};
    std::uint64_t revision{};
    std::uint32_t width{};
    std::uint32_t height{};
    float tint[4]{};
};

struct SpectraViewportCameraVisual {
    const char* name{};
    SpectraEntityRef owner{};
    float color[4]{};
    float width{};
    std::uint32_t width_mode{};
    std::uint32_t depth_mode{};
    float visual_near{};
    float visual_far{};
    std::uint32_t has_image{};
    SpectraViewportCameraVisualImage image{};
};

struct SpectraViewportCameraVisualSpan {
    const SpectraViewportCameraVisual* data{};
    std::uint64_t count{};
};

struct SpectraSceneItems {
    SpectraMaterialSpan materials{};
    SpectraLightSpan lights{};
    SpectraCameraSpan cameras{};
    SpectraMeshSpan meshes{};
    SpectraSphereSpan spheres{};
    SpectraPointCloudSpan point_clouds{};
    SpectraVolumeSpan volumes{};
    SpectraViewportSegmentSetSpan viewport_segment_sets{};
    SpectraViewportVoxelGridSpan viewport_voxel_grids{};
    SpectraViewportCameraVisualSpan viewport_camera_visuals{};
};

struct SpectraDocumentView {
    std::uint64_t struct_size{};
    const char* default_coordinate_system{};
    const char* active_camera_name{};
    SpectraSceneItems items{};
};

struct SpectraFrameInfo {
    double delta_seconds{};
    double time_seconds{};
    std::uint64_t frame_index{};
};

struct SpectraFrameView {
    std::uint64_t struct_size{};
    SpectraSceneItems items{};
};

typedef SpectraResult (*SpectraCreateFn)(const SpectraOpenInfo* open_info, SpectraInstance** instance);
typedef void (*SpectraDestroyFn)(SpectraInstance* instance);
typedef SpectraResult (*SpectraResetFn)(SpectraInstance* instance);
typedef SpectraResult (*SpectraUpdateFn)(SpectraInstance* instance, const SpectraUpdateInfo* update_info);
typedef SpectraResult (*SpectraDocumentFn)(SpectraInstance* instance, SpectraDocumentView* document);
typedef SpectraResult (*SpectraFrameFn)(SpectraInstance* instance, SpectraFrameInfo frame, SpectraFrameView* snapshot);
typedef SpectraResult (*SpectraSceneRevisionFn)(SpectraInstance* instance, std::uint64_t* revision);
typedef SpectraResult (*SpectraControlActionFn)(SpectraInstance* instance, const char* action_id, SpectraOptionSpan options);
typedef SpectraResult (*SpectraControlSettingUpdateFn)(SpectraInstance* instance, const char* key, const char* value);
typedef SpectraResult (*SpectraControlSnapshotFn)(SpectraInstance* instance, SpectraControlSnapshotView* snapshot);
typedef const char* (*SpectraLastErrorFn)(SpectraInstance* instance);

struct SpectraPlugin {
    std::uint32_t abi_version{};
    std::uint64_t struct_size{};
    const char* id{};
    const char* title{};
    const char* controls_panel_title{};
    const char* open_action_label{};
    const char* open_action_description{};
    const char* base_pbrt_path{};
    double frames_per_second{};
    SpectraOptionSchemaSpan open_options{};
    SpectraControlActionSpan control_actions{};
    SpectraOptionSchemaSpan control_settings{};
    SpectraCreateFn create{};
    SpectraDestroyFn destroy{};
    SpectraResetFn reset{};
    SpectraUpdateFn update{};
    SpectraDocumentFn document{};
    SpectraFrameFn frame{};
    SpectraSceneRevisionFn scene_revision{};
    SpectraControlActionFn control_action{};
    SpectraControlSettingUpdateFn control_setting_update{};
    SpectraControlSnapshotFn control_snapshot{};
    SpectraLastErrorFn last_error{};
};

namespace {
    struct OptionSchemaViews {
        std::vector<std::vector<SpectraOptionChoice>> choices{};
        std::vector<SpectraOptionSchema> schemas{};
    };

    struct DescriptorViews {
        OptionSchemaViews open_options{};
        std::vector<OptionSchemaViews> action_options{};
        std::vector<SpectraControlAction> control_actions{};
        OptionSchemaViews control_settings{};
    };

    struct SceneViewCache {
        instant_ngp::spectra_project::Document document{};
        instant_ngp::spectra_project::Frame frame{};
        std::vector<SpectraMaterial> material_views{};
        std::vector<SpectraLight> light_views{};
        std::vector<std::vector<SpectraVolumeChannel>> volume_channel_storage{};
        std::vector<SpectraVolume> volume_views{};
        std::vector<SpectraCamera> camera_views{};
        std::vector<SpectraViewportVoxelGrid> voxel_grid_views{};
        std::vector<SpectraViewportCameraVisual> camera_visual_views{};
    };

    struct ProjectStatusCache {
        instant_ngp::spectra_project::ProjectStatus status{};
        std::vector<SpectraControlMetric> metric_views{};
        std::vector<SpectraControlActionState> action_state_views{};
        SpectraControlStatusView status_view{};
    };

    struct ProjectSettingCache {
        std::vector<instant_ngp::spectra_project::ProjectSettingValue> settings{};
        std::vector<SpectraControlSettingValue> setting_views{};
    };

    struct ProjectLogCache {
        std::vector<instant_ngp::spectra_project::ProjectLogEntry> logs{};
        std::vector<SpectraControlLogEntry> log_views{};
    };

    struct ProjectImageCache {
        std::span<const instant_ngp::spectra_project::ProjectImage> images{};
        std::vector<SpectraControlImage> image_views{};
    };

    struct ProjectScalarSeriesCache {
        std::vector<instant_ngp::spectra_project::ProjectScalarSeries> series{};
        std::vector<std::vector<SpectraControlScalarSample>> sample_views{};
        std::vector<SpectraControlScalarSeries> series_views{};
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

    [[nodiscard]] std::vector<instant_ngp::spectra_project::Option> options_from_abi(const SpectraOptionSpan options, const std::string_view context) {
        if (options.count != 0u && options.data == nullptr) throw std::runtime_error(std::format("{} pointer is null", context));
        if (options.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::format("{} count is too large", context));
        std::vector<instant_ngp::spectra_project::Option> converted{};
        converted.reserve(static_cast<std::size_t>(options.count));
        const std::span<const SpectraOption> option_span{options.data, static_cast<std::size_t>(options.count)};
        for (const SpectraOption& option : option_span) {
            converted.push_back(instant_ngp::spectra_project::Option{
                .key = string_from_abi(option.key, std::format("{} key", context), false),
                .value = string_from_abi(option.value, std::format("{} value", context), true),
            });
        }
        return converted;
    }

    [[nodiscard]] std::string host_services_error(const SpectraHostServices& host_services) {
        if (host_services.last_error == nullptr) return "unknown host service error";
        std::string message = string_from_abi(host_services.last_error(host_services.user_data), "dynamic scene host services error", true);
        if (message.empty()) message = "unknown host service error";
        return message;
    }

    [[nodiscard]] instant_ngp::spectra_project::GpuResourceHandleKind gpu_handle_kind_from_abi(const std::uint32_t kind) {
        switch (kind) {
            case 1u: return instant_ngp::spectra_project::GpuResourceHandleKind::OpaqueWin32;
            case 2u: return instant_ngp::spectra_project::GpuResourceHandleKind::OpaqueFileDescriptor;
            default: throw std::runtime_error(std::format("unknown dynamic scene GPU resource handle kind {}", kind));
        }
    }

    [[nodiscard]] instant_ngp::spectra_project::GpuDeviceIdentity device_identity_from_abi(const SpectraGpuDeviceIdentity& identity) {
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
            for (const instant_ngp::spectra_project::OptionChoice& choice : schema.choices) views.choices[index].push_back(SpectraOptionChoice{.value = choice.value.c_str(), .label = choice.label.c_str()});
            views.schemas.push_back(SpectraOptionSchema{
                .key = schema.key.c_str(),
                .label = schema.label.c_str(),
                .description = schema.description.c_str(),
                .kind = static_cast<std::uint32_t>(schema.kind),
                .required = schema.required ? 1u : 0u,
                .default_value = schema.default_value.c_str(),
                .group = schema.group.c_str(),
                .advanced = schema.advanced ? 1u : 0u,
                .priority = schema.priority,
                .choices = SpectraOptionChoiceSpan{.data = views.choices[index].empty() ? nullptr : views.choices[index].data(), .count = static_cast<std::uint64_t>(views.choices[index].size())},
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
            views.control_actions.push_back(SpectraControlAction{
                .id = action.id.c_str(),
                .label = action.label.c_str(),
                .description = action.description.c_str(),
                .group = action.group,
                .priority = action.priority,
                .style = action.style,
                .options = SpectraOptionSchemaSpan{.data = action_options.schemas.empty() ? nullptr : action_options.schemas.data(), .count = static_cast<std::uint64_t>(action_options.schemas.size())},
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

    template <typename Value>
    [[nodiscard]] const Value* span_data(const std::vector<Value>& values) {
        return values.empty() ? nullptr : values.data();
    }

    template <typename Value>
    [[nodiscard]] std::uint64_t span_count(const std::vector<Value>& values) {
        return static_cast<std::uint64_t>(values.size());
    }

    [[nodiscard]] SpectraTransform make_transform_view(const instant_ngp::spectra_project::Transform& transform) {
        SpectraTransform view{};
        copy_array(view.position, transform.position);
        copy_array(view.rotation, transform.rotation);
        copy_array(view.scale, transform.scale);
        return view;
    }

    [[nodiscard]] SpectraEntityRef make_entity_ref_view(const instant_ngp::spectra_project::SceneEntityRef& ref) {
        return SpectraEntityRef{
            .kind = static_cast<std::uint32_t>(ref.kind),
            .name = ref.name.c_str(),
        };
    }

    [[nodiscard]] SpectraMaterial make_material_view(const instant_ngp::spectra_project::Material& material) {
        SpectraMaterial view{
            .name = material.name.c_str(),
            .model = material.model.c_str(),
            .alpha_mode = material.alpha_mode.c_str(),
            .emission_strength = material.emission_strength,
            .roughness = material.roughness,
            .metallic = material.metallic,
            .alpha_cutoff = material.alpha_cutoff,
            .volume_density_scale = material.volume_density_scale,
            .volume_temperature_scale = material.volume_temperature_scale,
        };
        copy_array(view.base_color, material.base_color);
        copy_array(view.emission_color, material.emission_color);
        return view;
    }

    void make_material_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::Material>& materials) {
        cache.material_views.clear();
        cache.material_views.reserve(materials.size());
        for (const instant_ngp::spectra_project::Material& material : materials) cache.material_views.push_back(make_material_view(material));
    }

    [[nodiscard]] SpectraLight make_light_view(const instant_ngp::spectra_project::Light& light) {
        SpectraLight view{
            .name = light.name.c_str(),
            .kind = light.kind.c_str(),
            .transform = make_transform_view(light.transform),
            .intensity = light.intensity,
            .cone_angle_degrees = light.cone_angle_degrees,
        };
        copy_array(view.color, light.color);
        return view;
    }

    void make_light_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::Light>& lights) {
        cache.light_views.clear();
        cache.light_views.reserve(lights.size());
        for (const instant_ngp::spectra_project::Light& light : lights) cache.light_views.push_back(make_light_view(light));
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
                SpectraVolumeChannel channel_view{
                    .name = channel.name.c_str(),
                    .values = SpectraFloatSpan{.data = channel.values.empty() ? nullptr : channel.values.data(), .count = static_cast<std::uint64_t>(channel.values.size())},
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

            SpectraVolume volume_view{
                .name = volume.name.c_str(),
                .channels = SpectraVolumeChannelSpan{.data = cache.volume_channel_storage[volume_index].empty() ? nullptr : cache.volume_channel_storage[volume_index].data(), .count = static_cast<std::uint64_t>(cache.volume_channel_storage[volume_index].size())},
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

    [[nodiscard]] SpectraCamera make_camera_view(const instant_ngp::spectra_project::Camera& camera) {
        SpectraCamera view{
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

    [[nodiscard]] SpectraViewportCameraVisual make_camera_visual_view(const instant_ngp::spectra_project::ViewportCameraVisual& visual) {
        SpectraViewportCameraVisual view{
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
            view.image = SpectraViewportCameraVisualImage{
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

    void make_camera_visual_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::ViewportCameraVisual>& visuals) {
        cache.camera_visual_views.clear();
        cache.camera_visual_views.reserve(visuals.size());
        for (const instant_ngp::spectra_project::ViewportCameraVisual& visual : visuals) cache.camera_visual_views.push_back(make_camera_visual_view(visual));
    }

    [[nodiscard]] SpectraViewportVoxelGrid make_voxel_grid_view(const instant_ngp::spectra_project::ViewportVoxelGrid& grid) {
        SpectraViewportVoxelGrid view{
            .name = grid.name.c_str(),
            .owner = make_entity_ref_view(grid.owner),
            .transform = make_transform_view(grid.transform),
            .cell_scale = grid.cell_scale,
            .depth_mode = grid.depth_mode,
            .source_kind = static_cast<std::uint32_t>(grid.source_kind),
            .index_encoding = static_cast<std::uint32_t>(grid.index_encoding),
            .buffer_id = grid.buffer_id,
            .source_byte_size = grid.source_byte_size,
            .index_count = grid.index_count,
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

    void make_voxel_grid_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::ViewportVoxelGrid>& grids) {
        cache.voxel_grid_views.clear();
        cache.voxel_grid_views.reserve(grids.size());
        for (const instant_ngp::spectra_project::ViewportVoxelGrid& grid : grids) cache.voxel_grid_views.push_back(make_voxel_grid_view(grid));
    }

    [[nodiscard]] SpectraDocumentView make_document_view(SceneViewCache& cache) {
        make_material_views(cache, cache.document.materials);
        make_light_views(cache, cache.document.lights);
        make_volume_views(cache, cache.document.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.document.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.document.cameras) cache.camera_views.push_back(make_camera_view(camera));
        make_voxel_grid_views(cache, cache.document.debug_attachments.viewport_voxel_grids);
        make_camera_visual_views(cache, cache.document.debug_attachments.viewport_camera_visuals);
        return SpectraDocumentView{
            .struct_size = sizeof(SpectraDocumentView),
            .default_coordinate_system = cache.document.default_coordinate_system.c_str(),
            .active_camera_name = cache.document.active_camera_name.c_str(),
            .items = SpectraSceneItems{
                .materials = SpectraMaterialSpan{.data = span_data(cache.material_views), .count = span_count(cache.material_views)},
                .lights = SpectraLightSpan{.data = span_data(cache.light_views), .count = span_count(cache.light_views)},
                .cameras = SpectraCameraSpan{.data = span_data(cache.camera_views), .count = span_count(cache.camera_views)},
                .volumes = SpectraVolumeSpan{.data = span_data(cache.volume_views), .count = span_count(cache.volume_views)},
                .viewport_voxel_grids = SpectraViewportVoxelGridSpan{.data = span_data(cache.voxel_grid_views), .count = span_count(cache.voxel_grid_views)},
                .viewport_camera_visuals = SpectraViewportCameraVisualSpan{.data = span_data(cache.camera_visual_views), .count = span_count(cache.camera_visual_views)},
            },
        };
    }

    [[nodiscard]] SpectraFrameView make_frame_view(SceneViewCache& cache) {
        make_volume_views(cache, cache.frame.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.frame.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.frame.cameras) cache.camera_views.push_back(make_camera_view(camera));
        make_voxel_grid_views(cache, cache.frame.debug_attachments.viewport_voxel_grids);
        make_camera_visual_views(cache, cache.frame.debug_attachments.viewport_camera_visuals);
        return SpectraFrameView{
            .struct_size = sizeof(SpectraFrameView),
            .items = SpectraSceneItems{
                .cameras = SpectraCameraSpan{.data = span_data(cache.camera_views), .count = span_count(cache.camera_views)},
                .volumes = SpectraVolumeSpan{.data = span_data(cache.volume_views), .count = span_count(cache.volume_views)},
                .viewport_voxel_grids = SpectraViewportVoxelGridSpan{.data = span_data(cache.voxel_grid_views), .count = span_count(cache.voxel_grid_views)},
                .viewport_camera_visuals = SpectraViewportCameraVisualSpan{.data = span_data(cache.camera_visual_views), .count = span_count(cache.camera_visual_views)},
            },
        };
    }

    [[nodiscard]] SpectraControlStatusView make_status_view(ProjectStatusCache& cache) {
        cache.metric_views.clear();
        cache.action_state_views.clear();
        cache.metric_views.reserve(cache.status.metrics.size());
        cache.action_state_views.reserve(cache.status.action_states.size());
        for (const instant_ngp::spectra_project::ProjectMetric& metric : cache.status.metrics) {
            cache.metric_views.push_back(SpectraControlMetric{
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
            cache.action_state_views.push_back(SpectraControlActionState{
                .action_id = action_state.action_id.c_str(),
                .enabled = action_state.enabled ? 1u : 0u,
                .disabled_reason = action_state.disabled_reason.c_str(),
            });
        }
        return SpectraControlStatusView{
            .struct_size = sizeof(SpectraControlStatusView),
            .phase = cache.status.phase.c_str(),
            .headline = cache.status.headline.c_str(),
            .detail = cache.status.detail.c_str(),
            .metrics = SpectraControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
            .action_states = SpectraControlActionStateSpan{.data = cache.action_state_views.empty() ? nullptr : cache.action_state_views.data(), .count = static_cast<std::uint64_t>(cache.action_state_views.size())},
        };
    }

    [[nodiscard]] std::span<const SpectraControlSettingValue> make_setting_view(ProjectSettingCache& cache) {
        cache.setting_views.clear();
        cache.setting_views.reserve(cache.settings.size());
        for (const instant_ngp::spectra_project::ProjectSettingValue& setting : cache.settings) {
            cache.setting_views.push_back(SpectraControlSettingValue{
                .key = setting.key.c_str(),
                .value = setting.value.c_str(),
            });
        }
        return cache.setting_views;
    }

    [[nodiscard]] std::span<const SpectraControlLogEntry> make_log_view(ProjectLogCache& cache) {
        cache.log_views.clear();
        cache.log_views.reserve(cache.logs.size());
        for (const instant_ngp::spectra_project::ProjectLogEntry& log : cache.logs) {
            cache.log_views.push_back(SpectraControlLogEntry{
                .sequence = log.sequence,
                .level = log.level.c_str(),
                .message = log.message.c_str(),
            });
        }
        return cache.log_views;
    }

    [[nodiscard]] std::span<const SpectraControlImage> make_image_view(ProjectImageCache& cache) {
        cache.image_views.clear();
        cache.image_views.reserve(cache.images.size());
        for (const instant_ngp::spectra_project::ProjectImage& image : cache.images) {
            cache.image_views.push_back(SpectraControlImage{
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

    [[nodiscard]] std::span<const SpectraControlScalarSeries> make_scalar_series_view(ProjectScalarSeriesCache& cache) {
        cache.sample_views.clear();
        cache.series_views.clear();
        cache.sample_views.resize(cache.series.size());
        cache.series_views.reserve(cache.series.size());
        for (std::size_t series_index = 0u; series_index < cache.series.size(); ++series_index) {
            const instant_ngp::spectra_project::ProjectScalarSeries& series = cache.series[series_index];
            std::vector<SpectraControlScalarSample>& samples = cache.sample_views[series_index];
            samples.reserve(series.samples.size());
            for (const instant_ngp::spectra_project::ProjectScalarSample& sample : series.samples) {
                samples.push_back(SpectraControlScalarSample{
                    .step = sample.step,
                    .time_seconds = sample.time_seconds,
                    .value = sample.value,
                });
            }
            SpectraControlScalarSeries view{
                .id = series.id.c_str(),
                .label = series.label.c_str(),
                .description = series.description.c_str(),
                .unit = series.unit.c_str(),
                .color = {},
                .group = series.group,
                .priority = series.priority,
                .revision = series.revision,
                .samples = SpectraControlScalarSampleSpan{.data = samples.empty() ? nullptr : samples.data(), .count = static_cast<std::uint64_t>(samples.size())},
            };
            copy_array(view.color, series.color);
            cache.series_views.push_back(view);
        }
        return cache.series_views;
    }

    [[nodiscard]] PluginInstance& checked_instance(SpectraInstance* instance, const std::string_view action) {
        if (instance == nullptr) throw std::runtime_error(std::format("{} instance pointer is null", action));
        return *reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraResult scene_create(const SpectraOpenInfo* open_info, SpectraInstance** instance) noexcept {
        try {
            if (open_info == nullptr) {
                global_error = "create open info pointer is null";
                return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
            }
            if (instance == nullptr) {
                global_error = "create instance output pointer is null";
                return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
            }
            *instance = nullptr;
            if (open_info->struct_size != sizeof(SpectraOpenInfo)) throw std::runtime_error("dynamic scene open info ABI size mismatch");
            static_cast<void>(string_from_abi(open_info->plugin_path, "dynamic scene plugin path", false));
            std::vector<instant_ngp::spectra_project::Option> options = options_from_abi(open_info->options, "dynamic scene open options");
            if (open_info->host_services == nullptr) throw std::runtime_error("dynamic scene open info host services pointer is null");
            if (open_info->host_services->struct_size != sizeof(SpectraHostServices)) throw std::runtime_error("dynamic scene host services ABI size mismatch");
            if (open_info->host_services->request_gpu_buffer == nullptr) throw std::runtime_error("dynamic scene host services request_gpu_buffer function is null");
            if (open_info->host_services->release_gpu_buffer == nullptr) throw std::runtime_error("dynamic scene host services release_gpu_buffer function is null");
            if (open_info->host_services->last_error == nullptr) throw std::runtime_error("dynamic scene host services last_error function is null");
            const SpectraHostServices* host_services_view = open_info->host_services;
            std::shared_ptr<instant_ngp::spectra_project::HostServices> host_services = std::make_shared<instant_ngp::spectra_project::HostServices>();
            host_services->request_gpu_buffer = [host_services_view](const std::uint32_t kind, const std::uint64_t byte_size, const std::string_view debug_name) {
                const std::string debug_name_text{debug_name};
                std::uint32_t abi_kind{};
                switch (kind) {
                    case instant_ngp::spectra_project::GpuBufferKindVolumeChannel:
                        abi_kind = SPECTRA_DYNAMIC_SCENE_GPU_BUFFER_VOLUME_CHANNEL;
                        break;
                    case instant_ngp::spectra_project::GpuBufferKindViewportVoxelGrid:
                        abi_kind = SPECTRA_DYNAMIC_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID;
                        break;
                    default:
                        throw std::runtime_error(std::format("unknown dynamic scene GPU buffer kind {}", kind));
                }
                SpectraGpuBufferRequest request{
                    .struct_size = sizeof(SpectraGpuBufferRequest),
                    .kind = abi_kind,
                    .byte_size = byte_size,
                    .debug_name = debug_name_text.c_str(),
                };
                SpectraGpuBufferAllocation allocation{};
                const SpectraResult result = host_services_view->request_gpu_buffer(host_services_view->user_data, &request, &allocation);
                if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
                if (allocation.struct_size != sizeof(SpectraGpuBufferAllocation)) throw std::runtime_error("dynamic scene GPU buffer allocation ABI size mismatch");
                if (allocation.kind != abi_kind) throw std::runtime_error(std::format("dynamic scene GPU buffer allocation kind {} does not match request kind {}", allocation.kind, abi_kind));
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
                const SpectraResult result = host_services_view->release_gpu_buffer(host_services_view->user_data, resource_id);
                if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
            };
            std::unique_ptr<PluginInstance> created = std::make_unique<PluginInstance>();
            created->project = instant_ngp::spectra_project::InstantNgpSpectraProject::open(options, std::move(host_services));
            *instance = reinterpret_cast<SpectraInstance*>(created.release());
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    void scene_destroy(SpectraInstance* instance) noexcept {
        delete reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraResult scene_reset(SpectraInstance* instance) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "reset");
            plugin_instance.last_error.clear();
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult scene_document(SpectraInstance* instance, SpectraDocumentView* document) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "document");
            if (document == nullptr) throw std::runtime_error("document output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.scene_cache.document = plugin_instance.project.document();
            *document = make_document_view(plugin_instance.scene_cache);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult scene_frame(SpectraInstance* instance, const SpectraFrameInfo frame, SpectraFrameView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "frame");
            if (snapshot == nullptr) throw std::runtime_error("frame output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.scene_cache.frame = plugin_instance.project.frame(instant_ngp::spectra_project::FrameInfo{
                .delta_seconds = frame.delta_seconds,
                .time_seconds = frame.time_seconds,
                .frame_index = frame.frame_index,
            });
            *snapshot = make_frame_view(plugin_instance.scene_cache);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const char* last_error(SpectraInstance* instance) noexcept {
        if (instance == nullptr) return global_error.c_str();
        return reinterpret_cast<PluginInstance*>(instance)->last_error.c_str();
    }

    [[nodiscard]] SpectraResult scene_update(SpectraInstance* instance, const SpectraUpdateInfo* update_info) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "scene_update");
            if (update_info == nullptr) throw std::runtime_error("scene_update info pointer is null");
            if (update_info->struct_size != sizeof(SpectraUpdateInfo)) throw std::runtime_error("scene_update info ABI size mismatch");
            plugin_instance.last_error.clear();
            plugin_instance.project.update(instant_ngp::spectra_project::UpdateInfo{
                .wall_delta_seconds = update_info->wall_delta_seconds,
                .scene_delta_seconds = update_info->scene_delta_seconds,
                .time_seconds = update_info->time_seconds,
                .frame_index = update_info->frame_index,
                .timeline_mode = update_info->timeline_mode,
                .timeline_playing = update_info->timeline_playing != 0u,
            });
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult scene_revision(SpectraInstance* instance, std::uint64_t* revision) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "scene_revision");
            if (revision == nullptr) throw std::runtime_error("scene_revision output pointer is null");
            plugin_instance.last_error.clear();
            *revision = plugin_instance.project.scene_revision();
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult control_action(SpectraInstance* instance, const char* action_id, const SpectraOptionSpan options) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_action");
            plugin_instance.last_error.clear();
            const std::string action = string_from_abi(action_id, "control action id", false);
            const std::vector<instant_ngp::spectra_project::Option> converted_options = options_from_abi(options, "control action options");
            plugin_instance.project.execute_action(action, converted_options);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult control_setting_update(SpectraInstance* instance, const char* key, const char* value) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_setting_update");
            plugin_instance.last_error.clear();
            const std::string setting_key = string_from_abi(key, "control setting key", false);
            const std::string setting_value = string_from_abi(value, "control setting value", true);
            plugin_instance.project.update_setting(setting_key, setting_value);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraResult control_snapshot(SpectraInstance* instance, SpectraControlSnapshotView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_snapshot");
            if (snapshot == nullptr) throw std::runtime_error("control_snapshot output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.setting_cache.settings = plugin_instance.project.settings();
            plugin_instance.status_cache.status = plugin_instance.project.status();
            plugin_instance.log_cache.logs = plugin_instance.project.logs();
            plugin_instance.image_cache.images = plugin_instance.project.images();
            plugin_instance.scalar_series_cache.series = plugin_instance.project.scalar_series();
            const std::span<const SpectraControlSettingValue> settings = make_setting_view(plugin_instance.setting_cache);
            plugin_instance.status_cache.status_view = make_status_view(plugin_instance.status_cache);
            const std::span<const SpectraControlLogEntry> logs = make_log_view(plugin_instance.log_cache);
            const std::span<const SpectraControlImage> images = make_image_view(plugin_instance.image_cache);
            const std::span<const SpectraControlScalarSeries> scalar_series = make_scalar_series_view(plugin_instance.scalar_series_cache);
            *snapshot = SpectraControlSnapshotView{
                .struct_size = sizeof(SpectraControlSnapshotView),
                .settings = SpectraControlSettingValueSpan{.data = settings.empty() ? nullptr : settings.data(), .count = static_cast<std::uint64_t>(settings.size())},
                .status = plugin_instance.status_cache.status_view,
                .logs = SpectraControlLogEntrySpan{.data = logs.empty() ? nullptr : logs.data(), .count = static_cast<std::uint64_t>(logs.size())},
                .images = SpectraControlImageSpan{.data = images.empty() ? nullptr : images.data(), .count = static_cast<std::uint64_t>(images.size())},
                .scalar_series = SpectraControlScalarSeriesSpan{.data = scalar_series.empty() ? nullptr : scalar_series.data(), .count = static_cast<std::uint64_t>(scalar_series.size())},
            };
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const SpectraPlugin& plugin() {
        const instant_ngp::spectra_project::Descriptor& descriptor = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor();
        const DescriptorViews& views = descriptor_views();
        static const SpectraPlugin value{
            .abi_version = plugin_abi_version,
            .struct_size = sizeof(SpectraPlugin),
            .id = descriptor.id.c_str(),
            .title = descriptor.title.c_str(),
            .controls_panel_title = descriptor.controls_panel_title.c_str(),
            .open_action_label = descriptor.open_action_label.c_str(),
            .open_action_description = descriptor.open_action_description.c_str(),
            .base_pbrt_path = descriptor.base_pbrt_path.c_str(),
            .frames_per_second = descriptor.frames_per_second,
            .open_options = SpectraOptionSchemaSpan{.data = views.open_options.schemas.empty() ? nullptr : views.open_options.schemas.data(), .count = static_cast<std::uint64_t>(views.open_options.schemas.size())},
            .control_actions = SpectraControlActionSpan{.data = views.control_actions.empty() ? nullptr : views.control_actions.data(), .count = static_cast<std::uint64_t>(views.control_actions.size())},
            .control_settings = SpectraOptionSchemaSpan{.data = views.control_settings.schemas.empty() ? nullptr : views.control_settings.schemas.data(), .count = static_cast<std::uint64_t>(views.control_settings.schemas.size())},
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

extern "C" SPECTRA_DYNAMIC_SCENE_EXPORT const SpectraPlugin* spectra_dynamic_scene_plugin_v29(void) {
    return &plugin();
}
