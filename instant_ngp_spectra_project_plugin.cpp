module;
#include <stdint.h>

module instant_ngp.spectra_project;

import std;

#if defined(_WIN32)
#define SPECTRA_DYNAMIC_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_DYNAMIC_SCENE_EXPORT __attribute__((visibility("default")))
#endif

#define SPECTRA_DYNAMIC_SCENE_ABI_VERSION 27u

typedef struct SpectraDynamicSceneOption {
    const char* key;
    const char* value;
} SpectraDynamicSceneOption;

typedef struct SpectraDynamicSceneOptionSpan {
    const SpectraDynamicSceneOption* data;
    uint64_t count;
} SpectraDynamicSceneOptionSpan;

static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_TEXT = 0u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_DIRECTORY_PATH = 1u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_FILE_PATH = 2u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_CHOICE = 3u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_BOOL = 4u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_FLOAT = 5u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_OPTION_UNSIGNED_INTEGER = 6u;

static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_MATERIAL = 0u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_LIGHT = 1u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_CAMERA = 2u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_VOLUME = 6u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_VOXEL_GRID = 101u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_CAMERA_VISUAL = 102u;

typedef struct SpectraDynamicSceneOptionChoice {
    const char* value;
    const char* label;
} SpectraDynamicSceneOptionChoice;

typedef struct SpectraDynamicSceneOptionChoiceSpan {
    const SpectraDynamicSceneOptionChoice* data;
    uint64_t count;
} SpectraDynamicSceneOptionChoiceSpan;

typedef struct SpectraDynamicSceneOptionSchema {
    const char* key;
    const char* label;
    const char* description;
    uint32_t kind;
    uint32_t required;
    const char* default_value;
    const char* group;
    uint32_t advanced;
    int32_t priority;
    SpectraDynamicSceneOptionChoiceSpan choices;
} SpectraDynamicSceneOptionSchema;

typedef struct SpectraDynamicSceneOptionSchemaSpan {
    const SpectraDynamicSceneOptionSchema* data;
    uint64_t count;
} SpectraDynamicSceneOptionSchemaSpan;

typedef struct SpectraDynamicSceneControlAction {
    const char* id;
    const char* label;
    const char* description;
    uint32_t group;
    int32_t priority;
    uint32_t style;
    SpectraDynamicSceneOptionSchemaSpan options;
} SpectraDynamicSceneControlAction;

typedef struct SpectraDynamicSceneControlActionSpan {
    const SpectraDynamicSceneControlAction* data;
    uint64_t count;
} SpectraDynamicSceneControlActionSpan;

typedef struct SpectraDynamicSceneControlSetting {
    const char* key;
    const char* label;
    const char* description;
    uint32_t kind;
    const char* value;
    const char* group;
    uint32_t advanced;
    int32_t priority;
    SpectraDynamicSceneOptionChoiceSpan choices;
} SpectraDynamicSceneControlSetting;

typedef struct SpectraDynamicSceneControlSettingSpan {
    const SpectraDynamicSceneControlSetting* data;
    uint64_t count;
} SpectraDynamicSceneControlSettingSpan;

typedef struct SpectraDynamicSceneControlSettingView {
    uint64_t struct_size;
    SpectraDynamicSceneControlSettingSpan settings;
} SpectraDynamicSceneControlSettingView;

typedef struct SpectraDynamicSceneControlMetric {
    const char* key;
    const char* label;
    const char* value;
    uint32_t placement_flags;
    int32_t priority;
    uint32_t has_color;
    float color[4];
} SpectraDynamicSceneControlMetric;

typedef struct SpectraDynamicSceneControlMetricSpan {
    const SpectraDynamicSceneControlMetric* data;
    uint64_t count;
} SpectraDynamicSceneControlMetricSpan;

typedef struct SpectraDynamicSceneControlDisabledAction {
    const char* action_id;
    const char* reason;
} SpectraDynamicSceneControlDisabledAction;

typedef struct SpectraDynamicSceneControlDisabledActionSpan {
    const SpectraDynamicSceneControlDisabledAction* data;
    uint64_t count;
} SpectraDynamicSceneControlDisabledActionSpan;

typedef struct SpectraDynamicSceneControlStatusView {
    uint64_t struct_size;
    const char* phase;
    const char* headline;
    const char* detail;
    SpectraDynamicSceneControlMetricSpan metrics;
    const char* const* enabled_action_ids;
    uint64_t enabled_action_id_count;
    SpectraDynamicSceneControlDisabledActionSpan disabled_actions;
} SpectraDynamicSceneControlStatusView;

typedef struct SpectraDynamicSceneControlLogEntry {
    uint64_t sequence;
    const char* level;
    const char* message;
} SpectraDynamicSceneControlLogEntry;

typedef struct SpectraDynamicSceneControlLogEntrySpan {
    const SpectraDynamicSceneControlLogEntry* data;
    uint64_t count;
} SpectraDynamicSceneControlLogEntrySpan;

typedef struct SpectraDynamicSceneControlLogView {
    uint64_t struct_size;
    SpectraDynamicSceneControlLogEntrySpan entries;
} SpectraDynamicSceneControlLogView;

typedef struct SpectraDynamicSceneControlImage {
    const char* id;
    const char* label;
    const char* description;
    const uint8_t* rgba8;
    uint64_t rgba8_size;
    uint64_t revision;
    uint32_t width;
    uint32_t height;
} SpectraDynamicSceneControlImage;

typedef struct SpectraDynamicSceneControlImageSpan {
    const SpectraDynamicSceneControlImage* data;
    uint64_t count;
} SpectraDynamicSceneControlImageSpan;

typedef struct SpectraDynamicSceneControlImageView {
    uint64_t struct_size;
    SpectraDynamicSceneControlImageSpan images;
} SpectraDynamicSceneControlImageView;

typedef struct SpectraDynamicSceneControlScalarSample {
    uint64_t step;
    double time_seconds;
    double value;
} SpectraDynamicSceneControlScalarSample;

typedef struct SpectraDynamicSceneControlScalarSampleSpan {
    const SpectraDynamicSceneControlScalarSample* data;
    uint64_t count;
} SpectraDynamicSceneControlScalarSampleSpan;

typedef struct SpectraDynamicSceneControlScalarSeries {
    const char* id;
    const char* label;
    const char* description;
    const char* unit;
    float color[4];
    uint32_t group;
    int32_t priority;
    uint64_t revision;
    SpectraDynamicSceneControlScalarSampleSpan samples;
} SpectraDynamicSceneControlScalarSeries;

typedef struct SpectraDynamicSceneControlScalarSeriesSpan {
    const SpectraDynamicSceneControlScalarSeries* data;
    uint64_t count;
} SpectraDynamicSceneControlScalarSeriesSpan;

typedef struct SpectraDynamicSceneControlScalarSeriesView {
    uint64_t struct_size;
    SpectraDynamicSceneControlScalarSeriesSpan series;
} SpectraDynamicSceneControlScalarSeriesView;

typedef struct SpectraDynamicSceneControlSnapshotView {
    uint64_t struct_size;
    SpectraDynamicSceneControlSettingView settings;
    SpectraDynamicSceneControlStatusView status;
    SpectraDynamicSceneControlLogView logs;
    SpectraDynamicSceneControlImageView images;
    SpectraDynamicSceneControlScalarSeriesView scalar_series;
} SpectraDynamicSceneControlSnapshotView;

typedef uint32_t SpectraDynamicSceneResult;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_RESULT_OK = 0u;
static constexpr uint32_t SPECTRA_DYNAMIC_SCENE_RESULT_ERROR = 1u;

typedef struct SpectraDynamicSceneGpuDeviceIdentity {
    uint32_t vendor_id;
    uint32_t device_id;
    uint8_t device_uuid[16];
    uint8_t device_luid[8];
    uint32_t device_node_mask;
} SpectraDynamicSceneGpuDeviceIdentity;

typedef struct SpectraDynamicSceneViewportVoxelBufferRequest {
    uint64_t struct_size;
    uint64_t byte_size;
    const char* debug_name;
} SpectraDynamicSceneViewportVoxelBufferRequest;

typedef struct SpectraDynamicSceneViewportVoxelBufferAllocation {
    uint64_t struct_size;
    uint64_t resource_id;
    uint64_t byte_size;
    uint32_t handle_kind;
    uintptr_t handle;
    SpectraDynamicSceneGpuDeviceIdentity device_identity;
} SpectraDynamicSceneViewportVoxelBufferAllocation;

typedef struct SpectraDynamicSceneVolumeBufferRequest {
    uint64_t struct_size;
    uint64_t byte_size;
    const char* debug_name;
} SpectraDynamicSceneVolumeBufferRequest;

typedef struct SpectraDynamicSceneVolumeBufferAllocation {
    uint64_t struct_size;
    uint64_t resource_id;
    uint64_t byte_size;
    uint32_t handle_kind;
    uintptr_t handle;
    SpectraDynamicSceneGpuDeviceIdentity device_identity;
} SpectraDynamicSceneVolumeBufferAllocation;

typedef SpectraDynamicSceneResult (*SpectraDynamicSceneRequestViewportVoxelBufferFn)(void* user_data, const SpectraDynamicSceneViewportVoxelBufferRequest* request, SpectraDynamicSceneViewportVoxelBufferAllocation* allocation);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneReleaseViewportVoxelBufferFn)(void* user_data, uint64_t resource_id);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneRequestVolumeBufferFn)(void* user_data, const SpectraDynamicSceneVolumeBufferRequest* request, SpectraDynamicSceneVolumeBufferAllocation* allocation);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneReleaseVolumeBufferFn)(void* user_data, uint64_t resource_id);
typedef const char* (*SpectraDynamicSceneHostLastErrorFn)(void* user_data);

typedef struct SpectraDynamicSceneHostServices {
    uint64_t struct_size;
    void* user_data;
    SpectraDynamicSceneRequestViewportVoxelBufferFn request_viewport_voxel_buffer;
    SpectraDynamicSceneReleaseViewportVoxelBufferFn release_viewport_voxel_buffer;
    SpectraDynamicSceneRequestVolumeBufferFn request_volume_buffer;
    SpectraDynamicSceneReleaseVolumeBufferFn release_volume_buffer;
    SpectraDynamicSceneHostLastErrorFn last_error;
} SpectraDynamicSceneHostServices;

typedef struct SpectraDynamicSceneOpenInfo {
    uint64_t struct_size;
    const char* plugin_path;
    SpectraDynamicSceneOptionSpan options;
    const SpectraDynamicSceneHostServices* host_services;
} SpectraDynamicSceneOpenInfo;

typedef struct SpectraDynamicSceneTransform {
    float position[3];
    float rotation[4];
    float scale[3];
} SpectraDynamicSceneTransform;

typedef struct SpectraDynamicSceneMaterial {
    const char* name;
    const char* model;
    const char* alpha_mode;
    float base_color[4];
    float emission_color[3];
    float emission_strength;
    float roughness;
    float metallic;
    float alpha_cutoff;
    float volume_density_scale;
    float volume_temperature_scale;
} SpectraDynamicSceneMaterial;

typedef struct SpectraDynamicSceneLight {
    const char* name;
    const char* kind;
    SpectraDynamicSceneTransform transform;
    float color[3];
    float intensity;
    float cone_angle_degrees;
} SpectraDynamicSceneLight;

typedef struct SpectraDynamicSceneCamera {
    const char* name;
    const char* local_coordinate_system;
    SpectraDynamicSceneTransform transform;
    float target[3];
    float up[3];
    uint32_t projection;
    float vertical_fov_degrees;
    uint32_t image_width;
    uint32_t image_height;
    float fx;
    float fy;
    float cx;
    float cy;
    float near_plane;
    float far_plane;
} SpectraDynamicSceneCamera;

typedef struct SpectraDynamicSceneFloatSpan {
    const float* data;
    uint64_t count;
} SpectraDynamicSceneFloatSpan;

typedef struct SpectraDynamicSceneVolumeChannel {
    const char* name;
    uint32_t dimensions[3];
    SpectraDynamicSceneFloatSpan values;
    uint32_t format;
    uint32_t source_kind;
    uint32_t index_encoding;
    uint64_t buffer_id;
    uintptr_t external_device_pointer;
    uint64_t source_byte_size;
    uint64_t revision;
} SpectraDynamicSceneVolumeChannel;

typedef struct SpectraDynamicSceneVolumeChannelSpan {
    const SpectraDynamicSceneVolumeChannel* data;
    uint64_t count;
} SpectraDynamicSceneVolumeChannelSpan;

typedef struct SpectraDynamicSceneVolume {
    const char* name;
    uint32_t dimensions[3];
    float origin[3];
    float voxel_size[3];
    SpectraDynamicSceneVolumeChannelSpan channels;
    const char* material_name;
} SpectraDynamicSceneVolume;

typedef struct SpectraDynamicSceneEntityRef {
    uint32_t kind;
    const char* name;
} SpectraDynamicSceneEntityRef;

typedef struct SpectraDynamicSceneViewportVoxelGrid {
    const char* name;
    SpectraDynamicSceneEntityRef owner;
    uint32_t dimensions[3];
    float origin[3];
    float voxel_size[3];
    SpectraDynamicSceneTransform transform;
    float color[4];
    float cell_scale;
    uint32_t depth_mode;
    uint32_t source_kind;
    uint32_t index_encoding;
    uint64_t buffer_id;
    uint64_t source_byte_size;
    uint64_t index_count;
    uint64_t revision;
} SpectraDynamicSceneViewportVoxelGrid;

typedef struct SpectraDynamicSceneViewportCameraVisualImage {
    const uint8_t* rgba8;
    uint64_t rgba8_size;
    uint64_t revision;
    uint32_t width;
    uint32_t height;
    float tint[4];
} SpectraDynamicSceneViewportCameraVisualImage;

typedef struct SpectraDynamicSceneViewportCameraVisual {
    const char* name;
    SpectraDynamicSceneEntityRef owner;
    float color[4];
    float width;
    uint32_t width_mode;
    uint32_t depth_mode;
    float visual_near;
    float visual_far;
    uint32_t has_image;
    SpectraDynamicSceneViewportCameraVisualImage image;
} SpectraDynamicSceneViewportCameraVisual;

typedef struct SpectraDynamicSceneTypedSpan {
    uint32_t kind;
    uint32_t item_size;
    const void* data;
    uint64_t count;
} SpectraDynamicSceneTypedSpan;

typedef struct SpectraDynamicSceneDocumentView {
    uint64_t struct_size;
    const char* default_coordinate_system;
    const char* active_camera_name;
    const SpectraDynamicSceneTypedSpan* items;
    uint64_t item_count;
} SpectraDynamicSceneDocumentView;

typedef struct SpectraDynamicSceneFrameInfo {
    double delta_seconds;
    double time_seconds;
    uint64_t frame_index;
} SpectraDynamicSceneFrameInfo;

typedef struct SpectraDynamicSceneFrameView {
    uint64_t struct_size;
    const SpectraDynamicSceneTypedSpan* items;
    uint64_t item_count;
} SpectraDynamicSceneFrameView;

typedef struct SpectraDynamicSceneUpdateInfo {
    uint64_t struct_size;
    double wall_delta_seconds;
    double scene_delta_seconds;
    double time_seconds;
    uint64_t frame_index;
    uint32_t timeline_mode;
    uint32_t timeline_playing;
} SpectraDynamicSceneUpdateInfo;

typedef void SpectraDynamicSceneInstance;

typedef SpectraDynamicSceneResult (*SpectraDynamicSceneCreateFn)(const SpectraDynamicSceneOpenInfo* open_info, SpectraDynamicSceneInstance** instance);
typedef void (*SpectraDynamicSceneDestroyFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneResetFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneUpdateFn)(SpectraDynamicSceneInstance* instance, const SpectraDynamicSceneUpdateInfo* update_info);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneDocumentFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneDocumentView* document);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneFrameFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneFrameInfo frame, SpectraDynamicSceneFrameView* snapshot);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlSceneRevisionFn)(SpectraDynamicSceneInstance* instance, uint64_t* revision);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlActionFn)(SpectraDynamicSceneInstance* instance, const char* action_id, SpectraDynamicSceneOptionSpan options);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlSettingUpdateFn)(SpectraDynamicSceneInstance* instance, const char* key, const char* value);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlSnapshotFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlSnapshotView* snapshot);
typedef const char* (*SpectraDynamicSceneLastErrorFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneGetApiFn)(const char* api_name, uint32_t api_version, const void** api);

typedef struct SpectraDynamicSceneSceneApi {
    uint64_t struct_size;
    const char* base_pbrt_path;
    double frames_per_second;
    SpectraDynamicSceneCreateFn create;
    SpectraDynamicSceneDestroyFn destroy;
    SpectraDynamicSceneResetFn reset;
    SpectraDynamicSceneUpdateFn update;
    SpectraDynamicSceneDocumentFn document;
    SpectraDynamicSceneFrameFn frame;
    SpectraDynamicSceneLastErrorFn last_error;
} SpectraDynamicSceneSceneApi;

typedef struct SpectraDynamicSceneControlsApi {
    uint64_t struct_size;
    SpectraDynamicSceneControlActionSpan control_actions;
    SpectraDynamicSceneControlSceneRevisionFn scene_revision;
    SpectraDynamicSceneControlActionFn control_action;
    SpectraDynamicSceneControlSettingUpdateFn control_setting_update;
    SpectraDynamicSceneControlSnapshotFn control_snapshot;
} SpectraDynamicSceneControlsApi;

typedef struct SpectraDynamicScenePlugin {
    uint32_t abi_version;
    uint64_t struct_size;
    const char* id;
    const char* title;
    const char* controls_panel_title;
    const char* open_action_label;
    const char* open_action_description;
    SpectraDynamicSceneOptionSchemaSpan open_options;
    SpectraDynamicSceneGetApiFn get_api;
} SpectraDynamicScenePlugin;


namespace {
    constexpr char scene_api_name[] = "spectra.dynamic_scene.scene";
    constexpr char controls_api_name[] = "spectra.dynamic_scene.controls";
    constexpr std::uint32_t scene_api_version = 1u;
    constexpr std::uint32_t controls_api_version = 1u;

    struct OptionSchemaViews {
        std::vector<std::vector<SpectraDynamicSceneOptionChoice>> choices{};
        std::vector<SpectraDynamicSceneOptionSchema> schemas{};
    };

    struct DescriptorViews {
        OptionSchemaViews open_options{};
        std::vector<OptionSchemaViews> action_options{};
        std::vector<SpectraDynamicSceneControlAction> control_actions{};
    };

    struct SceneViewCache {
        instant_ngp::spectra_project::Document document{};
        instant_ngp::spectra_project::Frame frame{};
        std::vector<SpectraDynamicSceneMaterial> material_views{};
        std::vector<SpectraDynamicSceneLight> light_views{};
        std::vector<std::vector<SpectraDynamicSceneVolumeChannel>> volume_channel_storage{};
        std::vector<SpectraDynamicSceneVolume> volume_views{};
        std::vector<SpectraDynamicSceneCamera> camera_views{};
        std::vector<SpectraDynamicSceneViewportVoxelGrid> voxel_grid_views{};
        std::vector<SpectraDynamicSceneViewportCameraVisual> camera_visual_views{};
        std::vector<SpectraDynamicSceneTypedSpan> document_items{};
        std::vector<SpectraDynamicSceneTypedSpan> frame_items{};
    };

    struct ProjectStatusCache {
        instant_ngp::spectra_project::ProjectStatus status{};
        std::vector<SpectraDynamicSceneControlMetric> metric_views{};
        std::vector<const char*> enabled_action_views{};
        std::vector<SpectraDynamicSceneControlDisabledAction> disabled_action_views{};
    };

    struct ProjectSettingCache {
        std::vector<instant_ngp::spectra_project::ProjectSetting> settings{};
        std::vector<std::vector<SpectraDynamicSceneOptionChoice>> choice_views{};
        std::vector<SpectraDynamicSceneControlSetting> setting_views{};
    };

    struct ProjectLogCache {
        std::vector<instant_ngp::spectra_project::ProjectLogEntry> logs{};
        std::vector<SpectraDynamicSceneControlLogEntry> log_views{};
    };

    struct ProjectImageCache {
        std::span<const instant_ngp::spectra_project::ProjectImage> images{};
        std::vector<SpectraDynamicSceneControlImage> image_views{};
    };

    struct ProjectScalarSeriesCache {
        std::vector<instant_ngp::spectra_project::ProjectScalarSeries> series{};
        std::vector<std::vector<SpectraDynamicSceneControlScalarSample>> sample_views{};
        std::vector<SpectraDynamicSceneControlScalarSeries> series_views{};
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

    [[nodiscard]] std::vector<instant_ngp::spectra_project::Option> options_from_abi(const SpectraDynamicSceneOptionSpan options, const std::string_view context) {
        if (options.count != 0u && options.data == nullptr) throw std::runtime_error(std::format("{} pointer is null", context));
        if (options.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::format("{} count is too large", context));
        std::vector<instant_ngp::spectra_project::Option> converted{};
        converted.reserve(static_cast<std::size_t>(options.count));
        const std::span<const SpectraDynamicSceneOption> option_span{options.data, static_cast<std::size_t>(options.count)};
        for (const SpectraDynamicSceneOption& option : option_span) {
            converted.push_back(instant_ngp::spectra_project::Option{
                .key = string_from_abi(option.key, std::format("{} key", context), false),
                .value = string_from_abi(option.value, std::format("{} value", context), true),
            });
        }
        return converted;
    }

    [[nodiscard]] std::string host_services_error(const SpectraDynamicSceneHostServices& host_services) {
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

    [[nodiscard]] instant_ngp::spectra_project::GpuDeviceIdentity device_identity_from_abi(const SpectraDynamicSceneGpuDeviceIdentity& identity) {
        instant_ngp::spectra_project::GpuDeviceIdentity converted{
            .vendor_id = identity.vendor_id,
            .device_id = identity.device_id,
            .device_node_mask = identity.device_node_mask,
        };
        for (std::size_t index = 0u; index < converted.device_uuid.size(); ++index) converted.device_uuid[index] = identity.device_uuid[index];
        for (std::size_t index = 0u; index < converted.device_luid.size(); ++index) converted.device_luid[index] = identity.device_luid[index];
        return converted;
    }

    class SpectraHostServicesAdapter final : public instant_ngp::spectra_project::HostServices {
    public:
        explicit SpectraHostServicesAdapter(const SpectraDynamicSceneHostServices* host_services) : host_services(host_services) {
            if (this->host_services == nullptr) throw std::runtime_error("dynamic scene open info host services pointer is null");
            if (this->host_services->struct_size != sizeof(SpectraDynamicSceneHostServices)) throw std::runtime_error("dynamic scene host services ABI size mismatch");
            if (this->host_services->request_viewport_voxel_buffer == nullptr) throw std::runtime_error("dynamic scene host services request_viewport_voxel_buffer function is null");
            if (this->host_services->release_viewport_voxel_buffer == nullptr) throw std::runtime_error("dynamic scene host services release_viewport_voxel_buffer function is null");
            if (this->host_services->request_volume_buffer == nullptr) throw std::runtime_error("dynamic scene host services request_volume_buffer function is null");
            if (this->host_services->release_volume_buffer == nullptr) throw std::runtime_error("dynamic scene host services release_volume_buffer function is null");
            if (this->host_services->last_error == nullptr) throw std::runtime_error("dynamic scene host services last_error function is null");
        }

        SpectraHostServicesAdapter(const SpectraHostServicesAdapter& other) = delete;
        SpectraHostServicesAdapter(SpectraHostServicesAdapter&& other) = delete;
        SpectraHostServicesAdapter& operator=(const SpectraHostServicesAdapter& other) = delete;
        SpectraHostServicesAdapter& operator=(SpectraHostServicesAdapter&& other) = delete;
        ~SpectraHostServicesAdapter() noexcept override = default;

        [[nodiscard]] instant_ngp::spectra_project::ViewportVoxelBufferAllocation request_viewport_voxel_buffer(const std::uint64_t byte_size, const std::string_view debug_name) override {
            const std::string debug_name_text{debug_name};
            SpectraDynamicSceneViewportVoxelBufferRequest request{
                .struct_size = sizeof(SpectraDynamicSceneViewportVoxelBufferRequest),
                .byte_size = byte_size,
                .debug_name = debug_name_text.c_str(),
            };
            SpectraDynamicSceneViewportVoxelBufferAllocation allocation{};
            const SpectraDynamicSceneResult result = this->host_services->request_viewport_voxel_buffer(this->host_services->user_data, &request, &allocation);
            if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*this->host_services));
            if (allocation.struct_size != sizeof(SpectraDynamicSceneViewportVoxelBufferAllocation)) throw std::runtime_error("dynamic scene viewport voxel buffer allocation ABI size mismatch");
            return instant_ngp::spectra_project::ViewportVoxelBufferAllocation{
                .resource_id = allocation.resource_id,
                .byte_size = allocation.byte_size,
                .handle_kind = gpu_handle_kind_from_abi(allocation.handle_kind),
                .handle = allocation.handle,
                .device_identity = device_identity_from_abi(allocation.device_identity),
            };
        }

        void release_viewport_voxel_buffer(const std::uint64_t resource_id) override {
            const SpectraDynamicSceneResult result = this->host_services->release_viewport_voxel_buffer(this->host_services->user_data, resource_id);
            if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*this->host_services));
        }

        [[nodiscard]] instant_ngp::spectra_project::VolumeBufferAllocation request_volume_buffer(const std::uint64_t byte_size, const std::string_view debug_name) override {
            const std::string debug_name_text{debug_name};
            SpectraDynamicSceneVolumeBufferRequest request{
                .struct_size = sizeof(SpectraDynamicSceneVolumeBufferRequest),
                .byte_size = byte_size,
                .debug_name = debug_name_text.c_str(),
            };
            SpectraDynamicSceneVolumeBufferAllocation allocation{};
            const SpectraDynamicSceneResult result = this->host_services->request_volume_buffer(this->host_services->user_data, &request, &allocation);
            if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*this->host_services));
            if (allocation.struct_size != sizeof(SpectraDynamicSceneVolumeBufferAllocation)) throw std::runtime_error("dynamic scene volume buffer allocation ABI size mismatch");
            return instant_ngp::spectra_project::VolumeBufferAllocation{
                .resource_id = allocation.resource_id,
                .byte_size = allocation.byte_size,
                .handle_kind = gpu_handle_kind_from_abi(allocation.handle_kind),
                .handle = allocation.handle,
                .device_identity = device_identity_from_abi(allocation.device_identity),
            };
        }

        void release_volume_buffer(const std::uint64_t resource_id) override {
            const SpectraDynamicSceneResult result = this->host_services->release_volume_buffer(this->host_services->user_data, resource_id);
            if (result != SPECTRA_DYNAMIC_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*this->host_services));
        }

    private:
        const SpectraDynamicSceneHostServices* host_services{};
    };

    [[nodiscard]] OptionSchemaViews make_option_schema_views(const std::vector<instant_ngp::spectra_project::OptionSchema>& schemas) {
        OptionSchemaViews views{};
        views.choices.resize(schemas.size());
        views.schemas.reserve(schemas.size());
        for (std::size_t index = 0u; index < schemas.size(); ++index) {
            const instant_ngp::spectra_project::OptionSchema& schema = schemas[index];
            views.choices[index].reserve(schema.choices.size());
            for (const instant_ngp::spectra_project::OptionChoice& choice : schema.choices) views.choices[index].push_back(SpectraDynamicSceneOptionChoice{.value = choice.value.c_str(), .label = choice.label.c_str()});
            views.schemas.push_back(SpectraDynamicSceneOptionSchema{
                .key = schema.key.c_str(),
                .label = schema.label.c_str(),
                .description = schema.description.c_str(),
                .kind = static_cast<std::uint32_t>(schema.kind),
                .required = schema.required ? 1u : 0u,
                .default_value = schema.default_value.c_str(),
                .group = schema.group.c_str(),
                .advanced = schema.advanced ? 1u : 0u,
                .priority = schema.priority,
                .choices = SpectraDynamicSceneOptionChoiceSpan{.data = views.choices[index].empty() ? nullptr : views.choices[index].data(), .count = static_cast<std::uint64_t>(views.choices[index].size())},
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
            views.control_actions.push_back(SpectraDynamicSceneControlAction{
                .id = action.id.c_str(),
                .label = action.label.c_str(),
                .description = action.description.c_str(),
                .group = action.group,
                .priority = action.priority,
                .style = action.style,
                .options = SpectraDynamicSceneOptionSchemaSpan{.data = action_options.schemas.empty() ? nullptr : action_options.schemas.data(), .count = static_cast<std::uint64_t>(action_options.schemas.size())},
            });
        }
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
    void append_scene_item(std::vector<SpectraDynamicSceneTypedSpan>& items, const std::uint32_t kind, const std::vector<Value>& values) {
        if (values.empty()) return;
        items.push_back(SpectraDynamicSceneTypedSpan{
            .kind = kind,
            .item_size = static_cast<std::uint32_t>(sizeof(Value)),
            .data = values.data(),
            .count = static_cast<std::uint64_t>(values.size()),
        });
    }

    [[nodiscard]] SpectraDynamicSceneTransform make_transform_view(const instant_ngp::spectra_project::Transform& transform) {
        SpectraDynamicSceneTransform view{};
        copy_array(view.position, transform.position);
        copy_array(view.rotation, transform.rotation);
        copy_array(view.scale, transform.scale);
        return view;
    }

    [[nodiscard]] SpectraDynamicSceneEntityRef make_entity_ref_view(const instant_ngp::spectra_project::SceneEntityRef& ref) {
        return SpectraDynamicSceneEntityRef{
            .kind = static_cast<std::uint32_t>(ref.kind),
            .name = ref.name.c_str(),
        };
    }

    [[nodiscard]] SpectraDynamicSceneMaterial make_material_view(const instant_ngp::spectra_project::Material& material) {
        SpectraDynamicSceneMaterial view{
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

    [[nodiscard]] SpectraDynamicSceneLight make_light_view(const instant_ngp::spectra_project::Light& light) {
        SpectraDynamicSceneLight view{
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
                SpectraDynamicSceneVolumeChannel channel_view{
                    .name = channel.name.c_str(),
                    .values = SpectraDynamicSceneFloatSpan{.data = channel.values.empty() ? nullptr : channel.values.data(), .count = static_cast<std::uint64_t>(channel.values.size())},
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

            SpectraDynamicSceneVolume volume_view{
                .name = volume.name.c_str(),
                .channels = SpectraDynamicSceneVolumeChannelSpan{.data = cache.volume_channel_storage[volume_index].empty() ? nullptr : cache.volume_channel_storage[volume_index].data(), .count = static_cast<std::uint64_t>(cache.volume_channel_storage[volume_index].size())},
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

    [[nodiscard]] SpectraDynamicSceneCamera make_camera_view(const instant_ngp::spectra_project::Camera& camera) {
        SpectraDynamicSceneCamera view{
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

    [[nodiscard]] SpectraDynamicSceneViewportCameraVisual make_camera_visual_view(const instant_ngp::spectra_project::ViewportCameraVisual& visual) {
        SpectraDynamicSceneViewportCameraVisual view{
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
            view.image = SpectraDynamicSceneViewportCameraVisualImage{
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

    [[nodiscard]] SpectraDynamicSceneViewportVoxelGrid make_voxel_grid_view(const instant_ngp::spectra_project::ViewportVoxelGrid& grid) {
        SpectraDynamicSceneViewportVoxelGrid view{
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

    [[nodiscard]] SpectraDynamicSceneDocumentView make_document_view(SceneViewCache& cache) {
        make_material_views(cache, cache.document.materials);
        make_light_views(cache, cache.document.lights);
        make_volume_views(cache, cache.document.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.document.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.document.cameras) cache.camera_views.push_back(make_camera_view(camera));
        make_voxel_grid_views(cache, cache.document.debug_attachments.viewport_voxel_grids);
        make_camera_visual_views(cache, cache.document.debug_attachments.viewport_camera_visuals);
        cache.document_items.clear();
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_MATERIAL, cache.material_views);
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_LIGHT, cache.light_views);
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_CAMERA, cache.camera_views);
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_VOLUME, cache.volume_views);
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_VOXEL_GRID, cache.voxel_grid_views);
        append_scene_item(cache.document_items, SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_CAMERA_VISUAL, cache.camera_visual_views);
        return SpectraDynamicSceneDocumentView{
            .struct_size = sizeof(SpectraDynamicSceneDocumentView),
            .default_coordinate_system = cache.document.default_coordinate_system.c_str(),
            .active_camera_name = cache.document.active_camera_name.c_str(),
            .items = cache.document_items.empty() ? nullptr : cache.document_items.data(),
            .item_count = static_cast<std::uint64_t>(cache.document_items.size()),
        };
    }

    [[nodiscard]] SpectraDynamicSceneFrameView make_frame_view(SceneViewCache& cache) {
        make_volume_views(cache, cache.frame.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.frame.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.frame.cameras) cache.camera_views.push_back(make_camera_view(camera));
        make_voxel_grid_views(cache, cache.frame.debug_attachments.viewport_voxel_grids);
        make_camera_visual_views(cache, cache.frame.debug_attachments.viewport_camera_visuals);
        cache.frame_items.clear();
        append_scene_item(cache.frame_items, SPECTRA_DYNAMIC_SCENE_ITEM_VOLUME, cache.volume_views);
        append_scene_item(cache.frame_items, SPECTRA_DYNAMIC_SCENE_ITEM_CAMERA, cache.camera_views);
        append_scene_item(cache.frame_items, SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_VOXEL_GRID, cache.voxel_grid_views);
        append_scene_item(cache.frame_items, SPECTRA_DYNAMIC_SCENE_ITEM_VIEWPORT_CAMERA_VISUAL, cache.camera_visual_views);
        return SpectraDynamicSceneFrameView{
            .struct_size = sizeof(SpectraDynamicSceneFrameView),
            .items = cache.frame_items.empty() ? nullptr : cache.frame_items.data(),
            .item_count = static_cast<std::uint64_t>(cache.frame_items.size()),
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlStatusView make_status_view(ProjectStatusCache& cache) {
        cache.metric_views.clear();
        cache.enabled_action_views.clear();
        cache.disabled_action_views.clear();
        cache.metric_views.reserve(cache.status.metrics.size());
        cache.enabled_action_views.reserve(cache.status.enabled_action_ids.size());
        cache.disabled_action_views.reserve(cache.status.disabled_actions.size());
        for (const instant_ngp::spectra_project::ProjectMetric& metric : cache.status.metrics) {
            cache.metric_views.push_back(SpectraDynamicSceneControlMetric{
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
        for (const std::string& action_id : cache.status.enabled_action_ids) cache.enabled_action_views.push_back(action_id.c_str());
        for (const instant_ngp::spectra_project::ProjectDisabledAction& disabled_action : cache.status.disabled_actions) {
            cache.disabled_action_views.push_back(SpectraDynamicSceneControlDisabledAction{
                .action_id = disabled_action.action_id.c_str(),
                .reason = disabled_action.reason.c_str(),
            });
        }
        return SpectraDynamicSceneControlStatusView{
            .struct_size = sizeof(SpectraDynamicSceneControlStatusView),
            .phase = cache.status.phase.c_str(),
            .headline = cache.status.headline.c_str(),
            .detail = cache.status.detail.c_str(),
            .metrics = SpectraDynamicSceneControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
            .enabled_action_ids = cache.enabled_action_views.empty() ? nullptr : cache.enabled_action_views.data(),
            .enabled_action_id_count = static_cast<std::uint64_t>(cache.enabled_action_views.size()),
            .disabled_actions = SpectraDynamicSceneControlDisabledActionSpan{.data = cache.disabled_action_views.empty() ? nullptr : cache.disabled_action_views.data(), .count = static_cast<std::uint64_t>(cache.disabled_action_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlSettingView make_setting_view(ProjectSettingCache& cache) {
        cache.choice_views.clear();
        cache.setting_views.clear();
        cache.choice_views.resize(cache.settings.size());
        cache.setting_views.reserve(cache.settings.size());
        for (std::size_t setting_index = 0u; setting_index < cache.settings.size(); ++setting_index) {
            const instant_ngp::spectra_project::ProjectSetting& setting = cache.settings[setting_index];
            std::vector<SpectraDynamicSceneOptionChoice>& choices = cache.choice_views[setting_index];
            choices.reserve(setting.choices.size());
            for (const instant_ngp::spectra_project::OptionChoice& choice : setting.choices) choices.push_back(SpectraDynamicSceneOptionChoice{.value = choice.value.c_str(), .label = choice.label.c_str()});
            cache.setting_views.push_back(SpectraDynamicSceneControlSetting{
                .key = setting.key.c_str(),
                .label = setting.label.c_str(),
                .description = setting.description.c_str(),
                .kind = static_cast<std::uint32_t>(setting.kind),
                .value = setting.value.c_str(),
                .group = setting.group.c_str(),
                .advanced = setting.advanced ? 1u : 0u,
                .priority = setting.priority,
                .choices = SpectraDynamicSceneOptionChoiceSpan{.data = choices.empty() ? nullptr : choices.data(), .count = static_cast<std::uint64_t>(choices.size())},
            });
        }
        return SpectraDynamicSceneControlSettingView{
            .struct_size = sizeof(SpectraDynamicSceneControlSettingView),
            .settings = SpectraDynamicSceneControlSettingSpan{.data = cache.setting_views.empty() ? nullptr : cache.setting_views.data(), .count = static_cast<std::uint64_t>(cache.setting_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlLogView make_log_view(ProjectLogCache& cache) {
        cache.log_views.clear();
        cache.log_views.reserve(cache.logs.size());
        for (const instant_ngp::spectra_project::ProjectLogEntry& log : cache.logs) {
            cache.log_views.push_back(SpectraDynamicSceneControlLogEntry{
                .sequence = log.sequence,
                .level = log.level.c_str(),
                .message = log.message.c_str(),
            });
        }
        return SpectraDynamicSceneControlLogView{
            .struct_size = sizeof(SpectraDynamicSceneControlLogView),
            .entries = SpectraDynamicSceneControlLogEntrySpan{.data = cache.log_views.empty() ? nullptr : cache.log_views.data(), .count = static_cast<std::uint64_t>(cache.log_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlImageView make_image_view(ProjectImageCache& cache) {
        cache.image_views.clear();
        cache.image_views.reserve(cache.images.size());
        for (const instant_ngp::spectra_project::ProjectImage& image : cache.images) {
            cache.image_views.push_back(SpectraDynamicSceneControlImage{
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
        return SpectraDynamicSceneControlImageView{
            .struct_size = sizeof(SpectraDynamicSceneControlImageView),
            .images = SpectraDynamicSceneControlImageSpan{.data = cache.image_views.empty() ? nullptr : cache.image_views.data(), .count = static_cast<std::uint64_t>(cache.image_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlScalarSeriesView make_scalar_series_view(ProjectScalarSeriesCache& cache) {
        cache.sample_views.clear();
        cache.series_views.clear();
        cache.sample_views.resize(cache.series.size());
        cache.series_views.reserve(cache.series.size());
        for (std::size_t series_index = 0u; series_index < cache.series.size(); ++series_index) {
            const instant_ngp::spectra_project::ProjectScalarSeries& series = cache.series[series_index];
            std::vector<SpectraDynamicSceneControlScalarSample>& samples = cache.sample_views[series_index];
            samples.reserve(series.samples.size());
            for (const instant_ngp::spectra_project::ProjectScalarSample& sample : series.samples) {
                samples.push_back(SpectraDynamicSceneControlScalarSample{
                    .step = sample.step,
                    .time_seconds = sample.time_seconds,
                    .value = sample.value,
                });
            }
            SpectraDynamicSceneControlScalarSeries view{
                .id = series.id.c_str(),
                .label = series.label.c_str(),
                .description = series.description.c_str(),
                .unit = series.unit.c_str(),
                .color = {},
                .group = series.group,
                .priority = series.priority,
                .revision = series.revision,
                .samples = SpectraDynamicSceneControlScalarSampleSpan{.data = samples.empty() ? nullptr : samples.data(), .count = static_cast<std::uint64_t>(samples.size())},
            };
            copy_array(view.color, series.color);
            cache.series_views.push_back(view);
        }
        return SpectraDynamicSceneControlScalarSeriesView{
            .struct_size = sizeof(SpectraDynamicSceneControlScalarSeriesView),
            .series = SpectraDynamicSceneControlScalarSeriesSpan{.data = cache.series_views.empty() ? nullptr : cache.series_views.data(), .count = static_cast<std::uint64_t>(cache.series_views.size())},
        };
    }

    [[nodiscard]] PluginInstance& checked_instance(SpectraDynamicSceneInstance* instance, const std::string_view action) {
        if (instance == nullptr) throw std::runtime_error(std::format("{} instance pointer is null", action));
        return *reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraDynamicSceneResult scene_create(const SpectraDynamicSceneOpenInfo* open_info, SpectraDynamicSceneInstance** instance) noexcept {
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
            if (open_info->struct_size != sizeof(SpectraDynamicSceneOpenInfo)) throw std::runtime_error("dynamic scene open info ABI size mismatch");
            static_cast<void>(string_from_abi(open_info->plugin_path, "dynamic scene plugin path", false));
            std::vector<instant_ngp::spectra_project::Option> options = options_from_abi(open_info->options, "dynamic scene open options");
            std::unique_ptr<PluginInstance> created = std::make_unique<PluginInstance>();
            created->project = instant_ngp::spectra_project::InstantNgpSpectraProject::open(options, std::make_shared<SpectraHostServicesAdapter>(open_info->host_services));
            *instance = reinterpret_cast<SpectraDynamicSceneInstance*>(created.release());
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    void scene_destroy(SpectraDynamicSceneInstance* instance) noexcept {
        delete reinterpret_cast<PluginInstance*>(instance);
    }

    [[nodiscard]] SpectraDynamicSceneResult scene_reset(SpectraDynamicSceneInstance* instance) noexcept {
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

    [[nodiscard]] SpectraDynamicSceneResult scene_document(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneDocumentView* document) noexcept {
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

    [[nodiscard]] SpectraDynamicSceneResult scene_frame(SpectraDynamicSceneInstance* instance, const SpectraDynamicSceneFrameInfo frame, SpectraDynamicSceneFrameView* snapshot) noexcept {
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

    [[nodiscard]] const char* last_error(SpectraDynamicSceneInstance* instance) noexcept {
        if (instance == nullptr) return global_error.c_str();
        return reinterpret_cast<PluginInstance*>(instance)->last_error.c_str();
    }

    [[nodiscard]] SpectraDynamicSceneResult scene_update(SpectraDynamicSceneInstance* instance, const SpectraDynamicSceneUpdateInfo* update_info) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "scene_update");
            if (update_info == nullptr) throw std::runtime_error("scene_update info pointer is null");
            if (update_info->struct_size != sizeof(SpectraDynamicSceneUpdateInfo)) throw std::runtime_error("scene_update info ABI size mismatch");
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

    [[nodiscard]] SpectraDynamicSceneResult scene_revision(SpectraDynamicSceneInstance* instance, std::uint64_t* revision) noexcept {
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

    [[nodiscard]] SpectraDynamicSceneResult control_action(SpectraDynamicSceneInstance* instance, const char* action_id, const SpectraDynamicSceneOptionSpan options) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_action");
            plugin_instance.last_error.clear();
            const std::string action = string_from_abi(action_id, "project action id", false);
            const std::vector<instant_ngp::spectra_project::Option> converted_options = options_from_abi(options, "project action options");
            plugin_instance.project.execute_action(action, converted_options);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraDynamicSceneResult control_setting_update(SpectraDynamicSceneInstance* instance, const char* key, const char* value) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_setting_update");
            plugin_instance.last_error.clear();
            const std::string setting_key = string_from_abi(key, "project setting key", false);
            const std::string setting_value = string_from_abi(value, "project setting value", true);
            plugin_instance.project.update_setting(setting_key, setting_value);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraDynamicSceneResult control_snapshot(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlSnapshotView* snapshot) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_snapshot");
            if (snapshot == nullptr) throw std::runtime_error("control_snapshot output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.setting_cache.settings = plugin_instance.project.settings();
            plugin_instance.status_cache.status = plugin_instance.project.status();
            plugin_instance.log_cache.logs = plugin_instance.project.logs();
            plugin_instance.image_cache.images = plugin_instance.project.images();
            plugin_instance.scalar_series_cache.series = plugin_instance.project.scalar_series();
            *snapshot = SpectraDynamicSceneControlSnapshotView{
                .struct_size = sizeof(SpectraDynamicSceneControlSnapshotView),
                .settings = make_setting_view(plugin_instance.setting_cache),
                .status = make_status_view(plugin_instance.status_cache),
                .logs = make_log_view(plugin_instance.log_cache),
                .images = make_image_view(plugin_instance.image_cache),
                .scalar_series = make_scalar_series_view(plugin_instance.scalar_series_cache),
            };
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const SpectraDynamicSceneSceneApi& scene_api() {
        static const SpectraDynamicSceneSceneApi api{
            .struct_size = sizeof(SpectraDynamicSceneSceneApi),
            .base_pbrt_path = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor().base_pbrt_path.c_str(),
            .frames_per_second = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor().frames_per_second,
            .create = scene_create,
            .destroy = scene_destroy,
            .reset = scene_reset,
            .update = scene_update,
            .document = scene_document,
            .frame = scene_frame,
            .last_error = last_error,
        };
        return api;
    }

    [[nodiscard]] const SpectraDynamicSceneControlsApi& controls_api() {
        const DescriptorViews& views = descriptor_views();
        static const SpectraDynamicSceneControlsApi api{
            .struct_size = sizeof(SpectraDynamicSceneControlsApi),
            .control_actions = SpectraDynamicSceneControlActionSpan{.data = views.control_actions.empty() ? nullptr : views.control_actions.data(), .count = static_cast<std::uint64_t>(views.control_actions.size())},
            .scene_revision = scene_revision,
            .control_action = control_action,
            .control_setting_update = control_setting_update,
            .control_snapshot = control_snapshot,
        };
        return api;
    }

    [[nodiscard]] SpectraDynamicSceneResult get_api(const char* api_name, const std::uint32_t api_version, const void** api) noexcept {
        try {
            if (api == nullptr) {
                global_error = "get_api output pointer is null";
                return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
            }
            *api = nullptr;
            const std::string name = string_from_abi(api_name, "api name", false);
            if (name == scene_api_name && api_version == scene_api_version) *api = &scene_api();
            else if (name == controls_api_name && api_version == controls_api_version) *api = &controls_api();
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] const SpectraDynamicScenePlugin& plugin() {
        const instant_ngp::spectra_project::Descriptor& descriptor = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor();
        const DescriptorViews& views = descriptor_views();
        static const SpectraDynamicScenePlugin value{
            .abi_version = SPECTRA_DYNAMIC_SCENE_ABI_VERSION,
            .struct_size = sizeof(SpectraDynamicScenePlugin),
            .id = descriptor.id.c_str(),
            .title = descriptor.title.c_str(),
            .controls_panel_title = descriptor.controls_panel_title.c_str(),
            .open_action_label = descriptor.open_action_label.c_str(),
            .open_action_description = descriptor.open_action_description.c_str(),
            .open_options = SpectraDynamicSceneOptionSchemaSpan{.data = views.open_options.schemas.empty() ? nullptr : views.open_options.schemas.data(), .count = static_cast<std::uint64_t>(views.open_options.schemas.size())},
            .get_api = get_api,
        };
        return value;
    }
}

extern "C" SPECTRA_DYNAMIC_SCENE_EXPORT const SpectraDynamicScenePlugin* spectra_dynamic_scene_plugin(void) {
    return &plugin();
}
