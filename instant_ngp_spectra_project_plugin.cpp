#include <stdint.h>

#if defined(_WIN32)
#define SPECTRA_DYNAMIC_SCENE_EXPORT __declspec(dllexport)
#else
#define SPECTRA_DYNAMIC_SCENE_EXPORT __attribute__((visibility("default")))
#endif

#define SPECTRA_DYNAMIC_SCENE_ABI_VERSION 21u

typedef struct SpectraDynamicSceneString {
    const char* data;
    uint64_t size;
} SpectraDynamicSceneString;

typedef struct SpectraDynamicSceneOption {
    SpectraDynamicSceneString key;
    SpectraDynamicSceneString value;
} SpectraDynamicSceneOption;

typedef struct SpectraDynamicSceneOptionSpan {
    const SpectraDynamicSceneOption* data;
    uint64_t count;
} SpectraDynamicSceneOptionSpan;

typedef enum SpectraDynamicSceneOpenOptionKind {
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_TEXT = 0,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_DIRECTORY_PATH = 1,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_FILE_PATH = 2,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_CHOICE = 3,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_BOOL = 4,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_FLOAT = 5,
    SPECTRA_DYNAMIC_SCENE_OPEN_OPTION_UNSIGNED_INTEGER = 6,
} SpectraDynamicSceneOpenOptionKind;

typedef struct SpectraDynamicSceneOpenOptionChoice {
    SpectraDynamicSceneString value;
    SpectraDynamicSceneString label;
} SpectraDynamicSceneOpenOptionChoice;

typedef struct SpectraDynamicSceneOpenOptionChoiceSpan {
    const SpectraDynamicSceneOpenOptionChoice* data;
    uint64_t count;
} SpectraDynamicSceneOpenOptionChoiceSpan;

typedef struct SpectraDynamicSceneOpenOptionSchema {
    SpectraDynamicSceneString key;
    SpectraDynamicSceneString label;
    SpectraDynamicSceneString description;
    uint32_t kind;
    uint32_t required;
    SpectraDynamicSceneString default_value;
    SpectraDynamicSceneOpenOptionChoiceSpan choices;
} SpectraDynamicSceneOpenOptionSchema;

typedef struct SpectraDynamicSceneOpenOptionSchemaSpan {
    const SpectraDynamicSceneOpenOptionSchema* data;
    uint64_t count;
} SpectraDynamicSceneOpenOptionSchemaSpan;

typedef struct SpectraDynamicSceneStringSpan {
    const SpectraDynamicSceneString* data;
    uint64_t count;
} SpectraDynamicSceneStringSpan;

typedef struct SpectraDynamicSceneControlAction {
    SpectraDynamicSceneString id;
    SpectraDynamicSceneString label;
    SpectraDynamicSceneString description;
    SpectraDynamicSceneOpenOptionSchemaSpan options;
} SpectraDynamicSceneControlAction;

typedef struct SpectraDynamicSceneControlActionSpan {
    const SpectraDynamicSceneControlAction* data;
    uint64_t count;
} SpectraDynamicSceneControlActionSpan;

typedef struct SpectraDynamicSceneControlMetric {
    SpectraDynamicSceneString key;
    SpectraDynamicSceneString label;
    SpectraDynamicSceneString value;
} SpectraDynamicSceneControlMetric;

typedef struct SpectraDynamicSceneControlMetricSpan {
    const SpectraDynamicSceneControlMetric* data;
    uint64_t count;
} SpectraDynamicSceneControlMetricSpan;

typedef struct SpectraDynamicSceneControlDisabledAction {
    SpectraDynamicSceneString action_id;
    SpectraDynamicSceneString reason;
} SpectraDynamicSceneControlDisabledAction;

typedef struct SpectraDynamicSceneControlDisabledActionSpan {
    const SpectraDynamicSceneControlDisabledAction* data;
    uint64_t count;
} SpectraDynamicSceneControlDisabledActionSpan;

typedef struct SpectraDynamicSceneControlStatusView {
    uint64_t struct_size;
    SpectraDynamicSceneString phase;
    SpectraDynamicSceneString headline;
    SpectraDynamicSceneString detail;
    SpectraDynamicSceneControlMetricSpan metrics;
    SpectraDynamicSceneStringSpan enabled_action_ids;
    SpectraDynamicSceneControlDisabledActionSpan disabled_actions;
} SpectraDynamicSceneControlStatusView;

typedef struct SpectraDynamicSceneControlLogEntry {
    uint64_t sequence;
    SpectraDynamicSceneString level;
    SpectraDynamicSceneString message;
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
    SpectraDynamicSceneString id;
    SpectraDynamicSceneString label;
    SpectraDynamicSceneString description;
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
    SpectraDynamicSceneString id;
    SpectraDynamicSceneString label;
    SpectraDynamicSceneString description;
    SpectraDynamicSceneString unit;
    float color[4];
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

typedef enum SpectraDynamicSceneResult {
    SPECTRA_DYNAMIC_SCENE_RESULT_OK = 0,
    SPECTRA_DYNAMIC_SCENE_RESULT_ERROR = 1,
} SpectraDynamicSceneResult;

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
    SpectraDynamicSceneString debug_name;
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
    SpectraDynamicSceneString debug_name;
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
typedef SpectraDynamicSceneString (*SpectraDynamicSceneHostLastErrorFn)(void* user_data);

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
    SpectraDynamicSceneString plugin_path;
    SpectraDynamicSceneOptionSpan options;
    const SpectraDynamicSceneHostServices* host_services;
} SpectraDynamicSceneOpenInfo;

typedef struct SpectraDynamicSceneTransform {
    float position[3];
    float rotation[4];
    float scale[3];
} SpectraDynamicSceneTransform;

typedef struct SpectraDynamicSceneMaterial {
    SpectraDynamicSceneString name;
    SpectraDynamicSceneString model;
    SpectraDynamicSceneString alpha_mode;
    float base_color[4];
    float emission_color[3];
    float emission_strength;
    float roughness;
    float metallic;
    float alpha_cutoff;
    float volume_density_scale;
    float volume_temperature_scale;
} SpectraDynamicSceneMaterial;

typedef struct SpectraDynamicSceneMaterialSpan {
    const SpectraDynamicSceneMaterial* data;
    uint64_t count;
} SpectraDynamicSceneMaterialSpan;

typedef struct SpectraDynamicSceneLight {
    SpectraDynamicSceneString name;
    SpectraDynamicSceneString kind;
    SpectraDynamicSceneTransform transform;
    float color[3];
    float intensity;
    float cone_angle_degrees;
} SpectraDynamicSceneLight;

typedef struct SpectraDynamicSceneLightSpan {
    const SpectraDynamicSceneLight* data;
    uint64_t count;
} SpectraDynamicSceneLightSpan;

typedef struct SpectraDynamicSceneCamera {
    SpectraDynamicSceneString name;
    SpectraDynamicSceneString local_coordinate_system;
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

typedef struct SpectraDynamicSceneCameraSpan {
    const SpectraDynamicSceneCamera* data;
    uint64_t count;
} SpectraDynamicSceneCameraSpan;

typedef struct SpectraDynamicSceneMeshVertex {
    float position[3];
    float normal[3];
} SpectraDynamicSceneMeshVertex;

typedef struct SpectraDynamicSceneMeshVertexSpan {
    const SpectraDynamicSceneMeshVertex* data;
    uint64_t count;
} SpectraDynamicSceneMeshVertexSpan;

typedef struct SpectraDynamicSceneUInt32Span {
    const uint32_t* data;
    uint64_t count;
} SpectraDynamicSceneUInt32Span;

typedef struct SpectraDynamicSceneMesh {
    SpectraDynamicSceneString name;
    SpectraDynamicSceneMeshVertexSpan vertices;
    SpectraDynamicSceneUInt32Span indices;
    SpectraDynamicSceneString material_name;
    SpectraDynamicSceneTransform transform;
} SpectraDynamicSceneMesh;

typedef struct SpectraDynamicSceneMeshSpan {
    const SpectraDynamicSceneMesh* data;
    uint64_t count;
} SpectraDynamicSceneMeshSpan;

typedef struct SpectraDynamicSceneSphere {
    SpectraDynamicSceneString name;
    float radius;
    SpectraDynamicSceneString material_name;
    SpectraDynamicSceneTransform transform;
} SpectraDynamicSceneSphere;

typedef struct SpectraDynamicSceneSphereSpan {
    const SpectraDynamicSceneSphere* data;
    uint64_t count;
} SpectraDynamicSceneSphereSpan;

typedef struct SpectraDynamicScenePoint {
    float position[3];
    float normal[3];
    float color[4];
    float radius;
} SpectraDynamicScenePoint;

typedef struct SpectraDynamicScenePointSpan {
    const SpectraDynamicScenePoint* data;
    uint64_t count;
} SpectraDynamicScenePointSpan;

typedef struct SpectraDynamicScenePointCloud {
    SpectraDynamicSceneString name;
    SpectraDynamicScenePointSpan points;
    SpectraDynamicSceneString material_name;
    SpectraDynamicSceneTransform transform;
} SpectraDynamicScenePointCloud;

typedef struct SpectraDynamicScenePointCloudSpan {
    const SpectraDynamicScenePointCloud* data;
    uint64_t count;
} SpectraDynamicScenePointCloudSpan;

typedef struct SpectraDynamicSceneFloatSpan {
    const float* data;
    uint64_t count;
} SpectraDynamicSceneFloatSpan;

typedef struct SpectraDynamicSceneVolumeChannel {
    SpectraDynamicSceneString name;
    uint32_t dimensions[3];
    SpectraDynamicSceneFloatSpan values;
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
    SpectraDynamicSceneString name;
    uint32_t dimensions[3];
    float origin[3];
    float voxel_size[3];
    SpectraDynamicSceneVolumeChannelSpan channels;
    SpectraDynamicSceneString material_name;
} SpectraDynamicSceneVolume;

typedef struct SpectraDynamicSceneVolumeSpan {
    const SpectraDynamicSceneVolume* data;
    uint64_t count;
} SpectraDynamicSceneVolumeSpan;

typedef struct SpectraDynamicSceneEntityRef {
    uint32_t kind;
    SpectraDynamicSceneString name;
} SpectraDynamicSceneEntityRef;

typedef struct SpectraDynamicSceneViewportSegment {
    float start[3];
    float end[3];
} SpectraDynamicSceneViewportSegment;

typedef struct SpectraDynamicSceneViewportSegmentSpan {
    const SpectraDynamicSceneViewportSegment* data;
    uint64_t count;
} SpectraDynamicSceneViewportSegmentSpan;

typedef struct SpectraDynamicSceneColor {
    float value[4];
} SpectraDynamicSceneColor;

typedef struct SpectraDynamicSceneColorSpan {
    const SpectraDynamicSceneColor* data;
    uint64_t count;
} SpectraDynamicSceneColorSpan;

typedef struct SpectraDynamicSceneViewportSegmentSet {
    SpectraDynamicSceneString name;
    SpectraDynamicSceneEntityRef owner;
    SpectraDynamicSceneViewportSegmentSpan segments;
    SpectraDynamicSceneColorSpan colors;
    SpectraDynamicSceneFloatSpan widths;
    float width;
    uint32_t width_mode;
    uint32_t depth_mode;
    SpectraDynamicSceneTransform transform;
} SpectraDynamicSceneViewportSegmentSet;

typedef struct SpectraDynamicSceneViewportSegmentSetSpan {
    const SpectraDynamicSceneViewportSegmentSet* data;
    uint64_t count;
} SpectraDynamicSceneViewportSegmentSetSpan;

typedef struct SpectraDynamicSceneViewportVoxelGrid {
    SpectraDynamicSceneString name;
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

typedef struct SpectraDynamicSceneViewportVoxelGridSpan {
    const SpectraDynamicSceneViewportVoxelGrid* data;
    uint64_t count;
} SpectraDynamicSceneViewportVoxelGridSpan;

typedef struct SpectraDynamicSceneViewportCameraVisualImage {
    const uint8_t* rgba8;
    uint64_t rgba8_size;
    uint64_t revision;
    uint32_t width;
    uint32_t height;
    float tint[4];
} SpectraDynamicSceneViewportCameraVisualImage;

typedef struct SpectraDynamicSceneViewportCameraVisual {
    SpectraDynamicSceneString name;
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

typedef struct SpectraDynamicSceneViewportCameraVisualSpan {
    const SpectraDynamicSceneViewportCameraVisual* data;
    uint64_t count;
} SpectraDynamicSceneViewportCameraVisualSpan;

typedef struct SpectraDynamicSceneDebugAttachmentSet {
    SpectraDynamicSceneViewportSegmentSetSpan viewport_segment_sets;
    SpectraDynamicSceneViewportVoxelGridSpan viewport_voxel_grids;
    SpectraDynamicSceneViewportCameraVisualSpan viewport_camera_visuals;
} SpectraDynamicSceneDebugAttachmentSet;

typedef struct SpectraDynamicSceneDocumentView {
    uint64_t struct_size;
    SpectraDynamicSceneString default_coordinate_system;
    SpectraDynamicSceneString active_camera_name;
    SpectraDynamicSceneCameraSpan cameras;
    SpectraDynamicSceneMaterialSpan materials;
    SpectraDynamicSceneLightSpan lights;
    SpectraDynamicSceneMeshSpan meshes;
    SpectraDynamicSceneSphereSpan spheres;
    SpectraDynamicScenePointCloudSpan point_clouds;
    SpectraDynamicSceneVolumeSpan volumes;
    SpectraDynamicSceneDebugAttachmentSet debug_attachments;
} SpectraDynamicSceneDocumentView;

typedef struct SpectraDynamicSceneFrameInfo {
    double delta_seconds;
    double time_seconds;
    uint64_t frame_index;
} SpectraDynamicSceneFrameInfo;

typedef struct SpectraDynamicSceneFrameView {
    uint64_t struct_size;
    SpectraDynamicSceneMeshSpan meshes;
    SpectraDynamicSceneSphereSpan spheres;
    SpectraDynamicScenePointCloudSpan point_clouds;
    SpectraDynamicSceneVolumeSpan volumes;
    SpectraDynamicSceneCameraSpan cameras;
    SpectraDynamicSceneDebugAttachmentSet debug_attachments;
} SpectraDynamicSceneFrameView;

typedef void SpectraDynamicSceneInstance;

typedef SpectraDynamicSceneResult (*SpectraDynamicSceneCreateFn)(const SpectraDynamicSceneOpenInfo* open_info, SpectraDynamicSceneInstance** instance);
typedef void (*SpectraDynamicSceneDestroyFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneResetFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneStepFn)(SpectraDynamicSceneInstance* instance, float delta_seconds);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneDocumentFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneDocumentView* document);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneFrameFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneFrameInfo frame, SpectraDynamicSceneFrameView* snapshot);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlUpdateFn)(SpectraDynamicSceneInstance* instance, float delta_seconds);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlSceneRevisionFn)(SpectraDynamicSceneInstance* instance, uint64_t* revision);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlActionFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneString action_id, SpectraDynamicSceneOptionSpan options);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlStatusFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlStatusView* status);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlLogsFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlLogView* logs);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlImagesFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlImageView* images);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneControlScalarSeriesFn)(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlScalarSeriesView* series);
typedef SpectraDynamicSceneString (*SpectraDynamicSceneLastErrorFn)(SpectraDynamicSceneInstance* instance);
typedef SpectraDynamicSceneResult (*SpectraDynamicSceneGetApiFn)(SpectraDynamicSceneString api_name, uint32_t api_version, const void** api);

typedef struct SpectraDynamicSceneSceneApi {
    uint64_t struct_size;
    SpectraDynamicSceneString base_pbrt_path;
    double frames_per_second;
    SpectraDynamicSceneCreateFn create;
    SpectraDynamicSceneDestroyFn destroy;
    SpectraDynamicSceneResetFn reset;
    SpectraDynamicSceneStepFn step;
    SpectraDynamicSceneDocumentFn document;
    SpectraDynamicSceneFrameFn frame;
    SpectraDynamicSceneLastErrorFn last_error;
} SpectraDynamicSceneSceneApi;

typedef struct SpectraDynamicSceneControlsApi {
    uint64_t struct_size;
    SpectraDynamicSceneControlActionSpan control_actions;
    SpectraDynamicSceneControlUpdateFn controls_update;
    SpectraDynamicSceneControlSceneRevisionFn scene_revision;
    SpectraDynamicSceneControlActionFn control_action;
    SpectraDynamicSceneControlStatusFn control_status;
    SpectraDynamicSceneControlLogsFn control_logs;
    SpectraDynamicSceneControlImagesFn control_images;
    SpectraDynamicSceneControlScalarSeriesFn control_scalar_series;
} SpectraDynamicSceneControlsApi;

typedef struct SpectraDynamicScenePlugin {
    uint32_t abi_version;
    uint64_t struct_size;
    SpectraDynamicSceneString id;
    SpectraDynamicSceneString title;
    SpectraDynamicSceneString controls_panel_title;
    SpectraDynamicSceneString open_action_label;
    SpectraDynamicSceneString open_action_description;
    SpectraDynamicSceneOpenOptionSchemaSpan open_options;
    SpectraDynamicSceneGetApiFn get_api;
} SpectraDynamicScenePlugin;

import std;
import instant_ngp.spectra_project;

namespace {
    constexpr char scene_api_name[] = "spectra.dynamic_scene.scene";
    constexpr char controls_api_name[] = "spectra.dynamic_scene.controls";
    constexpr std::uint32_t scene_api_version = 1u;
    constexpr std::uint32_t controls_api_version = 1u;

    struct OptionSchemaViews {
        std::vector<std::vector<SpectraDynamicSceneOpenOptionChoice>> choices{};
        std::vector<SpectraDynamicSceneOpenOptionSchema> schemas{};
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
        std::vector<std::vector<SpectraDynamicSceneViewportSegment>> segment_storage{};
        std::vector<std::vector<SpectraDynamicSceneColor>> color_storage{};
        std::vector<SpectraDynamicSceneViewportSegmentSet> segment_set_views{};
        std::vector<SpectraDynamicSceneViewportVoxelGrid> voxel_grid_views{};
        std::vector<SpectraDynamicSceneViewportCameraVisual> camera_visual_views{};
    };

    struct ProjectStatusCache {
        instant_ngp::spectra_project::ProjectStatus status{};
        std::vector<SpectraDynamicSceneControlMetric> metric_views{};
        std::vector<SpectraDynamicSceneString> enabled_action_views{};
        std::vector<SpectraDynamicSceneControlDisabledAction> disabled_action_views{};
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
        ProjectLogCache log_cache{};
        ProjectImageCache image_cache{};
        ProjectScalarSeriesCache scalar_series_cache{};
    };

    std::string global_error{};

    [[nodiscard]] SpectraDynamicSceneString abi_text(const std::string& text) {
        return SpectraDynamicSceneString{.data = text.data(), .size = static_cast<std::uint64_t>(text.size())};
    }

    [[nodiscard]] SpectraDynamicSceneString abi_text(const std::string_view text) {
        return SpectraDynamicSceneString{.data = text.data(), .size = static_cast<std::uint64_t>(text.size())};
    }

    [[nodiscard]] std::string_view string_view_from_abi(const SpectraDynamicSceneString value, const std::string_view context) {
        if (value.data == nullptr && value.size == 0u) return {};
        if (value.data == nullptr) throw std::runtime_error(std::format("{} string pointer is null", context));
        if (value.size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::format("{} string is too large", context));
        return std::string_view{value.data, static_cast<std::size_t>(value.size)};
    }

    [[nodiscard]] std::string string_from_abi(const SpectraDynamicSceneString value, const std::string_view context, const bool allow_empty) {
        const std::string_view view = string_view_from_abi(value, context);
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
            SpectraDynamicSceneViewportVoxelBufferRequest request{
                .struct_size = sizeof(SpectraDynamicSceneViewportVoxelBufferRequest),
                .byte_size = byte_size,
                .debug_name = abi_text(debug_name),
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
            SpectraDynamicSceneVolumeBufferRequest request{
                .struct_size = sizeof(SpectraDynamicSceneVolumeBufferRequest),
                .byte_size = byte_size,
                .debug_name = abi_text(debug_name),
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

    [[nodiscard]] SpectraDynamicSceneOpenOptionChoice make_choice_view(const instant_ngp::spectra_project::OptionChoice& choice) {
        return SpectraDynamicSceneOpenOptionChoice{.value = abi_text(choice.value), .label = abi_text(choice.label)};
    }

    [[nodiscard]] OptionSchemaViews make_option_schema_views(const std::vector<instant_ngp::spectra_project::OptionSchema>& schemas) {
        OptionSchemaViews views{};
        views.choices.resize(schemas.size());
        views.schemas.reserve(schemas.size());
        for (std::size_t index = 0u; index < schemas.size(); ++index) {
            const instant_ngp::spectra_project::OptionSchema& schema = schemas[index];
            views.choices[index].reserve(schema.choices.size());
            for (const instant_ngp::spectra_project::OptionChoice& choice : schema.choices) views.choices[index].push_back(make_choice_view(choice));
            views.schemas.push_back(SpectraDynamicSceneOpenOptionSchema{
                .key = abi_text(schema.key),
                .label = abi_text(schema.label),
                .description = abi_text(schema.description),
                .kind = static_cast<std::uint32_t>(schema.kind),
                .required = schema.required ? 1u : 0u,
                .default_value = abi_text(schema.default_value),
                .choices = SpectraDynamicSceneOpenOptionChoiceSpan{.data = views.choices[index].empty() ? nullptr : views.choices[index].data(), .count = static_cast<std::uint64_t>(views.choices[index].size())},
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
                .id = abi_text(action.id),
                .label = abi_text(action.label),
                .description = abi_text(action.description),
                .options = SpectraDynamicSceneOpenOptionSchemaSpan{.data = action_options.schemas.empty() ? nullptr : action_options.schemas.data(), .count = static_cast<std::uint64_t>(action_options.schemas.size())},
            });
        }
        return views;
    }

    [[nodiscard]] const DescriptorViews& descriptor_views() {
        static const DescriptorViews views = make_descriptor_views();
        return views;
    }

    void copy3(float (&output)[3], const std::array<float, 3u>& input) {
        output[0] = input[0];
        output[1] = input[1];
        output[2] = input[2];
    }

    void copy4(float (&output)[4], const std::array<float, 4u>& input) {
        output[0] = input[0];
        output[1] = input[1];
        output[2] = input[2];
        output[3] = input[3];
    }

    [[nodiscard]] SpectraDynamicSceneTransform make_transform_view(const instant_ngp::spectra_project::Transform& transform) {
        SpectraDynamicSceneTransform view{};
        copy3(view.position, transform.position);
        copy4(view.rotation, transform.rotation);
        copy3(view.scale, transform.scale);
        return view;
    }

    [[nodiscard]] SpectraDynamicSceneEntityRef make_entity_ref_view(const instant_ngp::spectra_project::SceneEntityRef& ref) {
        return SpectraDynamicSceneEntityRef{
            .kind = static_cast<std::uint32_t>(ref.kind),
            .name = abi_text(ref.name),
        };
    }

    [[nodiscard]] SpectraDynamicSceneMaterial make_material_view(const instant_ngp::spectra_project::Material& material) {
        SpectraDynamicSceneMaterial view{
            .name = abi_text(material.name),
            .model = abi_text(material.model),
            .alpha_mode = abi_text(material.alpha_mode),
            .emission_strength = material.emission_strength,
            .roughness = material.roughness,
            .metallic = material.metallic,
            .alpha_cutoff = material.alpha_cutoff,
            .volume_density_scale = material.volume_density_scale,
            .volume_temperature_scale = material.volume_temperature_scale,
        };
        copy4(view.base_color, material.base_color);
        copy3(view.emission_color, material.emission_color);
        return view;
    }

    void make_material_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::Material>& materials) {
        cache.material_views.clear();
        cache.material_views.reserve(materials.size());
        for (const instant_ngp::spectra_project::Material& material : materials) cache.material_views.push_back(make_material_view(material));
    }

    [[nodiscard]] SpectraDynamicSceneLight make_light_view(const instant_ngp::spectra_project::Light& light) {
        SpectraDynamicSceneLight view{
            .name = abi_text(light.name),
            .kind = abi_text(light.kind),
            .transform = make_transform_view(light.transform),
            .intensity = light.intensity,
            .cone_angle_degrees = light.cone_angle_degrees,
        };
        copy3(view.color, light.color);
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
                    .name = abi_text(channel.name),
                    .values = SpectraDynamicSceneFloatSpan{.data = channel.values.empty() ? nullptr : channel.values.data(), .count = static_cast<std::uint64_t>(channel.values.size())},
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
                .name = abi_text(volume.name),
                .channels = SpectraDynamicSceneVolumeChannelSpan{.data = cache.volume_channel_storage[volume_index].empty() ? nullptr : cache.volume_channel_storage[volume_index].data(), .count = static_cast<std::uint64_t>(cache.volume_channel_storage[volume_index].size())},
                .material_name = abi_text(volume.material_name),
            };
            volume_view.dimensions[0] = volume.dimensions[0];
            volume_view.dimensions[1] = volume.dimensions[1];
            volume_view.dimensions[2] = volume.dimensions[2];
            copy3(volume_view.origin, volume.origin);
            copy3(volume_view.voxel_size, volume.voxel_size);
            cache.volume_views.push_back(volume_view);
        }
    }

    [[nodiscard]] SpectraDynamicSceneCamera make_camera_view(const instant_ngp::spectra_project::Camera& camera) {
        SpectraDynamicSceneCamera view{
            .name = abi_text(camera.name),
            .local_coordinate_system = abi_text(camera.local_coordinate_system),
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
        copy3(view.target, camera.target);
        copy3(view.up, camera.up);
        return view;
    }

    [[nodiscard]] SpectraDynamicSceneViewportCameraVisual make_camera_visual_view(const instant_ngp::spectra_project::ViewportCameraVisual& visual) {
        SpectraDynamicSceneViewportCameraVisual view{
            .name = abi_text(visual.name),
            .owner = make_entity_ref_view(visual.owner),
            .width = visual.width,
            .width_mode = visual.width_mode,
            .depth_mode = visual.depth_mode,
            .visual_near = visual.visual_near,
            .visual_far = visual.visual_far,
            .has_image = visual.image.has_value() ? 1u : 0u,
        };
        copy4(view.color, visual.color);
        if (visual.image.has_value()) {
            const instant_ngp::spectra_project::ViewportCameraVisualImage& image = *visual.image;
            view.image = SpectraDynamicSceneViewportCameraVisualImage{
                .rgba8 = image.rgba8,
                .rgba8_size = image.rgba8_size,
                .revision = image.revision,
                .width = image.width,
                .height = image.height,
            };
            copy4(view.image.tint, image.tint);
        }
        return view;
    }

    void make_camera_visual_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::ViewportCameraVisual>& visuals) {
        cache.camera_visual_views.clear();
        cache.camera_visual_views.reserve(visuals.size());
        for (const instant_ngp::spectra_project::ViewportCameraVisual& visual : visuals) cache.camera_visual_views.push_back(make_camera_visual_view(visual));
    }

    void make_segment_set_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::ViewportSegmentSet>& sets) {
        cache.segment_storage.clear();
        cache.color_storage.clear();
        cache.segment_set_views.clear();
        cache.segment_storage.resize(sets.size());
        cache.color_storage.resize(sets.size());
        cache.segment_set_views.reserve(sets.size());
        for (std::size_t set_index = 0u; set_index < sets.size(); ++set_index) {
            const instant_ngp::spectra_project::ViewportSegmentSet& set = sets[set_index];
            cache.segment_storage[set_index].reserve(set.segments.size());
            for (const instant_ngp::spectra_project::ViewportSegment& segment : set.segments) {
                SpectraDynamicSceneViewportSegment segment_view{};
                copy3(segment_view.start, segment.start);
                copy3(segment_view.end, segment.end);
                cache.segment_storage[set_index].push_back(segment_view);
            }
            cache.color_storage[set_index].reserve(set.colors.size());
            for (const std::array<float, 4u>& color : set.colors) {
                SpectraDynamicSceneColor color_view{};
                copy4(color_view.value, color);
                cache.color_storage[set_index].push_back(color_view);
            }
            cache.segment_set_views.push_back(SpectraDynamicSceneViewportSegmentSet{
                .name = abi_text(set.name),
                .owner = make_entity_ref_view(set.owner),
                .segments = SpectraDynamicSceneViewportSegmentSpan{.data = cache.segment_storage[set_index].empty() ? nullptr : cache.segment_storage[set_index].data(), .count = static_cast<std::uint64_t>(cache.segment_storage[set_index].size())},
                .colors = SpectraDynamicSceneColorSpan{.data = cache.color_storage[set_index].empty() ? nullptr : cache.color_storage[set_index].data(), .count = static_cast<std::uint64_t>(cache.color_storage[set_index].size())},
                .widths = SpectraDynamicSceneFloatSpan{.data = set.widths.empty() ? nullptr : set.widths.data(), .count = static_cast<std::uint64_t>(set.widths.size())},
                .width = set.width,
                .width_mode = set.width_mode,
                .depth_mode = set.depth_mode,
                .transform = make_transform_view(set.transform),
            });
        }
    }

    [[nodiscard]] SpectraDynamicSceneViewportVoxelGrid make_voxel_grid_view(const instant_ngp::spectra_project::ViewportVoxelGrid& grid) {
        SpectraDynamicSceneViewportVoxelGrid view{
            .name = abi_text(grid.name),
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
        copy3(view.origin, grid.origin);
        copy3(view.voxel_size, grid.voxel_size);
        copy4(view.color, grid.color);
        return view;
    }

    void make_voxel_grid_views(SceneViewCache& cache, const std::vector<instant_ngp::spectra_project::ViewportVoxelGrid>& grids) {
        cache.voxel_grid_views.clear();
        cache.voxel_grid_views.reserve(grids.size());
        for (const instant_ngp::spectra_project::ViewportVoxelGrid& grid : grids) cache.voxel_grid_views.push_back(make_voxel_grid_view(grid));
    }

    [[nodiscard]] SpectraDynamicSceneDebugAttachmentSet make_debug_attachment_set_view(SceneViewCache& cache, const instant_ngp::spectra_project::DebugAttachmentSet& attachments) {
        make_segment_set_views(cache, attachments.viewport_segment_sets);
        make_voxel_grid_views(cache, attachments.viewport_voxel_grids);
        make_camera_visual_views(cache, attachments.viewport_camera_visuals);
        return SpectraDynamicSceneDebugAttachmentSet{
            .viewport_segment_sets = SpectraDynamicSceneViewportSegmentSetSpan{.data = cache.segment_set_views.empty() ? nullptr : cache.segment_set_views.data(), .count = static_cast<std::uint64_t>(cache.segment_set_views.size())},
            .viewport_voxel_grids = SpectraDynamicSceneViewportVoxelGridSpan{.data = cache.voxel_grid_views.empty() ? nullptr : cache.voxel_grid_views.data(), .count = static_cast<std::uint64_t>(cache.voxel_grid_views.size())},
            .viewport_camera_visuals = SpectraDynamicSceneViewportCameraVisualSpan{.data = cache.camera_visual_views.empty() ? nullptr : cache.camera_visual_views.data(), .count = static_cast<std::uint64_t>(cache.camera_visual_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneDocumentView make_document_view(SceneViewCache& cache) {
        make_material_views(cache, cache.document.materials);
        make_light_views(cache, cache.document.lights);
        make_volume_views(cache, cache.document.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.document.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.document.cameras) cache.camera_views.push_back(make_camera_view(camera));
        const SpectraDynamicSceneDebugAttachmentSet debug_attachments = make_debug_attachment_set_view(cache, cache.document.debug_attachments);
        return SpectraDynamicSceneDocumentView{
            .struct_size = sizeof(SpectraDynamicSceneDocumentView),
            .default_coordinate_system = abi_text(cache.document.default_coordinate_system),
            .active_camera_name = abi_text(cache.document.active_camera_name),
            .cameras = SpectraDynamicSceneCameraSpan{.data = cache.camera_views.empty() ? nullptr : cache.camera_views.data(), .count = static_cast<std::uint64_t>(cache.camera_views.size())},
            .materials = SpectraDynamicSceneMaterialSpan{.data = cache.material_views.empty() ? nullptr : cache.material_views.data(), .count = static_cast<std::uint64_t>(cache.material_views.size())},
            .lights = SpectraDynamicSceneLightSpan{.data = cache.light_views.empty() ? nullptr : cache.light_views.data(), .count = static_cast<std::uint64_t>(cache.light_views.size())},
            .volumes = SpectraDynamicSceneVolumeSpan{.data = cache.volume_views.empty() ? nullptr : cache.volume_views.data(), .count = static_cast<std::uint64_t>(cache.volume_views.size())},
            .debug_attachments = debug_attachments,
        };
    }

    [[nodiscard]] SpectraDynamicSceneFrameView make_frame_view(SceneViewCache& cache) {
        make_volume_views(cache, cache.frame.volumes);
        cache.camera_views.clear();
        cache.camera_views.reserve(cache.frame.cameras.size());
        for (const instant_ngp::spectra_project::Camera& camera : cache.frame.cameras) cache.camera_views.push_back(make_camera_view(camera));
        const SpectraDynamicSceneDebugAttachmentSet debug_attachments = make_debug_attachment_set_view(cache, cache.frame.debug_attachments);
        return SpectraDynamicSceneFrameView{
            .struct_size = sizeof(SpectraDynamicSceneFrameView),
            .volumes = SpectraDynamicSceneVolumeSpan{.data = cache.volume_views.empty() ? nullptr : cache.volume_views.data(), .count = static_cast<std::uint64_t>(cache.volume_views.size())},
            .cameras = SpectraDynamicSceneCameraSpan{.data = cache.camera_views.empty() ? nullptr : cache.camera_views.data(), .count = static_cast<std::uint64_t>(cache.camera_views.size())},
            .debug_attachments = debug_attachments,
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
                .key = abi_text(metric.key),
                .label = abi_text(metric.label),
                .value = abi_text(metric.value),
            });
        }
        for (const std::string& action_id : cache.status.enabled_action_ids) cache.enabled_action_views.push_back(abi_text(action_id));
        for (const instant_ngp::spectra_project::ProjectDisabledAction& disabled_action : cache.status.disabled_actions) {
            cache.disabled_action_views.push_back(SpectraDynamicSceneControlDisabledAction{
                .action_id = abi_text(disabled_action.action_id),
                .reason = abi_text(disabled_action.reason),
            });
        }
        return SpectraDynamicSceneControlStatusView{
            .struct_size = sizeof(SpectraDynamicSceneControlStatusView),
            .phase = abi_text(cache.status.phase),
            .headline = abi_text(cache.status.headline),
            .detail = abi_text(cache.status.detail),
            .metrics = SpectraDynamicSceneControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
            .enabled_action_ids = SpectraDynamicSceneStringSpan{.data = cache.enabled_action_views.empty() ? nullptr : cache.enabled_action_views.data(), .count = static_cast<std::uint64_t>(cache.enabled_action_views.size())},
            .disabled_actions = SpectraDynamicSceneControlDisabledActionSpan{.data = cache.disabled_action_views.empty() ? nullptr : cache.disabled_action_views.data(), .count = static_cast<std::uint64_t>(cache.disabled_action_views.size())},
        };
    }

    [[nodiscard]] SpectraDynamicSceneControlLogView make_log_view(ProjectLogCache& cache) {
        cache.log_views.clear();
        cache.log_views.reserve(cache.logs.size());
        for (const instant_ngp::spectra_project::ProjectLogEntry& log : cache.logs) {
            cache.log_views.push_back(SpectraDynamicSceneControlLogEntry{
                .sequence = log.sequence,
                .level = abi_text(log.level),
                .message = abi_text(log.message),
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
                .id = abi_text(image.id),
                .label = abi_text(image.label),
                .description = abi_text(image.description),
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
                .id = abi_text(series.id),
                .label = abi_text(series.label),
                .description = abi_text(series.description),
                .unit = abi_text(series.unit),
                .color = {},
                .revision = series.revision,
                .samples = SpectraDynamicSceneControlScalarSampleSpan{.data = samples.empty() ? nullptr : samples.data(), .count = static_cast<std::uint64_t>(samples.size())},
            };
            copy4(view.color, series.color);
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

    [[nodiscard]] SpectraDynamicSceneResult scene_step(SpectraDynamicSceneInstance* instance, const float delta_seconds) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "step");
            plugin_instance.last_error.clear();
            static_cast<void>(delta_seconds);
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

    [[nodiscard]] SpectraDynamicSceneString last_error(SpectraDynamicSceneInstance* instance) noexcept {
        if (instance == nullptr) return abi_text(global_error);
        return abi_text(reinterpret_cast<PluginInstance*>(instance)->last_error);
    }

    [[nodiscard]] SpectraDynamicSceneResult controls_update(SpectraDynamicSceneInstance* instance, const float delta_seconds) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "controls_update");
            plugin_instance.last_error.clear();
            plugin_instance.project.update(delta_seconds);
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

    [[nodiscard]] SpectraDynamicSceneResult control_action(SpectraDynamicSceneInstance* instance, const SpectraDynamicSceneString action_id, const SpectraDynamicSceneOptionSpan options) noexcept {
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

    [[nodiscard]] SpectraDynamicSceneResult control_status(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlStatusView* status) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_status");
            if (status == nullptr) throw std::runtime_error("control_status output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.status_cache.status = plugin_instance.project.status();
            *status = make_status_view(plugin_instance.status_cache);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraDynamicSceneResult control_logs(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlLogView* logs) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_logs");
            if (logs == nullptr) throw std::runtime_error("control_logs output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.log_cache.logs = plugin_instance.project.logs();
            *logs = make_log_view(plugin_instance.log_cache);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraDynamicSceneResult control_images(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlImageView* images) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_images");
            if (images == nullptr) throw std::runtime_error("control_images output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.image_cache.images = plugin_instance.project.images();
            *images = make_image_view(plugin_instance.image_cache);
            return SPECTRA_DYNAMIC_SCENE_RESULT_OK;
        } catch (const std::exception& error) {
            if (instance != nullptr) reinterpret_cast<PluginInstance*>(instance)->last_error = error.what();
            else global_error = error.what();
            return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
        }
    }

    [[nodiscard]] SpectraDynamicSceneResult control_scalar_series(SpectraDynamicSceneInstance* instance, SpectraDynamicSceneControlScalarSeriesView* series) noexcept {
        try {
            PluginInstance& plugin_instance = checked_instance(instance, "control_scalar_series");
            if (series == nullptr) throw std::runtime_error("control_scalar_series output pointer is null");
            plugin_instance.last_error.clear();
            plugin_instance.scalar_series_cache.series = plugin_instance.project.scalar_series();
            *series = make_scalar_series_view(plugin_instance.scalar_series_cache);
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
            .base_pbrt_path = abi_text(instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor().base_pbrt_path),
            .frames_per_second = instant_ngp::spectra_project::InstantNgpSpectraProject::descriptor().frames_per_second,
            .create = scene_create,
            .destroy = scene_destroy,
            .reset = scene_reset,
            .step = scene_step,
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
            .controls_update = controls_update,
            .scene_revision = scene_revision,
            .control_action = control_action,
            .control_status = control_status,
            .control_logs = control_logs,
            .control_images = control_images,
            .control_scalar_series = control_scalar_series,
        };
        return api;
    }

    [[nodiscard]] SpectraDynamicSceneResult get_api(const SpectraDynamicSceneString api_name, const std::uint32_t api_version, const void** api) noexcept {
        try {
            if (api == nullptr) {
                global_error = "get_api output pointer is null";
                return SPECTRA_DYNAMIC_SCENE_RESULT_ERROR;
            }
            *api = nullptr;
            const std::string_view name = string_view_from_abi(api_name, "api name");
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
            .id = abi_text(descriptor.id),
            .title = abi_text(descriptor.title),
            .controls_panel_title = abi_text(descriptor.controls_panel_title),
            .open_action_label = abi_text(descriptor.open_action_label),
            .open_action_description = abi_text(descriptor.open_action_description),
            .open_options = SpectraDynamicSceneOpenOptionSchemaSpan{.data = views.open_options.schemas.empty() ? nullptr : views.open_options.schemas.data(), .count = static_cast<std::uint64_t>(views.open_options.schemas.size())},
            .get_api = get_api,
        };
        return value;
    }
}

extern "C" SPECTRA_DYNAMIC_SCENE_EXPORT const SpectraDynamicScenePlugin* spectra_dynamic_scene_plugin(void) {
    return &plugin();
}
