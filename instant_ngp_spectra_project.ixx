export module instant_ngp.spectra_project;

import std;

namespace instant_ngp::spectra_project {
    export enum class OptionKind : std::uint32_t {
        Text = 0,
        DirectoryPath = 1,
        FilePath = 2,
        Choice = 3,
        Bool = 4,
        Float = 5,
        UnsignedInteger = 6,
    };

    export struct OptionChoice {
        std::string value{};
        std::string label{};
    };

    export struct OptionSchema {
        std::string key{};
        std::string label{};
        std::string description{};
        OptionKind kind{OptionKind::Text};
        bool required{};
        std::string default_value{};
        std::string group{};
        bool advanced{};
        std::int32_t priority{};
        std::vector<OptionChoice> choices{};
    };

    export struct Option {
        std::string key{};
        std::string value{};
    };

    export enum class GpuResourceHandleKind : std::uint32_t {
        OpaqueWin32 = 1u,
        OpaqueFileDescriptor = 2u,
    };

    export struct GpuDeviceIdentity {
        std::uint32_t vendor_id{};
        std::uint32_t device_id{};
        std::array<std::uint8_t, 16u> device_uuid{};
        std::array<std::uint8_t, 8u> device_luid{};
        std::uint32_t device_node_mask{};
    };

    export struct ViewportVoxelBufferAllocation {
        std::uint64_t resource_id{};
        std::uint64_t byte_size{};
        GpuResourceHandleKind handle_kind{GpuResourceHandleKind::OpaqueWin32};
        std::uintptr_t handle{};
        GpuDeviceIdentity device_identity{};
    };

    export struct VolumeBufferAllocation {
        std::uint64_t resource_id{};
        std::uint64_t byte_size{};
        GpuResourceHandleKind handle_kind{GpuResourceHandleKind::OpaqueWin32};
        std::uintptr_t handle{};
        GpuDeviceIdentity device_identity{};
    };

    export class HostServices {
    public:
        HostServices() = default;
        HostServices(const HostServices& other) = delete;
        HostServices(HostServices&& other) = delete;
        HostServices& operator=(const HostServices& other) = delete;
        HostServices& operator=(HostServices&& other) = delete;
        virtual ~HostServices() noexcept = default;

        [[nodiscard]] virtual ViewportVoxelBufferAllocation request_viewport_voxel_buffer(std::uint64_t byte_size, std::string_view debug_name) = 0;
        virtual void release_viewport_voxel_buffer(std::uint64_t resource_id) = 0;
        [[nodiscard]] virtual VolumeBufferAllocation request_volume_buffer(std::uint64_t byte_size, std::string_view debug_name) = 0;
        virtual void release_volume_buffer(std::uint64_t resource_id) = 0;
    };

    export inline constexpr std::uint32_t ControlPlacementViewportOverlay = 1u << 0u;
    export inline constexpr std::uint32_t ControlPlacementPanelSummary = 1u << 1u;
    export inline constexpr std::uint32_t ControlPlacementPanelDetail = 1u << 2u;
    export inline constexpr std::uint32_t ControlActionGroupRun = 0u;
    export inline constexpr std::uint32_t ControlActionGroupPreview = 1u;
    export inline constexpr std::uint32_t ControlActionGroupDebug = 2u;
    export inline constexpr std::uint32_t ControlActionGroupUtility = 3u;
    export inline constexpr std::uint32_t ControlActionStyleSecondary = 0u;
    export inline constexpr std::uint32_t ControlActionStylePrimary = 1u;
    export inline constexpr std::uint32_t ControlActionStyleDanger = 2u;
    export inline constexpr std::uint32_t ControlTimelineModeLive = 0u;
    export inline constexpr std::uint32_t ControlTimelineModeRecord = 1u;
    export inline constexpr std::uint32_t ControlTimelineModePlayback = 2u;

    export struct ProjectAction {
        std::string id{};
        std::string label{};
        std::string description{};
        std::uint32_t group{ControlActionGroupUtility};
        std::int32_t priority{};
        std::uint32_t style{ControlActionStyleSecondary};
        std::vector<OptionSchema> options{};
    };

    export struct ProjectSetting {
        std::string key{};
        std::string label{};
        std::string description{};
        OptionKind kind{OptionKind::Bool};
        std::string value{};
        std::string group{};
        bool advanced{};
        std::int32_t priority{};
        std::vector<OptionChoice> choices{};
    };

    export struct ProjectMetric {
        std::string key{};
        std::string label{};
        std::string value{};
        std::uint32_t placement_flags{ControlPlacementPanelDetail};
        std::int32_t priority{};
        bool has_color{};
        std::array<float, 4u> color{1.0f, 1.0f, 1.0f, 1.0f};
    };

    export struct ProjectDisabledAction {
        std::string action_id{};
        std::string reason{};
    };

    export struct ProjectStatus {
        std::string phase{};
        std::string headline{};
        std::string detail{};
        std::vector<ProjectMetric> metrics{};
        std::vector<std::string> enabled_action_ids{};
        std::vector<ProjectDisabledAction> disabled_actions{};
    };

    export struct ProjectUpdateInfo {
        double wall_delta_seconds{};
        double scene_delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
        std::uint32_t timeline_mode{ControlTimelineModeLive};
        bool timeline_playing{};
    };

    export struct ProjectLogEntry {
        std::uint64_t sequence{};
        std::string level{};
        std::string message{};
    };

    export struct ProjectImage {
        std::string id{};
        std::string label{};
        std::string description{};
        std::uint32_t width{};
        std::uint32_t height{};
        std::uint64_t revision{};
        std::vector<std::uint8_t> rgba8{};
    };

    export struct ProjectScalarSample {
        std::uint64_t step{};
        double time_seconds{};
        double value{};
    };

    export struct ProjectScalarSeries {
        std::string id{};
        std::string label{};
        std::string description{};
        std::string unit{};
        std::array<float, 4u> color{1.0f, 1.0f, 1.0f, 1.0f};
        std::uint32_t group{ControlActionGroupRun};
        std::int32_t priority{};
        std::uint64_t revision{};
        std::span<const ProjectScalarSample> samples{};
    };

    export struct Transform {
        std::array<float, 3u> position{};
        std::array<float, 4u> rotation{0.0f, 0.0f, 0.0f, 1.0f};
        std::array<float, 3u> scale{1.0f, 1.0f, 1.0f};
    };

    export enum class SceneEntityKind : std::uint32_t {
        Mesh = 0u,
        Sphere = 1u,
        PointCloud = 2u,
        VolumeGrid = 3u,
        Camera = 4u,
        Light = 5u,
    };

    export struct SceneEntityRef {
        SceneEntityKind kind{SceneEntityKind::Mesh};
        std::string name{};
    };

    export enum class CameraProjection : std::uint32_t {
        Perspective = 0,
        Pinhole = 1,
    };

    export struct Camera {
        std::string name{};
        std::string local_coordinate_system{};
        Transform transform{};
        std::array<float, 3u> target{};
        std::array<float, 3u> up{};
        CameraProjection projection{CameraProjection::Perspective};
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

    export struct Material {
        std::string name{};
        std::string model{"volume"};
        std::string alpha_mode{"blend"};
        std::array<float, 4u> base_color{1.0f, 1.0f, 1.0f, 1.0f};
        std::array<float, 3u> emission_color{};
        float emission_strength{};
        float roughness{0.5f};
        float metallic{};
        float alpha_cutoff{0.5f};
        float volume_density_scale{1.0f};
        float volume_temperature_scale{1.0f};
    };

    export struct Light {
        std::string name{};
        std::string kind{"environment"};
        Transform transform{};
        std::array<float, 3u> color{1.0f, 1.0f, 1.0f};
        float intensity{1.0f};
        float cone_angle_degrees{30.0f};
    };

    export enum class VolumeChannelSourceKind : std::uint32_t {
        Values = 0u,
        ExternalGpuBuffer = 1u,
    };

    export enum class VolumeChannelIndexEncoding : std::uint32_t {
        Linear = 0u,
        Morton3D = 1u,
    };

    export enum class VolumeChannelFormat : std::uint32_t {
        Float32 = 0u,
        Float32x3 = 1u,
    };

    export struct VolumeChannel {
        std::string name{};
        std::array<std::uint32_t, 3u> dimensions{};
        std::vector<float> values{};
        VolumeChannelFormat format{VolumeChannelFormat::Float32};
        VolumeChannelSourceKind source_kind{VolumeChannelSourceKind::Values};
        VolumeChannelIndexEncoding index_encoding{VolumeChannelIndexEncoding::Linear};
        std::uint64_t buffer_id{};
        std::uintptr_t external_device_pointer{};
        std::uint64_t source_byte_size{};
        std::uint64_t revision{};
    };

    export struct VolumeGrid {
        std::string name{};
        std::array<std::uint32_t, 3u> dimensions{};
        std::array<float, 3u> origin{};
        std::array<float, 3u> voxel_size{1.0f, 1.0f, 1.0f};
        std::vector<VolumeChannel> channels{};
        std::string material_name{};
    };

    export struct ViewportCameraVisualImage {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
        std::array<float, 4u> tint{};
    };

    export struct ViewportCameraVisual {
        std::string name{};
        SceneEntityRef owner{.kind = SceneEntityKind::Camera};
        std::array<float, 4u> color{};
        float width{};
        std::uint32_t width_mode{};
        std::uint32_t depth_mode{};
        float visual_near{};
        float visual_far{};
        std::optional<ViewportCameraVisualImage> image{};
    };

    export struct ViewportSegment {
        std::array<float, 3u> start{};
        std::array<float, 3u> end{};
    };

    export struct ViewportSegmentSet {
        std::string name{};
        SceneEntityRef owner{};
        std::vector<ViewportSegment> segments{};
        std::vector<std::array<float, 4u>> colors{};
        std::vector<float> widths{};
        float width{};
        std::uint32_t width_mode{};
        std::uint32_t depth_mode{};
        Transform transform{};
    };

    export enum class ViewportVoxelGridSourceKind : std::uint32_t {
        IndexList = 0u,
        Bitfield = 1u,
    };

    export enum class ViewportVoxelGridIndexEncoding : std::uint32_t {
        Linear = 0u,
        Morton3D = 1u,
    };

    export struct ViewportVoxelGrid {
        std::string name{};
        SceneEntityRef owner{};
        std::array<std::uint32_t, 3u> dimensions{};
        std::array<float, 3u> origin{};
        std::array<float, 3u> voxel_size{1.0f, 1.0f, 1.0f};
        Transform transform{};
        std::array<float, 4u> color{};
        float cell_scale{1.0f};
        std::uint32_t depth_mode{};
        ViewportVoxelGridSourceKind source_kind{ViewportVoxelGridSourceKind::IndexList};
        ViewportVoxelGridIndexEncoding index_encoding{ViewportVoxelGridIndexEncoding::Linear};
        std::uint64_t buffer_id{};
        std::uint64_t source_byte_size{};
        std::uint64_t index_count{};
        std::uint64_t revision{};
    };

    export struct DebugAttachmentSet {
        std::vector<ViewportSegmentSet> viewport_segment_sets{};
        std::vector<ViewportVoxelGrid> viewport_voxel_grids{};
        std::vector<ViewportCameraVisual> viewport_camera_visuals{};
    };

    export struct Document {
        std::string default_coordinate_system{};
        std::string active_camera_name{};
        std::vector<Camera> cameras{};
        std::vector<Material> materials{};
        std::vector<Light> lights{};
        std::vector<VolumeGrid> volumes{};
        DebugAttachmentSet debug_attachments{};
    };

    export struct FrameInfo {
        double delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
    };

    export struct Frame {
        std::vector<Camera> cameras{};
        std::vector<VolumeGrid> volumes{};
        DebugAttachmentSet debug_attachments{};
    };

    export struct Descriptor {
        std::string id{};
        std::string title{};
        std::string controls_panel_title{};
        std::string open_action_label{};
        std::string open_action_description{};
        std::string base_pbrt_path{};
        double frames_per_second{};
        std::vector<OptionSchema> open_options{};
        std::vector<ProjectAction> control_actions{};
    };

    export class InstantNgpSpectraProject final {
    public:
        struct State;

        InstantNgpSpectraProject();
        InstantNgpSpectraProject(const InstantNgpSpectraProject& other) = delete;
        InstantNgpSpectraProject(InstantNgpSpectraProject&& other) noexcept;
        InstantNgpSpectraProject& operator=(const InstantNgpSpectraProject& other) = delete;
        InstantNgpSpectraProject& operator=(InstantNgpSpectraProject&& other) noexcept;
        ~InstantNgpSpectraProject() noexcept;

        [[nodiscard]] static const Descriptor& descriptor();
        [[nodiscard]] static InstantNgpSpectraProject open(std::span<const Option> options, std::shared_ptr<HostServices> host_services);

        void update(const ProjectUpdateInfo& update);
        void execute_action(std::string_view action_id, std::span<const Option> options);
        [[nodiscard]] std::vector<ProjectSetting> settings() const;
        void update_setting(std::string_view key, std::string_view value);

        [[nodiscard]] std::uint64_t scene_revision() const;
        [[nodiscard]] ProjectStatus status() const;
        [[nodiscard]] std::vector<ProjectLogEntry> logs() const;
        [[nodiscard]] std::span<const ProjectImage> images() const;
        [[nodiscard]] std::vector<ProjectScalarSeries> scalar_series() const;
        [[nodiscard]] Document document() const;
        [[nodiscard]] Frame frame(const FrameInfo& frame_info) const;

    private:
        explicit InstantNgpSpectraProject(std::unique_ptr<State> state);

        std::unique_ptr<State> state{};
    };
}
