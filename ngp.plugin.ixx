export module ngp.plugin;

import std;

export namespace ngp::plugin {
    struct SpectraScenePlugin;

    enum class OptionKind : std::uint32_t {
        Text            = 0u,
        DirectoryPath   = 1u,
        FilePath        = 2u,
        Choice          = 3u,
        Bool            = 4u,
        Float           = 5u,
        UnsignedInteger = 6u,
    };

    inline constexpr std::uint32_t ControlOptionPresentationDefault = 0u;
    inline constexpr std::uint32_t ControlOptionPresentationSlider  = 1u;

    struct OptionChoice {
        std::string value{};
        std::string label{};
    };

    struct ControlSection {
        std::string id{};
        std::string label{};
    };

    struct OptionSchema {
        std::string key{};
        std::string label{};
        std::string description{};
        OptionKind kind{OptionKind::Text};
        bool required_flag{};
        std::string default_value{};
        std::string section_id{};
        std::vector<OptionChoice> choices{};
        std::uint32_t presentation{ControlOptionPresentationDefault};
        bool has_numeric_range{};
        float numeric_min{};
        float numeric_max{};
        float numeric_step{};

        OptionSchema& section(std::string value) & {
            this->section_id = std::move(value);
            return *this;
        }

        [[nodiscard]] OptionSchema section(std::string value) && {
            this->section_id = std::move(value);
            return std::move(*this);
        }

        OptionSchema& required_value(const bool value = true) & {
            this->required_flag = value;
            return *this;
        }

        [[nodiscard]] OptionSchema required_value(const bool value = true) && {
            this->required_flag = value;
            return std::move(*this);
        }

        OptionSchema& required() & {
            this->required_flag = true;
            return *this;
        }

        [[nodiscard]] OptionSchema required() && {
            this->required_flag = true;
            return std::move(*this);
        }

        OptionSchema& defaulted(std::string value) & {
            this->default_value = std::move(value);
            return *this;
        }

        [[nodiscard]] OptionSchema defaulted(std::string value) && {
            this->default_value = std::move(value);
            return std::move(*this);
        }

        OptionSchema& describe(std::string value) & {
            this->description = std::move(value);
            return *this;
        }

        [[nodiscard]] OptionSchema describe(std::string value) && {
            this->description = std::move(value);
            return std::move(*this);
        }

        OptionSchema& slider(const float min, const float max, const float step) & {
            this->presentation = ControlOptionPresentationSlider;
            this->has_numeric_range = true;
            this->numeric_min = min;
            this->numeric_max = max;
            this->numeric_step = step;
            return *this;
        }

        [[nodiscard]] OptionSchema slider(const float min, const float max, const float step) && {
            this->presentation = ControlOptionPresentationSlider;
            this->has_numeric_range = true;
            this->numeric_min = min;
            this->numeric_max = max;
            this->numeric_step = step;
            return std::move(*this);
        }
    };

    struct Option {
        std::string key{};
        std::string value{};
    };

    struct OpenContext {
        std::vector<Option> options{};
        std::shared_ptr<struct HostServices> host_services{};
    };

    struct ActionContext {
        std::vector<Option> options{};
    };

    enum class GpuResourceHandleKind : std::uint32_t {
        OpaqueWin32          = 1u,
        OpaqueFileDescriptor = 2u,
    };

    struct GpuDeviceIdentity {
        std::uint32_t vendor_id{};
        std::uint32_t device_id{};
        std::array<std::uint8_t, 16u> device_uuid{};
        std::array<std::uint8_t, 8u> device_luid{};
        std::uint32_t device_node_mask{};
    };

    inline constexpr std::uint32_t GpuBufferKindVolumeChannel     = 0u;
    inline constexpr std::uint32_t GpuBufferKindViewportVoxelGrid = 1u;

    struct GpuBufferAllocation {
        std::uint64_t resource_id{};
        std::uint64_t byte_size{};
        std::uint32_t kind{};
        GpuResourceHandleKind handle_kind{GpuResourceHandleKind::OpaqueWin32};
        std::uintptr_t handle{};
        GpuDeviceIdentity device_identity{};
    };

    struct HostServices {
        std::move_only_function<GpuBufferAllocation(std::uint32_t, std::uint64_t, std::string_view)> request_gpu_buffer{};
        std::move_only_function<void(std::uint64_t)> release_gpu_buffer{};
    };

    inline constexpr std::uint32_t ControlMetricDisplayPrimary = 1u << 0u;

    struct Action {
        std::string id{};
        std::string label{};
        std::string description{};
        std::string section_id{};
        std::vector<OptionSchema> options{};
    };

    struct Metric {
        std::string key{};
        std::string label{};
        std::string value{};
        std::string section_id{};
        std::uint32_t display_flags{};
        bool has_color{};
        std::array<float, 4u> color{1.0f, 1.0f, 1.0f, 1.0f};
    };

    struct ActionState {
        std::string action_id{};
        bool enabled{};
        std::string disabled_reason{};
    };

    struct ControlState {
        std::string phase{};
        std::string headline{};
        std::string detail{};
        std::vector<Metric> metrics{};
        std::vector<ActionState> action_states{};
    };

    struct UpdateInfo {
        double wall_delta_seconds{};
        double scene_delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
        bool timeline_playing{};
    };

    struct Transform {
        std::array<float, 3u> position{};
        std::array<float, 4u> rotation{0.0f, 0.0f, 0.0f, 1.0f};
        std::array<float, 3u> scale{1.0f, 1.0f, 1.0f};
    };

    enum class SceneEntityKind : std::uint32_t {
        Mesh       = 0u,
        Sphere     = 1u,
        PointCloud = 2u,
        VolumeGrid = 3u,
        Camera     = 4u,
        Light      = 5u,
    };

    struct SceneEntityRef {
        SceneEntityKind kind{SceneEntityKind::Mesh};
        std::string name{};
    };

    enum class CameraProjection : std::uint32_t {
        Perspective = 0u,
        Pinhole     = 1u,
    };

    struct CameraImage {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
    };

    struct Camera {
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
        std::optional<CameraImage> image{};
    };

    struct Material {
        std::string name{};
        std::string model{"volume"};
        std::string alpha_mode{"blend"};
        std::array<float, 4u> base_color{1.0f, 1.0f, 1.0f, 1.0f};
        float roughness{0.5f};
        float volume_density_scale{1.0f};
        float volume_temperature_scale{1.0f};
    };

    struct Light {
        std::string name{};
        std::string kind{"environment"};
        Transform transform{};
        std::array<float, 3u> color{1.0f, 1.0f, 1.0f};
        float intensity{1.0f};
    };

    enum class VolumeChannelSourceKind : std::uint32_t {
        Values            = 0u,
        ExternalGpuBuffer = 1u,
    };

    enum class VolumeChannelIndexEncoding : std::uint32_t {
        Linear   = 0u,
        Morton3D = 1u,
    };

    enum class VolumeChannelFormat : std::uint32_t {
        Float32   = 0u,
        Float32x3 = 1u,
    };

    struct VolumeChannel {
        std::string name{};
        std::array<std::uint32_t, 3u> dimensions{};
        VolumeChannelFormat format{VolumeChannelFormat::Float32};
        VolumeChannelSourceKind source_kind{VolumeChannelSourceKind::Values};
        VolumeChannelIndexEncoding index_encoding{VolumeChannelIndexEncoding::Linear};
        std::uint64_t buffer_id{};
        std::uintptr_t external_device_pointer{};
        std::uint64_t source_byte_size{};
        std::uint64_t revision{};
    };

    struct VolumeGrid {
        std::string name{};
        std::array<std::uint32_t, 3u> dimensions{};
        std::array<float, 3u> origin{};
        std::array<float, 3u> voxel_size{1.0f, 1.0f, 1.0f};
        std::vector<VolumeChannel> channels{};
        std::string material_name{};
    };

    enum class ViewportVoxelGridSourceKind : std::uint32_t {
        IndexList = 0u,
        Bitfield  = 1u,
    };

    enum class ViewportVoxelGridIndexEncoding : std::uint32_t {
        Linear   = 0u,
        Morton3D = 1u,
    };

    struct ViewportVoxelGrid {
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
        std::uint64_t revision{};
    };

    struct DebugAttachmentSet {
        std::vector<ViewportVoxelGrid> viewport_voxel_grids{};
    };

    struct Document {
        std::string default_coordinate_system{};
        std::string active_camera_name{};
        std::vector<Camera> cameras{};
        std::vector<Material> materials{};
        std::vector<Light> lights{};
        std::vector<VolumeGrid> volumes{};
        DebugAttachmentSet debug_attachments{};
    };

    class SceneBuilder final {
    public:
        SceneBuilder& coordinate_system(std::string value) {
            this->value.default_coordinate_system = std::move(value);
            return *this;
        }

        SceneBuilder& active_camera(std::string value) {
            this->value.active_camera_name = std::move(value);
            return *this;
        }

        SceneBuilder& set_document(Document document) {
            this->value = std::move(document);
            return *this;
        }

        [[nodiscard]] const Document& document() const {
            return this->value;
        }

    private:
        Document value{};
    };

    class ControlBuilder final {
    public:
        class MetricHandle final {
        public:
            MetricHandle(ControlBuilder& owner, const std::size_t index) : owner(&owner), index(index) {}

            MetricHandle& section(std::string value) {
                this->owner->value.metrics[this->index].section_id = std::move(value);
                return *this;
            }

            MetricHandle& display_primary() {
                this->owner->value.metrics[this->index].display_flags |= ControlMetricDisplayPrimary;
                return *this;
            }

            MetricHandle& color(const std::array<float, 4u> value) {
                Metric& metric   = this->owner->value.metrics[this->index];
                metric.has_color = true;
                metric.color     = value;
                return *this;
            }

        private:
            ControlBuilder* owner{};
            std::size_t index{};
        };

        ControlBuilder& phase(std::string value) {
            this->value.phase = std::move(value);
            return *this;
        }

        ControlBuilder& headline(std::string value) {
            this->value.headline = std::move(value);
            return *this;
        }

        ControlBuilder& message(std::string value) {
            this->value.detail = std::move(value);
            return *this;
        }

        [[nodiscard]] MetricHandle metric(std::string key, std::string label, std::string metric_value) {
            this->value.metrics.push_back(Metric{
                .key   = std::move(key),
                .label = std::move(label),
                .value = std::move(metric_value),
            });
            return MetricHandle{*this, this->value.metrics.size() - 1u};
        }

        [[nodiscard]] MetricHandle metric(std::string key, std::string label, const std::string_view value) {
            return this->metric(std::move(key), std::move(label), std::string{value});
        }

        [[nodiscard]] MetricHandle metric(std::string key, std::string label, const char* value) {
            return this->metric(std::move(key), std::move(label), value == nullptr ? std::string{} : std::string{value});
        }

        template <typename Value>
            requires (std::integral<Value> || std::floating_point<Value>)
        [[nodiscard]] MetricHandle metric(std::string key, std::string label, const Value value) {
            return this->metric(std::move(key), std::move(label), std::format("{}", value));
        }

        ControlBuilder& enable(std::string action_id) {
            this->value.action_states.push_back(ActionState{.action_id = std::move(action_id), .enabled = true});
            return *this;
        }

        ControlBuilder& disable(std::string action_id, std::string reason) {
            this->value.action_states.push_back(ActionState{.action_id = std::move(action_id), .enabled = false, .disabled_reason = std::move(reason)});
            return *this;
        }

        [[nodiscard]] const ControlState& state() const {
            return this->value;
        }

    private:
        friend class MetricHandle;

        ControlState value{};
    };

    [[nodiscard]] ControlSection section(std::string id, std::string label) {
        return ControlSection{.id = std::move(id), .label = std::move(label)};
    }

    [[nodiscard]] OptionChoice option_choice(std::string value) {
        return OptionChoice{.value = value, .label = std::move(value)};
    }

    [[nodiscard]] OptionChoice option_choice(std::string value, std::string label) {
        return OptionChoice{.value = std::move(value), .label = std::move(label)};
    }

    [[nodiscard]] OptionSchema text(std::string key, std::string label) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Text};
    }

    [[nodiscard]] OptionSchema directory(std::string key, std::string label) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::DirectoryPath};
    }

    [[nodiscard]] OptionSchema file(std::string key, std::string label) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::FilePath};
    }

    [[nodiscard]] OptionSchema choice(std::string key, std::string label, const std::initializer_list<std::string> values) {
        OptionSchema schema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Choice};
        schema.choices.reserve(values.size());
        for (const std::string& value : values) schema.choices.push_back(option_choice(value));
        return schema;
    }

    [[nodiscard]] OptionSchema toggle(std::string key, std::string label, const bool default_value) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Bool}.defaulted(default_value ? "true" : "false");
    }

    [[nodiscard]] OptionSchema float_option(std::string key, std::string label, const float default_value) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::Float}.defaulted(std::format("{:.9g}", default_value));
    }

    [[nodiscard]] OptionSchema unsigned_integer(std::string key, std::string label, const std::uint64_t default_value) {
        return OptionSchema{.key = std::move(key), .label = std::move(label), .kind = OptionKind::UnsignedInteger}.defaulted(std::format("{}", default_value));
    }

    template <typename Project>
    struct ActionBinding {
        Action schema{};
        std::function<void(Project&, ActionContext)> invoke{};

        ActionBinding& description(std::string value) & {
            this->schema.description = std::move(value);
            return *this;
        }

        [[nodiscard]] ActionBinding description(std::string value) && {
            this->schema.description = std::move(value);
            return std::move(*this);
        }

        ActionBinding& section(std::string value) & {
            this->schema.section_id = std::move(value);
            return *this;
        }

        [[nodiscard]] ActionBinding section(std::string value) && {
            this->schema.section_id = std::move(value);
            return std::move(*this);
        }

        ActionBinding& option(OptionSchema value) & {
            this->schema.options.push_back(std::move(value));
            return *this;
        }

        [[nodiscard]] ActionBinding option(OptionSchema value) && {
            this->schema.options.push_back(std::move(value));
            return std::move(*this);
        }
    };

    template <typename Project>
    [[nodiscard]] ActionBinding<Project> action(std::string id, std::string label, void (Project::*handler)()) {
        return ActionBinding<Project>{
            .schema = Action{.id = std::move(id), .label = std::move(label)},
            .invoke = [handler](Project& project, ActionContext) { (project.*handler)(); },
        };
    }

    template <typename Project>
    [[nodiscard]] ActionBinding<Project> action(std::string id, std::string label, void (Project::*handler)(ActionContext)) {
        return ActionBinding<Project>{
            .schema = Action{.id = std::move(id), .label = std::move(label)},
            .invoke = [handler](Project& project, ActionContext context) { (project.*handler)(std::move(context)); },
        };
    }

    template <typename Project>
    struct SettingBinding {
        OptionSchema schema{};
        std::function<void(Project&, std::string_view)> update{};

        SettingBinding& section(std::string value) & {
            this->schema.section_id = std::move(value);
            return *this;
        }

        [[nodiscard]] SettingBinding section(std::string value) && {
            this->schema.section_id = std::move(value);
            return std::move(*this);
        }

        SettingBinding& slider(const float min, const float max, const float step) & {
            this->schema.slider(min, max, step);
            return *this;
        }

        [[nodiscard]] SettingBinding slider(const float min, const float max, const float step) && {
            this->schema.slider(min, max, step);
            return std::move(*this);
        }
    };

    [[nodiscard]] bool parse_bool(const std::string_view text) {
        if (text == "true") return true;
        if (text == "false") return false;
        throw std::runtime_error(std::format("expected bool setting value, got '{}'", text));
    }

    [[nodiscard]] float parse_float(const std::string_view text) {
        float value{};
        const char* const begin             = text.data();
        const char* const end               = text.data() + text.size();
        const std::from_chars_result result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end || !std::isfinite(value)) throw std::runtime_error(std::format("expected finite float setting value, got '{}'", text));
        return value;
    }

    [[nodiscard]] std::uint64_t parse_unsigned_integer(const std::string_view text) {
        std::uint64_t value{};
        const char* const begin             = text.data();
        const char* const end               = text.data() + text.size();
        const std::from_chars_result result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end) throw std::runtime_error(std::format("expected unsigned integer setting value, got '{}'", text));
        return value;
    }

    template <typename Project>
    [[nodiscard]] SettingBinding<Project> toggle(std::string key, std::string label, const bool default_value, void (Project::*handler)(bool)) {
        return SettingBinding<Project>{
            .schema = toggle(std::move(key), std::move(label), default_value),
            .update = [handler](Project& project, const std::string_view value) { (project.*handler)(parse_bool(value)); },
        };
    }

    template <typename Project>
    [[nodiscard]] SettingBinding<Project> float_value(std::string key, std::string label, const float default_value, void (Project::*handler)(float)) {
        return SettingBinding<Project>{
            .schema = float_option(std::move(key), std::move(label), default_value),
            .update = [handler](Project& project, const std::string_view value) { (project.*handler)(parse_float(value)); },
        };
    }

    template <typename Project>
    [[nodiscard]] SettingBinding<Project> unsigned_integer_value(std::string key, std::string label, const std::uint64_t default_value, void (Project::*handler)(std::uint64_t)) {
        return SettingBinding<Project>{
            .schema = unsigned_integer(std::move(key), std::move(label), default_value),
            .update = [handler](Project& project, const std::string_view value) { (project.*handler)(parse_unsigned_integer(value)); },
        };
    }

    template <typename Project>
    struct PluginDefinition {
        std::string id{};
        std::string title{};
        std::string open_action_label{};
        double frames_per_second{};
        std::vector<ControlSection> sections{};
        std::vector<OptionSchema> open_options{};
        std::vector<ActionBinding<Project>> actions{};
        std::vector<SettingBinding<Project>> settings{};
    };

    struct TypeErasedAction {
        Action schema{};
        std::function<void(void*, ActionContext)> invoke{};
    };

    struct TypeErasedSetting {
        OptionSchema schema{};
        std::function<void(void*, std::string_view)> update{};
    };

    struct TypeErasedPluginDefinition {
        std::string id{};
        std::string title{};
        std::string open_action_label{};
        double frames_per_second{};
        std::vector<ControlSection> sections{};
        std::vector<OptionSchema> open_options{};
        std::vector<TypeErasedAction> actions{};
        std::vector<TypeErasedSetting> settings{};
        std::function<void*(OpenContext)> open{};
        std::function<void(void*)> destroy{};
        std::function<void(void*, const UpdateInfo&)> update{};
        std::function<std::uint64_t(void*)> revision{};
        std::function<void(void*, SceneBuilder&)> write_scene{};
        std::function<void(void*, ControlBuilder&)> write_controls{};
    };

    template <typename Project>
    [[nodiscard]] TypeErasedPluginDefinition erase_plugin_definition(const PluginDefinition<Project>& definition) {
        TypeErasedPluginDefinition erased{
            .id                      = definition.id,
            .title                   = definition.title,
            .open_action_label       = definition.open_action_label,
            .frames_per_second       = definition.frames_per_second,
            .sections                = definition.sections,
            .open_options            = definition.open_options,
            .open                    = [](OpenContext context) -> void* { return new Project{Project::open(std::move(context))}; },
            .destroy                 = [](void* project) { delete static_cast<Project*>(project); },
            .update                  = [](void* project, const UpdateInfo& update) { static_cast<Project*>(project)->update(update); },
            .revision                = [](void* project) -> std::uint64_t { return static_cast<Project*>(project)->revision(); },
            .write_scene             = [](void* project, SceneBuilder& scene) { static_cast<Project*>(project)->write_scene(scene); },
            .write_controls          = [](void* project, ControlBuilder& controls) { static_cast<Project*>(project)->write_controls(controls); },
        };
        erased.actions.reserve(definition.actions.size());
        for (const ActionBinding<Project>& action : definition.actions) {
            erased.actions.push_back(TypeErasedAction{
                .schema = action.schema,
                .invoke = [invoke = action.invoke](void* project, ActionContext context) { invoke(*static_cast<Project*>(project), std::move(context)); },
            });
        }
        erased.settings.reserve(definition.settings.size());
        for (const SettingBinding<Project>& setting : definition.settings) {
            erased.settings.push_back(TypeErasedSetting{
                .schema = setting.schema,
                .update = [update = setting.update](void* project, const std::string_view value) { update(*static_cast<Project*>(project), value); },
            });
        }
        return erased;
    }

    [[nodiscard]] const SpectraScenePlugin* export_type_erased_plugin(const TypeErasedPluginDefinition& definition);

    template <typename Project>
    [[nodiscard]] const SpectraScenePlugin* export_plugin() {
        static const TypeErasedPluginDefinition definition = erase_plugin_definition(Project::plugin());
        return export_type_erased_plugin(definition);
    }
} // namespace ngp::plugin

namespace ngp::plugin {
    constexpr std::uint32_t plugin_abi_version = 10u;
    typedef void SpectraSceneInstance;

    typedef std::uint32_t SpectraSceneResult;
    constexpr std::uint32_t SPECTRA_SCENE_RESULT_OK                      = 0u;
    constexpr std::uint32_t SPECTRA_SCENE_RESULT_ERROR                   = 1u;
    constexpr std::uint32_t SPECTRA_SCENE_GPU_BUFFER_VOLUME_CHANNEL      = 0u;
    constexpr std::uint32_t SPECTRA_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID = 1u;
    constexpr std::uint32_t SPECTRA_SCENE_OPTION_PRESENTATION_DEFAULT    = 0u;
    constexpr std::uint32_t SPECTRA_SCENE_OPTION_PRESENTATION_SLIDER     = 1u;

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

    struct SpectraSceneControlSection {
        const char* id{};
        const char* label{};
    };

    struct SpectraSceneControlSectionSpan {
        const SpectraSceneControlSection* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlOptionSchema {
        const char* key{};
        const char* label{};
        const char* description{};
        std::uint32_t kind{};
        std::uint32_t required{};
        const char* default_value{};
        const char* section_id{};
        SpectraSceneControlOptionChoiceSpan choices{};
        std::uint32_t presentation{};
        std::uint32_t has_numeric_range{};
        float numeric_min{};
        float numeric_max{};
        float numeric_step{};
    };

    struct SpectraSceneControlOptionSchemaSpan {
        const SpectraSceneControlOptionSchema* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlAction {
        const char* id{};
        const char* label{};
        const char* description{};
        const char* section_id{};
        SpectraSceneControlOptionSchemaSpan options{};
    };

    struct SpectraSceneControlActionSpan {
        const SpectraSceneControlAction* data{};
        std::uint64_t count{};
    };

    struct SpectraSceneControlMetric {
        const char* key{};
        const char* label{};
        const char* value{};
        const char* section_id{};
        std::uint32_t display_flags{};
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

    struct SpectraSceneControlStateView {
        std::uint64_t struct_size{};
        const char* phase{};
        const char* headline{};
        const char* detail{};
        SpectraSceneControlMetricSpan metrics{};
        SpectraSceneControlActionStateSpan action_states{};
    };

    struct SpectraSceneUpdateInfo {
        std::uint64_t struct_size{};
        double wall_delta_seconds{};
        double scene_delta_seconds{};
        double time_seconds{};
        std::uint64_t frame_index{};
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

    struct SpectraSceneCameraImage {
        const std::uint8_t* rgba8{};
        std::uint64_t rgba8_size{};
        std::uint64_t revision{};
        std::uint32_t width{};
        std::uint32_t height{};
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
        std::uint32_t has_image{};
        SpectraSceneCameraImage image{};
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
    typedef SpectraSceneResult (*SpectraSceneUpdateFn)(SpectraSceneInstance* instance, const SpectraSceneUpdateInfo* update_info);
    typedef SpectraSceneResult (*SpectraSceneDocumentFn)(SpectraSceneInstance* instance, SpectraSceneDocumentView* document);
    typedef SpectraSceneResult (*SpectraSceneFrameFn)(SpectraSceneInstance* instance, SpectraSceneFrameInfo frame, SpectraSceneFrameView* snapshot);
    typedef SpectraSceneResult (*SpectraSceneRevisionFn)(SpectraSceneInstance* instance, std::uint64_t* revision);
    typedef SpectraSceneResult (*SpectraSceneControlActionFn)(SpectraSceneInstance* instance, const char* action_id, SpectraSceneOptionSpan options);
    typedef SpectraSceneResult (*SpectraSceneControlSettingUpdateFn)(SpectraSceneInstance* instance, const char* key, const char* value);
    typedef SpectraSceneResult (*SpectraSceneControlStateFn)(SpectraSceneInstance* instance, SpectraSceneControlStateView* state);
    typedef const char* (*SpectraSceneLastErrorFn)(SpectraSceneInstance* instance);

    struct SpectraScenePlugin {
        std::uint32_t abi_version{};
        std::uint64_t struct_size{};
        const char* id{};
        const char* title{};
        const char* open_action_label{};
        double frames_per_second{};
        SpectraSceneControlSectionSpan sections{};
        SpectraSceneControlOptionSchemaSpan open_options{};
        SpectraSceneControlActionSpan control_actions{};
        SpectraSceneControlOptionSchemaSpan control_settings{};
        SpectraSceneCreateFn create{};
        SpectraSceneDestroyFn destroy{};
        SpectraSceneUpdateFn update{};
        SpectraSceneDocumentFn document{};
        SpectraSceneFrameFn frame{};
        SpectraSceneRevisionFn scene_revision{};
        SpectraSceneControlActionFn control_action{};
        SpectraSceneControlSettingUpdateFn control_setting_update{};
        SpectraSceneControlStateFn control_state{};
        SpectraSceneLastErrorFn last_error{};
    };

    namespace {
        struct OptionSchemaAbiStorage {
            std::vector<std::vector<SpectraSceneControlOptionChoice>> choices{};
            std::vector<SpectraSceneControlOptionSchema> schemas{};
        };

        struct PluginDescriptorStorage {
            std::vector<SpectraSceneControlSection> sections{};
            OptionSchemaAbiStorage open_options{};
            std::vector<OptionSchemaAbiStorage> action_options{};
            std::vector<SpectraSceneControlAction> control_actions{};
            std::vector<OptionSchema> control_setting_schemas{};
            OptionSchemaAbiStorage control_settings{};
        };

        struct SceneAbiStorage {
            Document document{};
            std::vector<SpectraSceneMaterial> material_views{};
            std::vector<SpectraSceneLight> light_views{};
            std::vector<std::vector<SpectraSceneVolumeChannel>> volume_channel_storage{};
            std::vector<SpectraSceneVolume> volume_views{};
            std::vector<SpectraSceneCamera> camera_views{};
            std::vector<SpectraSceneViewportVoxelGrid> voxel_grid_views{};
        };

        struct ControlStateAbiStorage {
            ControlState state{};
            std::vector<SpectraSceneControlMetric> metric_views{};
            std::vector<SpectraSceneControlActionState> action_state_views{};
        };

        struct PluginExportState {
            explicit PluginExportState(const TypeErasedPluginDefinition& plugin_definition);

            [[nodiscard]] static PluginExportState& instance(const TypeErasedPluginDefinition* plugin_definition = nullptr);

            const TypeErasedPluginDefinition& definition;
            PluginDescriptorStorage descriptor_storage{};
            SpectraScenePlugin plugin{};
            std::string export_error{};
        };

        struct PluginInstance {
            const TypeErasedPluginDefinition* definition{};
            void* project{};
            std::string last_error{};
            SceneAbiStorage scene_abi{};
            ControlStateAbiStorage control_state_abi{};
        };

        [[nodiscard]] std::string string_from_abi(const char* value, const std::string_view context, const bool allow_empty) {
            const std::string_view view = value == nullptr ? std::string_view{} : std::string_view{value};
            if (!allow_empty && view.empty()) throw std::runtime_error(std::format("{} must not be empty", context));
            return std::string{view};
        }

        [[nodiscard]] std::vector<Option> options_from_abi(const SpectraSceneOptionSpan options, const std::string_view context) {
            if (options.count != 0u && options.data == nullptr) throw std::runtime_error(std::format("{} pointer is null", context));
            if (options.count > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max())) throw std::runtime_error(std::format("{} count is too large", context));
            std::vector<Option> converted{};
            converted.reserve(static_cast<std::size_t>(options.count));
            const std::span<const SpectraSceneOption> option_span{options.data, static_cast<std::size_t>(options.count)};
            for (const SpectraSceneOption& option : option_span) {
                converted.push_back(Option{
                    .key   = string_from_abi(option.key, std::format("{} key", context), false),
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

        [[nodiscard]] GpuResourceHandleKind gpu_handle_kind_from_abi(const std::uint32_t kind) {
            switch (kind) {
            case 1u: return GpuResourceHandleKind::OpaqueWin32;
            case 2u: return GpuResourceHandleKind::OpaqueFileDescriptor;
            default: throw std::runtime_error(std::format("unknown scene plugin GPU resource handle kind {}", kind));
            }
        }

        [[nodiscard]] GpuDeviceIdentity device_identity_from_abi(const SpectraSceneGpuDeviceIdentity& identity) {
            GpuDeviceIdentity converted{
                .vendor_id        = identity.vendor_id,
                .device_id        = identity.device_id,
                .device_node_mask = identity.device_node_mask,
            };
            for (std::size_t index = 0u; index < converted.device_uuid.size(); ++index) converted.device_uuid[index] = identity.device_uuid[index];
            for (std::size_t index = 0u; index < converted.device_luid.size(); ++index) converted.device_luid[index] = identity.device_luid[index];
            return converted;
        }

        [[nodiscard]] OptionSchemaAbiStorage make_option_schema_abi_storage(const std::vector<OptionSchema>& schemas) {
            OptionSchemaAbiStorage storage{};
            storage.choices.resize(schemas.size());
            storage.schemas.reserve(schemas.size());
            for (std::size_t index = 0u; index < schemas.size(); ++index) {
                const OptionSchema& schema = schemas[index];
                storage.choices[index].reserve(schema.choices.size());
                for (const OptionChoice& choice : schema.choices) storage.choices[index].push_back(SpectraSceneControlOptionChoice{.value = choice.value.c_str(), .label = choice.label.c_str()});
                storage.schemas.push_back(SpectraSceneControlOptionSchema{
                    .key           = schema.key.c_str(),
                    .label         = schema.label.c_str(),
                    .description   = schema.description.c_str(),
                    .kind          = static_cast<std::uint32_t>(schema.kind),
                    .required      = schema.required_flag ? 1u : 0u,
                    .default_value = schema.default_value.c_str(),
                    .section_id    = schema.section_id.c_str(),
                    .choices       = SpectraSceneControlOptionChoiceSpan{.data = storage.choices[index].empty() ? nullptr : storage.choices[index].data(), .count = static_cast<std::uint64_t>(storage.choices[index].size())},
                    .presentation  = schema.presentation,
                    .has_numeric_range = schema.has_numeric_range ? 1u : 0u,
                    .numeric_min   = schema.numeric_min,
                    .numeric_max   = schema.numeric_max,
                    .numeric_step  = schema.numeric_step,
                });
            }
            return storage;
        }

        [[nodiscard]] PluginDescriptorStorage make_plugin_descriptor_storage(const TypeErasedPluginDefinition& descriptor) {
            PluginDescriptorStorage storage{};
            storage.sections.reserve(descriptor.sections.size());
            for (const ControlSection& section : descriptor.sections) {
                storage.sections.push_back(SpectraSceneControlSection{
                    .id    = section.id.c_str(),
                    .label = section.label.c_str(),
                });
            }
            storage.open_options = make_option_schema_abi_storage(descriptor.open_options);
            storage.action_options.reserve(descriptor.actions.size());
            storage.control_actions.reserve(descriptor.actions.size());
            for (const TypeErasedAction& action : descriptor.actions) {
                storage.action_options.push_back(make_option_schema_abi_storage(action.schema.options));
                const OptionSchemaAbiStorage& action_options = storage.action_options.back();
                storage.control_actions.push_back(SpectraSceneControlAction{
                    .id          = action.schema.id.c_str(),
                    .label       = action.schema.label.c_str(),
                    .description = action.schema.description.c_str(),
                    .section_id  = action.schema.section_id.c_str(),
                    .options     = SpectraSceneControlOptionSchemaSpan{.data = action_options.schemas.empty() ? nullptr : action_options.schemas.data(), .count = static_cast<std::uint64_t>(action_options.schemas.size())},
                });
            }
            storage.control_setting_schemas.reserve(descriptor.settings.size());
            for (const TypeErasedSetting& setting : descriptor.settings) storage.control_setting_schemas.push_back(setting.schema);
            storage.control_settings = make_option_schema_abi_storage(storage.control_setting_schemas);
            return storage;
        }

        template <std::size_t Count>
        void copy_array(float (&output)[Count], const std::array<float, Count>& input) {
            for (std::size_t index = 0u; index < Count; ++index) output[index] = input[index];
        }

        [[nodiscard]] SpectraSceneTransform make_transform_view(const Transform& transform) {
            SpectraSceneTransform view{};
            copy_array(view.position, transform.position);
            copy_array(view.rotation, transform.rotation);
            copy_array(view.scale, transform.scale);
            return view;
        }

        [[nodiscard]] SpectraSceneEntityRef make_entity_ref_view(const SceneEntityRef& ref) {
            return SpectraSceneEntityRef{
                .kind = static_cast<std::uint32_t>(ref.kind),
                .name = ref.name.c_str(),
            };
        }

        [[nodiscard]] SpectraSceneMaterial make_material_view(const Material& material) {
            SpectraSceneMaterial view{
                .name                     = material.name.c_str(),
                .model                    = material.model.c_str(),
                .alpha_mode               = material.alpha_mode.c_str(),
                .roughness                = material.roughness,
                .alpha_cutoff             = 0.5f,
                .volume_density_scale     = material.volume_density_scale,
                .volume_temperature_scale = material.volume_temperature_scale,
            };
            copy_array(view.base_color, material.base_color);
            return view;
        }

        [[nodiscard]] SpectraSceneLight make_light_view(const Light& light) {
            SpectraSceneLight view{
                .name               = light.name.c_str(),
                .kind               = light.kind.c_str(),
                .transform          = make_transform_view(light.transform),
                .intensity          = light.intensity,
                .cone_angle_degrees = 30.0f,
            };
            copy_array(view.color, light.color);
            return view;
        }

        void make_volume_abi_views(SceneAbiStorage& cache, const std::vector<VolumeGrid>& volumes) {
            cache.volume_channel_storage.clear();
            cache.volume_views.clear();
            cache.volume_channel_storage.resize(volumes.size());
            cache.volume_views.reserve(volumes.size());
            for (std::size_t volume_index = 0u; volume_index < volumes.size(); ++volume_index) {
                const VolumeGrid& volume = volumes[volume_index];
                cache.volume_channel_storage[volume_index].reserve(volume.channels.size());
                for (const VolumeChannel& channel : volume.channels) {
                    SpectraSceneVolumeChannel channel_view{
                        .name                    = channel.name.c_str(),
                        .format                  = static_cast<std::uint32_t>(channel.format),
                        .source_kind             = static_cast<std::uint32_t>(channel.source_kind),
                        .index_encoding          = static_cast<std::uint32_t>(channel.index_encoding),
                        .buffer_id               = channel.buffer_id,
                        .external_device_pointer = channel.external_device_pointer,
                        .source_byte_size        = channel.source_byte_size,
                        .revision                = channel.revision,
                    };
                    channel_view.dimensions[0] = channel.dimensions[0];
                    channel_view.dimensions[1] = channel.dimensions[1];
                    channel_view.dimensions[2] = channel.dimensions[2];
                    cache.volume_channel_storage[volume_index].push_back(channel_view);
                }

                SpectraSceneVolume volume_view{
                    .name          = volume.name.c_str(),
                    .channels      = SpectraSceneVolumeChannelSpan{.data = cache.volume_channel_storage[volume_index].empty() ? nullptr : cache.volume_channel_storage[volume_index].data(), .count = static_cast<std::uint64_t>(cache.volume_channel_storage[volume_index].size())},
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

        [[nodiscard]] SpectraSceneCamera make_camera_view(const Camera& camera) {
            SpectraSceneCamera view{
                .name                    = camera.name.c_str(),
                .local_coordinate_system = camera.local_coordinate_system.c_str(),
                .transform               = make_transform_view(camera.transform),
                .projection              = static_cast<std::uint32_t>(camera.projection),
                .vertical_fov_degrees    = camera.vertical_fov_degrees,
                .image_width             = camera.image_width,
                .image_height            = camera.image_height,
                .fx                      = camera.fx,
                .fy                      = camera.fy,
                .cx                      = camera.cx,
                .cy                      = camera.cy,
                .near_plane              = camera.near_plane,
                .far_plane               = camera.far_plane,
                .has_image               = camera.image.has_value() ? 1u : 0u,
            };
            copy_array(view.target, camera.target);
            copy_array(view.up, camera.up);
            if (camera.image.has_value()) {
                const CameraImage& image = *camera.image;
                view.image = SpectraSceneCameraImage{
                    .rgba8      = image.rgba8,
                    .rgba8_size = image.rgba8_size,
                    .revision   = image.revision,
                    .width      = image.width,
                    .height     = image.height,
                };
            }
            return view;
        }

        [[nodiscard]] SpectraSceneViewportVoxelGrid make_voxel_grid_view(const ViewportVoxelGrid& grid) {
            SpectraSceneViewportVoxelGrid view{
                .name             = grid.name.c_str(),
                .owner            = make_entity_ref_view(grid.owner),
                .transform        = make_transform_view(grid.transform),
                .cell_scale       = grid.cell_scale,
                .depth_mode       = grid.depth_mode,
                .source_kind      = static_cast<std::uint32_t>(grid.source_kind),
                .index_encoding   = static_cast<std::uint32_t>(grid.index_encoding),
                .buffer_id        = grid.buffer_id,
                .source_byte_size = grid.source_byte_size,
                .revision         = grid.revision,
            };
            view.dimensions[0] = grid.dimensions[0];
            view.dimensions[1] = grid.dimensions[1];
            view.dimensions[2] = grid.dimensions[2];
            copy_array(view.origin, grid.origin);
            copy_array(view.voxel_size, grid.voxel_size);
            copy_array(view.color, grid.color);
            return view;
        }

        [[nodiscard]] SpectraSceneDocumentView make_document_abi_view(SceneAbiStorage& cache) {
            cache.material_views.clear();
            cache.material_views.reserve(cache.document.materials.size());
            for (const Material& material : cache.document.materials) cache.material_views.push_back(make_material_view(material));
            cache.light_views.clear();
            cache.light_views.reserve(cache.document.lights.size());
            for (const Light& light : cache.document.lights) cache.light_views.push_back(make_light_view(light));
            make_volume_abi_views(cache, cache.document.volumes);
            cache.camera_views.clear();
            cache.camera_views.reserve(cache.document.cameras.size());
            for (const Camera& camera : cache.document.cameras) cache.camera_views.push_back(make_camera_view(camera));
            cache.voxel_grid_views.clear();
            cache.voxel_grid_views.reserve(cache.document.debug_attachments.viewport_voxel_grids.size());
            for (const ViewportVoxelGrid& grid : cache.document.debug_attachments.viewport_voxel_grids) cache.voxel_grid_views.push_back(make_voxel_grid_view(grid));
            return SpectraSceneDocumentView{
                .struct_size               = sizeof(SpectraSceneDocumentView),
                .default_coordinate_system = cache.document.default_coordinate_system.c_str(),
                .active_camera_name        = cache.document.active_camera_name.c_str(),
                .items =
                    SpectraSceneItems{
                        .materials               = SpectraSceneMaterialSpan{.data = cache.material_views.empty() ? nullptr : cache.material_views.data(), .count = static_cast<std::uint64_t>(cache.material_views.size())},
                        .lights                  = SpectraSceneLightSpan{.data = cache.light_views.empty() ? nullptr : cache.light_views.data(), .count = static_cast<std::uint64_t>(cache.light_views.size())},
                        .cameras                 = SpectraSceneCameraSpan{.data = cache.camera_views.empty() ? nullptr : cache.camera_views.data(), .count = static_cast<std::uint64_t>(cache.camera_views.size())},
                        .volumes                 = SpectraSceneVolumeSpan{.data = cache.volume_views.empty() ? nullptr : cache.volume_views.data(), .count = static_cast<std::uint64_t>(cache.volume_views.size())},
                        .viewport_voxel_grids    = SpectraSceneViewportVoxelGridSpan{.data = cache.voxel_grid_views.empty() ? nullptr : cache.voxel_grid_views.data(), .count = static_cast<std::uint64_t>(cache.voxel_grid_views.size())},
                    },
            };
        }

        [[nodiscard]] SpectraSceneControlStateView make_control_state_abi_view(ControlStateAbiStorage& cache) {
            cache.metric_views.clear();
            cache.action_state_views.clear();
            cache.metric_views.reserve(cache.state.metrics.size());
            cache.action_state_views.reserve(cache.state.action_states.size());
            for (const Metric& metric : cache.state.metrics) {
                cache.metric_views.push_back(SpectraSceneControlMetric{
                    .key             = metric.key.c_str(),
                    .label           = metric.label.c_str(),
                    .value           = metric.value.c_str(),
                    .section_id      = metric.section_id.c_str(),
                    .display_flags   = metric.display_flags,
                    .has_color       = metric.has_color ? 1u : 0u,
                    .color           = {},
                });
                copy_array(cache.metric_views.back().color, metric.color);
            }
            for (const ActionState& action_state : cache.state.action_states) {
                cache.action_state_views.push_back(SpectraSceneControlActionState{
                    .action_id       = action_state.action_id.c_str(),
                    .enabled         = action_state.enabled ? 1u : 0u,
                    .disabled_reason = action_state.disabled_reason.c_str(),
                });
            }
            return SpectraSceneControlStateView{
                .struct_size   = sizeof(SpectraSceneControlStateView),
                .phase         = cache.state.phase.c_str(),
                .headline      = cache.state.headline.c_str(),
                .detail        = cache.state.detail.c_str(),
                .metrics       = SpectraSceneControlMetricSpan{.data = cache.metric_views.empty() ? nullptr : cache.metric_views.data(), .count = static_cast<std::uint64_t>(cache.metric_views.size())},
                .action_states = SpectraSceneControlActionStateSpan{.data = cache.action_state_views.empty() ? nullptr : cache.action_state_views.data(), .count = static_cast<std::uint64_t>(cache.action_state_views.size())},
            };
        }

        [[nodiscard]] PluginInstance& checked_instance(SpectraSceneInstance* instance, const std::string_view action) {
            if (instance == nullptr) throw std::runtime_error(std::format("{} instance pointer is null", action));
            return *static_cast<PluginInstance*>(instance);
        }

        [[nodiscard]] SpectraSceneResult scene_create(const SpectraSceneOpenInfo* open_info, SpectraSceneInstance** instance) noexcept {
            try {
                if (open_info == nullptr) {
                    PluginExportState::instance().export_error = "create open info pointer is null";
                    return SPECTRA_SCENE_RESULT_ERROR;
                }
                if (instance == nullptr) {
                    PluginExportState::instance().export_error = "create instance output pointer is null";
                    return SPECTRA_SCENE_RESULT_ERROR;
                }
                *instance = nullptr;
                if (open_info->struct_size != sizeof(SpectraSceneOpenInfo)) throw std::runtime_error("scene plugin open info ABI size mismatch");
                const TypeErasedPluginDefinition& definition = PluginExportState::instance().definition;
                std::vector<Option> options                  = options_from_abi(open_info->options, "scene plugin open options");
                if (open_info->host_services == nullptr) throw std::runtime_error("scene plugin open info host services pointer is null");
                if (open_info->host_services->struct_size != sizeof(SpectraSceneHostServices)) throw std::runtime_error("scene plugin host services ABI size mismatch");
                if (open_info->host_services->request_gpu_buffer == nullptr) throw std::runtime_error("scene plugin host services request_gpu_buffer function is null");
                if (open_info->host_services->release_gpu_buffer == nullptr) throw std::runtime_error("scene plugin host services release_gpu_buffer function is null");
                if (open_info->host_services->last_error == nullptr) throw std::runtime_error("scene plugin host services last_error function is null");
                const SpectraSceneHostServices* host_services_view = open_info->host_services;
                auto host_services                                 = std::make_shared<HostServices>();
                host_services->request_gpu_buffer                  = [host_services_view](const std::uint32_t kind, const std::uint64_t byte_size, const std::string_view debug_name) {
                    const std::string debug_name_text{debug_name};
                    std::uint32_t abi_kind{};
                    switch (kind) {
                    case GpuBufferKindVolumeChannel: abi_kind = SPECTRA_SCENE_GPU_BUFFER_VOLUME_CHANNEL; break;
                    case GpuBufferKindViewportVoxelGrid: abi_kind = SPECTRA_SCENE_GPU_BUFFER_VIEWPORT_VOXEL_GRID; break;
                    default: throw std::runtime_error(std::format("unknown scene plugin GPU buffer kind {}", kind));
                    }
                    const SpectraSceneGpuBufferRequest request{
                                         .struct_size = sizeof(SpectraSceneGpuBufferRequest),
                                         .kind        = abi_kind,
                                         .byte_size   = byte_size,
                                         .debug_name  = debug_name_text.c_str(),
                    };
                    SpectraSceneGpuBufferAllocation allocation{};
                    const SpectraSceneResult result = host_services_view->request_gpu_buffer(host_services_view->user_data, &request, &allocation);
                    if (result != SPECTRA_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
                    if (allocation.struct_size != sizeof(SpectraSceneGpuBufferAllocation)) throw std::runtime_error("scene plugin GPU buffer allocation ABI size mismatch");
                    if (allocation.kind != abi_kind) throw std::runtime_error(std::format("scene plugin GPU buffer allocation kind {} does not match request kind {}", allocation.kind, abi_kind));
                    return GpuBufferAllocation{
                                         .resource_id     = allocation.resource_id,
                                         .byte_size       = allocation.byte_size,
                                         .kind            = kind,
                                         .handle_kind     = gpu_handle_kind_from_abi(allocation.handle_kind),
                                         .handle          = allocation.handle,
                                         .device_identity = device_identity_from_abi(allocation.device_identity),
                    };
                };
                host_services->release_gpu_buffer = [host_services_view](const std::uint64_t resource_id) {
                    const SpectraSceneResult result = host_services_view->release_gpu_buffer(host_services_view->user_data, resource_id);
                    if (result != SPECTRA_SCENE_RESULT_OK) throw std::runtime_error(host_services_error(*host_services_view));
                };
                auto created        = std::make_unique<PluginInstance>();
                created->definition = &definition;
                created->project    = definition.open(OpenContext{.options = std::move(options), .host_services = std::move(host_services)});
                *instance           = reinterpret_cast<SpectraSceneInstance*>(created.release());
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        void scene_destroy(SpectraSceneInstance* instance) noexcept {
            const auto plugin_instance = static_cast<PluginInstance*>(instance);
            if (plugin_instance == nullptr) return;
            if (plugin_instance->definition != nullptr && plugin_instance->project != nullptr) plugin_instance->definition->destroy(plugin_instance->project);
            delete plugin_instance;
        }

        [[nodiscard]] SpectraSceneResult scene_document(SpectraSceneInstance* instance, SpectraSceneDocumentView* document) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "document");
                if (document == nullptr) throw std::runtime_error("document output pointer is null");
                plugin_instance.last_error.clear();
                SceneBuilder builder{};
                plugin_instance.definition->write_scene(plugin_instance.project, builder);
                plugin_instance.scene_abi.document = builder.document();
                *document                          = make_document_abi_view(plugin_instance.scene_abi);
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
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
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        [[nodiscard]] const char* last_error(SpectraSceneInstance* instance) noexcept {
            if (instance == nullptr) return PluginExportState::instance().export_error.c_str();
            return static_cast<PluginInstance*>(instance)->last_error.c_str();
        }

        [[nodiscard]] SpectraSceneResult scene_update(SpectraSceneInstance* instance, const SpectraSceneUpdateInfo* update_info) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "scene_update");
                if (update_info == nullptr) throw std::runtime_error("scene_update info pointer is null");
                if (update_info->struct_size != sizeof(SpectraSceneUpdateInfo)) throw std::runtime_error("scene_update info ABI size mismatch");
                plugin_instance.last_error.clear();
                plugin_instance.definition->update(plugin_instance.project, UpdateInfo{
                                                                                .wall_delta_seconds  = update_info->wall_delta_seconds,
                                                                                .scene_delta_seconds = update_info->scene_delta_seconds,
                                                                                .time_seconds        = update_info->time_seconds,
                                                                                .frame_index         = update_info->frame_index,
                                                                                .timeline_playing    = update_info->timeline_playing != 0u,
                                                                            });
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        [[nodiscard]] SpectraSceneResult scene_revision(SpectraSceneInstance* instance, std::uint64_t* revision) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "scene_revision");
                if (revision == nullptr) throw std::runtime_error("scene_revision output pointer is null");
                plugin_instance.last_error.clear();
                *revision = plugin_instance.definition->revision(plugin_instance.project);
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        [[nodiscard]] SpectraSceneResult control_action(SpectraSceneInstance* instance, const char* action_id, const SpectraSceneOptionSpan options) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "control_action");
                plugin_instance.last_error.clear();
                const std::string action                    = string_from_abi(action_id, "control action id", false);
                const std::vector<Option> converted_options = options_from_abi(options, "control action options");
                const auto action_binding                   = std::ranges::find_if(plugin_instance.definition->actions, [&action](const TypeErasedAction& candidate) { return candidate.schema.id == action; });
                if (action_binding == plugin_instance.definition->actions.end()) throw std::runtime_error(std::format("unknown control action '{}'", action));
                action_binding->invoke(plugin_instance.project, ActionContext{.options = converted_options});
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        [[nodiscard]] SpectraSceneResult control_setting_update(SpectraSceneInstance* instance, const char* key, const char* value) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "control_setting_update");
                plugin_instance.last_error.clear();
                const std::string setting_key   = string_from_abi(key, "control setting key", false);
                const std::string setting_value = string_from_abi(value, "control setting value", true);
                const auto setting              = std::ranges::find_if(plugin_instance.definition->settings, [&setting_key](const TypeErasedSetting& candidate) { return candidate.schema.key == setting_key; });
                if (setting == plugin_instance.definition->settings.end()) throw std::runtime_error(std::format("unknown control setting '{}'", setting_key));
                setting->update(plugin_instance.project, setting_value);
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        [[nodiscard]] SpectraSceneResult control_state(SpectraSceneInstance* instance, SpectraSceneControlStateView* state) noexcept {
            try {
                PluginInstance& plugin_instance = checked_instance(instance, "control_state");
                if (state == nullptr) throw std::runtime_error("control_state output pointer is null");
                plugin_instance.last_error.clear();
                ControlBuilder builder{};
                plugin_instance.definition->write_controls(plugin_instance.project, builder);
                plugin_instance.control_state_abi.state = builder.state();
                *state                                  = make_control_state_abi_view(plugin_instance.control_state_abi);
                return SPECTRA_SCENE_RESULT_OK;
            } catch (const std::exception& error) {
                if (instance != nullptr)
                    static_cast<PluginInstance*>(instance)->last_error = error.what();
                else
                    PluginExportState::instance().export_error = error.what();
                return SPECTRA_SCENE_RESULT_ERROR;
            }
        }

        PluginExportState::PluginExportState(const TypeErasedPluginDefinition& plugin_definition)
            : definition(plugin_definition), descriptor_storage(make_plugin_descriptor_storage(plugin_definition)), plugin(SpectraScenePlugin{
                                                                                                                        .abi_version             = plugin_abi_version,
                                                                                                                        .struct_size             = sizeof(SpectraScenePlugin),
                                                                                                                        .id                      = plugin_definition.id.c_str(),
                                                                                                                        .title                   = plugin_definition.title.c_str(),
                                                                                                                        .open_action_label       = plugin_definition.open_action_label.c_str(),
                                                                                                                        .frames_per_second       = plugin_definition.frames_per_second,
                                                                                                                        .sections                = SpectraSceneControlSectionSpan{.data = descriptor_storage.sections.empty() ? nullptr : descriptor_storage.sections.data(), .count = static_cast<std::uint64_t>(descriptor_storage.sections.size())},
                                                                                                                        .open_options            = SpectraSceneControlOptionSchemaSpan{.data = descriptor_storage.open_options.schemas.empty() ? nullptr : descriptor_storage.open_options.schemas.data(), .count = static_cast<std::uint64_t>(descriptor_storage.open_options.schemas.size())},
                                                                                                                        .control_actions         = SpectraSceneControlActionSpan{.data = descriptor_storage.control_actions.empty() ? nullptr : descriptor_storage.control_actions.data(), .count = static_cast<std::uint64_t>(descriptor_storage.control_actions.size())},
                                                                                                                        .control_settings        = SpectraSceneControlOptionSchemaSpan{.data = descriptor_storage.control_settings.schemas.empty() ? nullptr : descriptor_storage.control_settings.schemas.data(), .count = static_cast<std::uint64_t>(descriptor_storage.control_settings.schemas.size())},
                                                                                                                        .create                  = scene_create,
                                                                                                                        .destroy                 = scene_destroy,
                                                                                                                        .update                  = scene_update,
                                                                                                                        .document                = scene_document,
                                                                                                                        .frame                   = scene_frame,
                                                                                                                        .scene_revision          = scene_revision,
                                                                                                                        .control_action          = control_action,
                                                                                                                        .control_setting_update  = control_setting_update,
                                                                                                                        .control_state           = control_state,
                                                                                                                        .last_error              = last_error,
                                                                                                                    }) {}

        PluginExportState& PluginExportState::instance(const TypeErasedPluginDefinition* plugin_definition) {
            static std::optional<PluginExportState> state{};
            if (!state.has_value()) {
                if (plugin_definition == nullptr) throw std::runtime_error("scene plugin export state is not initialized");
                state.emplace(*plugin_definition);
            }
            if (plugin_definition != nullptr && &state->definition != plugin_definition) throw std::runtime_error("scene plugin export state is already initialized with a different definition");
            return *state;
        }
    } // namespace

    const SpectraScenePlugin* export_type_erased_plugin(const TypeErasedPluginDefinition& definition) {
        const PluginExportState& state = PluginExportState::instance(&definition);
        return &state.plugin;
    }

} // namespace ngp::plugin
