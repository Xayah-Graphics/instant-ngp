export module ngp.project;

import ngp.plugin;
import std;

export namespace ngp::project {
    class Project final {
    public:
        struct State;

        Project();
        Project(const Project& other) = delete;
        Project(Project&& other) noexcept;
        Project& operator=(const Project& other) = delete;
        Project& operator=(Project&& other) noexcept;
        ~Project() noexcept;

        [[nodiscard]] static const plugin::PluginDefinition<Project>& plugin();
        [[nodiscard]] static Project open(plugin::OpenContext context);

        void update(const plugin::UpdateInfo& update);
        [[nodiscard]] std::uint64_t revision() const;
        void write_document(plugin::SceneBuilder& scene) const;
        void write_frame(plugin::SceneBuilder& scene, plugin::FrameInfo frame) const;
        void write_controls(plugin::ControlBuilder& controls) const;

        void render_preview(plugin::ActionContext context);

        void set_update_occupancy_grid(bool value);
        void set_show_volume(bool value);
        void set_show_occupancy(bool value);
        void set_occupancy_alpha(float value);
        void set_occupancy_cell_scale(float value);
        void set_show_sampler(bool value);
        void set_show_sampler_points(bool value);
        void set_show_sampler_rays(bool value);
        void set_sampler_point_radius(float value);
        void set_sampler_ray_width(float value);

    private:
        explicit Project(std::unique_ptr<State> state);

        std::unique_ptr<State> state{};
    };
}
