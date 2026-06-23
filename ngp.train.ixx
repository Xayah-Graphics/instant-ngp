export module ngp.train;
import std;

namespace ngp::train {
    export template <typename Frame>
    concept FrameLike =
        requires(const Frame& frame) {
            requires std::ranges::contiguous_range<decltype((frame.rgba))>;
            requires std::same_as<std::ranges::range_value_t<decltype((frame.rgba))>, std::uint8_t>;
            requires std::ranges::contiguous_range<decltype((frame.camera))>;
            requires std::same_as<std::ranges::range_value_t<decltype((frame.camera))>, float>;
            { frame.width } -> std::convertible_to<std::uint32_t>;
            { frame.height } -> std::convertible_to<std::uint32_t>;
            { frame.focal_x } -> std::convertible_to<float>;
            { frame.focal_y } -> std::convertible_to<float>;
            { frame.principal_x } -> std::convertible_to<float>;
            { frame.principal_y } -> std::convertible_to<float>;
        };

    export template <typename FrameSet>
    concept FrameSetLike =
        requires(const FrameSet& frame_set) {
            { std::string_view{frame_set.name} } -> std::same_as<std::string_view>;
            requires std::ranges::input_range<decltype((frame_set.frames))>;
            requires FrameLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((frame_set.frames))>>>;
        };

    export template <typename Dataset>
    concept DatasetLike =
        requires(const Dataset& dataset) {
            { static_cast<float>(dataset.scene_scale) } -> std::same_as<float>;
            requires std::ranges::input_range<decltype((dataset.frame_sets))>;
            requires FrameSetLike<std::remove_cvref_t<std::ranges::range_reference_t<decltype((dataset.frame_sets))>>>;
        };

    export struct FrameView final {
        std::span<const std::uint8_t> rgba;
        std::span<const float> camera;
        std::uint32_t width = 0u;
        std::uint32_t height = 0u;
        float focal_x = 0.0f;
        float focal_y = 0.0f;
        float principal_x = 0.0f;
        float principal_y = 0.0f;
    };

    export struct FrameSetView final {
        std::string_view name;
        std::span<const FrameView> frames;
    };

    export struct OptimizationRequest final {
        std::string_view frame_set;
        std::int32_t iterations = 1;
    };

    export struct EvaluationRequest final {
        std::string_view frame_set;
        std::optional<std::filesystem::path> comparison_output_dir;
    };

    export struct OptimizationStats final {
        std::uint32_t step                                    = 0u;
        std::uint32_t next_rays_per_batch                     = 0u;
        std::uint32_t measured_sample_count_before_compaction = 0u;
        std::uint32_t measured_sample_count                   = 0u;
        std::uint32_t density_grid_occupied_cells             = 0u;
        float loss                                            = 0.0f;
        float elapsed_ms                                      = 0.0f;
        float sample_efficiency_ratio                         = 0.0f;
        float density_grid_occupancy_ratio                    = 0.0f;
    };

    export struct EvaluationStats final {
        std::string frame_set;
        std::uint32_t step                   = 0u;
        std::uint32_t image_count            = 0u;
        std::uint32_t comparison_image_count = 0u;
        std::uint64_t pixel_count            = 0u;
        float mse                            = 0.0f;
        float psnr                           = 0.0f;
        float elapsed_ms                     = 0.0f;
        std::filesystem::path output_dir;
    };

    export struct InstantNGP final {
        template <DatasetLike Dataset>
        explicit InstantNGP(const Dataset& dataset) {
            std::vector<std::string_view> frame_set_names;
            std::vector<std::vector<FrameView>> frame_views_by_set;
            for (const auto& frame_set : dataset.frame_sets) {
                frame_set_names.push_back(std::string_view{frame_set.name});
                std::vector<FrameView>& frame_views = frame_views_by_set.emplace_back();
                for (const auto& frame : frame_set.frames) {
                    frame_views.push_back(FrameView{
                        .rgba = std::span<const std::uint8_t>{std::ranges::data(frame.rgba), std::ranges::size(frame.rgba)},
                        .camera = std::span<const float>{std::ranges::data(frame.camera), std::ranges::size(frame.camera)},
                        .width = static_cast<std::uint32_t>(frame.width),
                        .height = static_cast<std::uint32_t>(frame.height),
                        .focal_x = static_cast<float>(frame.focal_x),
                        .focal_y = static_cast<float>(frame.focal_y),
                        .principal_x = static_cast<float>(frame.principal_x),
                        .principal_y = static_cast<float>(frame.principal_y),
                    });
                }
            }

            std::vector<FrameSetView> frame_set_views;
            frame_set_views.reserve(frame_views_by_set.size());
            for (const std::size_t frame_set_index : std::views::iota(0uz, frame_views_by_set.size())) {
                frame_set_views.push_back(FrameSetView{
                    .name = frame_set_names[frame_set_index],
                    .frames = std::span<const FrameView>{frame_views_by_set[frame_set_index]},
                });
            }
            this->initialize(frame_set_views, static_cast<float>(dataset.scene_scale));
        }

        ~InstantNGP() noexcept;
        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = delete;
        InstantNGP& operator=(InstantNGP&&) noexcept = delete;

        std::expected<OptimizationStats, std::string> optimize(OptimizationRequest request);
        std::expected<EvaluationStats, std::string> evaluate(EvaluationRequest request) const;
        std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        std::expected<void, std::string> load_weights(const std::filesystem::path& path);

        void initialize(std::span<const FrameSetView> frame_sets, float scene_scale);

        struct HostFrameSet final {
            std::string name;
            std::uint32_t frame_count = 0u;
            std::uint32_t width       = 0u;
            std::uint32_t height      = 0u;
            float focal_x             = 0.0f;
            float focal_y             = 0.0f;
            float principal_x         = 0.0f;
            float principal_y         = 0.0f;
        };

        struct HostData {
            // Stable after construction: dataset metadata.
            std::vector<HostFrameSet> frame_sets = {};
            std::uint32_t comparison_width       = 0u;
            std::uint32_t comparison_height      = 0u;
            float scene_scale                    = 0.0f;

            // Mutated by optimize(): step, adaptive batch shape, and latest counters.
            std::uint32_t current_step                            = 0u;
            std::uint32_t rays_per_batch                          = 0u;
            std::uint32_t inference_sample_count                  = 0u;
            std::uint32_t measured_sample_count_before_compaction = 0u;
            std::uint32_t measured_sample_count                   = 0u;
            mutable std::uint32_t density_grid_ema_step           = 0u;
            std::uint32_t density_grid_occupied_cells             = 0u;
        } host;

        struct DeviceFrameSet final {
            const std::uint8_t* pixels = nullptr;
            const float* camera        = nullptr;
        };

        struct DeviceData {
            // Dataset.
            std::vector<DeviceFrameSet> frame_sets = {};

            // Sampler.
            std::uint8_t* occupancy                    = nullptr;
            float* sample_coords                       = nullptr;
            float* rays                                = nullptr;
            std::uint32_t* ray_indices                 = nullptr;
            std::uint32_t* numsteps                    = nullptr;
            std::uint32_t* ray_counter                 = nullptr;
            std::uint32_t* sample_counter              = nullptr;
            float* density_grid_values                 = nullptr;
            float* density_grid_scratch                = nullptr;
            std::uint32_t* density_grid_indices        = nullptr;
            float* density_grid_mean                   = nullptr;
            std::uint32_t* density_grid_occupied_count = nullptr;

            // Loss and compaction.
            std::uint32_t* compacted_sample_counter    = nullptr;
            float* compacted_sample_coords             = nullptr;
            float* loss_values                         = nullptr;
            std::uint16_t* network_output_gradients    = nullptr;
            std::uint32_t* evaluation_numsteps         = nullptr;
            std::uint32_t* evaluation_sample_counter   = nullptr;
            std::uint32_t* evaluation_overflow_counter = nullptr;
            double* evaluation_loss_sum                = nullptr;
            std::uint8_t* comparison_pixels            = nullptr;

            // Network.
            std::uint16_t* density_input           = nullptr;
            std::uint16_t* rgb_input               = nullptr;
            std::uint16_t* network_output          = nullptr;
            std::uint16_t* rgb_output_gradients    = nullptr;
            std::uint16_t* rgb_input_gradients     = nullptr;
            std::uint16_t* density_input_gradients = nullptr;
            std::uint16_t* density_forward_hidden  = nullptr;
            std::uint16_t* rgb_forward_hidden      = nullptr;
            std::uint16_t* density_backward_hidden = nullptr;
            std::uint16_t* rgb_backward_hidden     = nullptr;
            void* cublaslt_handle                  = nullptr;
            std::uint8_t* cublaslt_workspace       = nullptr;

            // Trainable parameters.
            float* params_full_precision   = nullptr;
            std::uint16_t* params          = nullptr;
            std::uint16_t* param_gradients = nullptr;

            // Optimizer.
            float* optimizer_first_moments       = nullptr;
            float* optimizer_second_moments      = nullptr;
            std::uint32_t* optimizer_param_steps = nullptr;
        } device;
    };
} // namespace ngp::train
