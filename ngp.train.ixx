module;
#include "ngp.train.h"
export module ngp.train;
export import ngp.dataset;
import std;

namespace ngp::train {
    export struct OptimizationRequest final {
        std::string_view frame_set;
        std::int32_t iterations = 1;
    };

    export struct EvaluationRequest final {
        std::string_view frame_set;
        std::optional<std::filesystem::path> comparison_output_dir;
        bool refresh_acceleration = false;
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

    export class InstantNGP final {
    public:
        explicit InstantNGP(const dataset::NGPDataset& dataset);
        ~InstantNGP() noexcept;
        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = delete;
        InstantNGP& operator=(InstantNGP&&) noexcept = delete;

        std::expected<OptimizationStats, std::string> optimize(OptimizationRequest request);
        std::expected<EvaluationStats, std::string> evaluate(EvaluationRequest request) const;
        std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        std::expected<void, std::string> load_weights(const std::filesystem::path& path);

    private:
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
            std::uint32_t rays_per_batch                          = cuda::config::INITIAL_RAYS_PER_BATCH;
            std::uint32_t inference_sample_count                  = cuda::config::MAX_SAMPLES;
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
