module;
#include "ngp.train.h"
export module ngp.train;
export import ngp.dataset;
import std;

namespace ngp::train {
    export struct TrainStats final {
        std::uint32_t step                                    = 0u;
        std::uint32_t rays_per_batch                          = 0u;
        std::uint32_t measured_sample_count_before_compaction = 0u;
        std::uint32_t measured_sample_count                   = 0u;
        std::uint32_t density_grid_occupied_cells             = 0u;
        float loss                                            = 0.0f;
        float elapsed_ms                                      = 0.0f;
        float density_grid_update_ms                          = 0.0f;
        float density_grid_occupancy_ratio                    = 0.0f;
    };

    export struct ValidationStats final {
        std::uint32_t step        = 0u;
        std::uint32_t image_count = 0u;
        std::uint64_t pixel_count = 0u;
        float mse                 = 0.0f;
        float psnr                = 0.0f;
        float elapsed_ms          = 0.0f;
    };

    export struct TestStats final {
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

        std::expected<TrainStats, std::string> train(std::int32_t iters);
        std::expected<ValidationStats, std::string> validate() const;
        std::expected<TestStats, std::string> test() const;
        std::expected<void, std::string> export_weights(const std::filesystem::path& path) const;
        std::expected<void, std::string> load_weights(const std::filesystem::path& path);

    private:
        struct HostData {
            // Stable after construction: dataset metadata.
            std::uint32_t frame_count            = 0u;
            std::uint32_t width                  = 0u;
            std::uint32_t height                 = 0u;
            float focal_x                        = 0.0f;
            float focal_y                        = 0.0f;
            float principal_x                    = 0.0f;
            float principal_y                    = 0.0f;
            std::uint32_t validation_frame_count = 0u;
            std::uint32_t validation_width       = 0u;
            std::uint32_t validation_height      = 0u;
            float validation_focal_x             = 0.0f;
            float validation_focal_y             = 0.0f;
            float validation_principal_x         = 0.0f;
            float validation_principal_y         = 0.0f;
            std::uint32_t test_frame_count       = 0u;
            std::uint32_t test_width             = 0u;
            std::uint32_t test_height            = 0u;
            float test_focal_x                   = 0.0f;
            float test_focal_y                   = 0.0f;
            float test_principal_x               = 0.0f;
            float test_principal_y               = 0.0f;
            float scene_scale                    = 0.0f;

            // Mutated by train(): step, adaptive batch shape, and latest counters.
            std::uint32_t current_step                            = 0u;
            std::uint32_t rays_per_batch                          = cuda::config::INITIAL_RAYS_PER_BATCH;
            std::uint32_t inference_sample_count                  = cuda::config::MAX_SAMPLES;
            std::uint32_t measured_sample_count_before_compaction = 0u;
            std::uint32_t measured_sample_count                   = 0u;
            std::uint32_t density_grid_ema_step                   = 0u;
            std::uint32_t density_grid_occupied_cells             = 0u;
            float density_grid_update_ms                          = 0.0f;
        } host;

        struct DeviceData {
            // Dataset.
            const std::uint8_t* pixels            = nullptr;
            const float* camera                   = nullptr;
            const std::uint8_t* validation_pixels = nullptr;
            const float* validation_camera        = nullptr;
            const std::uint8_t* test_pixels       = nullptr;
            const float* test_camera              = nullptr;

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
            std::uint8_t* test_comparison_pixels       = nullptr;

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
