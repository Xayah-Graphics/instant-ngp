#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include "common.cuh"
#include <cstdint>
#include <filesystem>
#include <optional>
#include <vector>

namespace ngp {
    class InstantNGP final {
    public:
        enum class ActivationMode { None, ReLU, Exponential, Sigmoid, Squareplus, Softplus, Tanh, LeakyReLU };

        struct NetworkConfig final {
            struct HashGridConfig final {
                std::uint32_t n_levels               = 8u;
                std::uint32_t n_features_per_level   = 4u;
                std::uint32_t log2_hashmap_size      = 19u;
                std::uint32_t base_resolution        = 16u;
                std::optional<float> per_level_scale = {};
                bool stochastic_interpolation        = false;
            } encoding = {};

            struct DirectionEncodingConfig final {
                std::uint32_t sh_degree = 4u;
            } direction_encoding = {};

            struct FullyFusedMlpConfig final {
                std::uint32_t n_hidden_layers    = 1u;
                ActivationMode activation        = ActivationMode::ReLU;
                ActivationMode output_activation = ActivationMode::None;
            } density_network = {}, rgb_network = {};

            struct AdamConfig final {
                float learning_rate = 1e-2f;
                float beta1         = 0.9f;
                float beta2         = 0.99f;
                float epsilon       = 1e-15f;
                float l2_reg        = 1e-6f;
            } optimizer = {};
        };

        struct InferenceCamera final {
            legacy::math::ivec2 resolution = {};
            float focal_length             = 0.0f;
            legacy::math::mat4x3 camera    = {};
        };

        struct TrainResult final {
            float loss                                          = 0.0f;
            float train_ms                                      = 0.0f;
            float prep_ms                                       = 0.0f;
            std::uint32_t step                                  = 0u;
            std::uint32_t batch_size                            = 0u;
            std::uint32_t rays_per_batch                        = 0u;
            std::uint32_t measured_batch_size                   = 0u;
            std::uint32_t measured_batch_size_before_compaction = 0u;
        };

        struct ValidateResult final {
            std::uint32_t image_count  = 0u;
            std::uint64_t total_pixels = 0u;
            float mean_mse             = 0.0f;
            float mean_psnr            = 0.0f;
            float split_psnr           = 0.0f;
            float min_psnr             = 0.0f;
            float max_psnr             = 0.0f;
            float benchmark_ms         = 0.0f;
        };

        struct TestResult final {
            std::uint32_t image_count  = 0u;
            std::uint64_t total_pixels = 0u;
            float mean_mse             = 0.0f;
            float mean_psnr            = 0.0f;
            float split_psnr           = 0.0f;
            float min_psnr             = 0.0f;
            float max_psnr             = 0.0f;
            float benchmark_ms         = 0.0f;
        };

        struct InferenceResult final {
            std::uint32_t width  = 0u;
            std::uint32_t height = 0u;
            float render_ms      = 0.0f;
        };

        struct DatasetState final {
            struct HostData final {
                struct Frame final {
                    std::vector<std::uint8_t> rgba = {};
                    legacy::math::ivec2 resolution = {};
                    float focal_length             = 0.0f;
                    legacy::math::mat4x3 camera    = {};
                };

                std::vector<Frame> train      = {};
                std::vector<Frame> validation = {};
                std::vector<Frame> test       = {};
            } host = {};

            struct DeviceData final {
                struct GpuFrame final {
                    const std::uint8_t* pixels     = nullptr;
                    legacy::math::ivec2 resolution = {};
                    float focal_length             = 0.0f;
                    legacy::math::mat4x3 camera    = {};
                };

                std::vector<legacy::GpuBuffer<std::uint8_t>> pixels = {};
                legacy::GpuBuffer<GpuFrame> frames                  = {};
            } device = {};
        };

        explicit InstantNGP(const NetworkConfig& network_config);
        ~InstantNGP() noexcept;
        InstantNGP(const InstantNGP&)                      = delete;
        InstantNGP& operator=(const InstantNGP&)           = delete;
        InstantNGP(InstantNGP&& other) noexcept            = delete;
        InstantNGP& operator=(InstantNGP&& other) noexcept = delete;

        void load_dataset(const std::filesystem::path& dataset_path);
        [[nodiscard]] auto train(std::int32_t iters) -> TrainResult;
        [[nodiscard]] auto validate(const std::filesystem::path& report_path) -> ValidateResult;
        [[nodiscard]] auto test(const std::filesystem::path& report_path) -> TestResult;
        [[nodiscard]] auto inference(const std::filesystem::path& output_path, const InferenceCamera& camera) const -> InferenceResult;

        DatasetState dataset = {};

    private:
        void density(cudaStream_t stream, const legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic>& output) const;
        void inference(cudaStream_t stream, const legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic>& output) const;
        void forward(cudaStream_t stream, const legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic>& input, legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic>* output);
        void backward(cudaStream_t stream, const legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic>& input, const legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic>& output, const legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic>& dL_doutput);

        struct ModelState;
        struct ModelScratch;
        struct Optimizer;

        struct TrainPlan final {
            struct NetworkStage final {
                std::uint32_t n_pos_dims               = 0u;
                std::uint32_t n_dir_dims               = 0u;
                std::uint32_t dir_offset               = 0u;
                std::uint32_t density_input_dims       = 0u;
                std::uint32_t density_output_dims      = 0u;
                std::uint32_t rgb_input_dims           = 0u;
                std::uint32_t rgb_output_dims          = 3u;
            } network = {};

            struct TrainingStage final {
                std::uint32_t batch_size          = 0u;
                std::uint32_t floats_per_coord    = 0u;
                std::uint32_t padded_output_width = 0u;
                std::uint32_t max_samples         = 0u;
            } training = {};

            struct TrainingPrepStage final {
                std::uint32_t warmup_steps              = 256u;
                std::uint32_t skip_growth_interval      = 16u;
                std::uint32_t max_skip                  = 16u;
                std::uint32_t uniform_samples_warmup    = 0u;
                std::uint32_t uniform_samples_steady    = 0u;
                std::uint32_t nonuniform_samples_steady = 0u;
            } prep = {};

            struct DensityGridStage final {
                std::uint32_t padded_output_width = 0u;
                std::uint32_t query_batch_size    = 0u;
                std::uint32_t n_elements          = 0u;
            } density_grid = {};

            struct ValidationStage final {
                std::uint32_t tile_rays           = 4096u;
                std::uint32_t max_samples_per_ray = 1024u;
                std::uint32_t floats_per_coord    = 0u;
                std::uint32_t padded_output_width = 0u;
                std::uint32_t max_samples         = 0u;
            } validation = {};
        };

        struct SamplingState final {
            struct DensityGrid final {
                legacy::GpuBuffer<float> values           = {};
                legacy::GpuBuffer<std::uint8_t> occupancy = {};
                legacy::GpuBuffer<float> reduction        = {};
                std::uint32_t ema_step                    = 0u;
                float ema_decay                           = 0.95f;
            } density = {};

            struct UpdateWorkspace final {
                legacy::GpuBuffer<char> arena = {};
                float* positions              = nullptr;
                std::uint32_t* indices        = nullptr;
                float* density_scratch        = nullptr;
                __half* mlp_out               = nullptr;
            } update = {};

            legacy::math::vec3 aabb_min     = legacy::math::vec3(0.0f);
            legacy::math::vec3 aabb_max     = legacy::math::vec3(1.0f);
            bool snap_to_pixel_centers      = true;
            float near_distance             = 0.1f;
            legacy::math::pcg32 density_rng = {};
        };

        struct TrainingState final {
            struct Counters final {
                legacy::GpuBuffer<std::uint32_t> numsteps_counter           = {};
                legacy::GpuBuffer<std::uint32_t> numsteps_counter_compacted = {};
                legacy::GpuBuffer<float> loss                               = {};

                std::uint32_t rays_per_batch                        = 1u << 12;
                std::uint32_t measured_batch_size                   = 0u;
                std::uint32_t measured_batch_size_before_compaction = 0u;
            } counters = {};

            struct StepWorkspace final {
                legacy::GpuBuffer<char> arena                           = {};
                std::uint32_t* ray_indices                              = nullptr;
                void* rays_unnormalized                                 = nullptr;
                std::uint32_t* numsteps                                 = nullptr;
                float* coords                                           = nullptr;
                __half* mlp_out                                         = nullptr;
                __half* dloss_dmlp_out                                  = nullptr;
                float* coords_compacted                                 = nullptr;
                std::uint32_t* ray_counter                              = nullptr;
                std::uint32_t max_samples                               = 0u;
                std::uint32_t max_inference                             = 0u;
                std::uint32_t floats_per_coord                          = 0u;
                std::uint32_t padded_output_width                       = 0u;
                legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic> coords_matrix           = {};
                legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic> rgbsigma_matrix        = {};
                legacy::GPUMatrix<float, legacy::MatrixLayout::Dynamic> compacted_coords_matrix = {};
                legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic> gradient_matrix        = {};
                legacy::GPUMatrix<__half, legacy::MatrixLayout::Dynamic> compacted_output       = {};
            } workspace = {};

            legacy::GpuBuffer<float> loss_reduction = {};
            legacy::math::pcg32 rng                 = {};
            std::uint32_t step                      = 0u;
            float last_prep_ms                      = 0.0f;
            float last_train_ms                     = 0.0f;
            float last_loss                         = 0.0f;
        };

        struct RenderWorkspace final {
            legacy::GpuBuffer<legacy::math::vec3> rendered    = {};
            legacy::GpuBuffer<std::uint32_t> tile_numsteps    = {};
            legacy::GpuBuffer<float> tile_coords              = {};
            legacy::GpuBuffer<__half> tile_mlp_out            = {};
            legacy::GpuBuffer<std::uint32_t> sample_counter   = {};
            legacy::GpuBuffer<std::uint32_t> overflow_counter = {};
        };

        NetworkConfig network_config             = {};
        TrainPlan train_plan                     = {};
        std::uint32_t seed                       = 1337u;
        SamplingState sampling                   = {};
        TrainingState training                   = {};
        mutable RenderWorkspace render_workspace = {};
        cudaStream_t stream                      = {};
        std::vector<cudaStream_t> aux_streams    = {};
        std::vector<cudaEvent_t> aux_events      = {};
        legacy::GpuBuffer<char> parameter_buffer = {};
        float* full_precision_params             = nullptr;
        __half* network_params                   = nullptr;
        __half* network_param_gradients          = nullptr;
        ModelState* model                        = nullptr;
        ModelScratch* model_scratch              = nullptr;
        Optimizer* optimizer                     = nullptr;
        cudaGraph_t graph                        = nullptr;
        cudaGraphExec_t graph_instance           = nullptr;
        bool synchronize_when_capture_done       = false;
    };

} // namespace ngp

#endif // INSTANT_NGP_H
