#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H

#include "common.cuh"
#include <array>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <vector>

namespace ngp {

    namespace network {
        template <typename T>
        struct TrainerState;
    } // namespace network

    struct TrainerStateDeleter final {
        void operator()(network::TrainerState<__half>* trainer) const;
    };

    namespace network::detail {
        void free_aux_stream_pool(cudaStream_t parent_stream);
    } // namespace network::detail

    struct StreamState {
        StreamState();
        ~StreamState();
        StreamState& operator=(const StreamState&) = delete;
        StreamState(const StreamState&)            = delete;
        StreamState& operator=(StreamState&& other) noexcept;
        StreamState(StreamState&& other) noexcept;

        cudaStream_t stream = {};
        cudaEvent_t event   = {};
    };

    class InstantNGP final {
    private:
        struct TrainingStepWorkspace;

    public:
        enum class DatasetType { NerfSynthetic };
        enum class ActivationMode { None, ReLU, Exponential, Sigmoid, Squareplus, Softplus, Tanh, LeakyReLU };
        enum class GridStorage { Hash, Dense, Tiled };

        struct GpuFrame final {
            const std::uint8_t* pixels     = nullptr;
            legacy::math::ivec2 resolution = {};
            float focal_length             = 0.0f;
            legacy::math::mat4x3 camera    = {};
        };

        void load_dataset(const std::filesystem::path& dataset_path, DatasetType dataset_type);
        void train(std::int32_t iters);

        struct NetworkConfig {
            struct HashGridConfig {
                uint32_t n_levels             = 8;
                uint32_t n_features_per_level = 4;
                uint32_t log2_hashmap_size    = 19;
                uint32_t base_resolution      = 16;
                std::optional<float> per_level_scale;
                bool stochastic_interpolation = false;
                GridStorage storage           = GridStorage::Hash;
            } encoding;

            struct DirectionEncodingConfig {
                uint32_t sh_degree = 4;
            } direction_encoding;

            struct FullyFusedMlpConfig {
                uint32_t n_hidden_layers         = 1;
                ActivationMode activation        = ActivationMode::ReLU;
                ActivationMode output_activation = ActivationMode::None;
            } density_network, rgb_network;

            struct AdamConfig {
                float learning_rate = 1e-2f;
                float beta1         = 0.9f;
                float beta2         = 0.99f;
                float epsilon       = 1e-15f;
                float l2_reg        = 1e-6f;
            } optimizer;
        };

        explicit InstantNGP(const NetworkConfig& network_config);
        ~InstantNGP();
        InstantNGP(const InstantNGP&)                = delete;
        InstantNGP& operator=(const InstantNGP&)     = delete;
        InstantNGP(InstantNGP&&) noexcept            = default;
        InstantNGP& operator=(InstantNGP&&) noexcept = default;

    protected:
        void run_training_prep();
        void update_density_grid();
        [[nodiscard]] auto begin_training_step() -> TrainingStepWorkspace;

    private:
        struct Dataset final {
            struct CPU final {
                struct Frame final {
                    std::vector<std::uint8_t> rgba             = {};
                    std::uint32_t width                        = 0;
                    std::uint32_t height                       = 0;
                    float focal_length_x                       = 0.0f;
                    float focal_length_y                       = 0.0f;
                    std::array<float, 16> transform_matrix_4x4 = {};
                };

                std::vector<Frame> train      = {};
                std::vector<Frame> validation = {};
                std::vector<Frame> test       = {};
            };

            struct GPU final {
                struct Train final {
                    std::vector<legacy::GpuBuffer<std::uint8_t>> pixels = {};
                    legacy::GpuBuffer<GpuFrame> frames                  = {};
                };

                Train train = {};
            };

            CPU cpu = {};
            GPU gpu = {};
        } dataset = {};

        struct TrainPlan {
            struct NetworkStage {
                uint32_t n_pos_dims               = 0;
                uint32_t n_dir_dims               = 0;
                uint32_t dir_offset               = 0;
                uint32_t density_alignment        = 0;
                uint32_t density_input_dims       = 0;
                uint32_t density_output_dims      = 0;
                uint32_t dir_encoding_output_dims = 0;
                uint32_t rgb_alignment            = 0;
                uint32_t rgb_input_dims           = 0;
                uint32_t rgb_output_dims          = 3;
            } network;

            struct TrainingStage {
                uint32_t batch_size          = 0;
                uint32_t floats_per_coord    = 0;
                uint32_t padded_output_width = 0;
                uint32_t max_samples         = 0;
            } training;

            struct TrainingPrepStage {
                uint32_t warmup_steps              = 256;
                uint32_t skip_growth_interval      = 16;
                uint32_t max_skip                  = 16;
                uint32_t uniform_samples_warmup    = 0;
                uint32_t uniform_samples_steady    = 0;
                uint32_t nonuniform_samples_steady = 0;
            } prep;

            struct DensityGridStage {
                uint32_t padded_output_width = 0;
                uint32_t query_batch_size    = 0;
                uint32_t n_elements          = 0;
            } density_grid;

            struct ValidationStage {
                uint32_t tile_rays           = 4096;
                uint32_t max_samples_per_ray = 96;
                uint32_t floats_per_coord    = 0;
                uint32_t padded_output_width = 0;
                uint32_t max_samples         = 0;
            } validation;
        } plan;

        struct NerfCounters final {
            legacy::GpuBuffer<std::uint32_t> numsteps_counter           = {};
            legacy::GpuBuffer<std::uint32_t> numsteps_counter_compacted = {};
            legacy::GpuBuffer<float> loss                               = {};

            std::uint32_t rays_per_batch                        = 1u << 12;
            std::uint32_t n_rays_total                          = 0u;
            std::uint32_t measured_batch_size                   = 0u;
            std::uint32_t measured_batch_size_before_compaction = 0u;
        };

        struct TrainingStepWorkspace final {
            legacy::GpuAllocation alloc                         = {};
            std::uint32_t* ray_indices                          = nullptr;
            void* rays_unnormalized                             = nullptr;
            std::uint32_t* numsteps                             = nullptr;
            float* coords                                       = nullptr;
            __half* mlp_out                                     = nullptr;
            __half* dloss_dmlp_out                              = nullptr;
            float* coords_compacted                             = nullptr;
            std::uint32_t* ray_counter                          = nullptr;
            std::uint32_t max_samples                           = 0u;
            std::uint32_t max_inference                         = 0u;
            std::uint32_t floats_per_coord                      = 0u;
            std::uint32_t padded_output_width                   = 0u;
            std::uint32_t n_rays_total                          = 0u;
            legacy::GPUMatrixDynamic<float> coords_matrix       = {};
            legacy::GPUMatrixDynamic<__half> rgbsigma_matrix    = {};
            legacy::GPUMatrixDynamic<float> compacted_coords_matrix = {};
            legacy::GPUMatrixDynamic<__half> gradient_matrix    = {};
            legacy::GPUMatrixDynamic<__half> compacted_output   = {};
        };

        NetworkConfig network_config         = {};
        std::uint32_t seed                   = 1337;
        legacy::math::pcg32 rng              = legacy::math::pcg32{seed};
        legacy::math::pcg32 density_grid_rng = {};

        NerfCounters counters_rgb = {};
        bool snap_to_pixel_centers = true;
        float near_distance = 0.1f;

        legacy::GpuBuffer<float> density_grid                 = {};
        legacy::GpuBuffer<std::uint8_t> density_grid_bitfield = {};
        legacy::GpuBuffer<float> density_grid_mean            = {};
        std::uint32_t density_grid_ema_step                   = 0;

        legacy::BoundingBox aabb = legacy::BoundingBox{legacy::math::vec3(0.0f), legacy::math::vec3(1.0f)};
        float density_grid_decay = 0.95f;

        std::unique_ptr<network::TrainerState<__half>, TrainerStateDeleter> trainer = {};
        uint32_t training_step                                                      = 0;
        float training_prep_ms                                                      = 0.0f;
        float training_ms                                                           = 0.0f;
        float loss_scalar                                                           = 0.0f;

        StreamState stream;
    };

} // namespace ngp

#endif // INSTANT_NGP_H
