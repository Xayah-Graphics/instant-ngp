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

    class InstantNGP final {
    public:
        enum class ActivationMode { None, ReLU, Exponential, Sigmoid, Squareplus, Softplus, Tanh, LeakyReLU };
        enum class GridStorage { Hash, Dense, Tiled };

        struct GpuFrame final {
            const std::uint8_t* pixels     = nullptr;
            legacy::math::ivec2 resolution = {};
            float focal_length             = 0.0f;
            legacy::math::mat4x3 camera    = {};
        };

        struct TrainStats final {
            float loss                                          = 0.0f;
            float train_ms                                      = 0.0f;
            float prep_ms                                       = 0.0f;
            std::uint32_t training_step                         = 0u;
            std::uint32_t batch_size                            = 0u;
            std::uint32_t rays_per_batch                        = 0u;
            std::uint32_t measured_batch_size                   = 0u;
            std::uint32_t measured_batch_size_before_compaction = 0u;
        };

        struct ValidationResult final {
            std::uint32_t width      = 0u;
            std::uint32_t height     = 0u;
            float mse                = 0.0f;
            float psnr               = 0.0f;
            std::int32_t image_index = -1;
        };

        void load_dataset(const std::filesystem::path& dataset_path);
        void train(std::int32_t iters);
        [[nodiscard]] auto read_train_stats() const -> TrainStats;
        [[nodiscard]] auto render_validation_image(const std::filesystem::path& output_path, std::uint32_t validation_image_index) -> ValidationResult;

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
        ~InstantNGP() noexcept;
        InstantNGP(const InstantNGP&)            = delete;
        InstantNGP& operator=(const InstantNGP&) = delete;
        InstantNGP(InstantNGP&& other) noexcept;
        InstantNGP& operator=(InstantNGP&& other) noexcept;

    private:
        // These markers describe steady-state runtime behavior, not C++ constness.
        // runtime-* covers train/validation/stats execution after setup is complete.
        // setup-only covers constructor/load_dataset staging that steady-state runtime does not touch.
        // Owning handles are classified by the mutability of the state they own.
        struct Dataset final {
            struct CPU final {
                struct Frame final {
                    std::vector<std::uint8_t> rgba             = {}; // runtime-immutable
                    std::uint32_t width                        = 0; // runtime-immutable
                    std::uint32_t height                       = 0; // runtime-immutable
                    float focal_length_x                       = 0.0f; // runtime-immutable
                    float focal_length_y                       = 0.0f; // runtime-immutable
                    std::array<float, 16> transform_matrix_4x4 = {}; // runtime-immutable
                };

                std::vector<Frame> train      = {}; // setup-only
                std::vector<Frame> validation = {}; // runtime-immutable
                std::vector<Frame> test       = {}; // setup-only
            };

            struct GPU final {
                std::vector<legacy::GpuBuffer<std::uint8_t>> pixels = {}; // runtime-immutable
                legacy::GpuBuffer<GpuFrame> frames                  = {}; // runtime-immutable
            };

            CPU cpu = {}; // runtime-immutable
            GPU gpu = {}; // runtime-immutable
        };

        struct TrainPlan final {
            struct NetworkStage {
                uint32_t n_pos_dims               = 0; // runtime-immutable
                uint32_t n_dir_dims               = 0; // runtime-immutable
                uint32_t dir_offset               = 0; // runtime-immutable
                uint32_t density_alignment        = 0; // runtime-immutable
                uint32_t density_input_dims       = 0; // runtime-immutable
                uint32_t density_output_dims      = 0; // runtime-immutable
                uint32_t dir_encoding_output_dims = 0; // runtime-immutable
                uint32_t rgb_alignment            = 0; // runtime-immutable
                uint32_t rgb_input_dims           = 0; // runtime-immutable
                uint32_t rgb_output_dims          = 3; // runtime-immutable
            } network = {}; // runtime-immutable

            struct TrainingStage {
                uint32_t batch_size          = 0; // runtime-immutable
                uint32_t floats_per_coord    = 0; // runtime-immutable
                uint32_t padded_output_width = 0; // runtime-immutable
                uint32_t max_samples         = 0; // runtime-immutable
            } training = {}; // runtime-immutable

            struct TrainingPrepStage {
                uint32_t warmup_steps              = 256; // runtime-immutable
                uint32_t skip_growth_interval      = 16; // runtime-immutable
                uint32_t max_skip                  = 16; // runtime-immutable
                uint32_t uniform_samples_warmup    = 0; // runtime-immutable
                uint32_t uniform_samples_steady    = 0; // runtime-immutable
                uint32_t nonuniform_samples_steady = 0; // runtime-immutable
            } prep = {}; // runtime-immutable

            struct DensityGridStage {
                uint32_t padded_output_width = 0; // runtime-immutable
                uint32_t query_batch_size    = 0; // runtime-immutable
                uint32_t n_elements          = 0; // runtime-immutable
            } density_grid = {}; // runtime-immutable

            struct ValidationStage {
                uint32_t tile_rays           = 4096; // runtime-immutable
                uint32_t max_samples_per_ray = 96; // runtime-immutable
                uint32_t floats_per_coord    = 0; // runtime-immutable
                uint32_t padded_output_width = 0; // runtime-immutable
                uint32_t max_samples         = 0; // runtime-immutable
            } validation = {}; // runtime-immutable
        };

        struct TrainCounters final {
            legacy::GpuBuffer<std::uint32_t> numsteps_counter           = {}; // runtime-mutable
            legacy::GpuBuffer<std::uint32_t> numsteps_counter_compacted = {}; // runtime-mutable
            legacy::GpuBuffer<float> loss                               = {}; // runtime-mutable

            std::uint32_t rays_per_batch                        = 1u << 12; // runtime-mutable
            std::uint32_t n_rays_total                          = 0u; // runtime-mutable
            std::uint32_t measured_batch_size                   = 0u; // runtime-mutable
            std::uint32_t measured_batch_size_before_compaction = 0u; // runtime-mutable
        };

        struct Spec final {
            NetworkConfig network_config = {}; // runtime-immutable
            TrainPlan plan               = {}; // runtime-immutable
            std::uint32_t seed           = 1337u; // setup-only
        };

        struct SamplerState final {
            struct DensityGrid final {
                legacy::GpuBuffer<float> values                = {}; // runtime-mutable
                legacy::GpuBuffer<std::uint8_t> occupancy_bits = {}; // runtime-mutable
                legacy::GpuBuffer<float> reduction_workspace   = {}; // runtime-mutable
                std::uint32_t ema_step                         = 0u; // runtime-mutable
                float ema_decay                                = 0.95f; // runtime-immutable
            } density = {}; // runtime-mutable

            legacy::BoundingBox aabb        = legacy::BoundingBox{legacy::math::vec3(0.0f), legacy::math::vec3(1.0f)}; // runtime-immutable
            bool snap_to_pixel_centers      = true; // runtime-immutable
            float near_distance             = 0.1f; // runtime-immutable
            legacy::math::pcg32 density_rng = {}; // runtime-mutable
        };

        struct TrainingState final {
            TrainCounters counters  = {}; // runtime-mutable
            legacy::math::pcg32 rng = {}; // runtime-mutable
            std::uint32_t step      = 0u; // runtime-mutable
            float last_prep_ms      = 0.0f; // runtime-mutable
            float last_train_ms     = 0.0f; // runtime-mutable
            float last_loss         = 0.0f; // runtime-mutable
        };

        struct DeviceState final {
            std::unique_ptr<network::TrainerState<__half>, void (*)(network::TrainerState<__half>*)> trainer = {nullptr, nullptr}; // runtime-mutable
            cudaStream_t stream                                                                              = {}; // runtime-immutable
        };

        Spec spec              = {}; // runtime-immutable
        Dataset dataset        = {}; // runtime-immutable
        SamplerState sampler   = {}; // runtime-mutable
        TrainingState training = {}; // runtime-mutable
        DeviceState device     = {}; // runtime-mutable
    };

} // namespace ngp

#endif // INSTANT_NGP_H
