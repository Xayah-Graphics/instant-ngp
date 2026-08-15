#ifndef NGP_TRAIN_CONFIG_H
#define NGP_TRAIN_CONFIG_H

#include <array>
#include <cstdint>
#include <limits>
#include <string_view>

namespace ngp::train::config {
    struct GridEncodingConfig final {
        std::uint32_t n_levels = 8u;
        std::uint32_t features_per_level = 4u;
        std::uint32_t base_resolution = 16u;
        std::uint32_t log2_hashmap_size = 19u;
    };

    struct MlpConfig final {
        std::uint32_t width = 64u;
        std::uint32_t density_hidden_layers = 1u;
        std::uint32_t rgb_hidden_layers = 2u;
        std::uint32_t density_output_width = 16u;
        std::uint32_t direction_output_width = 16u;
        std::uint32_t network_output_width = 16u;
    };

    struct BatchConfig final {
        std::uint32_t network_batch_size = 1u << 18u;
        std::uint32_t initial_rays_per_batch = 1u << 12u;
        std::uint32_t max_sample_multiplier = 16u;
    };

    struct RaymarchConfig final {
        std::uint32_t nerf_grid_size = 128u;
        std::uint32_t nerf_steps = 1024u;
        std::uint32_t max_random_samples_per_ray = 16u;
        std::uint32_t random_values_per_thread = 4u;
        std::uint32_t sample_coord_floats = 7u;
        std::uint32_t ray_floats = 6u;
        float nerf_min_optical_thickness = 0.01f;
        bool snap_to_pixel_centers = true;
        float transmittance_epsilon = 1e-4f;
    };

    struct DensityGridConfig final {
        std::uint32_t warmup_steps = 256u;
        std::uint32_t skip_interval = 16u;
        std::uint32_t max_skip = 16u;
        std::uint32_t steady_uniform_sample_divisor = 4u;
        std::uint32_t steady_nonuniform_sample_divisor = 4u;
        float decay = 0.95f;
    };

    struct EvaluationConfig final {
        std::uint32_t tile_rays = 4096u;
    };

    struct KernelConfig final {
        std::uint32_t threads_per_block = 128u;
        std::uint32_t grid_forward_threads = 512u;
        std::uint32_t grid_backward_threads = 256u;
        std::uint32_t grid_backward_features = 2u;
        std::uint32_t mlp_forward_iters = 8u;
        std::uint32_t mlp_skew = 8u;
        std::uint32_t mlp_input_skew = 8u;
        std::size_t cublaslt_workspace_bytes = static_cast<std::size_t>(64u) * 1024u * 1024u;
    };

    struct OptimizerConfig final {
        float learning_rate = 1e-2f;
        float beta1 = 0.9f;
        float beta2 = 0.99f;
        float epsilon = 1e-15f;
        float l2_reg = 1e-6f;
        float loss_scale = 128.0f;
        float density_gradient_clamp_min = -15.0f;
        float density_gradient_clamp_max = 15.0f;
        float density_regularization_threshold = -10.0f;
        float density_regularization_max_depth = 0.1f;
        float density_regularization_strength = 1e-4f;
    };

    struct RandomConfig final {
        std::uint64_t pcg32_default_state = 0x853c49e6748fea9bULL;
        std::uint64_t pcg32_default_stream = 0xda3e39cb94b95bdbULL;
        std::uint64_t pcg32_mult = 0x5851f42d4c957f2dULL;
    };

    struct TrainProfile final {
        GridEncodingConfig grid = {};
        MlpConfig mlp = {};
        BatchConfig batch = {};
        RaymarchConfig raymarch = {};
        DensityGridConfig density_grid = {};
        EvaluationConfig evaluation = {};
        KernelConfig kernel = {};
        OptimizerConfig optimizer = {};
        RandomConfig random = {};
    };

    namespace profiles {
        inline constexpr TrainProfile baseline = {};

        inline constexpr TrainProfile batch_network_2e17 = {
            .batch = {.network_batch_size = 1u << 17u, .initial_rays_per_batch = 1u << 12u, .max_sample_multiplier = 32u},
        };
        inline constexpr TrainProfile batch_network_2e19 = {
            .batch = {.network_batch_size = 1u << 19u, .initial_rays_per_batch = 1u << 12u, .max_sample_multiplier = 16u},
        };
        inline constexpr TrainProfile batch_samples_x8 = {
            .batch = {.network_batch_size = 1u << 18u, .initial_rays_per_batch = 1u << 12u, .max_sample_multiplier = 8u},
            .evaluation = {.tile_rays = 2048u},
        };
        inline constexpr TrainProfile batch_samples_x24 = {
            .batch = {.network_batch_size = 1u << 18u, .initial_rays_per_batch = 1u << 12u, .max_sample_multiplier = 24u},
        };
        inline constexpr TrainProfile batch_rays_2e11 = {
            .batch = {.network_batch_size = 1u << 18u, .initial_rays_per_batch = 1u << 11u, .max_sample_multiplier = 16u},
        };
        inline constexpr TrainProfile batch_rays_2e13 = {
            .batch = {.network_batch_size = 1u << 18u, .initial_rays_per_batch = 1u << 13u, .max_sample_multiplier = 16u},
        };

        inline constexpr TrainProfile density_skip_8 = {
            .density_grid = {.skip_interval = 8u, .max_skip = 8u},
        };
        inline constexpr TrainProfile density_skip_32 = {
            .density_grid = {.skip_interval = 32u, .max_skip = 32u},
        };
        inline constexpr TrainProfile density_uniform_div2 = {
            .density_grid = {.steady_uniform_sample_divisor = 2u},
        };
        inline constexpr TrainProfile density_uniform_div8 = {
            .density_grid = {.steady_uniform_sample_divisor = 8u},
        };
        inline constexpr TrainProfile density_nonuniform_div2 = {
            .density_grid = {.steady_nonuniform_sample_divisor = 2u},
        };
        inline constexpr TrainProfile density_nonuniform_div8 = {
            .density_grid = {.steady_nonuniform_sample_divisor = 8u},
        };
        inline constexpr TrainProfile density_decay_090 = {
            .density_grid = {.decay = 0.90f},
        };
        inline constexpr TrainProfile density_decay_099 = {
            .density_grid = {.decay = 0.99f},
        };

        inline constexpr TrainProfile raymarch_steps_512 = {
            .raymarch = {.nerf_steps = 512u},
        };
        inline constexpr TrainProfile raymarch_steps_2048 = {
            .raymarch = {.nerf_steps = 2048u},
            .evaluation = {.tile_rays = 2048u},
        };
        inline constexpr TrainProfile raymarch_random_samples_8 = {
            .raymarch = {.max_random_samples_per_ray = 8u},
        };
        inline constexpr TrainProfile raymarch_random_samples_32 = {
            .raymarch = {.max_random_samples_per_ray = 32u},
        };
        inline constexpr TrainProfile raymarch_transmittance_1em3 = {
            .raymarch = {.transmittance_epsilon = 1e-3f},
        };
        inline constexpr TrainProfile raymarch_transmittance_1em5 = {
            .raymarch = {.transmittance_epsilon = 1e-5f},
        };
        inline constexpr TrainProfile raymarch_thickness_0p005 = {
            .raymarch = {.nerf_min_optical_thickness = 0.005f},
        };
        inline constexpr TrainProfile raymarch_thickness_0p02 = {
            .raymarch = {.nerf_min_optical_thickness = 0.02f},
        };

        inline constexpr TrainProfile hash_log2_18 = {
            .grid = {.log2_hashmap_size = 18u},
        };
        inline constexpr TrainProfile hash_log2_20 = {
            .grid = {.log2_hashmap_size = 20u},
        };
        inline constexpr TrainProfile hash_base_8 = {
            .grid = {.base_resolution = 8u},
        };
        inline constexpr TrainProfile hash_base_32 = {
            .grid = {.base_resolution = 32u},
        };

        inline constexpr TrainProfile optimizer_lr_5em3 = {
            .optimizer = {.learning_rate = 5e-3f},
        };
        inline constexpr TrainProfile optimizer_lr_2em2 = {
            .optimizer = {.learning_rate = 2e-2f},
        };
        inline constexpr TrainProfile optimizer_l2_0 = {
            .optimizer = {.l2_reg = 0.0f},
        };
        inline constexpr TrainProfile optimizer_l2_1em5 = {
            .optimizer = {.l2_reg = 1e-5f},
        };
        inline constexpr TrainProfile optimizer_loss_scale_64 = {
            .optimizer = {.loss_scale = 64.0f},
        };
        inline constexpr TrainProfile optimizer_loss_scale_256 = {
            .optimizer = {.loss_scale = 256.0f},
        };

        inline constexpr TrainProfile kernel_threads_256 = {
            .kernel = {.threads_per_block = 256u},
        };
        inline constexpr TrainProfile kernel_grid_forward_256 = {
            .kernel = {.grid_forward_threads = 256u},
        };
        inline constexpr TrainProfile kernel_grid_backward_128 = {
            .kernel = {.grid_backward_threads = 128u},
        };
        inline constexpr TrainProfile kernel_grid_backward_512 = {
            .kernel = {.grid_backward_threads = 512u},
        };
        inline constexpr TrainProfile kernel_mlp_forward_iters_4 = {
            .kernel = {.mlp_forward_iters = 4u},
        };
    }

#if defined(NGP_TRAIN_PROFILE_BASELINE) + \
    defined(NGP_TRAIN_PROFILE_BATCH_NETWORK_2E17) + \
    defined(NGP_TRAIN_PROFILE_BATCH_NETWORK_2E19) + \
    defined(NGP_TRAIN_PROFILE_BATCH_SAMPLES_X8) + \
    defined(NGP_TRAIN_PROFILE_BATCH_SAMPLES_X24) + \
    defined(NGP_TRAIN_PROFILE_BATCH_RAYS_2E11) + \
    defined(NGP_TRAIN_PROFILE_BATCH_RAYS_2E13) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_SKIP_8) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_SKIP_32) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_UNIFORM_DIV2) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_UNIFORM_DIV8) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_NONUNIFORM_DIV2) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_NONUNIFORM_DIV8) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_DECAY_090) + \
    defined(NGP_TRAIN_PROFILE_DENSITY_DECAY_099) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_STEPS_512) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_STEPS_2048) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_RANDOM_SAMPLES_8) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_RANDOM_SAMPLES_32) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_TRANSMITTANCE_1EM3) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_TRANSMITTANCE_1EM5) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_THICKNESS_0P005) + \
    defined(NGP_TRAIN_PROFILE_RAYMARCH_THICKNESS_0P02) + \
    defined(NGP_TRAIN_PROFILE_HASH_LOG2_18) + \
    defined(NGP_TRAIN_PROFILE_HASH_LOG2_20) + \
    defined(NGP_TRAIN_PROFILE_HASH_BASE_8) + \
    defined(NGP_TRAIN_PROFILE_HASH_BASE_32) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_LR_5EM3) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_LR_2EM2) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_L2_0) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_L2_1EM5) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_LOSS_SCALE_64) + \
    defined(NGP_TRAIN_PROFILE_OPTIMIZER_LOSS_SCALE_256) + \
    defined(NGP_TRAIN_PROFILE_KERNEL_THREADS_256) + \
    defined(NGP_TRAIN_PROFILE_KERNEL_GRID_FORWARD_256) + \
    defined(NGP_TRAIN_PROFILE_KERNEL_GRID_BACKWARD_128) + \
    defined(NGP_TRAIN_PROFILE_KERNEL_GRID_BACKWARD_512) + \
    defined(NGP_TRAIN_PROFILE_KERNEL_MLP_FORWARD_ITERS_4) != 1
#error "Exactly one NGP_TRAIN_PROFILE_* compile-time profile must be selected."
#endif

#if defined(NGP_TRAIN_PROFILE_BASELINE)
    inline constexpr TrainProfile active_profile = profiles::baseline;
    inline constexpr std::string_view active_profile_name = "baseline";
#elif defined(NGP_TRAIN_PROFILE_BATCH_NETWORK_2E17)
    inline constexpr TrainProfile active_profile = profiles::batch_network_2e17;
    inline constexpr std::string_view active_profile_name = "batch_network_2e17";
#elif defined(NGP_TRAIN_PROFILE_BATCH_NETWORK_2E19)
    inline constexpr TrainProfile active_profile = profiles::batch_network_2e19;
    inline constexpr std::string_view active_profile_name = "batch_network_2e19";
#elif defined(NGP_TRAIN_PROFILE_BATCH_SAMPLES_X8)
    inline constexpr TrainProfile active_profile = profiles::batch_samples_x8;
    inline constexpr std::string_view active_profile_name = "batch_samples_x8";
#elif defined(NGP_TRAIN_PROFILE_BATCH_SAMPLES_X24)
    inline constexpr TrainProfile active_profile = profiles::batch_samples_x24;
    inline constexpr std::string_view active_profile_name = "batch_samples_x24";
#elif defined(NGP_TRAIN_PROFILE_BATCH_RAYS_2E11)
    inline constexpr TrainProfile active_profile = profiles::batch_rays_2e11;
    inline constexpr std::string_view active_profile_name = "batch_rays_2e11";
#elif defined(NGP_TRAIN_PROFILE_BATCH_RAYS_2E13)
    inline constexpr TrainProfile active_profile = profiles::batch_rays_2e13;
    inline constexpr std::string_view active_profile_name = "batch_rays_2e13";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_SKIP_8)
    inline constexpr TrainProfile active_profile = profiles::density_skip_8;
    inline constexpr std::string_view active_profile_name = "density_skip_8";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_SKIP_32)
    inline constexpr TrainProfile active_profile = profiles::density_skip_32;
    inline constexpr std::string_view active_profile_name = "density_skip_32";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_UNIFORM_DIV2)
    inline constexpr TrainProfile active_profile = profiles::density_uniform_div2;
    inline constexpr std::string_view active_profile_name = "density_uniform_div2";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_UNIFORM_DIV8)
    inline constexpr TrainProfile active_profile = profiles::density_uniform_div8;
    inline constexpr std::string_view active_profile_name = "density_uniform_div8";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_NONUNIFORM_DIV2)
    inline constexpr TrainProfile active_profile = profiles::density_nonuniform_div2;
    inline constexpr std::string_view active_profile_name = "density_nonuniform_div2";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_NONUNIFORM_DIV8)
    inline constexpr TrainProfile active_profile = profiles::density_nonuniform_div8;
    inline constexpr std::string_view active_profile_name = "density_nonuniform_div8";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_DECAY_090)
    inline constexpr TrainProfile active_profile = profiles::density_decay_090;
    inline constexpr std::string_view active_profile_name = "density_decay_090";
#elif defined(NGP_TRAIN_PROFILE_DENSITY_DECAY_099)
    inline constexpr TrainProfile active_profile = profiles::density_decay_099;
    inline constexpr std::string_view active_profile_name = "density_decay_099";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_STEPS_512)
    inline constexpr TrainProfile active_profile = profiles::raymarch_steps_512;
    inline constexpr std::string_view active_profile_name = "raymarch_steps_512";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_STEPS_2048)
    inline constexpr TrainProfile active_profile = profiles::raymarch_steps_2048;
    inline constexpr std::string_view active_profile_name = "raymarch_steps_2048";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_RANDOM_SAMPLES_8)
    inline constexpr TrainProfile active_profile = profiles::raymarch_random_samples_8;
    inline constexpr std::string_view active_profile_name = "raymarch_random_samples_8";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_RANDOM_SAMPLES_32)
    inline constexpr TrainProfile active_profile = profiles::raymarch_random_samples_32;
    inline constexpr std::string_view active_profile_name = "raymarch_random_samples_32";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_TRANSMITTANCE_1EM3)
    inline constexpr TrainProfile active_profile = profiles::raymarch_transmittance_1em3;
    inline constexpr std::string_view active_profile_name = "raymarch_transmittance_1em3";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_TRANSMITTANCE_1EM5)
    inline constexpr TrainProfile active_profile = profiles::raymarch_transmittance_1em5;
    inline constexpr std::string_view active_profile_name = "raymarch_transmittance_1em5";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_THICKNESS_0P005)
    inline constexpr TrainProfile active_profile = profiles::raymarch_thickness_0p005;
    inline constexpr std::string_view active_profile_name = "raymarch_thickness_0p005";
#elif defined(NGP_TRAIN_PROFILE_RAYMARCH_THICKNESS_0P02)
    inline constexpr TrainProfile active_profile = profiles::raymarch_thickness_0p02;
    inline constexpr std::string_view active_profile_name = "raymarch_thickness_0p02";
#elif defined(NGP_TRAIN_PROFILE_HASH_LOG2_18)
    inline constexpr TrainProfile active_profile = profiles::hash_log2_18;
    inline constexpr std::string_view active_profile_name = "hash_log2_18";
#elif defined(NGP_TRAIN_PROFILE_HASH_LOG2_20)
    inline constexpr TrainProfile active_profile = profiles::hash_log2_20;
    inline constexpr std::string_view active_profile_name = "hash_log2_20";
#elif defined(NGP_TRAIN_PROFILE_HASH_BASE_8)
    inline constexpr TrainProfile active_profile = profiles::hash_base_8;
    inline constexpr std::string_view active_profile_name = "hash_base_8";
#elif defined(NGP_TRAIN_PROFILE_HASH_BASE_32)
    inline constexpr TrainProfile active_profile = profiles::hash_base_32;
    inline constexpr std::string_view active_profile_name = "hash_base_32";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_LR_5EM3)
    inline constexpr TrainProfile active_profile = profiles::optimizer_lr_5em3;
    inline constexpr std::string_view active_profile_name = "optimizer_lr_5em3";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_LR_2EM2)
    inline constexpr TrainProfile active_profile = profiles::optimizer_lr_2em2;
    inline constexpr std::string_view active_profile_name = "optimizer_lr_2em2";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_L2_0)
    inline constexpr TrainProfile active_profile = profiles::optimizer_l2_0;
    inline constexpr std::string_view active_profile_name = "optimizer_l2_0";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_L2_1EM5)
    inline constexpr TrainProfile active_profile = profiles::optimizer_l2_1em5;
    inline constexpr std::string_view active_profile_name = "optimizer_l2_1em5";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_LOSS_SCALE_64)
    inline constexpr TrainProfile active_profile = profiles::optimizer_loss_scale_64;
    inline constexpr std::string_view active_profile_name = "optimizer_loss_scale_64";
#elif defined(NGP_TRAIN_PROFILE_OPTIMIZER_LOSS_SCALE_256)
    inline constexpr TrainProfile active_profile = profiles::optimizer_loss_scale_256;
    inline constexpr std::string_view active_profile_name = "optimizer_loss_scale_256";
#elif defined(NGP_TRAIN_PROFILE_KERNEL_THREADS_256)
    inline constexpr TrainProfile active_profile = profiles::kernel_threads_256;
    inline constexpr std::string_view active_profile_name = "kernel_threads_256";
#elif defined(NGP_TRAIN_PROFILE_KERNEL_GRID_FORWARD_256)
    inline constexpr TrainProfile active_profile = profiles::kernel_grid_forward_256;
    inline constexpr std::string_view active_profile_name = "kernel_grid_forward_256";
#elif defined(NGP_TRAIN_PROFILE_KERNEL_GRID_BACKWARD_128)
    inline constexpr TrainProfile active_profile = profiles::kernel_grid_backward_128;
    inline constexpr std::string_view active_profile_name = "kernel_grid_backward_128";
#elif defined(NGP_TRAIN_PROFILE_KERNEL_GRID_BACKWARD_512)
    inline constexpr TrainProfile active_profile = profiles::kernel_grid_backward_512;
    inline constexpr std::string_view active_profile_name = "kernel_grid_backward_512";
#elif defined(NGP_TRAIN_PROFILE_KERNEL_MLP_FORWARD_ITERS_4)
    inline constexpr TrainProfile active_profile = profiles::kernel_mlp_forward_iters_4;
    inline constexpr std::string_view active_profile_name = "kernel_mlp_forward_iters_4";
#endif

    inline constexpr std::uint32_t grid_n_levels = active_profile.grid.n_levels;
    inline constexpr std::uint32_t grid_features_per_level = active_profile.grid.features_per_level;
    inline constexpr std::uint32_t grid_base_resolution = active_profile.grid.base_resolution;
    inline constexpr std::uint32_t grid_log2_hashmap_size = active_profile.grid.log2_hashmap_size;
    inline constexpr std::uint32_t grid_output_width = grid_n_levels * grid_features_per_level;

    inline constexpr std::uint32_t mlp_width = active_profile.mlp.width;
    inline constexpr std::uint32_t density_hidden_layers = active_profile.mlp.density_hidden_layers;
    inline constexpr std::uint32_t rgb_hidden_layers = active_profile.mlp.rgb_hidden_layers;
    inline constexpr std::uint32_t density_output_width = active_profile.mlp.density_output_width;
    inline constexpr std::uint32_t direction_output_width = active_profile.mlp.direction_output_width;
    inline constexpr std::uint32_t rgb_input_width = density_output_width + direction_output_width;
    inline constexpr std::uint32_t network_output_width = active_profile.mlp.network_output_width;

    struct NetworkParameterLayout final {
        std::array<std::uint32_t, grid_n_levels + 1u> grid_offsets = {};
        std::uint32_t density_param_offset = 0u;
        std::uint32_t density_input_weight_offset = 0u;
        std::uint32_t density_output_weight_offset = 0u;
        std::uint32_t density_param_count = 0u;
        std::uint32_t rgb_param_offset = 0u;
        std::uint32_t rgb_input_weight_offset = 0u;
        std::uint32_t rgb_hidden_weight_offset = 0u;
        std::uint32_t rgb_output_weight_offset = 0u;
        std::uint32_t rgb_param_count = 0u;
        std::uint32_t mlp_param_count = 0u;
        std::uint32_t grid_param_offset = 0u;
        std::uint32_t grid_param_count = 0u;
        std::uint32_t total_param_count = 0u;
    };

    constexpr NetworkParameterLayout make_network_parameter_layout() {
        NetworkParameterLayout layout = {};
        std::uint32_t grid_cursor = 0u;
        constexpr std::uint32_t grid_hashmap_size = 1u << grid_log2_hashmap_size;
        constexpr std::uint64_t grid_max_positions = std::numeric_limits<std::uint32_t>::max() / 2ull;

        for (std::uint32_t level = 0u; level < grid_n_levels; ++level) {
            const std::uint32_t resolution = grid_base_resolution << level;
            const std::uint64_t dense = static_cast<std::uint64_t>(resolution) * resolution * resolution;
            std::uint64_t positions = dense > grid_max_positions ? grid_max_positions : dense;
            positions = ((positions + 7u) / 8u) * 8u;
            if (positions > grid_hashmap_size) positions = grid_hashmap_size;
            layout.grid_offsets[level] = grid_cursor;
            grid_cursor += static_cast<std::uint32_t>(positions);
        }

        layout.grid_offsets[grid_n_levels] = grid_cursor;

        std::uint32_t param_cursor = 0u;
        layout.density_param_offset = param_cursor;
        layout.density_input_weight_offset = param_cursor;
        param_cursor += mlp_width * grid_output_width;
        layout.density_output_weight_offset = param_cursor;
        param_cursor += density_output_width * mlp_width;
        layout.density_param_count = param_cursor - layout.density_param_offset;

        layout.rgb_param_offset = param_cursor;
        layout.rgb_input_weight_offset = param_cursor;
        param_cursor += mlp_width * rgb_input_width;
        layout.rgb_hidden_weight_offset = param_cursor;
        param_cursor += mlp_width * mlp_width;
        layout.rgb_output_weight_offset = param_cursor;
        param_cursor += network_output_width * mlp_width;
        layout.rgb_param_count = param_cursor - layout.rgb_param_offset;

        layout.mlp_param_count = param_cursor;
        layout.grid_param_offset = param_cursor;
        layout.grid_param_count = layout.grid_offsets[grid_n_levels] * grid_features_per_level;
        param_cursor += layout.grid_param_count;
        layout.total_param_count = param_cursor;
        return layout;
    }

    inline constexpr NetworkParameterLayout network_parameter_layout = make_network_parameter_layout();

    inline constexpr std::uint32_t network_batch_size = active_profile.batch.network_batch_size;
    inline constexpr std::uint32_t network_batch_granularity = 16u * active_profile.kernel.mlp_forward_iters;
    inline constexpr std::uint32_t initial_rays_per_batch = active_profile.batch.initial_rays_per_batch;
    inline constexpr std::uint32_t max_samples = network_batch_size * active_profile.batch.max_sample_multiplier;

    inline constexpr std::uint32_t nerf_grid_size = active_profile.raymarch.nerf_grid_size;
    inline constexpr std::uint32_t nerf_grid_cells = nerf_grid_size * nerf_grid_size * nerf_grid_size;
    inline constexpr std::uint32_t nerf_steps = active_profile.raymarch.nerf_steps;
    inline constexpr std::uint32_t max_random_samples_per_ray = active_profile.raymarch.max_random_samples_per_ray;
    inline constexpr std::uint32_t random_values_per_thread = active_profile.raymarch.random_values_per_thread;
    inline constexpr std::uint32_t sample_coord_floats = active_profile.raymarch.sample_coord_floats;
    inline constexpr std::uint32_t ray_floats = active_profile.raymarch.ray_floats;
    inline constexpr float min_cone_stepsize = 1.73205080757f / static_cast<float>(nerf_steps);
    inline constexpr float nerf_min_optical_thickness = active_profile.raymarch.nerf_min_optical_thickness;
    inline constexpr bool snap_to_pixel_centers = active_profile.raymarch.snap_to_pixel_centers;
    inline constexpr float transmittance_epsilon = active_profile.raymarch.transmittance_epsilon;

    inline constexpr std::uint32_t density_grid_warmup_steps = active_profile.density_grid.warmup_steps;
    inline constexpr std::uint32_t density_grid_skip_interval = active_profile.density_grid.skip_interval;
    inline constexpr std::uint32_t density_grid_max_skip = active_profile.density_grid.max_skip;
    inline constexpr float density_grid_decay = active_profile.density_grid.decay;
    inline constexpr std::uint32_t density_grid_warmup_samples = nerf_grid_cells;
    inline constexpr std::uint32_t density_grid_steady_uniform_samples = nerf_grid_cells / active_profile.density_grid.steady_uniform_sample_divisor;
    inline constexpr std::uint32_t density_grid_steady_nonuniform_samples = nerf_grid_cells / active_profile.density_grid.steady_nonuniform_sample_divisor;

    inline constexpr std::uint32_t evaluation_tile_rays = active_profile.evaluation.tile_rays;
    inline constexpr std::uint32_t evaluation_max_samples = evaluation_tile_rays * nerf_steps;

    inline constexpr std::uint32_t threads_per_block = active_profile.kernel.threads_per_block;
    inline constexpr std::uint32_t grid_forward_threads = active_profile.kernel.grid_forward_threads;
    inline constexpr std::uint32_t grid_backward_threads = active_profile.kernel.grid_backward_threads;
    inline constexpr std::uint32_t grid_backward_features = active_profile.kernel.grid_backward_features;
    inline constexpr std::uint32_t mlp_forward_iters = active_profile.kernel.mlp_forward_iters;
    inline constexpr std::uint32_t mlp_width_blocks = mlp_width / 16u;
    inline constexpr std::uint32_t mlp_skew = active_profile.kernel.mlp_skew;
    inline constexpr std::uint32_t mlp_input_skew = active_profile.kernel.mlp_input_skew;
    inline constexpr std::size_t cublaslt_workspace_bytes = active_profile.kernel.cublaslt_workspace_bytes;

    inline constexpr float optimizer_learning_rate = active_profile.optimizer.learning_rate;
    inline constexpr float optimizer_beta1 = active_profile.optimizer.beta1;
    inline constexpr float optimizer_beta2 = active_profile.optimizer.beta2;
    inline constexpr float optimizer_epsilon = active_profile.optimizer.epsilon;
    inline constexpr float optimizer_l2_reg = active_profile.optimizer.l2_reg;
    inline constexpr float optimizer_loss_scale = active_profile.optimizer.loss_scale;
    inline constexpr float density_gradient_clamp_min = active_profile.optimizer.density_gradient_clamp_min;
    inline constexpr float density_gradient_clamp_max = active_profile.optimizer.density_gradient_clamp_max;
    inline constexpr float density_regularization_threshold = active_profile.optimizer.density_regularization_threshold;
    inline constexpr float density_regularization_max_depth = active_profile.optimizer.density_regularization_max_depth;
    inline constexpr float density_regularization_strength = active_profile.optimizer.density_regularization_strength;

    inline constexpr std::uint64_t pcg32_default_state = active_profile.random.pcg32_default_state;
    inline constexpr std::uint64_t pcg32_default_stream = active_profile.random.pcg32_default_stream;
    inline constexpr std::uint64_t pcg32_mult = active_profile.random.pcg32_mult;

    static_assert(grid_n_levels == 8u, "The handwritten grid kernels currently pass exactly eight hash-grid levels.");
    static_assert(grid_features_per_level == 4u, "The handwritten grid kernels currently accumulate exactly four features per level.");
    static_assert(grid_output_width % grid_features_per_level == 0u);
    static_assert(grid_output_width == rgb_input_width);
    static_assert(mlp_width == 64u, "The handwritten MLP kernels currently support exactly width 64.");
    static_assert(density_hidden_layers == 1u, "The handwritten density MLP path currently supports exactly one hidden layer.");
    static_assert(rgb_hidden_layers == 2u, "The handwritten RGB MLP path currently supports exactly two hidden layers.");
    static_assert(density_output_width == 16u);
    static_assert(direction_output_width == 16u);
    static_assert(density_output_width == network_output_width);
    static_assert(network_output_width == 16u);
    static_assert(mlp_width % 16u == 0u);
    static_assert(nerf_grid_size != 0u);
    static_assert((nerf_grid_size & (nerf_grid_size - 1u)) == 0u, "The occupancy grid expects a power-of-two resolution.");
    static_assert(nerf_grid_cells % 8u == 0u);
    static_assert(sample_coord_floats == 7u);
    static_assert(ray_floats == 6u);
    static_assert(network_batch_size != 0u);
    static_assert(network_batch_granularity != 0u);
    static_assert(network_batch_size % network_batch_granularity == 0u);
    static_assert(max_samples % network_batch_granularity == 0u);
    static_assert(network_batch_size % (16u * mlp_forward_iters) == 0u);
    static_assert(evaluation_max_samples <= max_samples);
    static_assert(evaluation_max_samples % network_batch_granularity == 0u);
    static_assert(density_grid_warmup_samples <= max_samples);
    static_assert(density_grid_steady_uniform_samples <= max_samples);
    static_assert(density_grid_steady_nonuniform_samples <= max_samples);
    static_assert(grid_backward_features == 2u, "The handwritten grid backward kernel accumulates feature pairs with half2.");
    static_assert(network_parameter_layout.grid_offsets[0u] == 0u);
    static_assert(network_parameter_layout.density_param_offset == 0u);
    static_assert(network_parameter_layout.rgb_param_offset == network_parameter_layout.density_param_offset + network_parameter_layout.density_param_count);
    static_assert(network_parameter_layout.mlp_param_count == network_parameter_layout.rgb_param_offset + network_parameter_layout.rgb_param_count);
    static_assert(network_parameter_layout.grid_param_offset == network_parameter_layout.mlp_param_count);
    static_assert(network_parameter_layout.grid_param_count == network_parameter_layout.grid_offsets[grid_n_levels] * grid_features_per_level);
    static_assert(network_parameter_layout.total_param_count == network_parameter_layout.grid_param_offset + network_parameter_layout.grid_param_count);
} // namespace ngp::train::config

#endif // NGP_TRAIN_CONFIG_H
