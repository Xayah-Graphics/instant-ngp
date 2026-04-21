#include "instant-ngp.h"
import std;

int main(int argc, char* argv[]) {
    try {
        struct CliOptions final {
            std::filesystem::path scene               = "../data/nerf-synthetic/chair";
            std::uint32_t steps                       = 10u;
            std::uint32_t log_interval                = 0u;
            std::uint32_t validation_interval         = 0u;
            std::filesystem::path validation_dir      = "validation";
            std::uint32_t validation_image_index      = 0u;
            ngp::InstantNGP::NetworkConfig network    = {};
        };

        CliOptions options{};
        options.network.rgb_network.n_hidden_layers = 2u;

        const auto print_usage = [&] {
            std::println("Usage: {} [options]\n", argc > 0 ? argv[0] : "instant-ngp-app");
            std::println("Options:");
            std::println("  --scene <path>                  Dataset directory. Default: {}", options.scene.string());
            std::println("  --steps <count>                 Training steps. Default: {}", options.steps);
            std::println("  --log-interval <count>          Print a train log every N steps. Default: final step only");
            std::println("  --validation-interval <count>   Render a validation image every N steps. Default: disabled");
            std::println("  --validation-dir <path>         Validation output directory. Default: {}", options.validation_dir.string());
            std::println("  --validation-image-index <idx>  Validation image index. Default: {}", options.validation_image_index);
            std::println("  --grid-storage <hash|dense|tiled>");
            std::println("  --hash-levels <count>");
            std::println("  --hash-features <count>");
            std::println("  --hash-log2-size <count>");
            std::println("  --hash-base-res <count>");
            std::println("  --hash-scale <float>");
            std::println("  --stochastic-interp");
            std::println("  --sh-degree <count>");
            std::println("  --density-layers <count>        Default: {}", options.network.density_network.n_hidden_layers);
            std::println("  --rgb-layers <count>            Default: {}", options.network.rgb_network.n_hidden_layers);
            std::println("  --learning-rate <float>");
            std::println("  --beta1 <float>");
            std::println("  --beta2 <float>");
            std::println("  --epsilon <float>");
            std::println("  --l2-reg <float>");
            std::println("  --help");
        };

        const auto parse_u32 = [](const std::string_view name, const std::string_view value) {
            std::uint32_t parsed = 0u;
            const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (ec != std::errc{} || ptr != value.data() + value.size()) throw std::runtime_error{std::format("Invalid value for {}: '{}'.", name, value)};
            return parsed;
        };

        const auto parse_float = [](const std::string_view name, const std::string_view value) {
            float parsed         = 0.0f;
            const auto [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (ec != std::errc{} || ptr != value.data() + value.size() || !std::isfinite(parsed)) throw std::runtime_error{std::format("Invalid value for {}: '{}'.", name, value)};
            return parsed;
        };

        const auto require_value = [&](int& index, const std::string_view flag) {
            if (index + 1 >= argc) throw std::runtime_error{std::format("Missing value after {}.", flag)};
            return std::string_view{argv[++index]};
        };

        for (int i = 1; i < argc; ++i) {
            const std::string_view arg = argv[i];
            if (arg == "--help" || arg == "-h") {
                print_usage();
                return 0;
            }

            if (arg == "--scene") {
                options.scene = std::filesystem::path{std::string{require_value(i, arg)}};
            } else if (arg == "--steps") {
                options.steps = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--log-interval") {
                options.log_interval = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--validation-interval") {
                options.validation_interval = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--validation-dir") {
                options.validation_dir = std::filesystem::path{std::string{require_value(i, arg)}};
            } else if (arg == "--validation-image-index") {
                options.validation_image_index = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--grid-storage") {
                const std::string_view value = require_value(i, arg);
                if (value == "hash") options.network.encoding.storage = ngp::InstantNGP::GridStorage::Hash;
                else if (value == "dense") options.network.encoding.storage = ngp::InstantNGP::GridStorage::Dense;
                else if (value == "tiled") options.network.encoding.storage = ngp::InstantNGP::GridStorage::Tiled;
                else throw std::runtime_error{std::format("Unknown grid storage: '{}'.", value)};
            } else if (arg == "--hash-levels") {
                options.network.encoding.n_levels = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--hash-features") {
                options.network.encoding.n_features_per_level = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--hash-log2-size") {
                options.network.encoding.log2_hashmap_size = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--hash-base-res") {
                options.network.encoding.base_resolution = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--hash-scale") {
                options.network.encoding.per_level_scale = parse_float(arg, require_value(i, arg));
            } else if (arg == "--stochastic-interp") {
                options.network.encoding.stochastic_interpolation = true;
            } else if (arg == "--sh-degree") {
                options.network.direction_encoding.sh_degree = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--density-layers") {
                options.network.density_network.n_hidden_layers = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--rgb-layers") {
                options.network.rgb_network.n_hidden_layers = parse_u32(arg, require_value(i, arg));
            } else if (arg == "--learning-rate") {
                options.network.optimizer.learning_rate = parse_float(arg, require_value(i, arg));
            } else if (arg == "--beta1") {
                options.network.optimizer.beta1 = parse_float(arg, require_value(i, arg));
            } else if (arg == "--beta2") {
                options.network.optimizer.beta2 = parse_float(arg, require_value(i, arg));
            } else if (arg == "--epsilon") {
                options.network.optimizer.epsilon = parse_float(arg, require_value(i, arg));
            } else if (arg == "--l2-reg") {
                options.network.optimizer.l2_reg = parse_float(arg, require_value(i, arg));
            } else {
                throw std::runtime_error{std::format("Unknown argument: '{}'. Use --help for usage.", arg)};
            }
        }

        if (options.scene.empty()) throw std::runtime_error{"--scene must not be empty."};
        if (options.steps == 0u) throw std::runtime_error{"--steps must be greater than 0."};
        if (options.validation_interval > 0u && options.validation_dir.empty()) throw std::runtime_error{"--validation-dir must not be empty when validation is enabled."};

        std::println("scene={}", options.scene.string());
        std::println("steps={} log_interval={} validation_interval={}", options.steps, options.log_interval, options.validation_interval);
        std::println("grid={} levels={} features={} hash_log2={} base_res={} sh_degree={}",
            options.network.encoding.storage == ngp::InstantNGP::GridStorage::Hash ? "hash" : (options.network.encoding.storage == ngp::InstantNGP::GridStorage::Dense ? "dense" : "tiled"),
            options.network.encoding.n_levels,
            options.network.encoding.n_features_per_level,
            options.network.encoding.log2_hashmap_size,
            options.network.encoding.base_resolution,
            options.network.direction_encoding.sh_degree);
        std::println("density_layers={} rgb_layers={} lr={} beta1={} beta2={} epsilon={} l2_reg={}",
            options.network.density_network.n_hidden_layers,
            options.network.rgb_network.n_hidden_layers,
            options.network.optimizer.learning_rate,
            options.network.optimizer.beta1,
            options.network.optimizer.beta2,
            options.network.optimizer.epsilon,
            options.network.optimizer.l2_reg);

        ngp::InstantNGP ngp{options.network};
        ngp.load_dataset(options.scene, ngp::InstantNGP::DatasetType::NerfSynthetic);

        const auto total_start                      = std::chrono::steady_clock::now();
        auto interval_start                        = total_start;
        const std::uint32_t initial_training_step  = ngp.read_train_stats().training_step;
        std::uint32_t interval_training_step       = initial_training_step;
        const std::uint32_t target_training_step   = initial_training_step + options.steps;

        while (true) {
            const ngp::InstantNGP::TrainStats before = ngp.read_train_stats();
            if (before.training_step >= target_training_step) break;

            std::uint32_t chunk = target_training_step - before.training_step;
            if (options.log_interval > 0u) {
                const std::uint32_t progressed = before.training_step - initial_training_step;
                const std::uint32_t until_log  = progressed % options.log_interval == 0u ? options.log_interval : options.log_interval - (progressed % options.log_interval);
                if (until_log < chunk) chunk = until_log;
            }
            if (options.validation_interval > 0u) {
                const std::uint32_t progressed         = before.training_step - initial_training_step;
                const std::uint32_t until_validation   = progressed % options.validation_interval == 0u ? options.validation_interval : options.validation_interval - (progressed % options.validation_interval);
                if (until_validation < chunk) chunk = until_validation;
            }

            ngp.train(static_cast<std::int32_t>(chunk));
            const ngp::InstantNGP::TrainStats after = ngp.read_train_stats();
            const std::uint32_t progressed          = after.training_step - initial_training_step;

            const bool should_log = after.training_step == target_training_step || (options.log_interval > 0u && progressed % options.log_interval == 0u);
            if (should_log) {
                const auto now                       = std::chrono::steady_clock::now();
                const float interval_seconds         = std::chrono::duration<float>(now - interval_start).count();
                const std::uint32_t interval_steps   = after.training_step - interval_training_step;
                const float steps_per_second         = interval_seconds > 0.0f ? static_cast<float>(interval_steps) / interval_seconds : 0.0f;
                const float elapsed_seconds          = std::chrono::duration<float>(now - total_start).count();
                std::println("step={} loss={} steps_per_second={} train_ms={} prep_ms={} elapsed_s={}",
                    after.training_step,
                    after.loss,
                    steps_per_second,
                    after.train_ms,
                    after.prep_ms,
                    elapsed_seconds);
                interval_start         = now;
                interval_training_step = after.training_step;
            }

            const bool should_validate = options.validation_interval > 0u && progressed % options.validation_interval == 0u;
            if (should_validate) {
                const std::filesystem::path output_path = options.validation_dir / std::format("step_{}.png", after.training_step);
                const ngp::InstantNGP::ValidationResult validation = ngp.render_validation_image(output_path, options.validation_image_index);
                std::println("validation step={} image_index={} resolution={}x{} mse={} psnr={} path={}",
                    after.training_step,
                    validation.image_index,
                    validation.width,
                    validation.height,
                    validation.mse,
                    validation.psnr,
                    output_path.string());
            }
        }

        return 0;
    } catch (const std::exception& e) {
        std::println(stderr, "Error: {}", e.what());
        return 1;
    }
}
