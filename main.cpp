import std;
import ngp.dataset;
import ngp.train;

struct AppConfig {
    std::filesystem::path dataset_path      = "../data/nerf-synthetic/lego";
    std::int32_t total_steps                = 200000;
    std::int32_t chunk_steps                = 1000;
    std::uint32_t validation_interval_steps = 5000u;
    std::uint32_t early_stop_patience       = 5u;
    float early_stop_min_delta_mse          = 1e-6f;
};

struct CliError {
    std::string message;
    int exit_code = 2;
};

static std::string make_usage(const std::string_view executable_name) {
    return std::format("Usage: {} [dataset-path] [options]\n"
                       "\n"
                       "Options:\n"
                       "  --dataset <path>                  NeRF synthetic dataset root. Default: ../data/nerf-synthetic/lego\n"
                       "  --steps <count>                   Total training steps. Default: 30000\n"
                       "  --chunk-steps <count>             Training steps per progress print. Default: 1000\n"
                       "  --validation-interval <count>     Full validation interval in steps. Default: 5000\n"
                       "  --early-stop-patience <count>     Validation checks without improvement before stopping. Default: 5\n"
                       "  --early-stop-min-delta <mse>      Minimum validation MSE improvement. Default: 1e-6\n"
                       "  -h, --help                        Print this help.\n",
        executable_name);
}

template <typename T, typename Predicate>
static std::expected<T, std::string> parse_number(const std::string_view text, const std::string_view name, Predicate&& predicate, const std::string_view requirement) {
    T value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);

    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::unexpected{std::format("{} must be {}.", name, requirement)};
    }

    if constexpr (std::floating_point<T>) {
        if (!std::isfinite(value)) {
            return std::unexpected{std::format("{} must be finite.", name)};
        }
    }

    if (!std::invoke(std::forward<Predicate>(predicate), value)) {
        return std::unexpected{std::format("{} must be {}.", name, requirement)};
    }

    return value;
}

static std::expected<AppConfig, CliError> parse_cli(const std::span<const char* const> args) {
    AppConfig config;
    bool dataset_path_was_set = false;

    for (std::size_t i = 1uz; i < args.size(); ++i) {
        const std::string_view raw{args[i]};

        if (raw == "-h" || raw == "--help") {
            return std::unexpected{CliError{.message = "", .exit_code = 0}};
        }

        if (raw == "--") {
            ++i;

            if (i >= args.size()) {
                break;
            }

            if (dataset_path_was_set) {
                return std::unexpected{CliError{.message = "dataset path was provided more than once."}};
            }

            config.dataset_path  = std::filesystem::path{args[i]};
            dataset_path_was_set = true;
            ++i;

            if (i != args.size()) {
                return std::unexpected{CliError{.message = std::format("unexpected argument '{}'.", args[i])}};
            }

            break;
        }

        const auto equal_pos                               = raw.find('=');
        const std::string_view option                      = equal_pos == std::string_view::npos ? raw : raw.substr(0uz, equal_pos);
        const std::optional<std::string_view> inline_value = equal_pos == std::string_view::npos ? std::nullopt : std::optional{raw.substr(equal_pos + 1uz)};

        auto take_value = [&](const std::string_view name) -> std::expected<std::string_view, CliError> {
            if (inline_value.has_value()) {
                if (inline_value->empty()) {
                    return std::unexpected{CliError{.message = std::format("{} requires a non-empty value.", name)}};
                }

                return *inline_value;
            }

            if (i + 1uz >= args.size()) {
                return std::unexpected{CliError{.message = std::format("{} requires a value.", name)}};
            }

            return std::string_view{args[++i]};
        };

        if (option == "--dataset") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            if (dataset_path_was_set) {
                return std::unexpected{CliError{.message = "dataset path was provided more than once."}};
            }

            config.dataset_path  = std::filesystem::path{*value};
            dataset_path_was_set = true;
            continue;
        }

        if (option == "--steps") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            const auto parsed = parse_number<std::int32_t>(*value, option, [](const auto x) { return x > 0; }, "a positive integer");
            if (!parsed) {
                return std::unexpected{CliError{.message = parsed.error()}};
            }

            config.total_steps = *parsed;
            continue;
        }

        if (option == "--chunk-steps") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            const auto parsed = parse_number<std::int32_t>(*value, option, [](const auto x) { return x > 0; }, "a positive integer");
            if (!parsed) {
                return std::unexpected{CliError{.message = parsed.error()}};
            }

            config.chunk_steps = *parsed;
            continue;
        }

        if (option == "--validation-interval") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            const auto parsed = parse_number<std::uint32_t>(*value, option, [](const auto x) { return x > 0u; }, "a positive integer");
            if (!parsed) {
                return std::unexpected{CliError{.message = parsed.error()}};
            }

            config.validation_interval_steps = *parsed;
            continue;
        }

        if (option == "--early-stop-patience") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            const auto parsed = parse_number<std::uint32_t>(*value, option, [](const auto x) { return x > 0u; }, "a positive integer");
            if (!parsed) {
                return std::unexpected{CliError{.message = parsed.error()}};
            }

            config.early_stop_patience = *parsed;
            continue;
        }

        if (option == "--early-stop-min-delta") {
            const auto value = take_value(option);
            if (!value) {
                return std::unexpected{value.error()};
            }

            const auto parsed = parse_number<float>(*value, option, [](const auto x) { return x >= 0.0f; }, "a finite non-negative number");
            if (!parsed) {
                return std::unexpected{CliError{.message = parsed.error()}};
            }

            config.early_stop_min_delta_mse = *parsed;
            continue;
        }

        if (!raw.starts_with("-") && !dataset_path_was_set) {
            config.dataset_path  = std::filesystem::path{raw};
            dataset_path_was_set = true;
            continue;
        }

        return std::unexpected{CliError{.message = std::format("unknown argument '{}'.", raw)}};
    }

    if (config.dataset_path.empty()) {
        return std::unexpected{CliError{.message = "dataset path must not be empty."}};
    }

    return config;
}

int main(const int argc, const char* const* const argv) {
    const auto args                   = std::span<const char* const>{argv, static_cast<std::size_t>(argc)};
    const std::string executable_name = argc > 0 && argv[0] != nullptr ? std::filesystem::path{argv[0]}.filename().string() : "instant-ngp-app";

    const auto usage         = make_usage(executable_name);
    const auto config_result = parse_cli(args);

    if (!config_result) {
        if (config_result.error().exit_code == 0) {
            std::println("{}", usage);
            return 0;
        }

        std::cerr << std::format("error: {}\n{}", config_result.error().message, usage);
        return config_result.error().exit_code;
    }

    const AppConfig config = *config_result;

    std::error_code filesystem_error;
    if (!std::filesystem::is_directory(config.dataset_path, filesystem_error)) {
        if (filesystem_error) {
            std::cerr << std::format("error: failed to inspect dataset path '{}': {}\n", config.dataset_path.string(), filesystem_error.message());
        } else {
            std::cerr << std::format("error: dataset path '{}' is not a directory.\n", config.dataset_path.string());
        }

        return 2;
    }

    std::println("config dataset={} steps={} chunk_steps={} validation_interval={} early_stop_patience={} early_stop_min_delta_mse={}", config.dataset_path.string(), config.total_steps, config.chunk_steps, config.validation_interval_steps, config.early_stop_patience, config.early_stop_min_delta_mse);

    const auto result = ngp::dataset::load_nerf_synthetic(config.dataset_path)
                            .and_then([](const auto& dataset) -> std::expected<std::unique_ptr<ngp::train::InstantNGP>, std::string> {
                                try {
                                    return std::make_unique<ngp::train::InstantNGP>(dataset);
                                } catch (const std::exception& error) {
                                    return std::unexpected{std::string{error.what()}};
                                }
                            })
                            .and_then([&](auto&& ngp) -> std::expected<void, std::string> {
                                float first_loss                                    = 0.0f;
                                float last_loss                                     = 0.0f;
                                float total_ms                                      = 0.0f;
                                float best_validation_mse                           = std::numeric_limits<float>::infinity();
                                float best_validation_psnr                          = 0.0f;
                                std::uint32_t final_step                            = 0u;
                                std::uint32_t best_validation_step                  = 0u;
                                std::uint32_t validation_checks_without_improvement = 0u;
                                bool stopped_early                                  = false;
                                std::uint32_t next_validation_step                  = config.validation_interval_steps;

                                for (std::int32_t trained_steps = 0; trained_steps < config.total_steps;) {
                                    const std::int32_t requested_steps = std::min(config.chunk_steps, config.total_steps - trained_steps);
                                    const auto stats                   = ngp->train(requested_steps);

                                    if (!stats) {
                                        return std::unexpected{stats.error()};
                                    }

                                    if (trained_steps == 0) {
                                        first_loss = stats->loss;
                                    }

                                    last_loss = stats->loss;
                                    total_ms += stats->elapsed_ms;
                                    final_step = stats->step;
                                    trained_steps += requested_steps;

                                    std::println("step={} loss={:.6f} chunk_ms={:.3f} density_grid_ms={:.3f} steps/s={:.2f} rays={} samples={}/{} occupied={} occupancy={:.4f}", stats->step, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio);

                                    if (stats->step >= next_validation_step || stats->step >= static_cast<std::uint32_t>(config.total_steps)) {
                                        const auto validation = ngp->validate();

                                        if (!validation) {
                                            return std::unexpected{validation.error()};
                                        }

                                        if (validation->mse < best_validation_mse - config.early_stop_min_delta_mse) {
                                            best_validation_mse                   = validation->mse;
                                            best_validation_psnr                  = validation->psnr;
                                            best_validation_step                  = validation->step;
                                            validation_checks_without_improvement = 0u;
                                        } else {
                                            ++validation_checks_without_improvement;
                                        }

                                        std::println("validation step={} images={} pixels={} mse={:.8f} psnr={:.2f} validation_ms={:.3f} best_mse={:.8f} best_step={} patience={}/{}", validation->step, validation->image_count, validation->pixel_count, validation->mse, validation->psnr, validation->elapsed_ms, best_validation_mse, best_validation_step, validation_checks_without_improvement, config.early_stop_patience);

                                        if (validation_checks_without_improvement >= config.early_stop_patience) {
                                            stopped_early = true;
                                            break;
                                        }

                                        while (next_validation_step <= stats->step) {
                                            next_validation_step += config.validation_interval_steps;
                                        }
                                    }
                                }

                                std::println("summary steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} total_ms={:.3f} avg_steps/s={:.2f} best_validation_step={} best_validation_mse={:.8f} best_validation_psnr={:.2f}", final_step, stopped_early, first_loss, last_loss, total_ms, static_cast<float>(final_step) * 1000.0f / total_ms, best_validation_step, best_validation_mse, best_validation_psnr);

                                return {};
                            });

    if (!result) {
        std::cerr << std::format("Pipeline failed: {}\n", result.error());
        return 1;
    }

    std::println("Pipeline succeeded");
    return 0;
}
