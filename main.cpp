import std;
import ngp.dataset;
import ngp.train;

int main(const int argc, const char* const* const argv) {
    std::filesystem::path dataset_path      = "../data/nerf-synthetic/lego";
    std::int32_t total_steps                = 200000;
    std::int32_t chunk_steps                = 1000;
    std::uint32_t validation_interval_steps = 5000u;
    std::uint32_t early_stop_patience       = 5u;
    float early_stop_min_delta_mse          = 1e-6f;
    bool dataset_path_was_set               = false;
    const std::string executable_name       = argc > 0 && argv[0] != nullptr ? std::filesystem::path{argv[0]}.filename().string() : "instant-ngp-app";
    const std::string usage                 = std::format("Usage: {} [dataset-path] [options]\n"
                                                                          "\n"
                                                                          "Options:\n"
                                                                          "  --dataset                   NeRF synthetic dataset root. Default: ../data/nerf-synthetic/lego\n"
                                                                          "  --steps                    Total training steps. Default: 30000\n"
                                                                          "  --chunk-steps              Training steps per progress print. Default: 1000\n"
                                                                          "  --validation-interval      Full validation interval in steps. Default: 5000\n"
                                                                          "  --early-stop-patience      Validation checks without improvement before stopping. Default: 5\n"
                                                                          "  --early-stop-min-delta       Minimum validation MSE improvement. Default: 1e-6\n"
                                                                          "  -h, --help                        Print this help.\n",
                        executable_name);

    for (std::size_t i = 1uz; i < static_cast<std::size_t>(argc); ++i) {
        const std::string_view argument{argv[i]};
        if (argument == "-h" || argument == "--help") {
            std::println("{}", usage);
            return 0;
        }

        if (argument == "--dataset") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --dataset requires a value.\n{}", usage);
                return 2;
            }
            if (dataset_path_was_set) {
                std::println("error: dataset path was provided more than once.\n{}", usage);
                return 2;
            }
            dataset_path         = std::filesystem::path{argv[++i]};
            dataset_path_was_set = true;
            continue;
        }

        if (argument == "--steps") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --steps requires a value.\n{}", usage);
                return 2;
            }
            const std::string_view value{argv[++i]};
            std::int32_t parsed = 0;
            const auto result   = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed <= 0) {
                std::println("error: --steps must be a positive integer.\n{}", usage);
                return 2;
            }
            total_steps = parsed;
            continue;
        }

        if (argument == "--chunk-steps") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --chunk-steps requires a value.\n{}", usage);
                return 2;
            }
            const std::string_view value{argv[++i]};
            std::int32_t parsed = 0;
            const auto result   = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed <= 0) {
                std::println("error: --chunk-steps must be a positive integer.\n{}", usage);
                return 2;
            }
            chunk_steps = parsed;
            continue;
        }

        if (argument == "--validation-interval") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --validation-interval requires a value.\n{}", usage);
                return 2;
            }
            const std::string_view value{argv[++i]};
            std::uint32_t parsed = 0u;
            const auto result    = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0u) {
                std::println("error: --validation-interval must be a positive integer.\n{}", usage);
                return 2;
            }
            validation_interval_steps = parsed;
            continue;
        }

        if (argument == "--early-stop-patience") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --early-stop-patience requires a value.\n{}", usage);
                return 2;
            }
            const std::string_view value{argv[++i]};
            std::uint32_t parsed = 0u;
            const auto result    = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0u) {
                std::println("error: --early-stop-patience must be a positive integer.\n{}", usage);
                return 2;
            }
            early_stop_patience = parsed;
            continue;
        }

        if (argument == "--early-stop-min-delta") {
            if (i + 1uz >= static_cast<std::size_t>(argc)) {
                std::println("error: --early-stop-min-delta requires a value.\n{}", usage);
                return 2;
            }
            const std::string_view value{argv[++i]};
            float parsed      = 0.0f;
            const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
            if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(parsed) || parsed < 0.0f) {
                std::println("error: --early-stop-min-delta must be a finite non-negative number.\n{}", usage);
                return 2;
            }
            early_stop_min_delta_mse = parsed;
            continue;
        }

        if (!argument.starts_with("-") && !dataset_path_was_set) {
            dataset_path         = std::filesystem::path{argument};
            dataset_path_was_set = true;
            continue;
        }

        std::println("error: unknown argument '{}'.\n{}", argument, usage);
        return 2;
    }

    if (dataset_path.empty()) {
        std::println("error: dataset path must not be empty.\n{}", usage);
        return 2;
    }
    if (!std::filesystem::is_directory(dataset_path)) {
        std::println("error: dataset path '{}' is not a directory.", dataset_path.string());
        return 2;
    }

    std::println("config dataset={} steps={} chunk_steps={} validation_interval={} early_stop_patience={} early_stop_min_delta_mse={}", dataset_path.string(), total_steps, chunk_steps, validation_interval_steps, early_stop_patience, early_stop_min_delta_mse);

    const auto result = ngp::dataset::load_nerf_synthetic(dataset_path)
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
                                std::uint32_t next_validation_step                  = validation_interval_steps;

                                for (std::int32_t trained_steps = 0; trained_steps < total_steps;) {
                                    const std::int32_t requested_steps = std::min(chunk_steps, total_steps - trained_steps);
                                    const auto stats                   = ngp->train(requested_steps);
                                    if (!stats) return std::unexpected{stats.error()};

                                    if (trained_steps == 0) first_loss = stats->loss;
                                    last_loss = stats->loss;
                                    total_ms += stats->elapsed_ms;
                                    final_step = stats->step;
                                    trained_steps += requested_steps;
                                    std::println("step={} loss={:.6f} chunk_ms={:.3f} density_grid_ms={:.3f} steps/s={:.2f} rays={} samples={}/{} occupied={} occupancy={:.4f}", stats->step, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio);

                                    if (stats->step >= next_validation_step || stats->step >= static_cast<std::uint32_t>(total_steps)) {
                                        const auto validation = ngp->validate();
                                        if (!validation) return std::unexpected{validation.error()};

                                        if (validation->mse < best_validation_mse - early_stop_min_delta_mse) {
                                            best_validation_mse                   = validation->mse;
                                            best_validation_psnr                  = validation->psnr;
                                            best_validation_step                  = validation->step;
                                            validation_checks_without_improvement = 0u;
                                        } else {
                                            ++validation_checks_without_improvement;
                                        }

                                        std::println("validation step={} images={} pixels={} mse={:.8f} psnr={:.2f} validation_ms={:.3f} best_mse={:.8f} best_step={} patience={}/{}", validation->step, validation->image_count, validation->pixel_count, validation->mse, validation->psnr, validation->elapsed_ms, best_validation_mse, best_validation_step, validation_checks_without_improvement, early_stop_patience);
                                        if (validation_checks_without_improvement >= early_stop_patience) {
                                            stopped_early = true;
                                            break;
                                        }
                                        while (next_validation_step <= stats->step) next_validation_step += validation_interval_steps;
                                    }
                                }

                                std::println("summary steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} total_ms={:.3f} avg_steps/s={:.2f} best_validation_step={} best_validation_mse={:.8f} best_validation_psnr={:.2f}", final_step, stopped_early, first_loss, last_loss, total_ms, static_cast<float>(final_step) * 1000.0f / total_ms, best_validation_step, best_validation_mse, best_validation_psnr);
                                return {};
                            });

    std::println("Pipeline {}", result.has_value() ? "succeeded" : std::format("failed: {}", result.error()));
    return result.has_value() ? 0 : 1;
}
