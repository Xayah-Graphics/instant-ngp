import std;
import ngp.dataset;
import ngp.train;

int main() {
    constexpr std::int32_t total_steps                = 30000;
    constexpr std::int32_t chunk_steps                = 1000;
    constexpr std::uint32_t validation_interval_steps = 5000u;
    constexpr std::uint32_t early_stop_patience       = 5u;
    constexpr float early_stop_min_delta_mse          = 1e-6f;

    const auto result = ngp::dataset::load_nerf_synthetic("../data/nerf-synthetic/lego")
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

                                for (std::int32_t step = 0; step < total_steps; step += chunk_steps) {
                                    const auto stats = ngp->train(chunk_steps);
                                    if (!stats) return std::unexpected{stats.error()};

                                    if (step == 0) first_loss = stats->loss;
                                    last_loss = stats->loss;
                                    total_ms += stats->elapsed_ms;
                                    final_step = stats->step;
                                    std::println("step={} loss={:.6f} chunk_ms={:.3f} density_grid_ms={:.3f} steps/s={:.2f} rays={} samples={}/{} occupied={} occupancy={:.4f}", stats->step, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(chunk_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio);

                                    if (stats->step % validation_interval_steps == 0u || stats->step >= static_cast<std::uint32_t>(total_steps)) {
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
                                    }
                                }

                                std::println("summary steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} total_ms={:.3f} avg_steps/s={:.2f} best_validation_step={} best_validation_mse={:.8f} best_validation_psnr={:.2f}", final_step, stopped_early, first_loss, last_loss, total_ms, static_cast<float>(final_step) * 1000.0f / total_ms, best_validation_step, best_validation_mse, best_validation_psnr);
                                return {};
                            });

    std::println("Pipeline {}", result.has_value() ? "succeeded" : std::format("failed: {}", result.error()));
    return 0;
}
