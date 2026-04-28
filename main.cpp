import std;
import ngp.dataset;
import ngp.train;

int main() {
    constexpr std::int32_t total_steps = 30000;
    constexpr std::int32_t chunk_steps = 1000;

    const auto result = ngp::dataset::load_nerf_synthetic("../data/nerf-synthetic/lego")
                            .and_then([](const auto& dataset) -> std::expected<std::unique_ptr<ngp::train::InstantNGP>, std::string> {
                                try {
                                    return std::make_unique<ngp::train::InstantNGP>(dataset);
                                } catch (const std::exception& error) {
                                    return std::unexpected{std::string{error.what()}};
                                }
                            })
                            .and_then([&](auto&& ngp) -> std::expected<void, std::string> {
                                float first_loss = 0.0f;
                                float last_loss  = 0.0f;
                                float total_ms   = 0.0f;

                                for (std::int32_t step = 0; step < total_steps; step += chunk_steps) {
                                    const auto stats = ngp->train(chunk_steps);
                                    if (!stats) return std::unexpected{stats.error()};

                                    if (step == 0) first_loss = stats->loss;
                                    last_loss = stats->loss;
                                    total_ms += stats->elapsed_ms;
                                    std::println("step={} loss={:.6f} chunk_ms={:.3f} density_grid_ms={:.3f} steps/s={:.2f} rays={} samples={}/{} occupied={} occupancy={:.4f}", stats->step, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(chunk_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio);
                                }

                                std::println("summary steps={} first_loss={:.6f} last_loss={:.6f} total_ms={:.3f} avg_steps/s={:.2f}", total_steps, first_loss, last_loss, total_ms, static_cast<float>(total_steps) * 1000.0f / total_ms);
                                return {};
                            });

    std::println("Pipeline {}", result.has_value() ? "succeeded" : std::format("failed: {}", result.error()));
    return 0;
}
