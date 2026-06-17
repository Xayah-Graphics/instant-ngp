import std;
import xcli;
import ngp.dataset;
import ngp.train;

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    constexpr std::string_view ansi_reset = "\x1b[0m";
    constexpr std::string_view ansi_dim = "\x1b[2m";
    constexpr std::string_view ansi_bold = "\x1b[1m";
    constexpr std::string_view ansi_cyan = "\x1b[36m";
    constexpr std::string_view ansi_green = "\x1b[32m";
    constexpr std::string_view ansi_yellow = "\x1b[33m";
    constexpr std::string_view ansi_red = "\x1b[31m";
    constexpr std::string_view ansi_validation_badge = "\x1b[1;37;45m";
    constexpr std::string_view ansi_validation_metric = "\x1b[1;95m";
    constexpr std::string_view ansi_validation_best = "\x1b[1;33m";
    constexpr std::string_view ansi_test_badge = "\x1b[1;37;44m";
    constexpr std::string_view ansi_test_metric = "\x1b[1;96m";

    std::filesystem::path dataset_path = "../data/nerf-synthetic/lego";
    std::int32_t steps = 200000;
    std::int32_t chunk_steps = 1000;
    std::uint32_t validation_interval_steps = 5000u;
    std::uint32_t early_stop_patience = 5u;
    float scene_scale = ngp::dataset::DEFAULT_SCENE_SCALE;
    float early_stop_min_delta_mse = 1e-6f;
    std::optional<std::filesystem::path> load_weights_path;
    std::optional<std::filesystem::path> export_weights_path;
    std::string_view dataset_format;

    xcli::Command command{"Train Instant NGP."};
    command.add_positional({.name = "dataset-path", .description = "NeRF synthetic or DD-NeRF dataset root"}, dataset_path, {.requirement = xcli::PathRequirement::existing_directory});
    command.add_option({.long_name = "dataset", .value_name = "path", .description = "NeRF synthetic or DD-NeRF dataset root"}, dataset_path, {.requirement = xcli::PathRequirement::existing_directory});
    command.add_option({.long_name = "scene-scale", .value_name = "value", .description = "camera normalization scene scale"}, scene_scale, {.minimum = 0.0, .minimum_is_exclusive = true});
    command.add_option({.long_name = "steps", .value_name = "count", .description = "total training steps"}, steps, {.minimum = 1.0});
    command.add_option({.long_name = "chunk-steps", .value_name = "count", .description = "training steps per progress log"}, chunk_steps, {.minimum = 1.0});
    command.add_option({.long_name = "validation-interval", .value_name = "count", .description = "full validation interval in steps"}, validation_interval_steps, {.minimum = 1.0});
    command.add_option({.long_name = "early-stop-patience", .value_name = "count", .description = "validation checks without improvement before stopping"}, early_stop_patience, {.minimum = 1.0});
    command.add_option({.long_name = "early-stop-min-delta", .value_name = "mse", .description = "minimum validation MSE improvement"}, early_stop_min_delta_mse, {.minimum = 0.0});
    command.add_option({.long_name = "load-weights", .value_name = "path", .description = "load safetensors weights before training"}, load_weights_path, {.requirement = xcli::PathRequirement::existing_file});
    command.add_option({.long_name = "export-weights", .value_name = "path", .description = "export final safetensors weights"}, export_weights_path, {.requirement = xcli::PathRequirement::existing_parent_directory});
    command.add_example("../data/nerf-synthetic/lego --steps 30000");
    command.add_example("../data/dd-nerf-dataset/house1 --steps 30000");
    command.add_example("--dataset=../data/nerf-synthetic/lego --validation-interval=5000");
    command.add_example("--steps 1 --export-weights build-codex/weights.safetensors");
    command.add_example("--load-weights build-codex/weights.safetensors --steps 30000");
    command.add_validator("dataset-marker-set", [&dataset_path, &dataset_format] -> std::expected<void, std::string> {
        const bool has_nerf_synthetic_dataset =
            std::filesystem::status(dataset_path / "transforms_train.json").type() == std::filesystem::file_type::regular &&
            std::filesystem::status(dataset_path / "transforms_val.json").type() == std::filesystem::file_type::regular &&
            std::filesystem::status(dataset_path / "transforms_test.json").type() == std::filesystem::file_type::regular;
        const bool has_dd_nerf_dataset =
            std::filesystem::status(dataset_path / "cameras.json").type() == std::filesystem::file_type::regular &&
            std::filesystem::status(dataset_path / "images").type() == std::filesystem::file_type::directory;
        if (has_nerf_synthetic_dataset == has_dd_nerf_dataset) return std::unexpected{std::format("dataset path '{}' must contain exactly one supported dataset marker set: NeRF synthetic transforms_*.json or DD-NeRF cameras.json + images/.", dataset_path.string())};
        dataset_format = has_nerf_synthetic_dataset ? "nerf-synthetic" : "dd-nerf-dataset";
        return {};
    });

    const std::string usage = command.help(arguments);

    const auto cli_result = command.parse(arguments);
    if (!cli_result) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, cli_result.error());
        std::println("{}", usage);
        return 2;
    }
    if (cli_result->help_requested) {
        std::println("{}", usage);
        return 0;
    }

    const auto validation_result = command.validate();
    if (!validation_result) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, validation_result.error());
        return 2;
    }

    const auto config_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} dataset={} format={} scene_scale={} steps={} chunk={} validation_interval={} patience={} min_delta_mse={} test_output=test load_weights={} export_weights={}", ansi_dim, config_timestamp, ansi_reset, ansi_cyan, "CONFIG", ansi_reset, dataset_path.string(), dataset_format, scene_scale, steps, chunk_steps, validation_interval_steps, early_stop_patience, early_stop_min_delta_mse, load_weights_path.has_value() ? load_weights_path->string() : "none", export_weights_path.has_value() ? export_weights_path->string() : "none");

    const auto load_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} loading dataset", ansi_dim, load_timestamp, ansi_reset, ansi_cyan, "INFO", ansi_reset);

    std::optional<std::string> pipeline_error;
    std::unique_ptr<ngp::train::InstantNGP> ngp;

    const auto dataset = dataset_format == "nerf-synthetic" ? ngp::dataset::load_nerf_synthetic(dataset_path, scene_scale) : ngp::dataset::load_dd_nerf_dataset(dataset_path, scene_scale);
    if (!dataset) {
        pipeline_error = dataset.error();
    } else {
        try {
            ngp = std::make_unique<ngp::train::InstantNGP>(*dataset);
        } catch (const std::exception& error) {
            pipeline_error = std::string{error.what()};
        }
    }

    if (!pipeline_error && load_weights_path.has_value()) {
        const auto loaded_weights = ngp->load_weights(*load_weights_path);
        if (!loaded_weights) pipeline_error = loaded_weights.error();
        else {
            const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<7}{} loaded={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, load_weights_path->string());
        }
    }

    if (!pipeline_error) {
        float first_loss = 0.0f;
        float last_loss = 0.0f;
        float train_ms = 0.0f;
        float best_validation_mse = std::numeric_limits<float>::infinity();
        float best_validation_psnr = 0.0f;
        std::uint32_t final_step = 0u;
        std::uint32_t best_validation_step = 0u;
        std::uint32_t validation_checks_without_improvement = 0u;
        bool stopped_early = false;
        std::uint32_t next_validation_step = validation_interval_steps;

        for (std::int32_t trained_steps = 0; trained_steps < steps;) {
            const std::int32_t requested_steps = std::min(chunk_steps, steps - trained_steps);
            const auto stats = ngp->train(requested_steps);
            if (!stats) {
                pipeline_error = stats.error();
                break;
            }

            if (trained_steps == 0) first_loss = stats->loss;
            last_loss = stats->loss;
            train_ms += stats->elapsed_ms;
            final_step = stats->step;
            trained_steps += requested_steps;
            const auto train_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<7}{} step={:>6}/{} loss={:>10.6f} chunk={:>8.3f}ms grid={:>7.3f}ms rate={:>7.2f} step/s rays={:>6} samples={:>7}/{:<7} occupied={:>7} occupancy={:>6.2f}%", ansi_dim, train_timestamp, ansi_reset, ansi_green, "TRAIN", ansi_reset, stats->step, steps, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio * 100.0f);

            if (stats->step >= next_validation_step || stats->step >= static_cast<std::uint32_t>(steps)) {
                const auto validation = ngp->validate();
                if (!validation) {
                    pipeline_error = validation.error();
                    break;
                }

                const bool validation_improved = validation->mse < best_validation_mse - early_stop_min_delta_mse;
                if (validation_improved) {
                    best_validation_mse = validation->mse;
                    best_validation_psnr = validation->psnr;
                    best_validation_step = validation->step;
                    validation_checks_without_improvement = 0u;
                } else {
                    ++validation_checks_without_improvement;
                }

                const auto validation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {} {:<7} {} step={:>6} status={}{}{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | {}BEST={:.8f}@{}{} | patience={}{}{}/{} | images={:>3} pixels={} val={:>8.3f}ms", ansi_dim, validation_timestamp, ansi_reset, ansi_validation_badge, "VALID", ansi_reset, validation->step, validation_improved ? ansi_green : ansi_yellow, validation_improved ? "improved" : "stalled", ansi_reset, ansi_validation_metric, validation->mse, ansi_reset, ansi_cyan, validation->psnr, ansi_reset, ansi_validation_best, best_validation_mse, best_validation_step, ansi_reset, validation_checks_without_improvement == 0u ? ansi_green : ansi_yellow, validation_checks_without_improvement, ansi_reset, early_stop_patience, validation->image_count, validation->pixel_count, validation->elapsed_ms);

                if (validation_checks_without_improvement >= early_stop_patience) {
                    stopped_early = true;
                    break;
                }
                while (next_validation_step <= stats->step) next_validation_step += validation_interval_steps;
            }
        }

        if (!pipeline_error) {
            const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<7}{} steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} train={:.3f}s avg={:.2f} step/s best_validation={:.8f}@{} psnr={:.2f}", ansi_dim, summary_timestamp, ansi_reset, stopped_early ? ansi_yellow : ansi_bold, "SUMMARY", ansi_reset, final_step, stopped_early, first_loss, last_loss, train_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / train_ms, best_validation_mse, best_validation_step, best_validation_psnr);

            const auto test = ngp->test();
            if (!test) pipeline_error = test.error();
            else {
                const auto test_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {} {:<7} {} step={:>6} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3} saved={} pixels={} output={} test={:>8.3f}ms", ansi_dim, test_timestamp, ansi_reset, ansi_test_badge, "TEST", ansi_reset, test->step, ansi_test_metric, test->mse, ansi_reset, ansi_cyan, test->psnr, ansi_reset, test->image_count, test->comparison_image_count, test->pixel_count, test->output_dir.string(), test->elapsed_ms);
            }
        }

        if (!pipeline_error && export_weights_path.has_value()) {
            const auto exported_weights = ngp->export_weights(*export_weights_path);
            if (!exported_weights) pipeline_error = exported_weights.error();
            else {
                const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<7}{} exported={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, export_weights_path->string());
            }
        }
    }

    const auto finish_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    if (!pipeline_error) std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=succeeded", ansi_dim, finish_timestamp, ansi_reset, ansi_bold, "DONE", ansi_reset);
    else std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=failed error=\"{}\"", ansi_dim, finish_timestamp, ansi_reset, ansi_red, "ERROR", ansi_reset, *pipeline_error);
    return !pipeline_error ? 0 : 1;
}
