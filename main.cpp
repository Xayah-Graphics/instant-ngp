import std;
import xlog;
import xcli;
import ngp.dataset;
import ngp.train;

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};

    std::filesystem::path dataset_path = "../data/nerf-synthetic/lego";
    std::int32_t steps = 200000;
    std::int32_t chunk_steps = 1000;
    std::uint32_t validation_interval_steps = 5000u;
    std::uint32_t early_stop_patience = 5u;
    float scene_scale = ngp::dataset::DEFAULT_SCENE_SCALE;
    float early_stop_min_delta_mse = 1e-6f;
    std::optional<std::filesystem::path> load_weights_path;
    std::optional<std::filesystem::path> export_weights_path;
    std::optional<std::filesystem::path> log_file_path;
    std::string_view dataset_format;

    xcli::Command command{"Train Instant NGP."};
    command.add_positional({.name = "dataset-path", .description = "NeRF synthetic or DD-NeRF dataset root"}, dataset_path, {.reject_empty = true, .requirement = xcli::PathRequirement::existing_directory});
    command.add_option({.long_name = "dataset", .value_name = "path", .description = "NeRF synthetic or DD-NeRF dataset root"}, dataset_path, {.reject_empty = true, .requirement = xcli::PathRequirement::existing_directory});
    command.add_option({.long_name = "scene-scale", .value_name = "value", .description = "camera normalization scene scale"}, scene_scale, {.minimum = 0.0, .minimum_is_exclusive = true});
    command.add_option({.long_name = "steps", .value_name = "count", .description = "total training steps"}, steps, {.minimum = 1.0});
    command.add_option({.long_name = "chunk-steps", .value_name = "count", .description = "training steps per progress log"}, chunk_steps, {.minimum = 1.0});
    command.add_option({.long_name = "validation-interval", .value_name = "count", .description = "full validation interval in steps"}, validation_interval_steps, {.minimum = 1.0});
    command.add_option({.long_name = "early-stop-patience", .value_name = "count", .description = "validation checks without improvement before stopping"}, early_stop_patience, {.minimum = 1.0});
    command.add_option({.long_name = "early-stop-min-delta", .value_name = "mse", .description = "minimum validation MSE improvement"}, early_stop_min_delta_mse, {.minimum = 0.0});
    command.add_option({.long_name = "load-weights", .value_name = "path", .description = "load safetensors weights before training"}, load_weights_path, {.reject_empty = true, .requirement = xcli::PathRequirement::existing_file});
    command.add_option({.long_name = "export-weights", .value_name = "path", .description = "export final safetensors weights"}, export_weights_path, {.reject_empty = true, .requirement = xcli::PathRequirement::existing_parent_directory});
    command.add_option({.long_name = "log-file", .value_name = "path", .description = "write plain text logs to a file"}, log_file_path);
    command.add_example("../data/nerf-synthetic/lego --steps 30000");
    command.add_example("../data/dd-nerf-dataset/house1 --steps 30000");
    command.add_example("--dataset=../data/nerf-synthetic/lego --validation-interval=5000");
    command.add_example("--steps 1 --export-weights build-codex/weights.safetensors");
    command.add_example("--steps 1 --log-file build/instant-ngp.log");
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

    const xcli::HelpStyle help_style{
        .reset = xlog::ansi::reset,
        .dim = xlog::ansi::dim,
        .bold = xlog::ansi::bold,
        .heading = xlog::ansi::bold,
        .executable = xlog::ansi::cyan,
        .option = xlog::ansi::green,
        .value = xlog::ansi::yellow,
        .default_label = xlog::ansi::dim,
    };
    const std::string usage = command.help(arguments, help_style);

    auto logger_result = xlog::Logger::create(xlog::LoggerConfig{});
    if (!logger_result) throw std::runtime_error{logger_result.error()};
    xlog::Logger logger = std::move(*logger_result);
    logger.set_tag_style("CONFIG", {.begin = xlog::ansi::cyan, .end = xlog::ansi::reset});
    logger.set_tag_style("INFO", {.begin = xlog::ansi::cyan, .end = xlog::ansi::reset});
    logger.set_tag_style("WEIGHT", {.begin = xlog::ansi::yellow, .end = xlog::ansi::reset});
    logger.set_tag_style("TRAIN", {.begin = xlog::ansi::green, .end = xlog::ansi::reset});
    logger.set_tag_style("VALID", {.begin = xlog::ansi::validation_badge, .end = xlog::ansi::reset});
    logger.set_tag_style("TEST", {.begin = xlog::ansi::test_badge, .end = xlog::ansi::reset});
    logger.set_tag_style("SUMMARY", {.begin = xlog::ansi::bold, .end = xlog::ansi::reset});
    logger.set_tag_style("DONE", {.begin = xlog::ansi::bold, .end = xlog::ansi::reset});
    logger.set_tag_style("ERROR", {.begin = xlog::ansi::red, .end = xlog::ansi::reset});

    const auto cli_result = command.parse(arguments);
    if (!cli_result) {
        logger.error("ERROR", "{}", cli_result.error());
        std::println("{}", usage);
        return 2;
    }
    if (cli_result->help_requested) {
        std::println("{}", usage);
        return 0;
    }

    if (log_file_path.has_value()) {
        const auto file_sink = logger.add_file_sink({.path = *log_file_path});
        if (!file_sink) {
            logger.error("ERROR", "{}", file_sink.error());
            return 2;
        }
    }

    const auto validation_result = command.validate();
    if (!validation_result) {
        logger.error("ERROR", "{}", validation_result.error());
        return 2;
    }

    logger.info("CONFIG", "dataset={} format={} scene_scale={} steps={} chunk={} validation_interval={} patience={} min_delta_mse={} test_output=test load_weights={} export_weights={} log_file={}", dataset_path.string(), dataset_format, scene_scale, steps, chunk_steps, validation_interval_steps, early_stop_patience, early_stop_min_delta_mse, load_weights_path.has_value() ? load_weights_path->string() : "none", export_weights_path.has_value() ? export_weights_path->string() : "none", log_file_path.has_value() ? log_file_path->string() : "none");
    logger.info("INFO", "loading dataset");

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
        else logger.info("WEIGHT", "loaded={}", load_weights_path->string());
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
            logger.info("TRAIN", "step={:>6}/{} loss={:>10.6f} chunk={:>8.3f}ms grid={:>7.3f}ms rate={:>7.2f} step/s rays={:>6} samples={:>7}/{:<7} occupied={:>7} occupancy={:>6.2f}%", stats->step, steps, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio * 100.0f);

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

                if (validation_improved) logger.info("VALID", "step={:>6} status=improved mse={:.8f} psnr={:>5.2f} best={:.8f}@{} patience={}/{} images={:>3} pixels={} val={:>8.3f}ms", validation->step, validation->mse, validation->psnr, best_validation_mse, best_validation_step, validation_checks_without_improvement, early_stop_patience, validation->image_count, validation->pixel_count, validation->elapsed_ms);
                else logger.warn("VALID", "step={:>6} status=stalled mse={:.8f} psnr={:>5.2f} best={:.8f}@{} patience={}/{} images={:>3} pixels={} val={:>8.3f}ms", validation->step, validation->mse, validation->psnr, best_validation_mse, best_validation_step, validation_checks_without_improvement, early_stop_patience, validation->image_count, validation->pixel_count, validation->elapsed_ms);

                if (validation_checks_without_improvement >= early_stop_patience) {
                    stopped_early = true;
                    break;
                }
                while (next_validation_step <= stats->step) next_validation_step += validation_interval_steps;
            }
        }

        if (!pipeline_error) {
            if (stopped_early) logger.warn("SUMMARY", "steps={} stopped_early=true first_loss={:.6f} last_loss={:.6f} train={:.3f}s avg={:.2f} step/s best_validation={:.8f}@{} psnr={:.2f}", final_step, first_loss, last_loss, train_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / train_ms, best_validation_mse, best_validation_step, best_validation_psnr);
            else logger.info("SUMMARY", "steps={} stopped_early=false first_loss={:.6f} last_loss={:.6f} train={:.3f}s avg={:.2f} step/s best_validation={:.8f}@{} psnr={:.2f}", final_step, first_loss, last_loss, train_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / train_ms, best_validation_mse, best_validation_step, best_validation_psnr);

            const auto test = ngp->test();
            if (!test) pipeline_error = test.error();
            else logger.info("TEST", "step={:>6} mse={:.8f} psnr={:>5.2f} images={:>3} saved={} pixels={} output={} test={:>8.3f}ms", test->step, test->mse, test->psnr, test->image_count, test->comparison_image_count, test->pixel_count, test->output_dir.string(), test->elapsed_ms);
        }

        if (!pipeline_error && export_weights_path.has_value()) {
            const auto exported_weights = ngp->export_weights(*export_weights_path);
            if (!exported_weights) pipeline_error = exported_weights.error();
            else logger.info("WEIGHT", "exported={}", export_weights_path->string());
        }
    }

    if (!pipeline_error) logger.info("DONE", "pipeline=succeeded");
    else logger.error("ERROR", "pipeline=failed error=\"{}\"", *pipeline_error);
    logger.flush();
    return !pipeline_error ? 0 : 1;
}
