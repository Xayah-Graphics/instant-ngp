#include <cuda_runtime_api.h>
#include "json/json.hpp"
import std;
import xcli;
import nerf_synthetic;
import dd_nerf;
import ngp.train;

#ifndef NGP_TRAIN_PROFILE_NAME
#error "NGP_TRAIN_PROFILE_NAME must be provided by the benchmark target."
#endif

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    constexpr std::string_view active_train_profile_name = NGP_TRAIN_PROFILE_NAME;
    constexpr std::string_view ansi_reset = "\x1b[0m";
    constexpr std::string_view ansi_dim = "\x1b[2m";
    constexpr std::string_view ansi_bold = "\x1b[1m";
    constexpr std::string_view ansi_cyan = "\x1b[36m";
    constexpr std::string_view ansi_green = "\x1b[32m";
    constexpr std::string_view ansi_yellow = "\x1b[33m";
    constexpr std::string_view ansi_red = "\x1b[31m";
    constexpr std::string_view ansi_evaluation_badge = "\x1b[1;37;45m";
    constexpr std::string_view ansi_evaluation_metric = "\x1b[1;95m";
    constexpr std::string_view ansi_evaluation_best = "\x1b[1;33m";
    struct DatasetProvider final {
        std::string_view name;
        bool (*matches)(const std::filesystem::path&);
        std::expected<std::unique_ptr<ngp::train::InstantNGP>, std::string> (*create)(const std::filesystem::path&, const std::vector<std::string>&, float);
    };
    const std::array dataset_providers{
        DatasetProvider{
            .name = "nerf-synthetic",
            .matches = nerf_synthetic::is_dataset,
            .create = [](const std::filesystem::path& path, const std::vector<std::string>& frame_sets, const float scene_scale) -> std::expected<std::unique_ptr<ngp::train::InstantNGP>, std::string> {
                const auto dataset = nerf_synthetic::load(path, {.frame_sets = frame_sets, .scene_scale = scene_scale});
                if (!dataset) return std::unexpected{dataset.error()};
                try {
                    return std::make_unique<ngp::train::InstantNGP>(*dataset);
                } catch (const std::exception& error) {
                    return std::unexpected{std::string{error.what()}};
                }
            },
        },
        DatasetProvider{
            .name = "dd-nerf-dataset",
            .matches = dd_nerf::is_dataset,
            .create = [](const std::filesystem::path& path, const std::vector<std::string>& frame_sets, const float scene_scale) -> std::expected<std::unique_ptr<ngp::train::InstantNGP>, std::string> {
                const auto dataset = dd_nerf::load(path, {.frame_sets = frame_sets, .scene_scale = scene_scale});
                if (!dataset) return std::unexpected{dataset.error()};
                try {
                    return std::make_unique<ngp::train::InstantNGP>(*dataset);
                } catch (const std::exception& error) {
                    return std::unexpected{std::string{error.what()}};
                }
            },
        },
    };

    std::filesystem::path dataset_path;
    std::string optimize_frame_set = "train";
    std::vector<std::string> evaluation_frame_sets;
    std::int32_t steps = 200000;
    std::int32_t log_every_steps = 1000;
    std::uint32_t evaluate_every_steps = 5000u;
    std::uint32_t early_stop_patience = 5u;
    bool no_optimize = false;
    bool no_evaluation = false;
    constexpr float default_scene_scale = 0.33f;
    float scene_scale = default_scene_scale;
    float early_stop_min_delta_mse = 1e-6f;
    std::optional<std::filesystem::path> load_weights_path;
    std::optional<std::filesystem::path> save_weights_path;
    std::optional<std::filesystem::path> comparison_output_dir;
    std::optional<std::filesystem::path> benchmark_output_path;
    std::string_view dataset_format;

    xcli::Command command =
        xcli::Command{"Benchmark Instant NGP optimization and evaluation on named frame sets."}
        | xcli::positional({.name = "dataset-path", .description = "NeRF synthetic or DD-NeRF dataset root", .show_default = false, .required = true}, dataset_path, {.requirement = xcli::PathRequirement::existing_directory})
        | xcli::option({.long_name = "dataset", .value_name = "path", .description = "NeRF synthetic or DD-NeRF dataset root", .show_default = false}, dataset_path, {.requirement = xcli::PathRequirement::existing_directory})
        | xcli::option({.long_name = "optimize", .value_name = "frame-set", .description = "frame set used for parameter optimization"}, optimize_frame_set)
        | xcli::option({.long_name = "no-optimize", .description = "skip parameter optimization", .show_default = false}, no_optimize)
        | xcli::option({.long_name = "evaluate", .value_name = "frame-set", .description = "frame set to evaluate; repeat for multiple sets", .default_text = "validation"}, evaluation_frame_sets)
        | xcli::option({.long_name = "no-evaluation", .description = "skip evaluation and early stopping", .show_default = false}, no_evaluation)
        | xcli::option({.long_name = "evaluate-every", .value_name = "count", .description = "optimization steps per interval evaluation"}, evaluate_every_steps, {.minimum = 1.0})
        | xcli::option({.long_name = "steps", .value_name = "count", .description = "total optimization steps"}, steps, {.minimum = 1.0})
        | xcli::option({.long_name = "log-every", .value_name = "count", .description = "optimization steps per progress log"}, log_every_steps, {.minimum = 1.0})
        | xcli::option({.long_name = "scene-scale", .value_name = "value", .description = "camera normalization scene scale"}, scene_scale, {.minimum = 0.0, .minimum_is_exclusive = true})
        | xcli::option({.long_name = "early-stop-patience", .value_name = "count", .description = "first evaluation frame set checks without improvement before stopping; 0 disables early stop"}, early_stop_patience, {.minimum = 0.0})
        | xcli::option({.long_name = "early-stop-min-delta", .value_name = "mse", .description = "minimum first evaluation frame set MSE improvement"}, early_stop_min_delta_mse, {.minimum = 0.0})
        | xcli::option({.long_name = "load-weights", .value_name = "path", .description = "load safetensors weights before optimization or evaluation"}, load_weights_path, {.requirement = xcli::PathRequirement::existing_file})
        | xcli::option({.long_name = "save-weights", .value_name = "path", .description = "save final safetensors weights after optimization"}, save_weights_path, {.requirement = xcli::PathRequirement::existing_parent_directory})
        | xcli::option({.long_name = "comparison-output", .value_name = "dir", .description = "save final evaluation comparison images"}, comparison_output_dir, {.requirement = xcli::PathRequirement::existing_parent_directory})
        | xcli::option({.long_name = "benchmark-output", .value_name = "path", .description = "append one benchmark result row to .jsonl or .csv"}, benchmark_output_path, {.requirement = xcli::PathRequirement::existing_parent_directory})
        | xcli::example("../data/nerf-synthetic/lego --steps 30000")
        | xcli::example("../data/nerf-synthetic/lego --no-evaluation --save-weights build-benchmarks/model.safetensors")
        | xcli::example("../data/nerf-synthetic/lego --steps 1000 --evaluate validation --benchmark-output build-benchmarks/results.jsonl")
        | xcli::example("../data/nerf-synthetic/lego --no-optimize --evaluate validation --load-weights build-benchmarks/model.safetensors")
        | xcli::example("../data/nerf-synthetic/lego --no-optimize --evaluate test --load-weights build-benchmarks/model.safetensors --comparison-output build-benchmarks/test-lego")
        | xcli::example("../data/nerf-synthetic/lego --evaluate validation --evaluate test --steps 1");

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

    const auto path_validation = command.validate();
    if (!path_validation) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, path_validation.error());
        return 2;
    }

    if (!command.option_provided("evaluate") && !no_evaluation) evaluation_frame_sets.push_back("validation");

    std::optional<std::string> cli_error;
    if (optimize_frame_set != "train" && optimize_frame_set != "validation" && optimize_frame_set != "test") cli_error = std::format("--optimize must be one of train, validation, or test; got '{}'.", optimize_frame_set);
    else {
        for (const std::string& evaluation_frame_set : evaluation_frame_sets) {
            if (evaluation_frame_set != "train" && evaluation_frame_set != "validation" && evaluation_frame_set != "test") {
                cli_error = std::format("--evaluate must be one of train, validation, or test; got '{}'.", evaluation_frame_set);
                break;
            }
            std::uint32_t duplicate_count = 0u;
            for (const std::string& other_evaluation_frame_set : evaluation_frame_sets)
                if (other_evaluation_frame_set == evaluation_frame_set) ++duplicate_count;
            if (duplicate_count > 1u) {
                cli_error = std::format("frame set '{}' was provided to --evaluate more than once.", evaluation_frame_set);
                break;
            }
        }
    }
    if (!cli_error && no_optimize && !load_weights_path.has_value()) cli_error = "--no-optimize requires --load-weights.";
    else if (!cli_error && no_optimize && no_evaluation) cli_error = "--no-optimize cannot be combined with --no-evaluation.";
    else if (!cli_error && no_optimize && command.option_provided("optimize")) cli_error = "--optimize is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("steps")) cli_error = "--steps is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("log-every")) cli_error = "--log-every is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("save-weights")) cli_error = "--save-weights is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("evaluate-every")) cli_error = "--evaluate-every is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("early-stop-patience")) cli_error = "--early-stop-patience is not valid with --no-optimize.";
    else if (!cli_error && no_optimize && command.option_provided("early-stop-min-delta")) cli_error = "--early-stop-min-delta is not valid with --no-optimize.";
    else if (!cli_error && no_evaluation && command.option_provided("evaluate")) cli_error = "--evaluate is not valid with --no-evaluation.";
    else if (!cli_error && no_evaluation && comparison_output_dir.has_value()) cli_error = "--comparison-output is not valid with --no-evaluation.";
    else if (!cli_error && no_evaluation && command.option_provided("evaluate-every")) cli_error = "--evaluate-every is not valid with --no-evaluation.";
    else if (!cli_error && no_evaluation && command.option_provided("early-stop-patience")) cli_error = "--early-stop-patience is not valid with --no-evaluation.";
    else if (!cli_error && no_evaluation && command.option_provided("early-stop-min-delta")) cli_error = "--early-stop-min-delta is not valid with --no-evaluation.";
    else if (!cli_error && benchmark_output_path.has_value() && benchmark_output_path->extension() != ".jsonl" && benchmark_output_path->extension() != ".csv") cli_error = "--benchmark-output must use a .jsonl or .csv extension.";
    if (cli_error.has_value()) {
        std::println("{}error:{} {}", ansi_red, ansi_reset, *cli_error);
        return 2;
    }

    std::vector<std::string> requested_frame_sets;
    if (!no_optimize) requested_frame_sets.push_back(optimize_frame_set);
    if (!no_evaluation) {
        for (const std::string& evaluation_frame_set : evaluation_frame_sets) {
            bool already_requested = false;
            for (const std::string& requested_frame_set : requested_frame_sets)
                if (requested_frame_set == evaluation_frame_set) already_requested = true;
            if (!already_requested) requested_frame_sets.push_back(evaluation_frame_set);
        }
    }

    const DatasetProvider* dataset_provider = nullptr;
    std::vector<std::string_view> matched_dataset_formats;
    for (const DatasetProvider& provider : dataset_providers) {
        if (!provider.matches(dataset_path)) continue;
        matched_dataset_formats.push_back(provider.name);
        if (dataset_provider == nullptr) dataset_provider = std::addressof(provider);
    }
    if (dataset_provider == nullptr) {
        std::string supported_formats;
        for (const DatasetProvider& provider : dataset_providers) {
            if (!supported_formats.empty()) supported_formats += ", ";
            supported_formats += provider.name;
        }
        std::println("{}error:{} dataset path '{}' does not match any supported dataset format: {}.", ansi_red, ansi_reset, dataset_path.string(), supported_formats);
        return 2;
    }
    if (matched_dataset_formats.size() > 1uz) {
        std::string matched_formats;
        for (const std::string_view matched_format : matched_dataset_formats) {
            if (!matched_formats.empty()) matched_formats += ", ";
            matched_formats += matched_format;
        }
        std::println("{}error:{} dataset path '{}' matches multiple dataset formats: {}.", ansi_red, ansi_reset, dataset_path.string(), matched_formats);
        return 2;
    }
    dataset_format = dataset_provider->name;

    const bool evaluation_enabled = !no_evaluation;
    const bool optimization_enabled = !no_optimize;
    const bool early_stop_enabled = optimization_enabled && evaluation_enabled && early_stop_patience != 0u;

    std::string frame_set_stage;
    for (const std::string& requested_frame_set : requested_frame_sets) {
        if (!frame_set_stage.empty()) frame_set_stage += ",";
        frame_set_stage += requested_frame_set;
    }
    std::string evaluation_stage = "off";
    if (evaluation_enabled) {
        evaluation_stage = "sets:";
        for (std::size_t evaluation_index = 0uz; evaluation_index < evaluation_frame_sets.size(); ++evaluation_index) {
            if (evaluation_index != 0uz) evaluation_stage += ",";
            evaluation_stage += evaluation_frame_sets[evaluation_index];
        }
        if (optimization_enabled) evaluation_stage += std::format(",every:{}", evaluate_every_steps);
    }
    const std::string optimize_stage = optimization_enabled ? optimize_frame_set : "off";
    const std::string early_stop_stage = early_stop_enabled ? std::format("frame_set:{},patience:{},min_delta:{:.6g}", evaluation_frame_sets.front(), early_stop_patience, static_cast<double>(early_stop_min_delta_mse)) : std::string{"off"};
    const std::string comparison_stage = comparison_output_dir.has_value() ? std::format("comparison_output={}", comparison_output_dir->string()) : std::string{"comparison_output=off"};
    const std::string benchmark_output_stage = benchmark_output_path.has_value() ? std::format("benchmark_output={}", benchmark_output_path->string()) : std::string{"benchmark_output=off"};

    const auto config_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    if (optimization_enabled) std::println("{}[{:%F %T}]{} {}{:<8}{} profile={} dataset={} format={} scene_scale={} frame_sets={} optimize={} steps={} log_every={} evaluation={} early_stop={} {} {} load_weights={} save_weights={}", ansi_dim, config_timestamp, ansi_reset, ansi_cyan, "CONFIG", ansi_reset, active_train_profile_name, dataset_path.string(), dataset_format, scene_scale, frame_set_stage, optimize_stage, steps, log_every_steps, evaluation_stage, early_stop_stage, comparison_stage, benchmark_output_stage, load_weights_path.has_value() ? load_weights_path->string() : "none", save_weights_path.has_value() ? save_weights_path->string() : "none");
    else std::println("{}[{:%F %T}]{} {}{:<8}{} profile={} dataset={} format={} scene_scale={} frame_sets={} optimize=off evaluation={} {} {} load_weights={}", ansi_dim, config_timestamp, ansi_reset, ansi_cyan, "CONFIG", ansi_reset, active_train_profile_name, dataset_path.string(), dataset_format, scene_scale, frame_set_stage, evaluation_stage, comparison_stage, benchmark_output_stage, load_weights_path.has_value() ? load_weights_path->string() : "none");

    const auto load_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<8}{} loading dataset", ansi_dim, load_timestamp, ansi_reset, ansi_cyan, "INFO", ansi_reset);

    std::optional<std::string> pipeline_error;
    std::unique_ptr<ngp::train::InstantNGP> ngp;
    const bool benchmark_output_enabled = benchmark_output_path.has_value();
    const auto benchmark_start = std::chrono::steady_clock::now();
    std::uint64_t benchmark_peak_vram_bytes = 0u;
    std::uint32_t benchmark_final_step = 0u;
    float benchmark_optimization_elapsed_ms = 0.0f;
    float benchmark_sample_efficiency_ratio = std::numeric_limits<float>::quiet_NaN();
    float benchmark_density_grid_occupancy_ratio = std::numeric_limits<float>::quiet_NaN();
    std::optional<ngp::train::EvaluationStats> benchmark_primary_evaluation;
    const auto query_used_device_memory = [] -> std::expected<std::uint64_t, std::string> {
        try {
            std::size_t free_bytes = 0uz;
            std::size_t total_bytes = 0uz;
            if (const cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes); status != cudaSuccess) throw std::runtime_error{std::string{"cudaMemGetInfo failed: "} + cudaGetErrorString(status)};
            return static_cast<std::uint64_t>(total_bytes - free_bytes);
        } catch (const std::exception& error) {
            return std::unexpected{std::string{error.what()}};
        }
    };

    auto created_ngp = dataset_provider->create(dataset_path, requested_frame_sets, scene_scale);
    if (!created_ngp) pipeline_error = created_ngp.error();
    else ngp = std::move(*created_ngp);
    if (!pipeline_error && benchmark_output_enabled) {
        const auto memory = query_used_device_memory();
        if (!memory) pipeline_error = memory.error();
        else benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
    }

    if (!pipeline_error && load_weights_path.has_value()) {
        const auto loaded_weights = ngp->load_weights(*load_weights_path);
        if (!loaded_weights) pipeline_error = loaded_weights.error();
        else {
            if (benchmark_output_enabled) {
                const auto memory = query_used_device_memory();
                if (!memory) pipeline_error = memory.error();
                else benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
            }
            if (!pipeline_error) {
                const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<8}{} loaded={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, load_weights_path->string());
            }
        }
    }

    if (!pipeline_error && optimization_enabled) {
        float first_loss = 0.0f;
        float last_loss = 0.0f;
        float optimize_ms = 0.0f;
        float best_evaluation_mse = std::numeric_limits<float>::infinity();
        float best_evaluation_psnr = 0.0f;
        std::uint32_t final_step = 0u;
        std::uint32_t best_evaluation_step = 0u;
        std::uint32_t evaluation_checks_without_improvement = 0u;
        bool stopped_early = false;
        std::uint32_t next_evaluation_step = evaluate_every_steps;

        for (std::int32_t optimized_steps = 0; optimized_steps < steps;) {
            const std::int32_t requested_steps = std::min(log_every_steps, steps - optimized_steps);
            const auto stats = ngp->optimize({.frame_set = optimize_frame_set, .iterations = requested_steps});
            if (!stats) {
                pipeline_error = stats.error();
                break;
            }

            if (optimized_steps == 0) first_loss = stats->loss;
            last_loss = stats->loss;
            optimize_ms += stats->elapsed_ms;
            final_step = stats->step;
            if (benchmark_output_enabled) {
                benchmark_final_step = stats->step;
                benchmark_optimization_elapsed_ms = optimize_ms;
                benchmark_sample_efficiency_ratio = stats->sample_efficiency_ratio;
                benchmark_density_grid_occupancy_ratio = stats->density_grid_occupancy_ratio;
                const auto memory = query_used_device_memory();
                if (!memory) {
                    pipeline_error = memory.error();
                    break;
                }
                benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
            }
            optimized_steps += requested_steps;
            const auto optimize_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<8}{} frame_set={} step={:>6}/{} loss={:>10.6f} chunk={:>8.3f}ms rate={:>7.2f} step/s next_rays={:>6} samples={:>7}/{:<7} sample_eff={:>6.2f}% occupied={:>7} occupancy={:>6.2f}%", ansi_dim, optimize_timestamp, ansi_reset, ansi_green, "OPTIMIZE", ansi_reset, optimize_frame_set, stats->step, steps, stats->loss, stats->elapsed_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->next_rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->sample_efficiency_ratio * 100.0f, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio * 100.0f);

            if (evaluation_enabled && (stats->step >= next_evaluation_step || stats->step >= static_cast<std::uint32_t>(steps))) {
                bool first_evaluation_improved = false;
                for (std::size_t evaluation_index = 0uz; evaluation_index < evaluation_frame_sets.size(); ++evaluation_index) {
                    const auto evaluation = ngp->evaluate({.frame_set = evaluation_frame_sets[evaluation_index]});
                    if (!evaluation) {
                        pipeline_error = evaluation.error();
                        break;
                    }
                    if (benchmark_output_enabled) {
                        const auto memory = query_used_device_memory();
                        if (!memory) {
                            pipeline_error = memory.error();
                            break;
                        }
                        benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
                    }

                    const auto evaluation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                    if (evaluation_index == 0uz) {
                        if (benchmark_output_enabled) benchmark_primary_evaluation = *evaluation;
                        first_evaluation_improved = evaluation->mse < best_evaluation_mse - early_stop_min_delta_mse;
                        if (first_evaluation_improved) {
                            best_evaluation_mse = evaluation->mse;
                            best_evaluation_psnr = evaluation->psnr;
                            best_evaluation_step = evaluation->step;
                            evaluation_checks_without_improvement = 0u;
                        } else if (early_stop_enabled) {
                            ++evaluation_checks_without_improvement;
                        }

                        if (early_stop_enabled) std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}{}{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | {}BEST={:.8f}@{}{} | patience={}{}{}/{} | images={:>3} pixels={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, first_evaluation_improved ? ansi_green : ansi_yellow, first_evaluation_improved ? "improved" : "stalled", ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, ansi_evaluation_best, best_evaluation_mse, best_evaluation_step, ansi_reset, evaluation_checks_without_improvement == 0u ? ansi_green : ansi_yellow, evaluation_checks_without_improvement, ansi_reset, early_stop_patience, evaluation->image_count, evaluation->pixel_count, evaluation->elapsed_ms);
                        else std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}{}{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | {}BEST={:.8f}@{}{} | early_stop=off | images={:>3} pixels={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, first_evaluation_improved ? ansi_green : ansi_yellow, first_evaluation_improved ? "improved" : "stalled", ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, ansi_evaluation_best, best_evaluation_mse, best_evaluation_step, ansi_reset, evaluation->image_count, evaluation->pixel_count, evaluation->elapsed_ms);
                    } else {
                        std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}evaluated{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3} pixels={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, ansi_green, ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, evaluation->image_count, evaluation->pixel_count, evaluation->elapsed_ms);
                    }
                }
                if (pipeline_error) break;
                if (early_stop_enabled && !first_evaluation_improved && evaluation_checks_without_improvement >= early_stop_patience) {
                    stopped_early = true;
                    break;
                }
                while (next_evaluation_step <= stats->step) next_evaluation_step += evaluate_every_steps;
            }
        }

        if (!pipeline_error) {
            const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            if (evaluation_enabled) std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s best_evaluation={}:{:.8f}@{} psnr={:.2f}", ansi_dim, summary_timestamp, ansi_reset, stopped_early ? ansi_yellow : ansi_bold, "SUMMARY", ansi_reset, final_step, stopped_early, first_loss, last_loss, optimize_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / optimize_ms, evaluation_frame_sets.front(), best_evaluation_mse, best_evaluation_step, best_evaluation_psnr);
            else std::println("{}[{:%F %T}]{} {}{:<8}{} steps={} stopped_early=false first_loss={:.6f} last_loss={:.6f} optimize={:.3f}s avg={:.2f} step/s evaluation=off", ansi_dim, summary_timestamp, ansi_reset, ansi_bold, "SUMMARY", ansi_reset, final_step, first_loss, last_loss, optimize_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / optimize_ms);
        }

        if (!pipeline_error && save_weights_path.has_value()) {
            const auto saved_weights = ngp->export_weights(*save_weights_path);
            if (!saved_weights) pipeline_error = saved_weights.error();
            else {
                const auto weights_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<8}{} saved={}", ansi_dim, weights_timestamp, ansi_reset, ansi_yellow, "WEIGHT", ansi_reset, save_weights_path->string());
            }
        }

        if (!pipeline_error && comparison_output_dir.has_value()) {
            for (std::size_t evaluation_index = 0uz; evaluation_index < evaluation_frame_sets.size(); ++evaluation_index) {
                const std::string& evaluation_frame_set = evaluation_frame_sets[evaluation_index];
                const std::filesystem::path output_dir = evaluation_frame_sets.size() == 1uz ? *comparison_output_dir : *comparison_output_dir / evaluation_frame_set;
                const auto evaluation = ngp->evaluate({.frame_set = evaluation_frame_set, .comparison_output_dir = output_dir});
                if (!evaluation) {
                    pipeline_error = evaluation.error();
                    break;
                }
                if (benchmark_output_enabled) {
                    const auto memory = query_used_device_memory();
                    if (!memory) {
                        pipeline_error = memory.error();
                        break;
                    }
                    benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
                    if (evaluation_index == 0uz) benchmark_primary_evaluation = *evaluation;
                }
                const auto evaluation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}saved{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3} saved={} pixels={} output={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, ansi_green, ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, evaluation->image_count, evaluation->comparison_image_count, evaluation->pixel_count, evaluation->output_dir.string(), evaluation->elapsed_ms);
            }
        }
    }

    if (!pipeline_error && !optimization_enabled) {
        for (std::size_t evaluation_index = 0uz; evaluation_index < evaluation_frame_sets.size(); ++evaluation_index) {
            const std::string& evaluation_frame_set = evaluation_frame_sets[evaluation_index];
            const std::optional<std::filesystem::path> output_dir = comparison_output_dir.has_value() ? std::optional<std::filesystem::path>{evaluation_frame_sets.size() == 1uz ? *comparison_output_dir : *comparison_output_dir / evaluation_frame_set} : std::nullopt;
            const auto evaluation = ngp->evaluate({.frame_set = evaluation_frame_set, .comparison_output_dir = output_dir});
            if (!evaluation) {
                pipeline_error = evaluation.error();
                break;
            }
            if (benchmark_output_enabled) {
                const auto memory = query_used_device_memory();
                if (!memory) {
                    pipeline_error = memory.error();
                    break;
                }
                benchmark_peak_vram_bytes = std::max(benchmark_peak_vram_bytes, *memory);
                benchmark_final_step = evaluation->step;
                if (evaluation_index == 0uz) benchmark_primary_evaluation = *evaluation;
            }
            const auto evaluation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            if (output_dir.has_value()) std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}saved{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3} saved={} pixels={} output={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, ansi_green, ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, evaluation->image_count, evaluation->comparison_image_count, evaluation->pixel_count, evaluation->output_dir.string(), evaluation->elapsed_ms);
            else std::println("{}[{:%F %T}]{} {} {:<8} {} frame_set={} step={:>6} status={}evaluated{} | {}MSE={:.8f}{} {}PSNR={:>5.2f}{} | images={:>3} pixels={} eval={:>8.3f}ms", ansi_dim, evaluation_timestamp, ansi_reset, ansi_evaluation_badge, "EVAL", ansi_reset, evaluation->frame_set, evaluation->step, ansi_green, ansi_reset, ansi_evaluation_metric, evaluation->mse, ansi_reset, ansi_cyan, evaluation->psnr, ansi_reset, evaluation->image_count, evaluation->pixel_count, evaluation->elapsed_ms);
        }
    }

    if (!pipeline_error && benchmark_output_enabled) {
        try {
            const float benchmark_elapsed_ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - benchmark_start).count();
            const float benchmark_step_rate = benchmark_optimization_elapsed_ms > 0.0f ? static_cast<float>(benchmark_final_step) * 1000.0f / benchmark_optimization_elapsed_ms : 0.0f;
            const bool has_optimization_metrics = std::isfinite(benchmark_sample_efficiency_ratio) && std::isfinite(benchmark_density_grid_occupancy_ratio);
            const bool has_evaluation_metrics = benchmark_primary_evaluation.has_value();
            const bool has_json_mse = has_evaluation_metrics && std::isfinite(benchmark_primary_evaluation->mse);
            const bool has_json_psnr = has_evaluation_metrics && std::isfinite(benchmark_primary_evaluation->psnr);
            if (benchmark_output_path->extension() == ".jsonl") {
                nlohmann::json row = nlohmann::json::object();
                row["profile"] = std::string{active_train_profile_name};
                row["dataset"] = dataset_path.string();
                row["steps"] = benchmark_final_step;
                row["elapsed_ms"] = benchmark_elapsed_ms;
                row["step_per_second"] = benchmark_step_rate;
                row["sample_eff"] = nullptr;
                row["occupancy"] = nullptr;
                row["mse"] = nullptr;
                row["psnr"] = nullptr;
                if (has_optimization_metrics) row["sample_eff"] = benchmark_sample_efficiency_ratio;
                if (has_optimization_metrics) row["occupancy"] = benchmark_density_grid_occupancy_ratio;
                if (has_json_mse) row["mse"] = benchmark_primary_evaluation->mse;
                if (has_json_psnr) row["psnr"] = benchmark_primary_evaluation->psnr;
                row["peak_vram_bytes"] = benchmark_peak_vram_bytes;
                row["peak_vram_mib"] = static_cast<double>(benchmark_peak_vram_bytes) / (1024.0 * 1024.0);

                std::ofstream output{*benchmark_output_path, std::ios::app};
                if (!output) throw std::runtime_error{std::format("failed to open benchmark output '{}'.", benchmark_output_path->string())};
                output << row.dump() << '\n';
                if (!output) throw std::runtime_error{std::format("failed to write benchmark output '{}'.", benchmark_output_path->string())};
            } else {
                const bool write_header = !std::filesystem::exists(*benchmark_output_path) || std::filesystem::file_size(*benchmark_output_path) == 0u;
                std::ofstream output{*benchmark_output_path, std::ios::app};
                if (!output) throw std::runtime_error{std::format("failed to open benchmark output '{}'.", benchmark_output_path->string())};
                if (write_header) output << "profile,dataset,steps,elapsed_ms,step_per_second,sample_eff,occupancy,mse,psnr,peak_vram_bytes\n";
                output << std::quoted(std::string{active_train_profile_name}) << ',' << std::quoted(dataset_path.string()) << ',' << benchmark_final_step << ',' << std::setprecision(9) << benchmark_elapsed_ms << ',' << benchmark_step_rate << ',';
                if (has_optimization_metrics) output << benchmark_sample_efficiency_ratio;
                output << ',';
                if (has_optimization_metrics) output << benchmark_density_grid_occupancy_ratio;
                output << ',';
                if (has_evaluation_metrics) output << benchmark_primary_evaluation->mse;
                output << ',';
                if (has_evaluation_metrics) output << benchmark_primary_evaluation->psnr;
                output << ',' << benchmark_peak_vram_bytes << '\n';
                if (!output) throw std::runtime_error{std::format("failed to write benchmark output '{}'.", benchmark_output_path->string())};
            }
        } catch (const std::exception& error) {
            pipeline_error = std::string{error.what()};
        }
    }

    const auto finish_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    if (!pipeline_error) std::println("{}[{:%F %T}]{} {}{:<8}{} pipeline=succeeded", ansi_dim, finish_timestamp, ansi_reset, ansi_bold, "DONE", ansi_reset);
    else std::println("{}[{:%F %T}]{} {}{:<8}{} pipeline=failed error=\"{}\"", ansi_dim, finish_timestamp, ansi_reset, ansi_red, "ERROR", ansi_reset, *pipeline_error);
    return !pipeline_error ? 0 : 1;
}
