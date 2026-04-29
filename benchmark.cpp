import std;
import ngp.dataset;
import ngp.train;

namespace {
    constexpr std::string_view ansi_reset  = "\x1b[0m";
    constexpr std::string_view ansi_dim    = "\x1b[2m";
    constexpr std::string_view ansi_bold   = "\x1b[1m";
    constexpr std::string_view ansi_cyan   = "\x1b[36m";
    constexpr std::string_view ansi_green  = "\x1b[32m";
    constexpr std::string_view ansi_yellow = "\x1b[33m";
    constexpr std::string_view ansi_red    = "\x1b[31m";

    struct CandidateResult final {
        std::string_view stage             = "";
        std::uint32_t rank                 = 0u;
        float scene_scale                  = 0.0f;
        std::uint32_t final_step           = 0u;
        std::uint32_t best_validation_step = 0u;
        std::uint32_t occupied_cells       = 0u;
        std::uint32_t average_samples      = 0u;
        std::uint32_t average_compacted    = 0u;
        float final_loss                   = 0.0f;
        float final_mse                    = std::numeric_limits<float>::infinity();
        float final_psnr                   = -std::numeric_limits<float>::infinity();
        float best_mse                     = std::numeric_limits<float>::infinity();
        float best_psnr                    = -std::numeric_limits<float>::infinity();
        float occupancy_ratio              = 0.0f;
        float train_ms                     = 0.0f;
        float validation_ms                = 0.0f;
        float steps_per_second             = 0.0f;
        bool failed                        = false;
        std::string error;
    };
} // namespace

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    const std::string executable_name = !arguments.empty() && arguments.front() != nullptr ? std::filesystem::path{arguments.front()}.filename().string() : "instant-ngp-benchmark";
    const std::string usage           = std::format(
        R"({}Usage:{}
  {}{}{} {}--dataset <path>{} {}[options]{}

{}Options:{}
  {}--dataset <path>{}                  NeRF synthetic or DD-NeRF dataset root
  {}--steps <count>{}                   training steps per scene scale
                                    {}default:{} 10000
  {}--chunk-steps <count>{}             training steps per progress sample
                                    {}default:{} 1000
  {}--validation-interval <count>{}     validation interval in steps
                                    {}default:{} 5000
  {}--coarse-scales <csv>{}             first-stage scene_scale candidates
                                    {}default:{} 0.125,0.1767767,0.25,0.3535534,0.5,0.7071068,1,1.4142135,2,2.8284271,4
  {}--refine-count <odd>{}              second-stage candidate count around coarse best
                                    {}default:{} 9
  {}--output-csv <path>{}               benchmark CSV output
                                    {}default:{} scale-benchmark.csv
  {}-h, --help{}                        print this help
)",
        ansi_bold, ansi_reset, ansi_cyan, executable_name, ansi_reset, ansi_yellow, ansi_reset, ansi_dim, ansi_reset, ansi_bold, ansi_reset, ansi_green, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset, ansi_dim, ansi_reset, ansi_green, ansi_reset);

    std::filesystem::path dataset_path;
    std::filesystem::path output_csv_path = "scale-benchmark.csv";
    std::int32_t steps                    = 10000;
    std::int32_t chunk_steps              = 1000;
    std::uint32_t validation_interval     = 5000u;
    std::uint32_t refine_count            = 9u;
    std::string coarse_scales_text        = "0.125,0.1767767,0.25,0.3535534,0.5,0.7071068,1,1.4142135,2,2.8284271,4";
    bool dataset_was_set                  = false;

    for (std::size_t i = 1uz; i < arguments.size(); ++i) {
        const std::string_view argument{arguments[i]};
        const std::size_t assignment_position = argument.find('=');
        const std::string_view option_name    = assignment_position == std::string_view::npos ? argument : argument.substr(0uz, assignment_position);
        std::string_view inline_value;
        bool has_inline_value = false;
        if (assignment_position != std::string_view::npos) {
            inline_value     = argument.substr(assignment_position + 1uz);
            has_inline_value = true;
        }

        if (option_name == "-h" || option_name == "--help") {
            if (has_inline_value) {
                std::println("{}error:{} {} does not accept a value.\n{}", ansi_red, ansi_reset, option_name, usage);
                return 2;
            }
            std::println("{}", usage);
            return 0;
        }

        if (option_name == "--dataset") {
            if (dataset_was_set) {
                std::println("{}error:{} dataset path was provided more than once.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --dataset requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            if (value.empty()) {
                std::println("{}error:{} --dataset requires a non-empty value.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            dataset_path    = std::filesystem::path{value};
            dataset_was_set = true;
            continue;
        }

        if (option_name == "--steps") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --steps requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), steps);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || steps <= 0) {
                std::println("{}error:{} --steps must be a positive integer.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            continue;
        }

        if (option_name == "--chunk-steps") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --chunk-steps requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), chunk_steps);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || chunk_steps <= 0) {
                std::println("{}error:{} --chunk-steps must be a positive integer.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            continue;
        }

        if (option_name == "--validation-interval") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --validation-interval requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), validation_interval);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || validation_interval == 0u) {
                std::println("{}error:{} --validation-interval must be a positive integer.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            continue;
        }

        if (option_name == "--coarse-scales") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --coarse-scales requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            if (value.empty()) {
                std::println("{}error:{} --coarse-scales requires a non-empty CSV value.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            coarse_scales_text = std::string{value};
            continue;
        }

        if (option_name == "--refine-count") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --refine-count requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            const auto parsed = std::from_chars(value.data(), value.data() + value.size(), refine_count);
            if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size() || refine_count < 3u || refine_count % 2u == 0u) {
                std::println("{}error:{} --refine-count must be an odd integer >= 3.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            continue;
        }

        if (option_name == "--output-csv") {
            std::string_view value;
            if (has_inline_value) {
                value = inline_value;
            } else {
                if (i + 1uz >= arguments.size()) {
                    std::println("{}error:{} --output-csv requires a value.\n{}", ansi_red, ansi_reset, usage);
                    return 2;
                }
                value = std::string_view{arguments[++i]};
            }
            if (value.empty()) {
                std::println("{}error:{} --output-csv requires a non-empty value.\n{}", ansi_red, ansi_reset, usage);
                return 2;
            }
            output_csv_path = std::filesystem::path{value};
            continue;
        }

        if (!argument.starts_with("-") && !dataset_was_set) {
            dataset_path    = std::filesystem::path{argument};
            dataset_was_set = true;
            continue;
        }

        std::println("{}error:{} unknown argument '{}'.\n{}", ansi_red, ansi_reset, argument, usage);
        return 2;
    }

    if (dataset_path.empty()) {
        std::println("{}error:{} dataset path is required.\n{}", ansi_red, ansi_reset, usage);
        return 2;
    }
    if (!std::filesystem::is_directory(dataset_path)) {
        std::println("{}error:{} dataset path '{}' is not a directory.", ansi_red, ansi_reset, dataset_path.string());
        return 2;
    }
    if (!output_csv_path.parent_path().empty() && !std::filesystem::is_directory(output_csv_path.parent_path())) {
        std::println("{}error:{} output CSV parent directory '{}' does not exist.", ansi_red, ansi_reset, output_csv_path.parent_path().string());
        return 2;
    }

    std::vector<float> coarse_scene_scales;
    std::size_t token_begin = 0uz;
    while (token_begin <= coarse_scales_text.size()) {
        const std::size_t token_end = coarse_scales_text.find(',', token_begin);
        const std::size_t end       = token_end == std::string::npos ? coarse_scales_text.size() : token_end;
        const std::string_view token{coarse_scales_text.data() + token_begin, end - token_begin};
        float parsed_scale = 0.0f;
        const auto parsed  = std::from_chars(token.data(), token.data() + token.size(), parsed_scale);
        if (token.empty() || parsed.ec != std::errc{} || parsed.ptr != token.data() + token.size() || !std::isfinite(parsed_scale) || parsed_scale <= 0.0f) {
            std::println("{}error:{} --coarse-scales must contain finite positive scene_scale values.", ansi_red, ansi_reset);
            return 2;
        }
        coarse_scene_scales.push_back(parsed_scale);
        if (token_end == std::string::npos) break;
        token_begin = token_end + 1uz;
    }
    if (coarse_scene_scales.size() < 2uz) {
        std::println("{}error:{} --coarse-scales must contain at least two values.", ansi_red, ansi_reset);
        return 2;
    }
    std::ranges::sort(coarse_scene_scales);
    for (std::size_t i = 1uz; i < coarse_scene_scales.size(); ++i) {
        const float tolerance = std::max(1.0f, std::max(std::abs(coarse_scene_scales[i]), std::abs(coarse_scene_scales[i - 1uz]))) * 1e-6f;
        if (std::abs(coarse_scene_scales[i] - coarse_scene_scales[i - 1uz]) <= tolerance) {
            std::println("{}error:{} --coarse-scales contains duplicate scene_scale values.", ansi_red, ansi_reset);
            return 2;
        }
    }

    const bool has_nerf_synthetic_dataset = std::filesystem::is_regular_file(dataset_path / "transforms_train.json") && std::filesystem::is_regular_file(dataset_path / "transforms_val.json") && std::filesystem::is_regular_file(dataset_path / "transforms_test.json");
    const bool has_dd_nerf_dataset        = std::filesystem::is_regular_file(dataset_path / "cameras.json") && std::filesystem::is_directory(dataset_path / "images");
    if (has_nerf_synthetic_dataset == has_dd_nerf_dataset) {
        std::println("{}error:{} dataset path '{}' must contain exactly one supported dataset marker set.", ansi_red, ansi_reset, dataset_path.string());
        return 2;
    }

    std::vector<CandidateResult> results;
    results.reserve(coarse_scene_scales.size() + refine_count);
    std::vector<float> stage_scene_scales = coarse_scene_scales;
    std::string_view stage                = "coarse";

    const auto start_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<9}{} dataset={} format={} steps={} chunk={} validation_interval={} refine_count={} coarse_scales={}", ansi_dim, start_timestamp, ansi_reset, ansi_cyan, "BENCH", ansi_reset, dataset_path.string(), has_nerf_synthetic_dataset ? "nerf-synthetic" : "dd-nerf-dataset", steps, chunk_steps, validation_interval, refine_count, coarse_scales_text);

    for (std::uint32_t stage_index = 0u; stage_index < 2u; ++stage_index) {
        if (stage_index == 1u) {
            std::optional<std::size_t> best_coarse_index;
            for (std::size_t i = 0uz; i < results.size(); ++i) {
                if (results[i].stage != "coarse" || results[i].failed) continue;
                if (!best_coarse_index.has_value() || results[i].final_psnr > results[*best_coarse_index].final_psnr + 0.05f || (std::abs(results[i].final_psnr - results[*best_coarse_index].final_psnr) <= 0.05f && results[i].scene_scale < results[*best_coarse_index].scene_scale)) best_coarse_index = i;
            }
            if (!best_coarse_index.has_value()) break;

            const float best_coarse_scale    = results[*best_coarse_index].scene_scale;
            std::size_t best_coarse_position = 0uz;
            for (std::size_t i = 0uz; i < coarse_scene_scales.size(); ++i) {
                const float tolerance = std::max(1.0f, std::max(std::abs(best_coarse_scale), std::abs(coarse_scene_scales[i]))) * 1e-6f;
                if (std::abs(best_coarse_scale - coarse_scene_scales[i]) <= tolerance) {
                    best_coarse_position = i;
                    break;
                }
            }

            float lower_bound = 0.0f;
            float upper_bound = 0.0f;
            if (best_coarse_position == 0uz) {
                const float ratio = coarse_scene_scales[1uz] / coarse_scene_scales[0uz];
                lower_bound       = best_coarse_scale / ratio;
                upper_bound       = coarse_scene_scales[1uz];
            } else if (best_coarse_position + 1uz == coarse_scene_scales.size()) {
                const float ratio = coarse_scene_scales[best_coarse_position] / coarse_scene_scales[best_coarse_position - 1uz];
                lower_bound       = coarse_scene_scales[best_coarse_position - 1uz];
                upper_bound       = best_coarse_scale * ratio;
            } else {
                lower_bound = coarse_scene_scales[best_coarse_position - 1uz];
                upper_bound = coarse_scene_scales[best_coarse_position + 1uz];
            }
            if (!std::isfinite(lower_bound) || !std::isfinite(upper_bound) || lower_bound <= 0.0f || upper_bound <= lower_bound) {
                std::println("{}error:{} failed to construct a valid refine scene_scale range around {}.", ansi_red, ansi_reset, best_coarse_scale);
                return 1;
            }

            stage_scene_scales.clear();
            const std::uint32_t half_refine_count = refine_count / 2u;
            for (std::uint32_t i = 0u; i < half_refine_count; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(half_refine_count);
                stage_scene_scales.push_back(static_cast<float>(std::exp(std::log(static_cast<double>(lower_bound)) * (1.0 - t) + std::log(static_cast<double>(best_coarse_scale)) * t)));
            }
            stage_scene_scales.push_back(best_coarse_scale);
            for (std::uint32_t i = 1u; i <= half_refine_count; ++i) {
                const double t = static_cast<double>(i) / static_cast<double>(half_refine_count);
                stage_scene_scales.push_back(static_cast<float>(std::exp(std::log(static_cast<double>(best_coarse_scale)) * (1.0 - t) + std::log(static_cast<double>(upper_bound)) * t)));
            }
            stage                = "refine";
            const auto timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
            std::println("{}[{:%F %T}]{} {}{:<9}{} coarse_best_scene_scale={:.7f} refine_range=[{:.7f}, {:.7f}] refine_count={}", ansi_dim, timestamp, ansi_reset, ansi_cyan, "REFINE", ansi_reset, best_coarse_scale, lower_bound, upper_bound, refine_count);
        }

        for (std::size_t candidate_index = 0uz; candidate_index < stage_scene_scales.size(); ++candidate_index) {
            const float scene_scale = stage_scene_scales[candidate_index];
            CandidateResult result  = {};
            result.stage            = stage;
            result.rank             = static_cast<std::uint32_t>(candidate_index);
            result.scene_scale      = scene_scale;

            if (stage == "refine") {
                bool reused = false;
                for (const CandidateResult& existing : results) {
                    const float tolerance = std::max(1.0f, std::max(std::abs(existing.scene_scale), std::abs(scene_scale))) * 1e-6f;
                    if (std::abs(existing.scene_scale - scene_scale) <= tolerance) {
                        result       = existing;
                        result.stage = stage;
                        result.rank  = static_cast<std::uint32_t>(candidate_index);
                        reused       = true;
                        break;
                    }
                }
                if (reused) {
                    const auto timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                    if (result.failed)
                        std::println("{}[{:%F %T}]{} {}{:<9}{} stage={} rank={} scene_scale={:<10.7f} status=reused-failed error=\"{}\"", ansi_dim, timestamp, ansi_reset, ansi_yellow, "REUSE", ansi_reset, result.stage, result.rank, result.scene_scale, result.error);
                    else
                        std::println("{}[{:%F %T}]{} {}{:<9}{} stage={} rank={} scene_scale={:<10.7f} final_psnr={:>6.2f} best_psnr={:>6.2f} status=reused", ansi_dim, timestamp, ansi_reset, ansi_yellow, "REUSE", ansi_reset, result.stage, result.rank, result.scene_scale, result.final_psnr, result.best_psnr);
                    results.push_back(std::move(result));
                    continue;
                }
            }

            std::optional<std::string> failure;
            std::unique_ptr<ngp::train::InstantNGP> ngp;

            if (has_nerf_synthetic_dataset) {
                const auto dataset = ngp::dataset::load_nerf_synthetic(dataset_path, scene_scale);
                if (!dataset) {
                    failure = dataset.error();
                } else {
                    result.scene_scale = dataset->scene_scale;
                    try {
                        ngp = std::make_unique<ngp::train::InstantNGP>(*dataset);
                    } catch (const std::exception& error) {
                        failure = std::string{error.what()};
                    }
                }
            } else {
                const auto dataset = ngp::dataset::load_dd_nerf_dataset(dataset_path, scene_scale);
                if (!dataset) {
                    failure = dataset.error();
                } else {
                    result.scene_scale = dataset->scene_scale;
                    try {
                        ngp = std::make_unique<ngp::train::InstantNGP>(*dataset);
                    } catch (const std::exception& error) {
                        failure = std::string{error.what()};
                    }
                }
            }

            if (!failure) {
                std::uint64_t sample_sum           = 0u;
                std::uint64_t compacted_sum        = 0u;
                std::uint32_t train_chunk_count    = 0u;
                std::uint32_t next_validation_step = validation_interval;

                for (std::int32_t trained_steps = 0; trained_steps < steps;) {
                    const std::int32_t requested_steps = std::min(chunk_steps, steps - trained_steps);
                    const auto stats                   = ngp->train(requested_steps);
                    if (!stats) {
                        failure = stats.error();
                        break;
                    }

                    result.final_loss      = stats->loss;
                    result.final_step      = stats->step;
                    result.occupied_cells  = stats->density_grid_occupied_cells;
                    result.occupancy_ratio = stats->density_grid_occupancy_ratio;
                    result.train_ms += stats->elapsed_ms;
                    sample_sum += stats->measured_sample_count_before_compaction;
                    compacted_sum += stats->measured_sample_count;
                    ++train_chunk_count;
                    trained_steps += requested_steps;

                    if (stats->step >= next_validation_step || stats->step >= static_cast<std::uint32_t>(steps)) {
                        const auto validation = ngp->validate();
                        if (!validation) {
                            failure = validation.error();
                            break;
                        }
                        result.validation_ms += validation->elapsed_ms;
                        result.final_mse  = validation->mse;
                        result.final_psnr = validation->psnr;
                        if (validation->psnr > result.best_psnr) {
                            result.best_psnr            = validation->psnr;
                            result.best_mse             = validation->mse;
                            result.best_validation_step = validation->step;
                        }
                        while (next_validation_step <= stats->step) next_validation_step += validation_interval;
                    }
                }

                if (!failure) {
                    if (train_chunk_count == 0u) {
                        failure = "training produced no chunks.";
                    } else {
                        result.average_samples   = static_cast<std::uint32_t>(sample_sum / train_chunk_count);
                        result.average_compacted = static_cast<std::uint32_t>(compacted_sum / train_chunk_count);
                        if (result.train_ms > 0.0f) result.steps_per_second = static_cast<float>(result.final_step) * 1000.0f / result.train_ms;
                        if (!std::isfinite(result.final_psnr)) failure = "benchmark did not produce final validation PSNR.";
                    }
                }
            }

            if (failure) {
                result.failed        = true;
                result.error         = *failure;
                const auto timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<9}{} stage={} rank={} scene_scale={:<10.7f} status=failed error=\"{}\"", ansi_dim, timestamp, ansi_reset, ansi_red, "CAND", ansi_reset, result.stage, result.rank, result.scene_scale, result.error);
            } else {
                const auto timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                std::println("{}[{:%F %T}]{} {}{:<9}{} stage={} rank={} scene_scale={:<10.7f} final_psnr={:>6.2f} final_mse={:.8f} best_psnr={:>6.2f} best_step={:>6} loss={:.6f} occupancy={:>6.2f}% samples={:>7}/{:<7} rate={:>7.2f} step/s", ansi_dim, timestamp, ansi_reset, ansi_green, "CAND", ansi_reset, result.stage, result.rank, result.scene_scale, result.final_psnr, result.final_mse, result.best_psnr, result.best_validation_step, result.final_loss, result.occupancy_ratio * 100.0f, result.average_compacted, result.average_samples, result.steps_per_second);
            }

            results.push_back(std::move(result));
        }
    }

    std::ofstream csv{output_csv_path, std::ios::trunc};
    if (!csv) {
        std::println("{}error:{} failed to open output CSV '{}'.", ansi_red, ansi_reset, output_csv_path.string());
        return 1;
    }
    csv << "stage,rank,scene_scale,status,error,final_step,best_validation_step,final_psnr,final_mse,best_psnr,best_mse,final_loss,occupancy_ratio,occupied_cells,average_samples,average_compacted,train_ms,validation_ms,steps_per_second\n";
    for (const CandidateResult& result : results) {
        std::string escaped_error = result.error;
        for (std::size_t i = 0uz; i < escaped_error.size(); ++i)
            if (escaped_error[i] == '"') escaped_error.insert(i++, 1uz, '"');
        csv << result.stage << "," << result.rank << "," << result.scene_scale << "," << (result.failed ? "failed" : "ok") << ",\"" << escaped_error << "\"," << result.final_step << "," << result.best_validation_step << "," << result.final_psnr << "," << result.final_mse << "," << result.best_psnr << "," << result.best_mse << "," << result.final_loss << "," << result.occupancy_ratio << "," << result.occupied_cells << "," << result.average_samples << "," << result.average_compacted << "," << result.train_ms << "," << result.validation_ms << "," << result.steps_per_second << "\n";
    }
    if (!csv) {
        std::println("{}error:{} failed to write output CSV '{}'.", ansi_red, ansi_reset, output_csv_path.string());
        return 1;
    }

    std::optional<std::size_t> best_index;
    for (std::size_t i = 0uz; i < results.size(); ++i) {
        if (results[i].stage != "refine" || results[i].failed) continue;
        if (!best_index.has_value() || results[i].final_psnr > results[*best_index].final_psnr + 0.05f || (std::abs(results[i].final_psnr - results[*best_index].final_psnr) <= 0.05f && results[i].scene_scale < results[*best_index].scene_scale)) best_index = i;
    }

    if (!best_index.has_value()) {
        std::println("{}{:<9}{} no successful refined scene_scale candidates; csv={}", ansi_red, "BEST", ansi_reset, output_csv_path.string());
        return 1;
    }

    const CandidateResult& best = results[*best_index];
    std::println("{}{:<9}{} scene_scale={:.7f} final_psnr={:.2f} final_mse={:.8f} best_psnr={:.2f} best_step={} occupancy={:.2f}% csv={}", ansi_bold, "BEST", ansi_reset, best.scene_scale, best.final_psnr, best.final_mse, best.best_psnr, best.best_validation_step, best.occupancy_ratio * 100.0f, output_csv_path.string());
    return 0;
}
