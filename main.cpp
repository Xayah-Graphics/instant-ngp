import std;
import ngp.dataset;
import ngp.train;

namespace {
    struct CliOptions final {
        std::filesystem::path dataset_path      = "../data/nerf-synthetic/lego";
        std::int32_t steps                      = 200000;
        std::int32_t chunk_steps                = 1000;
        std::uint32_t validation_interval_steps = 5000u;
        std::uint32_t early_stop_patience       = 5u;
        float early_stop_min_delta_mse          = 1e-6f;
    };

    std::expected<std::string_view, std::string> parse_option_value(const std::span<const char* const> arguments, std::size_t& index, const std::string_view option_name, const std::optional<std::string_view> inline_value) {
        if (inline_value.has_value()) {
            if (inline_value->empty()) return std::unexpected{std::format("{} requires a value.", option_name)};
            return *inline_value;
        }

        if (index + 1uz >= arguments.size()) return std::unexpected{std::format("{} requires a value.", option_name)};

        const std::string_view value{arguments[++index]};
        if (value.empty()) return std::unexpected{std::format("{} requires a value.", option_name)};
        return value;
    }

    std::expected<std::int32_t, std::string> parse_positive_int32(const std::string_view option_name, const std::string_view value) {
        std::int32_t parsed = 0;
        const auto result   = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed <= 0) return std::unexpected{std::format("{} must be a positive integer.", option_name)};
        return parsed;
    }

    std::expected<std::uint32_t, std::string> parse_positive_uint32(const std::string_view option_name, const std::string_view value) {
        std::uint32_t parsed = 0u;
        const auto result    = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || parsed == 0u) return std::unexpected{std::format("{} must be a positive integer.", option_name)};
        return parsed;
    }

    std::expected<float, std::string> parse_nonnegative_float(const std::string_view option_name, const std::string_view value) {
        float parsed      = 0.0f;
        const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(parsed) || parsed < 0.0f) return std::unexpected{std::format("{} must be a finite non-negative number.", option_name)};
        return parsed;
    }

    std::expected<std::optional<CliOptions>, std::string> parse_cli(const std::span<const char* const> arguments) {
        CliOptions cli            = {};
        bool dataset_path_was_set = false;

        for (std::size_t i = 1uz; i < arguments.size(); ++i) {
            const std::string_view argument{arguments[i]};
            const std::size_t assignment_position = argument.find('=');
            const std::string_view option_name    = assignment_position == std::string_view::npos ? argument : argument.substr(0uz, assignment_position);
            std::optional<std::string_view> inline_value;
            if (assignment_position != std::string_view::npos) inline_value = argument.substr(assignment_position + 1uz);

            if (option_name == "-h" || option_name == "--help") {
                if (inline_value.has_value()) return std::unexpected{std::format("{} does not accept a value.", option_name)};
                return std::nullopt;
            }

            if (option_name == "--dataset") {
                if (dataset_path_was_set) return std::unexpected{"dataset path was provided more than once."};
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                cli.dataset_path     = std::filesystem::path{*value};
                dataset_path_was_set = true;
                continue;
            }

            if (option_name == "--steps") {
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                const auto parsed = parse_positive_int32(option_name, *value);
                if (!parsed) return std::unexpected{parsed.error()};
                cli.steps = *parsed;
                continue;
            }

            if (option_name == "--chunk-steps") {
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                const auto parsed = parse_positive_int32(option_name, *value);
                if (!parsed) return std::unexpected{parsed.error()};
                cli.chunk_steps = *parsed;
                continue;
            }

            if (option_name == "--validation-interval") {
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                const auto parsed = parse_positive_uint32(option_name, *value);
                if (!parsed) return std::unexpected{parsed.error()};
                cli.validation_interval_steps = *parsed;
                continue;
            }

            if (option_name == "--early-stop-patience") {
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                const auto parsed = parse_positive_uint32(option_name, *value);
                if (!parsed) return std::unexpected{parsed.error()};
                cli.early_stop_patience = *parsed;
                continue;
            }

            if (option_name == "--early-stop-min-delta") {
                const auto value = parse_option_value(arguments, i, option_name, inline_value);
                if (!value) return std::unexpected{value.error()};
                const auto parsed = parse_nonnegative_float(option_name, *value);
                if (!parsed) return std::unexpected{parsed.error()};
                cli.early_stop_min_delta_mse = *parsed;
                continue;
            }

            if (!argument.starts_with("-") && !dataset_path_was_set) {
                cli.dataset_path     = std::filesystem::path{argument};
                dataset_path_was_set = true;
                continue;
            }

            return std::unexpected{std::format("unknown argument '{}'.", argument)};
        }

        return cli;
    }
} // namespace

int main(const int argc, const char* const* const argv) {
    const std::span<const char* const> arguments{argv, static_cast<std::size_t>(argc)};
    const std::string executable_name = !arguments.empty() && arguments.front() != nullptr ? std::filesystem::path{arguments.front()}.filename().string() : "instant-ngp-app";
    const std::string usage           = std::format(
        R"(Usage:
  {} [dataset-path] [options]

Options:
  --dataset <path>                  NeRF synthetic dataset root
                                    default: ../data/nerf-synthetic/lego
  --steps <count>                   total training steps
                                    default: 200000
  --chunk-steps <count>             training steps per progress log
                                    default: 1000
  --validation-interval <count>     full validation interval in steps
                                    default: 5000
  --early-stop-patience <count>     validation checks without improvement before stopping
                                    default: 5
  --early-stop-min-delta <mse>      minimum validation MSE improvement
                                    default: 1e-6
  -h, --help                        print this help

Examples:
  {} ../data/nerf-synthetic/lego --steps 30000
  {} --dataset=../data/nerf-synthetic/lego --validation-interval=5000
)",
        executable_name, executable_name, executable_name);

    const auto cli_result = parse_cli(arguments);
    if (!cli_result) {
        std::println("error: {}\n{}", cli_result.error(), usage);
        return 2;
    }
    if (!cli_result->has_value()) {
        std::println("{}", usage);
        return 0;
    }
    const CliOptions cli = **cli_result;

    if (cli.dataset_path.empty()) {
        std::println("error: dataset path must not be empty.\n{}", usage);
        return 2;
    }
    if (!std::filesystem::is_directory(cli.dataset_path)) {
        std::println("error: dataset path '{}' is not a directory.", cli.dataset_path.string());
        return 2;
    }

    constexpr std::string_view reset   = "\x1b[0m";
    constexpr std::string_view dim     = "\x1b[2m";
    constexpr std::string_view bold    = "\x1b[1m";
    constexpr std::string_view cyan    = "\x1b[36m";
    constexpr std::string_view green   = "\x1b[32m";
    constexpr std::string_view magenta = "\x1b[35m";
    constexpr std::string_view yellow  = "\x1b[33m";
    constexpr std::string_view red     = "\x1b[31m";
    const auto config_timestamp        = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} dataset={} steps={} chunk={} validation_interval={} patience={} min_delta_mse={}", dim, config_timestamp, reset, cyan, "CONFIG", reset, cli.dataset_path.string(), cli.steps, cli.chunk_steps, cli.validation_interval_steps, cli.early_stop_patience, cli.early_stop_min_delta_mse);

    const auto load_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    std::println("{}[{:%F %T}]{} {}{:<7}{} loading dataset", dim, load_timestamp, reset, cyan, "INFO", reset);

    const auto result = ngp::dataset::load_nerf_synthetic(cli.dataset_path)
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
                                float train_ms                                      = 0.0f;
                                float best_validation_mse                           = std::numeric_limits<float>::infinity();
                                float best_validation_psnr                          = 0.0f;
                                std::uint32_t final_step                            = 0u;
                                std::uint32_t best_validation_step                  = 0u;
                                std::uint32_t validation_checks_without_improvement = 0u;
                                bool stopped_early                                  = false;
                                std::uint32_t next_validation_step                  = cli.validation_interval_steps;

                                for (std::int32_t trained_steps = 0; trained_steps < cli.steps;) {
                                    const std::int32_t requested_steps = std::min(cli.chunk_steps, cli.steps - trained_steps);
                                    const auto stats                   = ngp->train(requested_steps);
                                    if (!stats) return std::unexpected{stats.error()};

                                    if (trained_steps == 0) first_loss = stats->loss;
                                    last_loss = stats->loss;
                                    train_ms += stats->elapsed_ms;
                                    final_step = stats->step;
                                    trained_steps += requested_steps;
                                    const auto train_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                                    std::println("{}[{:%F %T}]{} {}{:<7}{} step={:>6}/{} loss={:>10.6f} chunk={:>8.3f}ms grid={:>7.3f}ms rate={:>7.2f} step/s rays={:>6} samples={:>7}/{:<7} occupied={:>7} occupancy={:>6.2f}%", dim, train_timestamp, reset, green, "TRAIN", reset, stats->step, cli.steps, stats->loss, stats->elapsed_ms, stats->density_grid_update_ms, static_cast<float>(requested_steps) * 1000.0f / stats->elapsed_ms, stats->rays_per_batch, stats->measured_sample_count, stats->measured_sample_count_before_compaction, stats->density_grid_occupied_cells, stats->density_grid_occupancy_ratio * 100.0f);

                                    if (stats->step >= next_validation_step || stats->step >= static_cast<std::uint32_t>(cli.steps)) {
                                        const auto validation = ngp->validate();
                                        if (!validation) return std::unexpected{validation.error()};

                                        if (validation->mse < best_validation_mse - cli.early_stop_min_delta_mse) {
                                            best_validation_mse                   = validation->mse;
                                            best_validation_psnr                  = validation->psnr;
                                            best_validation_step                  = validation->step;
                                            validation_checks_without_improvement = 0u;
                                        } else {
                                            ++validation_checks_without_improvement;
                                        }

                                        const auto validation_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                                        std::println("{}[{:%F %T}]{} {}{:<7}{} step={:>6} images={:>3} pixels={} mse={:.8f} psnr={:>5.2f} val={:>8.3f}ms best={:.8f}@{} patience={}/{}", dim, validation_timestamp, reset, magenta, "VALID", reset, validation->step, validation->image_count, validation->pixel_count, validation->mse, validation->psnr, validation->elapsed_ms, best_validation_mse, best_validation_step, validation_checks_without_improvement, cli.early_stop_patience);
                                        if (validation_checks_without_improvement >= cli.early_stop_patience) {
                                            stopped_early = true;
                                            break;
                                        }
                                        while (next_validation_step <= stats->step) next_validation_step += cli.validation_interval_steps;
                                    }
                                }

                                const auto summary_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
                                std::println("{}[{:%F %T}]{} {}{:<7}{} steps={} stopped_early={} first_loss={:.6f} last_loss={:.6f} train={:.3f}s avg={:.2f} step/s best_validation={:.8f}@{} psnr={:.2f}", dim, summary_timestamp, reset, stopped_early ? yellow : bold, "SUMMARY", reset, final_step, stopped_early, first_loss, last_loss, train_ms * 0.001f, static_cast<float>(final_step) * 1000.0f / train_ms, best_validation_mse, best_validation_step, best_validation_psnr);
                                return {};
                            });

    const auto finish_timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
    if (result.has_value()) {
        std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=succeeded", dim, finish_timestamp, reset, bold, "DONE", reset);
    } else {
        std::println("{}[{:%F %T}]{} {}{:<7}{} pipeline=failed error=\"{}\"", dim, finish_timestamp, reset, red, "ERROR", reset, result.error());
    }
    return result.has_value() ? 0 : 1;
}
