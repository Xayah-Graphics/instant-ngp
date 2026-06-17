module xlog;
import std;

namespace xlog {
    struct Logger::State final {
        struct StoredTagStyle final {
            std::string tag;
            std::string begin;
            std::string end;
        };

        LoggerConfig config;
        std::mutex mutex;
        std::ofstream file;
        bool file_enabled = false;
        std::vector<StoredTagStyle> tag_styles;
    };

    Logger::Logger(const LoggerConfig& config) : state{std::make_unique<State>()} {
        this->state->config = config;
    }

    Logger::~Logger() noexcept = default;
    Logger::Logger(Logger&&) noexcept = default;
    Logger& Logger::operator=(Logger&&) noexcept = default;

    std::expected<Logger, std::string> Logger::create(const LoggerConfig& config) {
        if (config.tag_width == 0uz) return std::unexpected{"log tag width must be positive."};
        Logger logger{config};
        if (config.file.has_value()) {
            const auto file_sink = logger.add_file_sink(*config.file);
            if (!file_sink) return std::unexpected{file_sink.error()};
        }
        return logger;
    }

    std::expected<void, std::string> Logger::add_file_sink(const FileSinkConfig& config) {
        if (this->state == nullptr) return std::unexpected{"logger is moved-from."};
        if (config.path.empty()) return std::unexpected{"log file path must not be empty."};
        if (!config.path.parent_path().empty() && !std::filesystem::is_directory(config.path.parent_path())) return std::unexpected{std::format("log file parent directory '{}' does not exist.", config.path.parent_path().string())};

        const std::scoped_lock lock{this->state->mutex};
        if (this->state->file.is_open()) this->state->file.close();

        const std::ios::openmode mode = config.mode == FileMode::append ? std::ios::out | std::ios::app : std::ios::out | std::ios::trunc;
        this->state->file.open(config.path, mode);
        if (!this->state->file) return std::unexpected{std::format("failed to open log file '{}'.", config.path.string())};
        this->state->file_enabled = true;
        this->state->config.file = config;
        return {};
    }

    void Logger::set_minimum_level(const Level level) {
        if (this->state == nullptr) throw std::runtime_error{"logger is moved-from."};
        const std::scoped_lock lock{this->state->mutex};
        this->state->config.minimum_level = level;
    }

    void Logger::set_tag_style(const std::string_view tag, const TagStyle style) {
        if (this->state == nullptr) throw std::runtime_error{"logger is moved-from."};
        if (tag.empty()) throw std::runtime_error{"log tag must not be empty."};
        const std::scoped_lock lock{this->state->mutex};
        for (State::StoredTagStyle& stored_style : this->state->tag_styles) {
            if (stored_style.tag == tag) {
                stored_style.begin = style.begin;
                stored_style.end = style.end;
                return;
            }
        }
        this->state->tag_styles.push_back({.tag = std::string{tag}, .begin = std::string{style.begin}, .end = std::string{style.end}});
    }

    void Logger::write(const Level level, const std::string_view tag, const std::string_view message) {
        if (this->state == nullptr) throw std::runtime_error{"logger is moved-from."};
        if (tag.empty()) throw std::runtime_error{"log tag must not be empty."};
        const std::scoped_lock lock{this->state->mutex};
        if (static_cast<int>(level) < static_cast<int>(this->state->config.minimum_level)) return;

        const auto timestamp = std::chrono::floor<std::chrono::seconds>(std::chrono::system_clock::now());
        const std::string timestamp_text = std::format("[{:%F %T}]", timestamp);
        const std::string tag_text{tag};
        const std::string plain_line = std::format("{} {:<{}} {}", timestamp_text, tag_text, this->state->config.tag_width, message);

        if (this->state->config.console.enabled) {
            std::string style_begin;
            std::string style_end;
            if (this->state->config.console.color_enabled) {
                for (const State::StoredTagStyle& stored_style : this->state->tag_styles) {
                    if (stored_style.tag == tag_text) {
                        style_begin = stored_style.begin;
                        style_end = stored_style.end;
                        break;
                    }
                }
                if (style_begin.empty() && level == Level::warn) {
                    style_begin = ansi::yellow;
                    style_end = ansi::reset;
                }
                if (style_begin.empty() && (level == Level::error || level == Level::fatal)) {
                    style_begin = ansi::red;
                    style_end = ansi::reset;
                }
            }

            const std::string console_line = this->state->config.console.color_enabled ? std::format("{}{}{} {}{:<{}}{} {}", ansi::dim, timestamp_text, ansi::reset, style_begin, tag_text, this->state->config.tag_width, style_end, message) : plain_line;
            if (this->state->config.console.stderr_for_error && (level == Level::error || level == Level::fatal)) {
                std::cerr << console_line << '\n' << std::flush;
            } else {
                std::cout << console_line << '\n' << std::flush;
            }
        }

        if (this->state->file_enabled) {
            this->state->file << plain_line << '\n' << std::flush;
            if (!this->state->file) throw std::runtime_error{"failed to write log file."};
        }
    }

    void Logger::flush() {
        if (this->state == nullptr) throw std::runtime_error{"logger is moved-from."};
        const std::scoped_lock lock{this->state->mutex};
        if (this->state->config.console.enabled) {
            std::cout << std::flush;
            std::cerr << std::flush;
        }
        if (this->state->file_enabled) this->state->file.flush();
    }
} // namespace xlog
