export module xlog;
import std;

namespace xlog {
    export namespace ansi {
        inline constexpr std::string_view reset             = "\x1b[0m";
        inline constexpr std::string_view dim               = "\x1b[2m";
        inline constexpr std::string_view bold              = "\x1b[1m";
        inline constexpr std::string_view cyan              = "\x1b[36m";
        inline constexpr std::string_view green             = "\x1b[32m";
        inline constexpr std::string_view yellow            = "\x1b[33m";
        inline constexpr std::string_view red               = "\x1b[31m";
        inline constexpr std::string_view validation_badge  = "\x1b[1;37;45m";
        inline constexpr std::string_view validation_metric = "\x1b[1;95m";
        inline constexpr std::string_view validation_best   = "\x1b[1;33m";
        inline constexpr std::string_view test_badge        = "\x1b[1;37;44m";
        inline constexpr std::string_view test_metric       = "\x1b[1;96m";
    } // namespace ansi

    export enum class Level {
        trace,
        debug,
        info,
        warn,
        error,
        fatal,
    };

    export enum class FileMode {
        append,
        truncate,
    };

    export struct ConsoleSinkConfig final {
        bool enabled          = true;
        bool color_enabled    = true;
        bool stderr_for_error = true;
    };

    export struct FileSinkConfig final {
        std::filesystem::path path;
        FileMode mode = FileMode::append;
    };

    export struct LoggerConfig final {
        Level minimum_level = Level::trace;
        std::size_t tag_width = 7uz;
        ConsoleSinkConfig console;
        std::optional<FileSinkConfig> file;
    };

    export struct TagStyle final {
        std::string_view begin;
        std::string_view end;
    };

    export class Logger final {
    public:
        ~Logger() noexcept;
        Logger(const Logger&)            = delete;
        Logger& operator=(const Logger&) = delete;
        Logger(Logger&&) noexcept;
        Logger& operator=(Logger&&) noexcept;

        static std::expected<Logger, std::string> create(const LoggerConfig& config);
        std::expected<void, std::string> add_file_sink(const FileSinkConfig& config);
        void set_minimum_level(Level level);
        void set_tag_style(std::string_view tag, TagStyle style);
        void write(Level level, std::string_view tag, std::string_view message);
        void flush();

        template <typename... Args>
        void trace(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::trace, tag, std::format(format, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void debug(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::debug, tag, std::format(format, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void info(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::info, tag, std::format(format, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void warn(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::warn, tag, std::format(format, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void error(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::error, tag, std::format(format, std::forward<Args>(args)...));
        }

        template <typename... Args>
        void fatal(const std::string_view tag, const std::format_string<Args...> format, Args&&... args) {
            this->write(Level::fatal, tag, std::format(format, std::forward<Args>(args)...));
        }

    private:
        struct State;

        explicit Logger(const LoggerConfig& config);

        std::unique_ptr<State> state;
    };
} // namespace xlog
