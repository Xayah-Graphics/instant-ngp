export module xcli;
import std;

namespace xcli {
    export struct OptionSpec final {
        std::string_view long_name;
        std::optional<char> short_name;
        std::string_view value_name;
        std::string_view description;
        std::optional<std::string_view> default_text;
        bool show_default = true;
        bool required     = false;
    };

    export struct PositionalSpec final {
        std::string_view name;
        std::string_view description;
        std::optional<std::string_view> default_text;
        bool show_default = true;
        bool required     = false;
    };

    export struct NumericRule final {
        std::optional<double> minimum;
        bool minimum_is_exclusive = false;
        std::optional<double> maximum;
        bool maximum_is_exclusive = false;
    };

    export enum class PathRequirement {
        none,
        existing_file,
        existing_directory,
        existing_parent_directory,
    };

    export struct PathRule final {
        bool reject_empty = true;
        PathRequirement requirement = PathRequirement::none;
    };

    export struct ParseResult final {
        bool help_requested = false;
    };

    export struct HelpStyle final {
        std::string_view reset;
        std::string_view dim;
        std::string_view bold;
        std::string_view heading;
        std::string_view executable;
        std::string_view option;
        std::string_view value;
        std::string_view default_label;
    };

    export struct CommandItem final {
        CommandItem(CommandItem&&) = default;
        CommandItem& operator=(CommandItem&&) = default;
        CommandItem(const CommandItem&) = delete;
        CommandItem& operator=(const CommandItem&) = delete;

    private:
        friend class Command;
        friend CommandItem option(const OptionSpec& spec, bool& target);
        friend CommandItem option(const OptionSpec& spec, std::string& target);
        friend CommandItem option(const OptionSpec& spec, std::vector<std::string>& target);
        friend CommandItem option(const OptionSpec& spec, std::filesystem::path& target, PathRule rule);
        friend CommandItem option(const OptionSpec& spec, std::optional<std::string>& target);
        friend CommandItem option(const OptionSpec& spec, std::optional<std::filesystem::path>& target, PathRule rule);
        friend CommandItem option(const OptionSpec& spec, std::int32_t& target, NumericRule rule);
        friend CommandItem option(const OptionSpec& spec, std::uint32_t& target, NumericRule rule);
        friend CommandItem option(const OptionSpec& spec, float& target, NumericRule rule);
        friend CommandItem positional(const PositionalSpec& spec, std::string& target);
        friend CommandItem positional(const PositionalSpec& spec, std::filesystem::path& target, PathRule rule);
        friend CommandItem example(std::string_view arguments);
        friend CommandItem validator(std::string_view name, std::move_only_function<std::expected<void, std::string>()> validator);

        struct StoredOptionSpec final {
            std::string long_name;
            std::optional<char> short_name;
            std::string value_name;
            std::string description;
            std::optional<std::string> default_text;
            bool show_default = true;
            bool required     = false;
        };

        struct StoredPositionalSpec final {
            std::string name;
            std::string description;
            std::optional<std::string> default_text;
            bool show_default = true;
            bool required     = false;
        };

        struct OptionAction final {
            StoredOptionSpec spec;
            bool requires_value = true;
            const void* target_address = nullptr;
            NumericRule numeric_rule;
            std::optional<PathRule> path_rule;
            std::variant<bool*, std::string*, std::vector<std::string>*, std::filesystem::path*, std::optional<std::string>*, std::optional<std::filesystem::path>*, std::int32_t*, std::uint32_t*, float*> target;
        };

        struct PositionalAction final {
            StoredPositionalSpec spec;
            const void* target_address = nullptr;
            std::optional<PathRule> path_rule;
            std::variant<std::string*, std::filesystem::path*> target;
        };

        struct ExampleAction final {
            std::string arguments;
        };

        struct ValidatorAction final {
            std::string name;
            std::move_only_function<std::expected<void, std::string>()> validator;
        };

        explicit CommandItem(OptionAction action);
        explicit CommandItem(PositionalAction action);
        explicit CommandItem(ExampleAction action);
        explicit CommandItem(ValidatorAction action);

        std::variant<OptionAction, PositionalAction, ExampleAction, ValidatorAction> action;
    };

    export class Command final {
    public:
        explicit Command(std::string_view description);
        Command(Command&&) = default;
        Command& operator=(Command&&) = default;
        Command(const Command&) = delete;
        Command& operator=(const Command&) = delete;

        std::expected<ParseResult, std::string> parse(std::span<const char* const> arguments);
        std::expected<void, std::string> validate();
        bool option_provided(std::string_view long_name) const;
        std::string help(std::span<const char* const> arguments) const;
        std::string help(std::span<const char* const> arguments, const HelpStyle& style) const;

    private:
        friend Command operator|(Command&& command, CommandItem&& item);
        friend Command& operator|(Command& command, CommandItem&& item);

        struct OptionBinding final {
            std::string long_name;
            std::optional<char> short_name;
            std::string value_name;
            std::string description;
            std::optional<std::string> default_text;
            bool show_default    = true;
            bool required        = false;
            bool seen            = false;
            bool requires_value  = true;
            const void* target_address = nullptr;
            NumericRule numeric_rule;
            std::optional<PathRule> path_rule;
            std::variant<bool*, std::string*, std::vector<std::string>*, std::filesystem::path*, std::optional<std::string>*, std::optional<std::filesystem::path>*, std::int32_t*, std::uint32_t*, float*> target;
        };

        struct PositionalBinding final {
            std::string name;
            std::string description;
            std::optional<std::string> default_text;
            bool show_default = true;
            bool required     = false;
            bool seen         = false;
            const void* target_address = nullptr;
            std::optional<PathRule> path_rule;
            std::variant<std::string*, std::filesystem::path*> target;
        };

        struct ValidatorBinding final {
            std::string name;
            std::move_only_function<std::expected<void, std::string>()> validator;
        };

        std::string description;
        std::vector<OptionBinding> options;
        std::vector<PositionalBinding> positionals;
        std::vector<std::string> examples;
        std::vector<ValidatorBinding> validators;

        Command& accept(CommandItem&& item);
        Command& bind_option(const OptionSpec& spec, bool& target);
        Command& bind_option(const OptionSpec& spec, std::string& target);
        Command& bind_option(const OptionSpec& spec, std::vector<std::string>& target);
        Command& bind_option(const OptionSpec& spec, std::filesystem::path& target, PathRule rule);
        Command& bind_option(const OptionSpec& spec, std::optional<std::string>& target);
        Command& bind_option(const OptionSpec& spec, std::optional<std::filesystem::path>& target, PathRule rule);
        Command& bind_option(const OptionSpec& spec, std::int32_t& target, NumericRule rule);
        Command& bind_option(const OptionSpec& spec, std::uint32_t& target, NumericRule rule);
        Command& bind_option(const OptionSpec& spec, float& target, NumericRule rule);
        Command& bind_positional(const PositionalSpec& spec, std::string& target);
        Command& bind_positional(const PositionalSpec& spec, std::filesystem::path& target, PathRule rule);
        Command& bind_example(std::string_view arguments);
        Command& bind_validator(std::string_view name, std::move_only_function<std::expected<void, std::string>()> validator);
    };

    export CommandItem option(const OptionSpec& spec, bool& target);
    export CommandItem option(const OptionSpec& spec, std::string& target);
    export CommandItem option(const OptionSpec& spec, std::vector<std::string>& target);
    export CommandItem option(const OptionSpec& spec, std::filesystem::path& target, PathRule rule = {});
    export CommandItem option(const OptionSpec& spec, std::optional<std::string>& target);
    export CommandItem option(const OptionSpec& spec, std::optional<std::filesystem::path>& target, PathRule rule = {});
    export CommandItem option(const OptionSpec& spec, std::int32_t& target, NumericRule rule = {});
    export CommandItem option(const OptionSpec& spec, std::uint32_t& target, NumericRule rule = {});
    export CommandItem option(const OptionSpec& spec, float& target, NumericRule rule = {});

    export CommandItem positional(const PositionalSpec& spec, std::string& target);
    export CommandItem positional(const PositionalSpec& spec, std::filesystem::path& target, PathRule rule = {});
    export CommandItem example(std::string_view arguments);
    export CommandItem validator(std::string_view name, std::move_only_function<std::expected<void, std::string>()> validator);

    export Command operator|(Command&& command, CommandItem&& item);
    export Command& operator|(Command& command, CommandItem&& item);
} // namespace xcli
