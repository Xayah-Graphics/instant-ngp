module xcli;
import std;

namespace xcli {
    Command::Command(const std::string_view description) : description{description} {}

    CommandItem::CommandItem(OptionAction value) : action{std::move(value)} {}
    CommandItem::CommandItem(PositionalAction value) : action{std::move(value)} {}
    CommandItem::CommandItem(ExampleAction value) : action{std::move(value)} {}
    CommandItem::CommandItem(ValidatorAction value) : action{std::move(value)} {}

    CommandItem option(const OptionSpec& spec, bool& target) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .requires_value = false,
            .target_address = std::addressof(target),
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::string& target) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::vector<std::string>& target) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::filesystem::path& target, const PathRule rule) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .path_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::optional<std::string>& target) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::optional<std::filesystem::path>& target, const PathRule rule) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .path_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::int32_t& target, const NumericRule rule) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .numeric_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, std::uint32_t& target, const NumericRule rule) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .numeric_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem option(const OptionSpec& spec, float& target, const NumericRule rule) {
        return CommandItem{CommandItem::OptionAction{
            .spec = {
                .long_name = std::string{spec.long_name},
                .short_name = spec.short_name,
                .value_name = std::string{spec.value_name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .numeric_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem positional(const PositionalSpec& spec, std::string& target) {
        return CommandItem{CommandItem::PositionalAction{
            .spec = {
                .name = std::string{spec.name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .target = std::addressof(target),
        }};
    }

    CommandItem positional(const PositionalSpec& spec, std::filesystem::path& target, const PathRule rule) {
        return CommandItem{CommandItem::PositionalAction{
            .spec = {
                .name = std::string{spec.name},
                .description = std::string{spec.description},
                .default_text = spec.default_text.has_value() ? std::optional<std::string>{std::string{*spec.default_text}} : std::nullopt,
                .show_default = spec.show_default,
                .required = spec.required,
            },
            .target_address = std::addressof(target),
            .path_rule = rule,
            .target = std::addressof(target),
        }};
    }

    CommandItem example(const std::string_view arguments) {
        return CommandItem{CommandItem::ExampleAction{.arguments = std::string{arguments}}};
    }

    CommandItem validator(const std::string_view name, std::move_only_function<std::expected<void, std::string>()> validator) {
        return CommandItem{CommandItem::ValidatorAction{.name = std::string{name}, .validator = std::move(validator)}};
    }

    Command operator|(Command&& command, CommandItem&& item) {
        command.accept(std::move(item));
        return std::move(command);
    }

    Command& operator|(Command& command, CommandItem&& item) {
        command.accept(std::move(item));
        return command;
    }

    Command& Command::accept(CommandItem&& item) {
        if (CommandItem::OptionAction* action = std::get_if<CommandItem::OptionAction>(std::addressof(item.action))) {
            std::optional<std::string_view> default_text;
            if (action->spec.default_text.has_value()) default_text = *action->spec.default_text;
            const OptionSpec spec{
                .long_name = action->spec.long_name,
                .short_name = action->spec.short_name,
                .value_name = action->spec.value_name,
                .description = action->spec.description,
                .default_text = default_text,
                .show_default = action->spec.show_default,
                .required = action->spec.required,
            };

            if (bool** target = std::get_if<bool*>(std::addressof(action->target))) return this->bind_option(spec, **target);
            if (std::string** target = std::get_if<std::string*>(std::addressof(action->target))) return this->bind_option(spec, **target);
            if (std::vector<std::string>** target = std::get_if<std::vector<std::string>*>(std::addressof(action->target))) return this->bind_option(spec, **target);
            if (std::filesystem::path** target = std::get_if<std::filesystem::path*>(std::addressof(action->target))) return this->bind_option(spec, **target, action->path_rule.value_or(PathRule{}));
            if (std::optional<std::string>** target = std::get_if<std::optional<std::string>*>(std::addressof(action->target))) return this->bind_option(spec, **target);
            if (std::optional<std::filesystem::path>** target = std::get_if<std::optional<std::filesystem::path>*>(std::addressof(action->target))) return this->bind_option(spec, **target, action->path_rule.value_or(PathRule{}));
            if (std::int32_t** target = std::get_if<std::int32_t*>(std::addressof(action->target))) return this->bind_option(spec, **target, action->numeric_rule);
            if (std::uint32_t** target = std::get_if<std::uint32_t*>(std::addressof(action->target))) return this->bind_option(spec, **target, action->numeric_rule);
            if (float** target = std::get_if<float*>(std::addressof(action->target))) return this->bind_option(spec, **target, action->numeric_rule);
            std::unreachable();
        }

        if (CommandItem::PositionalAction* action = std::get_if<CommandItem::PositionalAction>(std::addressof(item.action))) {
            std::optional<std::string_view> default_text;
            if (action->spec.default_text.has_value()) default_text = *action->spec.default_text;
            const PositionalSpec spec{
                .name = action->spec.name,
                .description = action->spec.description,
                .default_text = default_text,
                .show_default = action->spec.show_default,
                .required = action->spec.required,
            };

            if (std::string** target = std::get_if<std::string*>(std::addressof(action->target))) return this->bind_positional(spec, **target);
            if (std::filesystem::path** target = std::get_if<std::filesystem::path*>(std::addressof(action->target))) return this->bind_positional(spec, **target, action->path_rule.value_or(PathRule{}));
            std::unreachable();
        }

        if (CommandItem::ExampleAction* action = std::get_if<CommandItem::ExampleAction>(std::addressof(item.action))) return this->bind_example(action->arguments);
        if (CommandItem::ValidatorAction* action = std::get_if<CommandItem::ValidatorAction>(std::addressof(item.action))) return this->bind_validator(action->name, std::move(action->validator));
        std::unreachable();
    }

    Command& Command::bind_option(const OptionSpec& spec, bool& target) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.requires_value = false;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = target ? "true" : "false";
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (!binding.value_name.empty()) throw std::runtime_error{std::format("boolean option '--{}' must not declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::string& target) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = target;
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::vector<std::string>& target) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default && !target.empty()) {
            std::string default_text;
            for (const std::string& value : target) {
                if (!default_text.empty()) default_text += ",";
                default_text += value;
            }
            binding.default_text = std::move(default_text);
        }
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::filesystem::path& target, const PathRule rule) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.path_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = target.string();
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::optional<std::string>& target) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default && target.has_value()) binding.default_text = *target;
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::optional<std::filesystem::path>& target, const PathRule rule) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.path_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default && target.has_value()) binding.default_text = target->string();
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::int32_t& target, const NumericRule rule) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.numeric_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = std::format("{}", target);
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, std::uint32_t& target, const NumericRule rule) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.numeric_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = std::format("{}", target);
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_option(const OptionSpec& spec, float& target, const NumericRule rule) {
        OptionBinding binding = {};
        binding.long_name = spec.long_name;
        binding.short_name = spec.short_name;
        binding.value_name = spec.value_name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.numeric_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = std::format("{:.6g}", static_cast<double>(target));
        if (binding.long_name.empty()) throw std::runtime_error{"option long_name must not be empty."};
        if (binding.long_name.starts_with("-")) throw std::runtime_error{std::format("option '{}' must be registered without '-' prefixes.", binding.long_name)};
        if (binding.long_name == "help") throw std::runtime_error{"option '--help' is reserved."};
        if (binding.value_name.empty()) throw std::runtime_error{std::format("option '--{}' must declare value_name.", binding.long_name)};
        for (const char character : binding.long_name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("option '{}' contains invalid character '{}'.", binding.long_name, character)};
        }
        if (binding.short_name.has_value() && (*binding.short_name == 'h' || *binding.short_name == '-' || *binding.short_name == '=' || *binding.short_name == '\0')) throw std::runtime_error{std::format("option '--{}' uses invalid or reserved short name.", binding.long_name)};
        for (const OptionBinding& option : this->options) {
            if (option.long_name == binding.long_name) throw std::runtime_error{std::format("option '--{}' was registered more than once.", binding.long_name)};
            if (option.short_name.has_value() && binding.short_name.has_value() && *option.short_name == *binding.short_name) throw std::runtime_error{std::format("short option '-{}' was registered more than once.", *binding.short_name)};
        }
        this->options.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_positional(const PositionalSpec& spec, std::string& target) {
        PositionalBinding binding = {};
        binding.name = spec.name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = target;
        if (binding.name.empty()) throw std::runtime_error{"positional name must not be empty."};
        if (binding.name.starts_with("-")) throw std::runtime_error{std::format("positional '{}' must not start with '-'.", binding.name)};
        for (const char character : binding.name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("positional '{}' contains invalid character '{}'.", binding.name, character)};
        }
        for (const PositionalBinding& positional : this->positionals) {
            if (positional.name == binding.name) throw std::runtime_error{std::format("positional '{}' was registered more than once.", binding.name)};
            if (positional.target_address == binding.target_address) throw std::runtime_error{std::format("positional '{}' reuses an existing positional target.", binding.name)};
        }
        this->positionals.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_positional(const PositionalSpec& spec, std::filesystem::path& target, const PathRule rule) {
        PositionalBinding binding = {};
        binding.name = spec.name;
        binding.description = spec.description;
        binding.show_default = spec.show_default;
        binding.required = spec.required;
        binding.path_rule = rule;
        binding.target_address = std::addressof(target);
        binding.target = std::addressof(target);
        if (spec.default_text.has_value()) binding.default_text = std::string{*spec.default_text};
        else if (spec.show_default) binding.default_text = target.string();
        if (binding.name.empty()) throw std::runtime_error{"positional name must not be empty."};
        if (binding.name.starts_with("-")) throw std::runtime_error{std::format("positional '{}' must not start with '-'.", binding.name)};
        for (const char character : binding.name) {
            const bool valid_character = (character >= 'a' && character <= 'z') || (character >= 'A' && character <= 'Z') || (character >= '0' && character <= '9') || character == '-';
            if (!valid_character) throw std::runtime_error{std::format("positional '{}' contains invalid character '{}'.", binding.name, character)};
        }
        for (const PositionalBinding& positional : this->positionals) {
            if (positional.name == binding.name) throw std::runtime_error{std::format("positional '{}' was registered more than once.", binding.name)};
            if (positional.target_address == binding.target_address) throw std::runtime_error{std::format("positional '{}' reuses an existing positional target.", binding.name)};
        }
        this->positionals.push_back(std::move(binding));
        return *this;
    }

    Command& Command::bind_example(const std::string_view arguments) {
        if (arguments.empty()) throw std::runtime_error{"example arguments must not be empty."};
        this->examples.emplace_back(arguments);
        return *this;
    }

    Command& Command::bind_validator(const std::string_view name, std::move_only_function<std::expected<void, std::string>()> validator) {
        if (name.empty()) throw std::runtime_error{"validator name must not be empty."};
        if (!validator) throw std::runtime_error{std::format("validator '{}' must not be empty.", name)};
        for (const ValidatorBinding& existing_validator : this->validators) if (existing_validator.name == name) throw std::runtime_error{std::format("validator '{}' was registered more than once.", name)};
        this->validators.push_back({.name = std::string{name}, .validator = std::move(validator)});
        return *this;
    }

    std::expected<ParseResult, std::string> Command::parse(const std::span<const char* const> arguments) {
        if (arguments.empty()) return std::unexpected{"argv must contain executable path."};
        for (OptionBinding& option : this->options) option.seen = false;
        for (PositionalBinding& positional : this->positionals) positional.seen = false;

        std::size_t next_positional = 0uz;
        for (std::size_t index = 1uz; index < arguments.size(); ++index) {
            if (arguments[index] == nullptr) return std::unexpected{std::format("argument {} is null.", index)};
            const std::string_view argument{arguments[index]};
            if (argument.empty()) return std::unexpected{std::format("argument {} must not be empty.", index)};
            if (argument == "-h" || argument == "--help") return ParseResult{.help_requested = true};
            if (argument.starts_with("--help=") || argument.starts_with("-h=")) return std::unexpected{"--help does not accept a value."};

            if (argument.starts_with("--")) {
                const std::size_t assignment_position = argument.find('=');
                const std::string_view option_name = assignment_position == std::string_view::npos ? argument.substr(2uz) : argument.substr(2uz, assignment_position - 2uz);
                std::optional<std::string_view> inline_value;
                if (assignment_position != std::string_view::npos) inline_value = argument.substr(assignment_position + 1uz);
                if (option_name.empty()) return std::unexpected{std::format("unknown argument '{}'.", argument)};

                OptionBinding* matched_option = nullptr;
                for (OptionBinding& option : this->options) {
                    if (option.long_name == option_name) {
                        matched_option = std::addressof(option);
                        break;
                    }
                }
                if (matched_option == nullptr) return std::unexpected{std::format("unknown option '--{}'.", option_name)};
                const std::string display_name = std::format("--{}", matched_option->long_name);

                if (!matched_option->requires_value) {
                    if (inline_value.has_value()) return std::unexpected{std::format("{} does not accept a value.", display_name)};
                    for (const OptionBinding& option : this->options) if (option.target_address == matched_option->target_address && option.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    for (const PositionalBinding& positional : this->positionals) if (positional.target_address == matched_option->target_address && positional.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    **std::get_if<bool*>(std::addressof(matched_option->target)) = true;
                    matched_option->seen = true;
                    continue;
                }

                std::string_view value;
                if (inline_value.has_value()) {
                    if (inline_value->empty()) return std::unexpected{std::format("{} requires a value.", display_name)};
                    value = *inline_value;
                } else {
                    if (index + 1uz >= arguments.size()) return std::unexpected{std::format("{} requires a value.", display_name)};
                    if (arguments[index + 1uz] == nullptr) return std::unexpected{std::format("{} requires a value.", display_name)};
                    value = std::string_view{arguments[++index]};
                    if (value.empty()) return std::unexpected{std::format("{} requires a value.", display_name)};
                }

                const bool option_appends_values = std::holds_alternative<std::vector<std::string>*>(matched_option->target);
                if (!option_appends_values) {
                    for (const OptionBinding& option : this->options) if (option.target_address == matched_option->target_address && option.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    for (const PositionalBinding& positional : this->positionals) if (positional.target_address == matched_option->target_address && positional.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                }
                if (std::string** target = std::get_if<std::string*>(std::addressof(matched_option->target))) {
                    **target = std::string{value};
                } else if (std::vector<std::string>** target = std::get_if<std::vector<std::string>*>(std::addressof(matched_option->target))) {
                    (**target).push_back(std::string{value});
                } else if (std::filesystem::path** target = std::get_if<std::filesystem::path*>(std::addressof(matched_option->target))) {
                    **target = std::filesystem::path{value};
                } else if (std::optional<std::string>** target = std::get_if<std::optional<std::string>*>(std::addressof(matched_option->target))) {
                    **target = std::string{value};
                } else if (std::optional<std::filesystem::path>** target = std::get_if<std::optional<std::filesystem::path>*>(std::addressof(matched_option->target))) {
                    **target = std::filesystem::path{value};
                } else if (std::int32_t** target = std::get_if<std::int32_t*>(std::addressof(matched_option->target))) {
                    std::int32_t parsed = 0;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::unexpected{std::format("{} must be a signed 32-bit integer.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                } else if (std::uint32_t** target = std::get_if<std::uint32_t*>(std::addressof(matched_option->target))) {
                    std::uint32_t parsed = 0u;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::unexpected{std::format("{} must be an unsigned 32-bit integer.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                } else if (float** target = std::get_if<float*>(std::addressof(matched_option->target))) {
                    float parsed = 0.0f;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(parsed)) return std::unexpected{std::format("{} must be a finite number.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                }
                matched_option->seen = true;
                continue;
            }

            if (argument.starts_with("-")) {
                if (argument.size() != 2uz) return std::unexpected{std::format("unknown argument '{}'.", argument)};
                OptionBinding* matched_option = nullptr;
                for (OptionBinding& option : this->options) {
                    if (option.short_name.has_value() && *option.short_name == argument[1uz]) {
                        matched_option = std::addressof(option);
                        break;
                    }
                }
                if (matched_option == nullptr) return std::unexpected{std::format("unknown option '-{}'.", argument[1uz])};
                const std::string display_name = std::format("-{}", *matched_option->short_name);

                if (!matched_option->requires_value) {
                    for (const OptionBinding& option : this->options) if (option.target_address == matched_option->target_address && option.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    for (const PositionalBinding& positional : this->positionals) if (positional.target_address == matched_option->target_address && positional.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    **std::get_if<bool*>(std::addressof(matched_option->target)) = true;
                    matched_option->seen = true;
                    continue;
                }

                if (index + 1uz >= arguments.size()) return std::unexpected{std::format("{} requires a value.", display_name)};
                if (arguments[index + 1uz] == nullptr) return std::unexpected{std::format("{} requires a value.", display_name)};
                const std::string_view value{arguments[++index]};
                if (value.empty()) return std::unexpected{std::format("{} requires a value.", display_name)};
                const bool option_appends_values = std::holds_alternative<std::vector<std::string>*>(matched_option->target);
                if (!option_appends_values) {
                    for (const OptionBinding& option : this->options) if (option.target_address == matched_option->target_address && option.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                    for (const PositionalBinding& positional : this->positionals) if (positional.target_address == matched_option->target_address && positional.seen) return std::unexpected{std::format("{} was provided more than once.", display_name)};
                }
                if (std::string** target = std::get_if<std::string*>(std::addressof(matched_option->target))) {
                    **target = std::string{value};
                } else if (std::vector<std::string>** target = std::get_if<std::vector<std::string>*>(std::addressof(matched_option->target))) {
                    (**target).push_back(std::string{value});
                } else if (std::filesystem::path** target = std::get_if<std::filesystem::path*>(std::addressof(matched_option->target))) {
                    **target = std::filesystem::path{value};
                } else if (std::optional<std::string>** target = std::get_if<std::optional<std::string>*>(std::addressof(matched_option->target))) {
                    **target = std::string{value};
                } else if (std::optional<std::filesystem::path>** target = std::get_if<std::optional<std::filesystem::path>*>(std::addressof(matched_option->target))) {
                    **target = std::filesystem::path{value};
                } else if (std::int32_t** target = std::get_if<std::int32_t*>(std::addressof(matched_option->target))) {
                    std::int32_t parsed = 0;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::unexpected{std::format("{} must be a signed 32-bit integer.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                } else if (std::uint32_t** target = std::get_if<std::uint32_t*>(std::addressof(matched_option->target))) {
                    std::uint32_t parsed = 0u;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size()) return std::unexpected{std::format("{} must be an unsigned 32-bit integer.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                } else if (float** target = std::get_if<float*>(std::addressof(matched_option->target))) {
                    float parsed = 0.0f;
                    const auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
                    if (result.ec != std::errc{} || result.ptr != value.data() + value.size() || !std::isfinite(parsed)) return std::unexpected{std::format("{} must be a finite number.", display_name)};
                    const double numeric_value = static_cast<double>(parsed);
                    if (matched_option->numeric_rule.minimum.has_value() && (matched_option->numeric_rule.minimum_is_exclusive ? numeric_value <= *matched_option->numeric_rule.minimum : numeric_value < *matched_option->numeric_rule.minimum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.minimum_is_exclusive ? "greater than" : "greater than or equal to", *matched_option->numeric_rule.minimum)};
                    if (matched_option->numeric_rule.maximum.has_value() && (matched_option->numeric_rule.maximum_is_exclusive ? numeric_value >= *matched_option->numeric_rule.maximum : numeric_value > *matched_option->numeric_rule.maximum)) return std::unexpected{std::format("{} must be {} {}.", display_name, matched_option->numeric_rule.maximum_is_exclusive ? "less than" : "less than or equal to", *matched_option->numeric_rule.maximum)};
                    **target = parsed;
                }
                matched_option->seen = true;
                continue;
            }

            if (next_positional >= this->positionals.size()) return std::unexpected{std::format("unknown argument '{}'.", argument)};
            PositionalBinding& positional = this->positionals[next_positional++];
            if (argument.empty()) return std::unexpected{std::format("{} must not be empty.", positional.name)};
            for (const OptionBinding& option : this->options) if (option.target_address == positional.target_address && option.seen) return std::unexpected{std::format("{} was provided more than once.", positional.name)};
            for (const PositionalBinding& existing_positional : this->positionals) if (existing_positional.target_address == positional.target_address && existing_positional.seen) return std::unexpected{std::format("{} was provided more than once.", positional.name)};
            if (std::string** target = std::get_if<std::string*>(std::addressof(positional.target))) **target = std::string{argument};
            else if (std::filesystem::path** target = std::get_if<std::filesystem::path*>(std::addressof(positional.target))) **target = std::filesystem::path{argument};
            positional.seen = true;
        }

        for (const OptionBinding& option : this->options) if (option.required && !option.seen) return std::unexpected{std::format("--{} is required.", option.long_name)};
        for (const PositionalBinding& positional : this->positionals) {
            if (!positional.required || positional.seen) continue;
            bool same_target_option_seen = false;
            for (const OptionBinding& option : this->options) {
                if (option.target_address == positional.target_address && option.seen) {
                    same_target_option_seen = true;
                    break;
                }
            }
            if (!same_target_option_seen) return std::unexpected{std::format("{} is required.", positional.name)};
        }
        return ParseResult{};
    }

    std::expected<void, std::string> Command::validate() {
        for (const PositionalBinding& positional : this->positionals) {
            if (!positional.path_rule.has_value()) continue;
            bool same_target_option_seen = false;
            for (const OptionBinding& option : this->options) {
                if (option.target_address == positional.target_address && option.seen) {
                    same_target_option_seen = true;
                    break;
                }
            }
            if (same_target_option_seen && !positional.seen) continue;

            const std::filesystem::path* path = nullptr;
            if (std::filesystem::path* const* target = std::get_if<std::filesystem::path*>(std::addressof(positional.target))) path = *target;
            if (path == nullptr) continue;
            if (positional.path_rule->reject_empty && path->empty()) return std::unexpected{std::format("{} must not be empty.", positional.name)};
            if (path->empty()) continue;
            if (positional.path_rule->requirement == PathRequirement::none) continue;

            if (positional.path_rule->requirement == PathRequirement::existing_parent_directory) {
                const std::filesystem::path parent_path = path->parent_path();
                if (parent_path.empty()) continue;
                std::error_code status_error;
                const std::filesystem::file_status status = std::filesystem::status(parent_path, status_error);
                if (status_error) return std::unexpected{std::format("{} parent directory '{}' cannot be inspected: {}", positional.name, parent_path.string(), status_error.message())};
                if (!std::filesystem::is_directory(status)) return std::unexpected{std::format("{} parent directory '{}' does not exist.", positional.name, parent_path.string())};
                continue;
            }

            std::error_code status_error;
            const std::filesystem::file_status status = std::filesystem::status(*path, status_error);
            if (status_error) return std::unexpected{std::format("{} '{}' cannot be inspected: {}", positional.name, path->string(), status_error.message())};
            if (positional.path_rule->requirement == PathRequirement::existing_file && !std::filesystem::is_regular_file(status)) return std::unexpected{std::format("{} '{}' is not a regular file.", positional.name, path->string())};
            if (positional.path_rule->requirement == PathRequirement::existing_directory && !std::filesystem::is_directory(status)) return std::unexpected{std::format("{} '{}' is not a directory.", positional.name, path->string())};
        }

        for (const OptionBinding& option : this->options) {
            if (!option.path_rule.has_value()) continue;
            bool same_target_positional_exists = false;
            bool same_target_positional_seen = false;
            for (const PositionalBinding& positional : this->positionals) {
                if (positional.target_address == option.target_address) {
                    same_target_positional_exists = true;
                    if (positional.seen) same_target_positional_seen = true;
                }
            }
            if (same_target_positional_seen && !option.seen) continue;
            if (same_target_positional_exists && !option.seen) continue;

            const std::string display_name = std::format("--{}", option.long_name);
            const std::filesystem::path* path = nullptr;
            if (std::filesystem::path* const* target = std::get_if<std::filesystem::path*>(std::addressof(option.target))) {
                path = *target;
            } else if (std::optional<std::filesystem::path>* const* target = std::get_if<std::optional<std::filesystem::path>*>(std::addressof(option.target))) {
                if (!(**target).has_value()) continue;
                path = std::addressof((**target).value());
            }
            if (path == nullptr) continue;
            if (option.path_rule->reject_empty && path->empty()) return std::unexpected{std::format("{} must not be empty.", display_name)};
            if (path->empty()) continue;
            if (option.path_rule->requirement == PathRequirement::none) continue;

            if (option.path_rule->requirement == PathRequirement::existing_parent_directory) {
                const std::filesystem::path parent_path = path->parent_path();
                if (parent_path.empty()) continue;
                std::error_code status_error;
                const std::filesystem::file_status status = std::filesystem::status(parent_path, status_error);
                if (status_error) return std::unexpected{std::format("{} parent directory '{}' cannot be inspected: {}", display_name, parent_path.string(), status_error.message())};
                if (!std::filesystem::is_directory(status)) return std::unexpected{std::format("{} parent directory '{}' does not exist.", display_name, parent_path.string())};
                continue;
            }

            std::error_code status_error;
            const std::filesystem::file_status status = std::filesystem::status(*path, status_error);
            if (status_error) return std::unexpected{std::format("{} '{}' cannot be inspected: {}", display_name, path->string(), status_error.message())};
            if (option.path_rule->requirement == PathRequirement::existing_file && !std::filesystem::is_regular_file(status)) return std::unexpected{std::format("{} '{}' is not a regular file.", display_name, path->string())};
            if (option.path_rule->requirement == PathRequirement::existing_directory && !std::filesystem::is_directory(status)) return std::unexpected{std::format("{} '{}' is not a directory.", display_name, path->string())};
        }

        for (ValidatorBinding& validator : this->validators) {
            const auto validation = validator.validator();
            if (!validation) return std::unexpected{std::format("{}: {}", validator.name, validation.error())};
        }
        return {};
    }

    bool Command::option_provided(const std::string_view long_name) const {
        if (long_name.empty()) throw std::runtime_error{"option name must not be empty."};
        if (long_name.starts_with("-")) throw std::runtime_error{std::format("option name '{}' must not include '-' prefixes.", long_name)};
        for (const OptionBinding& option : this->options)
            if (option.long_name == long_name) return option.seen;
        throw std::runtime_error{std::format("option '{}' is not registered.", long_name)};
    }

    std::string Command::help(const std::span<const char* const> arguments) const {
        return this->help(arguments, {
            .reset = "\x1b[0m",
            .dim = "\x1b[2m",
            .bold = "\x1b[1m",
            .heading = "\x1b[1m",
            .executable = "\x1b[36m",
            .option = "\x1b[32m",
            .value = "\x1b[33m",
            .default_label = "\x1b[2m",
        });
    }

    std::string Command::help(const std::span<const char* const> arguments, const HelpStyle& style) const {
        if (arguments.empty() || arguments.front() == nullptr) throw std::runtime_error{"argv must contain executable path."};
        const std::string executable_name = std::filesystem::path{arguments.front()}.filename().string();
        constexpr std::size_t description_column = 42uz;

        std::string text;
        text += std::format("{}Usage:{}\n  {}{}{}", style.heading, style.reset, style.executable, executable_name, style.reset);
        for (const PositionalBinding& positional : this->positionals) {
            if (positional.required) text += std::format(" {}{}{}", style.value, positional.name, style.reset);
            else text += std::format(" [{}{}{}]", style.value, positional.name, style.reset);
        }
        if (!this->options.empty()) text += std::format(" {}[options]{}", style.dim, style.reset);
        text += "\n";
        if (!this->description.empty()) text += std::format("\n{}\n", this->description);

        if (!this->positionals.empty()) {
            text += std::format("\n{}Arguments:{}\n", style.heading, style.reset);
            for (const PositionalBinding& positional : this->positionals) {
                const std::string display = positional.name;
                const std::string styled_display = std::format("{}{}{}", style.value, positional.name, style.reset);
                text += "  ";
                text += styled_display;
                if (display.size() + 2uz < description_column) text += std::string(description_column - display.size() - 2uz, ' ');
                else text += "\n" + std::string(description_column, ' ');
                text += positional.description;
                text += "\n";
                if (positional.show_default && positional.default_text.has_value()) text += std::format("{}{}default:{} {}\n", std::string(description_column, ' '), style.default_label, style.reset, *positional.default_text);
            }
        }

        text += std::format("\n{}Options:{}\n", style.heading, style.reset);
        for (const OptionBinding& option : this->options) {
            std::string display;
            std::string styled_display;
            if (option.short_name.has_value()) {
                display = std::format("-{}, --{}", *option.short_name, option.long_name);
                styled_display = std::format("{}-{}, --{}{}", style.option, *option.short_name, option.long_name, style.reset);
            } else {
                display = std::format("--{}", option.long_name);
                styled_display = std::format("{}--{}{}", style.option, option.long_name, style.reset);
            }
            if (option.requires_value) {
                display += std::format(" <{}>", option.value_name);
                styled_display += std::format(" {}<{}>{}", style.value, option.value_name, style.reset);
            }
            text += "  ";
            text += styled_display;
            if (display.size() + 2uz < description_column) text += std::string(description_column - display.size() - 2uz, ' ');
            else text += "\n" + std::string(description_column, ' ');
            text += option.description;
            text += "\n";
            if (option.show_default && option.default_text.has_value()) text += std::format("{}{}default:{} {}\n", std::string(description_column, ' '), style.default_label, style.reset, *option.default_text);
        }
        text += std::format("  {}-h, --help{}{}print this help\n", style.option, style.reset, std::string(description_column - 12uz, ' '));

        if (!this->examples.empty()) {
            text += std::format("\n{}Examples:{}\n", style.heading, style.reset);
            for (const std::string& example : this->examples) text += std::format("  {}{}{} {}\n", style.executable, executable_name, style.reset, example);
        }

        return text;
    }
} // namespace xcli
