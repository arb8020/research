from __future__ import annotations

import sys


def auth_main(args: list[str]) -> int:
    """Handle auth subcommand: rollouts auth <login|status|switch>."""
    from ..credentials import (
        CREDENTIALS_FILE,
        KNOWN_PROVIDERS,
        PROVIDER_ENV_MAP,
        get_active_profile,
        key_preview,
        load_profiles,
        set_active_profile,
        set_profile_key,
    )

    if not args or args[0] in ("-h", "--help"):
        print("Usage: rollouts auth <command>")
        print()
        print("Commands:")
        print("  login <provider>   Save API key for a provider")
        print("  status             Show configured credentials")
        print("  switch <profile>   Switch active profile")
        print()
        print(f"Providers: {', '.join(sorted(KNOWN_PROVIDERS))}")
        print(f"Config: {CREDENTIALS_FILE}")
        return 0

    cmd = args[0]

    if cmd == "login":
        if len(args) < 2:
            print("Usage: rollouts auth login <provider> [--profile NAME]", file=sys.stderr)
            print(f"Providers: {', '.join(sorted(KNOWN_PROVIDERS))}", file=sys.stderr)
            return 1

        provider = args[1]
        profile = "default"
        if "--profile" in args:
            idx = args.index("--profile")
            if idx + 1 < len(args):
                profile = args[idx + 1]

        if provider not in KNOWN_PROVIDERS:
            print(f"Unknown provider: {provider}", file=sys.stderr)
            print(f"Known providers: {', '.join(sorted(KNOWN_PROVIDERS))}", file=sys.stderr)
            return 1

        import getpass

        api_key = getpass.getpass(f"Enter {provider} API key: ")
        if not api_key.strip():
            print("No API key provided", file=sys.stderr)
            return 1

        set_profile_key(profile, provider, api_key.strip())
        print(f"Saved {provider} API key to profile '{profile}'")
        return 0

    if cmd == "status":
        import os

        profiles = load_profiles()
        active_name, profile_creds = get_active_profile()
        del profiles

        for provider in sorted(KNOWN_PROVIDERS):
            env_var_name = PROVIDER_ENV_MAP.get(provider)
            env_val = os.environ.get(env_var_name) if env_var_name else None
            toml_val = profile_creds.get(provider)

            if not env_val and not toml_val:
                continue

            print(f"{provider.capitalize()} (in precedence order):")
            active_found = False

            if env_val:
                active_found = True
                print(f"  1. ${env_var_name}: {key_preview(env_val)} <- active")
            else:
                print(f"  1. ${env_var_name}: (not set)")

            if toml_val:
                marker = " <- active" if not active_found else ""
                print(
                    f"  2. credentials.toml: {key_preview(toml_val)} (profile:{active_name}){marker}"
                )
            else:
                print(f"  2. credentials.toml: (not set in profile:{active_name})")

            print()

        unconfigured = []
        for provider in sorted(KNOWN_PROVIDERS):
            env_var_name = PROVIDER_ENV_MAP.get(provider)
            env_val = os.environ.get(env_var_name) if env_var_name else None
            toml_val = profile_creds.get(provider)
            if not env_val and not toml_val:
                unconfigured.append(provider)

        if unconfigured:
            print(f"Not configured: {', '.join(unconfigured)}")
            print()

        print(f"Config: {CREDENTIALS_FILE}")
        return 0

    if cmd == "switch":
        if len(args) < 2:
            print("Usage: rollouts auth switch <profile>", file=sys.stderr)
            profiles = load_profiles()
            if profiles:
                print(f"Available: {', '.join(profiles.keys())}", file=sys.stderr)
            return 1

        profile = args[1]
        try:
            set_active_profile(profile)
            print(f"Switched to profile '{profile}'")
            return 0
        except ValueError as e:
            print(str(e), file=sys.stderr)
            return 1

    print(f"Unknown auth command: {cmd}", file=sys.stderr)
    print("Use: rollouts auth --help", file=sys.stderr)
    return 1
