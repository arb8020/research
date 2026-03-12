from __future__ import annotations

import sys

import trio


def cmd_oauth(login: bool, profile: str) -> int:
    from ..frontends.tui.oauth import OAuthError, logout
    from ..frontends.tui.oauth import login as do_login

    if not login:
        logout(profile)
        return 0

    async def oauth_action() -> int:
        try:
            await do_login(profile)
        except OAuthError as e:
            print(f"❌ OAuth error: {e}", file=sys.stderr)
            return 1
        except KeyboardInterrupt:
            print("\n⚠️  Login cancelled")
            return 1
        return 0

    return trio.run(oauth_action)


def cmd_list_profiles() -> int:
    from ..frontends.tui.oauth import list_profiles

    profiles = list_profiles()
    if not profiles:
        print("No Claude OAuth profiles found.")
        print("Run: rollouts --login-claude")
        return 0

    for profile in profiles:
        print(profile)
    return 0


def cmd_set_default_profile(profile: str) -> int:
    from ..frontends.tui.oauth import set_default_profile

    _, err = set_default_profile(profile)
    if err:
        print(f"❌ {err}", file=sys.stderr)
        return 1

    print(f"✅ Set '{profile}' as default profile")
    return 0
