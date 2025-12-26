#!/usr/bin/env python3
"""
Quick authentication checker for Rollouts CLI

Tells you definitively:
- Are you using OAuth or API key?
- Will you get invoiced by Anthropic?

Usage:
    python check_auth.py
    python check_auth.py --test-api-key  # Test with invalid key to prove it works
"""

import argparse
import os
import sys


def check_current_auth() -> bool:
    """Check what authentication is currently configured"""
    print("=" * 70)
    print("🔍 Rollouts Authentication Check")
    print("=" * 70)

    # Check 1: Environment variables
    print("\n1️⃣  Environment Variables:")
    api_key_env = os.environ.get("ANTHROPIC_API_KEY")
    if api_key_env:
        print(f"   ⚠️  ANTHROPIC_API_KEY is set: {api_key_env[:15]}...")
        print("   ⚠️  Used as fallback if OAuth not available")
    else:
        print("   ✅ ANTHROPIC_API_KEY not set")

    # Check 2: OAuth status
    print("\n2️⃣  OAuth Status:")
    try:
        from rollouts.frontends.tui.oauth import get_oauth_client

        client = get_oauth_client()
        tokens = client.tokens

        if tokens:
            print("   ✅ OAuth tokens found")
            print(f"   📝 Token: {tokens.access_token[:20]}...")

            if tokens.is_expired():
                print("   ⚠️  Token expired (will auto-refresh on use)")
            else:
                print("   ✅ Token valid")

            if tokens.access_token.startswith("sk-ant-oat01-"):
                print("   ✅ Correct OAuth format")
            else:
                print("   ❌ Wrong token format!")
        else:
            print("   ❌ No OAuth tokens found")
            print("   💡 Run: python -m rollouts.cli --login-claude")
            return False
    except Exception as e:
        print(f"   ❌ Error checking OAuth: {e}")
        return False

    # Check 3: What endpoint will use
    print("\n3️⃣  Endpoint Configuration Test:")
    print("   Creating: anthropic/claude-3-5-sonnet-20241022")

    try:
        import io
        from contextlib import redirect_stdout

        from rollouts.cli import create_endpoint

        output = io.StringIO()
        with redirect_stdout(output):
            endpoint = create_endpoint(
                "anthropic/claude-3-5-sonnet-20241022", api_key=None, thinking="disabled"
            )

        cli_output = output.getvalue()

        # Show CLI message (this is the key indicator)
        if cli_output.strip():
            print("\n   CLI Output:")
            for line in cli_output.strip().split("\n"):
                print(f"   {line}")
        else:
            print("\n   ⚠️  No CLI message (OAuth likely not used)")

        print("\n   Endpoint:")
        print(f"   • oauth_token: {endpoint.oauth_token[:20] if endpoint.oauth_token else 'EMPTY'}")
        print(f"   • api_key:     {endpoint.api_key[:20] if endpoint.api_key else 'EMPTY'}")

    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

    # Final verdict
    print("\n" + "=" * 70)
    print("📊 Final Verdict:")
    print("=" * 70)

    if endpoint.oauth_token and endpoint.oauth_token.startswith("sk-ant-oat01-"):
        print("✅ ✅ ✅ USING OAUTH ✅ ✅ ✅")
        print("\n💰 You will NOT get invoiced by Anthropic")
        print("\nAPI calls use:")
        print("  • AsyncAnthropic(auth_token='sk-ant-oat01-...')")
        print("  • Header: Authorization: Bearer sk-ant-oat01-...")
        print("  • Billing: Claude Pro/Max subscription")
        return True
    elif endpoint.api_key:
        print("❌ ❌ ❌ USING API KEY ❌ ❌ ❌")
        print("\n💰 YOU WILL GET INVOICED by Anthropic")
        print("\nAPI calls use:")
        print(f"  • AsyncAnthropic(api_key='{endpoint.api_key[:15]}...')")
        print(f"  • Header: x-api-key: {endpoint.api_key[:15]}...")
        print("  • Billing: Anthropic API account")
        print("\n💡 To use OAuth instead:")
        if api_key_env:
            print("  1. unset ANTHROPIC_API_KEY")
        print("  2. Ensure OAuth login: python -m rollouts.cli --login-claude")
        return False
    else:
        print("❌ No authentication configured")
        return False


def test_api_key_path() -> None:
    """Test that API key path actually uses API key (not OAuth)"""
    import trio

    from rollouts.dtypes import Actor, Endpoint, Message, StreamEvent, Trajectory
    from rollouts.providers.anthropic import rollout_anthropic

    async def noop(event: StreamEvent) -> None:
        pass

    async def test() -> bool | None:
        print("\n" + "=" * 70)
        print("🧪 Testing API Key Path (with invalid key)")
        print("=" * 70)
        print("\nThis proves API key is actually used when set...")

        endpoint = Endpoint(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-api03-INVALID",
            oauth_token="",
            max_tokens=100,
        )

        print("\nEndpoint: oauth_token=EMPTY, api_key=INVALID")
        print("Expected: 401 authentication error\n")

        actor = Actor(
            trajectory=Trajectory(messages=[Message(role="user", content="hi")]), endpoint=endpoint
        )

        try:
            await rollout_anthropic(actor=actor, on_chunk=noop)
            print("❌ Unexpected: call succeeded with invalid key!")
            return False
        except Exception as e:
            if "401" in str(e) or "authentication" in str(e).lower():
                print("✅ Got expected auth error:")
                print(f"   {str(e)[:100]}...")
                print("\n✅ This proves API key is used when set")
                print("💡 If the key were valid, you'd get invoiced")
                return True
            else:
                print(f"⚠️  Got different error: {e}")
                return False

    return trio.run(test)


def test_cli_output() -> None:
    """Test that CLI shows OAuth message"""
    import subprocess

    print("\n" + "=" * 70)
    print("🧪 Testing CLI Output")
    print("=" * 70)

    cmd = [
        sys.executable,
        "-m",
        "rollouts.cli",
        "--model",
        "anthropic/claude-3-5-sonnet-20241022",
        "-p",
        "say ok",
    ]

    print(f"\nCommand: {' '.join(cmd)}")
    print("\nRunning (with 3s timeout)...")
    print("Looking for: '🔐 Using OAuth authentication'\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)

        # Check for OAuth message (could be in stdout or stderr)
        output = result.stdout + result.stderr

        if "🔐 Using OAuth authentication" in output:
            print("✅ OAuth message found!")
            print("✅ CLI is using OAuth")
        else:
            print("⚠️  No OAuth message found")
            print(f"Output preview: {output[:300]}")

    except subprocess.TimeoutExpired as e:
        # Check partial output
        stderr = e.stderr.decode() if e.stderr else ""
        stdout = e.stdout.decode() if e.stdout else ""
        output = stdout + stderr

        if "🔐 Using OAuth authentication" in output:
            print("✅ OAuth message found in partial output")
            print("⏱️  Timed out during API call (expected)")
        else:
            print("⚠️  Timed out before seeing OAuth message")
            print(f"Output preview: {output[:300]}")
    except Exception as e:
        print(f"❌ Error: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Rollouts authentication method")
    parser.add_argument("--test-api-key", action="store_true", help="Test API key path")
    parser.add_argument("--test-json", action="store_true", help="Test with --json mode")
    args = parser.parse_args()

    # Main check
    oauth_ok = check_current_auth()

    # Optional tests
    if args.test_api_key:
        test_api_key_path()

    # TODO: test_with_json_mode was removed - remove this flag or reimplement
    # if args.test_json:
    #     test_with_json_mode()

    print("\n" + "=" * 70)
    sys.exit(0 if oauth_ok else 1)


if __name__ == "__main__":
    main()
