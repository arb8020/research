#!/usr/bin/env python3
"""
Fetch Prime Intellect dashboard session cookie using Playwright.

Two modes:
1. Connect to existing Chrome (must be launched with --remote-debugging-port=9222)
2. Launch new browser for manual login

Usage:
    # First, launch Chrome with debugging enabled:
    /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222

    # Then run this script to extract cookie from existing session:
    python scripts/prime_dashboard_auth.py

    # Or launch new browser for login:
    python scripts/prime_dashboard_auth.py --new-browser
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Cookie storage location
COOKIE_FILE = Path.home() / ".prime" / "dashboard_session.json"
SPINUP_CACHE_FILE = Path.home() / ".prime" / "spinup_times_cache.json"


def save_cookie(cookie_value: str) -> None:
    """Save session cookie to file."""
    COOKIE_FILE.parent.mkdir(parents=True, exist_ok=True)
    COOKIE_FILE.write_text(json.dumps({"session_token": cookie_value}))
    os.chmod(COOKIE_FILE, 0o600)
    print(f"Cookie saved to {COOKIE_FILE}")


def load_cookie() -> str | None:
    """Load session cookie from file."""
    if not COOKIE_FILE.exists():
        return None
    try:
        data = json.loads(COOKIE_FILE.read_text())
        return data.get("session_token")
    except (json.JSONDecodeError, KeyError):
        return None


def extract_from_existing_chrome() -> str | None:
    """Connect to existing Chrome and extract Prime Intellect session cookie."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright not installed. Run: uv pip install playwright && playwright install chromium"
        )
        sys.exit(1)

    print("Connecting to Chrome on localhost:9222...")
    print("(Make sure Chrome was launched with: --remote-debugging-port=9222)")

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
        except Exception as e:
            print(f"\nFailed to connect to Chrome: {e}")
            print("\nTo enable remote debugging, restart Chrome with:")
            print(
                "  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --remote-debugging-port=9222"
            )
            return None

        # Get all contexts (each window/profile is a context)
        contexts = browser.contexts
        if not contexts:
            print("No browser contexts found")
            return None

        # Search all contexts for Prime Intellect cookies
        for ctx in contexts:
            cookies = ctx.cookies(["https://app.primeintellect.ai"])
            for cookie in cookies:
                if cookie["name"] == "__Secure-authjs.session-token":
                    print(f"Found session cookie (expires: {cookie.get('expires', 'session')})")
                    return cookie["value"]

        print("No Prime Intellect session cookie found.")
        print("Make sure you're logged in to https://app.primeintellect.ai")
        return None


def login_with_new_browser() -> str | None:
    """Launch browser for manual login and extract cookie."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "Playwright not installed. Run: uv pip install playwright && playwright install chromium"
        )
        sys.exit(1)

    print("Launching browser for Prime Intellect login...")
    print("Log in with Google/GitHub, then press Enter in this terminal.\n")

    with sync_playwright() as p:
        # Launch visible browser
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        # Navigate to Prime Intellect login
        page.goto("https://app.primeintellect.ai/api/auth/signin")

        # Wait for user to log in
        input("\nPress Enter after logging in to Prime Intellect...")

        # Extract cookie
        cookies = context.cookies(["https://app.primeintellect.ai"])
        for cookie in cookies:
            if cookie["name"] == "__Secure-authjs.session-token":
                print("Found session cookie!")
                browser.close()
                return cookie["value"]

        print("No session cookie found. Did you complete login?")
        browser.close()
        return None


def fetch_spinup_times(session_token: str) -> list | None:
    """Fetch spinup times using session cookie.

    Returns list of dicts with provider_type, gpu_name, avg_provider_spinup_seconds, etc.
    """
    import httpx

    url = "https://app.primeintellect.ai/api/trpc/pod.getSpinupTimes?batch=1&input=%7B%220%22%3A%7B%7D%7D"

    cookies = {"__Secure-authjs.session-token": session_token}

    try:
        resp = httpx.get(url, cookies=cookies, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            # Navigate tRPC response structure: [0].result.data.json.data
            return data[0]["result"]["data"]["json"]["data"]
        else:
            print(f"Failed to fetch spinup times: {resp.status_code}")
            if resp.status_code == 401:
                print("Session expired. Re-run script to get new cookie.")
            return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


def aggregate_by_provider(entries: list) -> dict:
    """Aggregate spinup times by provider."""
    from collections import defaultdict

    provider_stats = defaultdict(list)
    for entry in entries:
        provider = entry.get("provider_type", "unknown")
        seconds = entry.get("avg_provider_spinup_seconds")
        if seconds is not None:
            provider_stats[provider].append(seconds)

    result = {}
    for provider, times in provider_stats.items():
        result[provider] = {
            "avg_seconds": sum(times) / len(times),
            "min_seconds": min(times),
            "max_seconds": max(times),
            "sample_count": len(times),
        }
    return result


def save_spinup_cache(stats: dict) -> None:
    """Save spinup stats with timestamp."""
    import time

    cache_data = {
        "fetched_at": time.time(),
        "stats": stats,
    }
    SPINUP_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPINUP_CACHE_FILE.write_text(json.dumps(cache_data, indent=2))


def load_spinup_cache() -> tuple[dict | None, float | None]:
    """Load cached spinup stats. Returns (stats, fetched_at_timestamp)."""
    if not SPINUP_CACHE_FILE.exists():
        return None, None
    try:
        data = json.loads(SPINUP_CACHE_FILE.read_text())
        return data.get("stats"), data.get("fetched_at")
    except (json.JSONDecodeError, KeyError):
        return None, None


def format_age(timestamp: float) -> str:
    """Format timestamp as human-readable age."""
    import time

    age_seconds = time.time() - timestamp
    if age_seconds < 60:
        return "just now"
    elif age_seconds < 3600:
        return f"{int(age_seconds / 60)} minutes ago"
    elif age_seconds < 86400:
        return f"{int(age_seconds / 3600)} hours ago"
    else:
        return f"{int(age_seconds / 86400)} days ago"


def print_spinup_stats(stats: dict, age_str: str) -> None:
    """Print spinup stats table."""
    print(f"\nSpinup times by provider (fetched {age_str}):")
    print("-" * 60)
    for provider, data in sorted(stats.items(), key=lambda x: x[1]["avg_seconds"]):
        avg_min = data["avg_seconds"] / 60
        min_min = data["min_seconds"] / 60
        max_min = data["max_seconds"] / 60
        count = data["sample_count"]
        print(
            f"  {provider:20} {avg_min:>5.1f} min  (range: {min_min:.1f}-{max_min:.1f}, {count} configs)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prime Intellect dashboard auth")
    parser.add_argument("--new-browser", action="store_true", help="Launch new browser for login")
    parser.add_argument(
        "--fetch", action="store_true", help="Fetch spinup times using saved cookie"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show cached spinup times (no network request)"
    )
    parser.add_argument("--show-cookie", action="store_true", help="Show saved cookie value")
    args = parser.parse_args()

    if args.show_cookie:
        cookie = load_cookie()
        if cookie:
            print(f"Saved cookie: {cookie[:20]}...{cookie[-10:]}")
        else:
            print("No saved cookie")
        return

    if args.show:
        stats, fetched_at = load_spinup_cache()
        if stats and fetched_at:
            print_spinup_stats(stats, format_age(fetched_at))
        else:
            print("No cached spinup data. Run with --fetch first.")
        return

    if args.fetch:
        cookie = load_cookie()
        if not cookie:
            print("No saved cookie. Run without --fetch first to get cookie.")
            sys.exit(1)

        print("Fetching spinup times...")
        entries = fetch_spinup_times(cookie)
        if entries:
            stats = aggregate_by_provider(entries)
            save_spinup_cache(stats)
            print_spinup_stats(stats, "just now")
        return

    # Get cookie
    if args.new_browser:
        cookie = login_with_new_browser()
    else:
        cookie = extract_from_existing_chrome()

    if cookie:
        save_cookie(cookie)
        print("\nTesting cookie by fetching spinup times...")
        data = fetch_spinup_times(cookie)
        if data:
            result = data[0].get("result", {}).get("data", {})
            print("\nSpinup times by provider:")
            print("-" * 50)
            for provider, stats in sorted(result.items()):
                avg = stats.get("avgSpinupTimeMinutes")
                count = stats.get("sampleCount", 0)
                avg_str = f"{avg:.1f}" if avg is not None else "N/A"
                print(f"  {provider:20} {avg_str:>6} min  ({count} samples)")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
