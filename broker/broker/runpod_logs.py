"""Fetch RunPod system logs via their internal API.

Uses a separate Chrome instance with remote debugging to grab the Clerk JWT
from an authenticated RunPod console session.
"""

import shutil
import subprocess
import time
from pathlib import Path

import httpx

# Chrome debug instance settings
CHROME_DEBUG_PORT = 9222
CHROME_USER_DATA_DIR = Path.home() / ".broker" / "chrome-debug"
MAIN_CHROME_DIR = Path.home() / "Library/Application Support/Google/Chrome"


def is_chrome_debug_running() -> bool:
    """Check if Chrome debug instance is already running."""
    try:
        response = httpx.get(f"http://localhost:{CHROME_DEBUG_PORT}/json/version", timeout=1)
        return response.status_code == 200
    except Exception:
        return False


def copy_cookies_from_main_chrome() -> None:
    """Copy cookies from main Chrome profile to debug profile.

    This allows the debug Chrome to inherit RunPod login session.
    """
    debug_default = CHROME_USER_DATA_DIR / "Default"
    debug_default.mkdir(parents=True, exist_ok=True)

    main_default = MAIN_CHROME_DIR / "Default"

    # Files to copy for session persistence
    files_to_copy = ["Cookies", "Login Data", "Web Data"]

    for filename in files_to_copy:
        src = main_default / filename
        dst = debug_default / filename
        if src.exists():
            try:
                shutil.copy2(src, dst)
            except Exception:
                # Might fail if Chrome has the file locked, that's ok
                pass


def launch_chrome_debug(copy_cookies: bool = True) -> subprocess.Popen:
    """Launch a separate Chrome instance with remote debugging enabled.

    Args:
        copy_cookies: If True, copy cookies from main Chrome profile first
    """
    CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if copy_cookies:
        copy_cookies_from_main_chrome()

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    proc = subprocess.Popen(
        [
            chrome_path,
            f"--remote-debugging-port={CHROME_DEBUG_PORT}",
            f"--user-data-dir={CHROME_USER_DATA_DIR}",
            "--no-first-run",
            "--no-default-browser-check",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for Chrome to be ready
    for _ in range(30):
        if is_chrome_debug_running():
            return proc
        time.sleep(0.2)

    raise RuntimeError("Chrome debug instance failed to start")


def get_runpod_jwt() -> str | None:
    """Get the Clerk JWT from RunPod console session.

    Returns None if not logged in or no RunPod tab found.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError("playwright not installed. Run: pip install playwright") from e

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(f"http://localhost:{CHROME_DEBUG_PORT}")

        # Find RunPod console tab
        for context in browser.contexts:
            for page in context.pages:
                if "console.runpod.io" in page.url:
                    # Try to get the session token from Clerk
                    # Clerk stores tokens in sessionStorage with keys like __clerk_*
                    token = page.evaluate("""
                        () => {
                            // Check sessionStorage for Clerk session
                            for (const key of Object.keys(sessionStorage)) {
                                if (key.includes('clerk') && key.includes('session')) {
                                    const data = sessionStorage.getItem(key);
                                    try {
                                        const parsed = JSON.parse(data);
                                        if (parsed && parsed.jwt) return parsed.jwt;
                                    } catch {}
                                }
                            }

                            // Try localStorage as fallback
                            for (const key of Object.keys(localStorage)) {
                                if (key.includes('clerk')) {
                                    const data = localStorage.getItem(key);
                                    try {
                                        const parsed = JSON.parse(data);
                                        if (parsed && parsed.jwt) return parsed.jwt;
                                    } catch {}
                                }
                            }

                            return null;
                        }
                    """)

                    if token:
                        return token

                    # Alternative: intercept a request to get fresh token
                    # Navigate to trigger a request that includes the JWT
                    page.reload()

                    # Get from cookies/auth header by evaluating Clerk's internal state
                    token = page.evaluate("""
                        async () => {
                            // Clerk exposes window.Clerk in the browser
                            if (window.Clerk && window.Clerk.session) {
                                const session = window.Clerk.session;
                                if (session.getToken) {
                                    try {
                                        return await session.getToken();
                                    } catch {}
                                }
                            }
                            return null;
                        }
                    """)

                    return token

        return None


def fetch_pod_logs(pod_id: str, log_type: str = "system") -> dict:
    """Fetch logs for a RunPod pod.

    Args:
        pod_id: The RunPod pod ID
        log_type: "system" or "container"

    Returns:
        Log data from RunPod API
    """
    if not is_chrome_debug_running():
        launch_chrome_debug()
        print("Chrome debug instance launched. Please log into RunPod console.")
        print("Navigate to: https://console.runpod.io")
        print("Then run this command again.")
        return {"error": "Please log into RunPod console in the Chrome debug window"}

    jwt = get_runpod_jwt()
    if not jwt:
        return {"error": "Not logged into RunPod console. Please log in and try again."}

    # Fetch logs from RunPod's internal API
    url = f"https://hapi.runpod.net/v1/pod/{pod_id}/logs"

    response = httpx.get(
        url,
        headers={
            "Authorization": f"Bearer {jwt}",
            "Content-Type": "application/json",
            "Origin": "https://console.runpod.io",
            "Referer": "https://console.runpod.io/",
        },
        timeout=30,
    )

    if response.status_code == 401:
        return {"error": "JWT expired. Please refresh the RunPod console page and try again."}

    response.raise_for_status()
    return response.json()
