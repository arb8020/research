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
    """Copy cookies and session data from main Chrome profile to debug profile.

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

    # Also copy Local Storage (includes localStorage for Clerk)
    main_local_storage = main_default / "Local Storage" / "leveldb"
    debug_local_storage = debug_default / "Local Storage" / "leveldb"
    if main_local_storage.exists():
        debug_local_storage.parent.mkdir(parents=True, exist_ok=True)
        try:
            if debug_local_storage.exists():
                shutil.rmtree(debug_local_storage)
            shutil.copytree(main_local_storage, debug_local_storage)
        except Exception:
            pass

    # Copy Session Storage (includes sessionStorage for Clerk - usually empty on disk)
    main_session_storage = main_default / "Session Storage"
    debug_session_storage = debug_default / "Session Storage"
    if main_session_storage.exists():
        try:
            if debug_session_storage.exists():
                shutil.rmtree(debug_session_storage)
            shutil.copytree(main_session_storage, debug_session_storage)
        except Exception:
            pass


def launch_chrome_debug(copy_cookies: bool = True, open_runpod: bool = True) -> subprocess.Popen:
    """Launch a separate Chrome instance with remote debugging enabled.

    Args:
        copy_cookies: If True, copy cookies from main Chrome profile first
        open_runpod: If True, open RunPod console on launch
    """
    CHROME_USER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if copy_cookies:
        copy_cookies_from_main_chrome()

    chrome_path = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

    cmd = [
        chrome_path,
        f"--remote-debugging-port={CHROME_DEBUG_PORT}",
        f"--user-data-dir={CHROME_USER_DATA_DIR}",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    if open_runpod:
        cmd.append("https://console.runpod.io")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for Chrome to be ready
    for _ in range(30):
        if is_chrome_debug_running():
            return proc
        time.sleep(0.2)

    raise RuntimeError("Chrome debug instance failed to start")


def get_runpod_auth() -> tuple[str | None, str | None]:
    """Get the Clerk JWT and team ID from RunPod console session.

    Returns (jwt, team_id) tuple. Either can be None if not found.
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
                    # Primary method: Get token from Clerk's API
                    # Clerk exposes window.Clerk in the browser
                    token = page.evaluate("""
                        async () => {
                            if (window.Clerk && window.Clerk.session) {
                                const session = window.Clerk.session;
                                if (session.getToken) {
                                    try {
                                        return await session.getToken();
                                    } catch (e) {
                                        console.error('Failed to get token:', e);
                                    }
                                }
                            }
                            return null;
                        }
                    """)

                    # Get team ID from localStorage
                    team_id = page.evaluate("""
                        () => {
                            try {
                                const teamData = localStorage.getItem('rp-selected-team');
                                if (teamData) {
                                    const parsed = JSON.parse(teamData);
                                    return parsed?.state?.selectedTeam || null;
                                }
                            } catch {}
                            return null;
                        }
                    """)

                    if token:
                        return token, team_id

                    # Fallback: Check sessionStorage for Clerk session
                    token = page.evaluate("""
                        () => {
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

                    return token, team_id

        return None, None


def ensure_runpod_tab_open() -> None:
    """Ensure there's a RunPod console tab open in debug Chrome."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        return

    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(f"http://localhost:{CHROME_DEBUG_PORT}")
        except Exception:
            return

        # Check if RunPod tab exists
        for context in browser.contexts:
            for page in context.pages:
                if "console.runpod.io" in page.url:
                    return  # Already have a tab

        # No RunPod tab - open one
        for context in browser.contexts:
            if context.pages:
                page = context.pages[0]
                page.goto("https://console.runpod.io")
                return


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
        print("Chrome debug instance launched.")
        print("Please log into RunPod console in the Chrome window that just opened.")
        print("Then run this command again.")
        return {"error": "Please log into RunPod console in the Chrome debug window"}

    # Make sure there's a RunPod tab
    ensure_runpod_tab_open()

    jwt, team_id = get_runpod_auth()
    if not jwt:
        print("Not logged in. Opening RunPod console...")
        ensure_runpod_tab_open()
        return {"error": "Not logged into RunPod console. Please log in and try again."}

    # Fetch logs from RunPod's internal API
    url = f"https://hapi.runpod.net/v1/pod/{pod_id}/logs"

    headers = {
        "Authorization": f"Bearer {jwt}",
        "Content-Type": "application/json",
        "Origin": "https://console.runpod.io",
        "Referer": "https://console.runpod.io/",
    }
    if team_id:
        headers["x-team-id"] = team_id

    response = httpx.get(url, headers=headers, timeout=30)

    if response.status_code == 401:
        return {"error": "JWT expired. Please refresh the RunPod console page and try again."}

    response.raise_for_status()
    return response.json()
