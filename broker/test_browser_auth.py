#!/usr/bin/env python3
"""
Test RunPod HAPI logs endpoint with browser-style authentication.

Usage:
    1. Open browser DevTools (F12)
    2. Go to https://www.runpod.io/console/pods
    3. Click "Logs" on a pod
    4. In Network tab, find the request to hapi.runpod.net
    5. Right-click → Copy → Copy as cURL
    6. Extract the headers and paste them below

Then run:
    python test_browser_auth.py <pod_id>
"""

import json
import sys

import requests


def test_with_browser_headers(pod_id: str, headers: dict[str, str]) -> None:
    """Test HAPI endpoint with headers copied from browser."""

    url = f"https://hapi.runpod.net/v1/pod/{pod_id}/logs"

    print(f"Testing HAPI logs endpoint for pod: {pod_id}")
    print(f"URL: {url}")
    print()
    print("Headers:")
    for key, value in headers.items():
        # Mask sensitive values
        if key.lower() in ['authorization', 'cookie', 'x-api-key']:
            display_value = f"...{value[-10:]}" if len(value) > 10 else "***"
        else:
            display_value = value
        print(f"  {key}: {display_value}")
    print()

    try:
        print("Sending GET request...")
        response = requests.get(
            url,
            headers=headers,
            timeout=(10, 30)
        )

        print(f"Response status code: {response.status_code}")
        print(f"Response headers: {dict(response.headers)}")
        print()

        if response.status_code == 200:
            print("✅ Success!")
            print()

            # Try to parse response
            content_type = response.headers.get('Content-Type', '')

            if 'application/json' in content_type:
                try:
                    data = response.json()
                    print("Response body (JSON):")
                    print(json.dumps(data, indent=2))
                except ValueError as e:
                    print(f"Failed to parse JSON: {e}")
                    print("Raw response:")
                    print(response.text[:2000])
            else:
                print("Response body (text):")
                print(response.text[:2000])
                if len(response.text) > 2000:
                    print(f"\n... (truncated, total length: {len(response.text)} chars)")
        else:
            print(f"❌ Failed with status {response.status_code}")
            print("Response body:")
            print(response.text[:500])

    except requests.Timeout:
        print("❌ Request timed out")
    except requests.RequestException as e:
        print(f"❌ Request failed: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nExample headers to paste:")
        print("""
headers = {
    "Accept": "application/json",
    "Authorization": "Bearer YOUR_TOKEN_HERE",
    "Cookie": "session=YOUR_SESSION_COOKIE",
    "User-Agent": "Mozilla/5.0...",
    # Add other headers from browser DevTools
}
""")
        sys.exit(1)

    pod_id = sys.argv[1]

    # TODO: Replace these with actual headers from browser DevTools
    # Instructions:
    # 1. Open RunPod console in browser
    # 2. Open DevTools (F12) → Network tab
    # 3. Click "Logs" button on a pod
    # 4. Find the request to hapi.runpod.net/v1/pod/.../logs
    # 5. Right-click → Copy → Copy all headers
    # 6. Paste them below

    headers = {
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Authorization": "Bearer eyJhbGciOiJSUzI1NiIsImNhdCI6ImNsX0I3ZDRQRDExMUFBQSIsImtpZCI6Imluc18yUE9SZ01ka295ZHE5clp2VWx5ZDBUdTlCeGkiLCJ0eXAiOiJKV1QifQ.eyJhdWQiOlsiaHR0cHM6Ly9ydW5wb2QuaW8iXSwiYXpwIjoiaHR0cHM6Ly9jb25zb2xlLnJ1bnBvZC5pbyIsImV4cCI6MTc2MDY2MTY4OCwiZnZhIjpbNTk5NSwtMV0sImh0dHBzOi8vcnVucG9kLmlvL2VtYWlsIjoiY2hpcmFhZ2JhbHVAYmVya2VsZXkuZWR1IiwiaHR0cHM6Ly9ydW5wb2QuaW8vZW1haWxfdmVyaWZpZWQiOnRydWUsImlhdCI6MTc2MDY2MTYyOCwiaXNzIjoiaHR0cHM6Ly9jbGVyay5ydW5wb2QuaW8iLCJqdGkiOiI2MjQxZTc2M2RmZDA0NmM4NjI5NCIsIm5iZiI6MTc2MDY2MTYxOCwic2lkIjoic2Vzc18zM3l2MmlkVmNwNko0Y2pOVXY0UlBEUG1HM0IiLCJzdHMiOiJhY3RpdmUiLCJzdWIiOiJ1c2VyXzJmTGk0dUhIQUppZ2pDbTJSeHJrR29CSkdMaCJ9.BgI37cJWrfnTrjofR1rgTaUGc_5Vcaa-uCTPxwhcqetZGD39d0rlp6U8JL8iLciDdGNoj8I44_vZQRW-uIr0QjoDoWMXvYzEdMsRSKLEb85es5tzG7NoVCeXL4BSmWkGvoLyyxmD6E0D-Aj16IFD4_bHss0m-Mf59RNdxoNPnR__2WCUzTvrQVVaLRnt0WZkfu05-YQiObFSxzZ0VNYjnQTJAb0FxI_mlJQEXpR38z87uzQwgI-7Y1RFEdJfBIbMOVF1XvxN4cfQDjWM5ujFKHlwTe2jBdwsnDWgYwCVfZWcwyzY-hZGhP1Xr57c5NsGxxfHgVhssQ0VvvpBaAwX-w",
        "Content-Type": "application/json",
        "Origin": "https://console.runpod.io",
        "Referer": "https://console.runpod.io/",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36",
    }

    if "Authorization" not in headers and "Cookie" not in headers:
        print("⚠️  WARNING: No Authorization or Cookie headers found!")
        print("Please edit this script and paste headers from browser DevTools")
        print()
        print("To get headers:")
        print("1. Go to https://www.runpod.io/console/pods")
        print("2. Open DevTools (F12) → Network tab")
        print("3. Click 'Logs' on any pod")
        print("4. Find request to hapi.runpod.net in Network tab")
        print("5. Right-click → Copy → Copy all headers")
        print("6. Paste into this script's 'headers' dict")
        print()
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    test_with_browser_headers(pod_id, headers)


if __name__ == "__main__":
    main()
