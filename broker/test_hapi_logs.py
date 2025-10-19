#!/usr/bin/env python3
"""
Smoke test for RunPod HAPI logs endpoint.

Usage:
    python test_hapi_logs.py <pod_id>

The script will attempt to fetch logs from the undocumented HAPI endpoint
and print the results or error details.
"""

import json
import os
import sys
from typing import Optional

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def test_hapi_logs_endpoint(pod_id: str, api_key: Optional[str] = None) -> None:
    """Test the HAPI logs endpoint with a given pod ID."""

    if not api_key:
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            print("ERROR: No API key provided. Set RUNPOD_API_KEY environment variable.")
            sys.exit(1)

    print(f"Testing HAPI logs endpoint for pod: {pod_id}")
    print(f"API key: ...{api_key[-4:]}")
    print()

    url = f"https://hapi.runpod.net/v1/pod/{pod_id}/logs"

    # Try multiple auth approaches
    auth_attempts = [
        {
            "name": "Bearer token with Content-Type",
            "headers": {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
        },
        {
            "name": "Bearer token without Content-Type",
            "headers": {
                "Authorization": f"Bearer {api_key}"
            }
        },
        {
            "name": "API key as query parameter",
            "headers": {},
            "params": {"api_key": api_key}
        },
        {
            "name": "x-api-key header",
            "headers": {
                "x-api-key": api_key
            }
        },
    ]

    for attempt in auth_attempts:
        print(f"\n{'='*60}")
        print(f"Attempt: {attempt['name']}")
        print(f"{'='*60}")

        headers = attempt["headers"]
        params = attempt.get("params", {})

        print(f"Request URL: {url}")
        print(f"Headers: {headers}")
        if params:
            print(f"Params: {params}")
        print()

        try:
            print("Sending GET request...")
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=(10, 30)
            )

            print(f"Response status code: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            print()

            if response.status_code == 200:
                print(f"✅ Success with: {attempt['name']}")
                print()

                # Try to parse response
                content_type = response.headers.get('Content-Type', '')
                print(f"Content-Type: {content_type}")
                print()

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

                # Success! Exit early
                return

            # Non-200 status - log and continue trying
            print(f"❌ Failed with status {response.status_code}")

        except requests.Timeout:
            print("❌ Request timed out")
        except requests.RequestException as e:
            print(f"❌ Request failed: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

    # If we get here, all attempts failed
    print(f"\n{'='*60}")
    print("All authentication attempts failed")
    print("The HAPI endpoint may not be publicly accessible")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_hapi_logs.py <pod_id>")
        print()
        print("Example:")
        print("  python test_hapi_logs.py 5fu3yfugp04ll7")
        sys.exit(1)

    pod_id = sys.argv[1]
    test_hapi_logs_endpoint(pod_id)
