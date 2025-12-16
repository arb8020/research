"""
Capture LeetGPU submission API requests using Playwright.
This script intercepts network requests to understand the API format.
"""

import asyncio
import json

from playwright.async_api import async_playwright


async def capture_api_requests():
    """Launch browser and capture API requests to LeetGPU."""

    captured_requests = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # Intercept all requests - simpler approach
        async def handle_request(request):
            # Capture ALL requests to api.leetgpu.com
            if "api.leetgpu.com" in request.url:
                print(f"\n🔍 Request: {request.method} {request.url}")

                if request.method == "POST":
                    print(f"Headers: {json.dumps(dict(request.headers), indent=2)}")

                    # Get the request body
                    post_data = request.post_data
                    if post_data:
                        print(f"\n📤 Request Body (raw):\n{post_data}\n")
                        try:
                            body_json = json.loads(post_data)
                            print(
                                f"📤 Request Body (formatted):\n{json.dumps(body_json, indent=2)}\n"
                            )
                        except:
                            pass

                    captured_requests.append({
                        "url": request.url,
                        "method": request.method,
                        "headers": dict(request.headers),
                        "body": post_data,
                    })

        # Also listen to responses
        async def handle_response(response):
            if "api.leetgpu.com" in response.url:
                print(f"✅ Response: {response.status} {response.url}")
                if response.request.method == "POST":
                    try:
                        body = await response.text()
                        print(f"📥 Response Body:\n{body}\n")
                        try:
                            response_json = json.loads(body)
                            print(
                                f"📥 Response Body (formatted):\n{json.dumps(response_json, indent=2)}\n"
                            )
                        except:
                            pass
                    except:
                        pass

        page.on("request", handle_request)
        page.on("response", handle_response)

        # Navigate to LeetGPU
        print("🌐 Opening LeetGPU...")
        await page.goto("https://leetgpu.com/challenges")

        print("\n📋 Instructions:")
        print("1. Navigate to a challenge")
        print("2. Submit some code")
        print("3. The API request will be captured and printed above")
        print("4. Press Ctrl+C when done\n")

        # Wait for user to interact
        try:
            await page.wait_for_timeout(300000)  # 5 minutes
        except KeyboardInterrupt:
            pass

        await browser.close()

    # Save captured requests
    if captured_requests:
        output_file = "leetgpu_api_captured.json"
        with open(output_file, "w") as f:
            json.dump(captured_requests, f, indent=2)
        print(f"\n💾 Saved captured requests to {output_file}")

    return captured_requests


if __name__ == "__main__":
    asyncio.run(capture_api_requests())
