#!/usr/bin/env python3
"""
Test to smoke out rate limit information from HuggingFace.
Attempts to download the model and captures detailed error information.
"""

import logging
import sys
from typing import cast

import httpx
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_rate_limit():
    """Attempt to load the model and capture all error details."""

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    print("=" * 80)
    print("RATE LIMIT SMOKE TEST")
    print("=" * 80)
    print(f"\nAttempting to load model: {model_name}")
    print("This will try to download from HuggingFace and capture error details...\n")

    try:
        model = SentenceTransformer(model_name, device='cpu')
        print("\n✅ SUCCESS: Model loaded without rate limiting!")
        print(f"Model device: {model.device}")
        print(f"Model max_seq_length: {model.max_seq_length}")
        return 0

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ ERROR CAPTURED")
        print("=" * 80)

        print(f"\nException type: {type(e).__name__}")
        print(f"\nException message:\n{str(e)}")

        # Try to extract additional info from the exception
        print("\n" + "-" * 80)
        print("DETAILED EXCEPTION ATTRIBUTES:")
        print("-" * 80)

        for attr in dir(e):
            if not attr.startswith('_'):
                try:
                    value = getattr(e, attr)
                    if not callable(value):
                        print(f"  {attr}: {value}")
                except:
                    pass

        # Check if there's response info (for HTTP errors)
        if hasattr(e, 'response'):
            print("\n" + "-" * 80)
            print("HTTP RESPONSE DETAILS:")
            print("-" * 80)
            response = cast(httpx.Response, e.response)

            print(f"  Status code: {response.status_code}")

            print("\n  Response headers:")
            for key, value in response.headers.items():
                print(f"    {key}: {value}")

            print(f"\n  Response body:\n{response.text[:1000]}")

        # Check for nested exceptions
        if hasattr(e, '__cause__') and e.__cause__:
            print("\n" + "-" * 80)
            print("ORIGINAL CAUSE:")
            print("-" * 80)
            print(f"  Type: {type(e.__cause__).__name__}")
            print(f"  Message: {str(e.__cause__)}")

            if hasattr(e.__cause__, 'response'):
                cause_response = cast(httpx.Response, e.__cause__.response)
                print("\n  Original response headers:")
                for key, value in cause_response.headers.items():
                    print(f"    {key}: {value}")

        print("\n" + "=" * 80)
        print("KEY FINDINGS:")
        print("=" * 80)

        # Extract key info
        findings = []

        if '429' in str(e):
            findings.append("✗ Rate limited (429 Too Many Requests)")

        if 'retry' in str(e).lower():
            findings.append("✓ Retry information present")
        else:
            findings.append("✗ No retry information found")

        if 'rate limit' in str(e).lower():
            findings.append("✓ Explicit rate limit message")

        if hasattr(e, 'response') and hasattr(e.response, 'headers'):
            headers = cast(httpx.Response, e.response).headers
            if 'retry-after' in headers or 'Retry-After' in headers:
                retry_after = headers.get('retry-after') or headers.get('Retry-After')
                findings.append(f"✓ Retry-After header: {retry_after}")
            else:
                findings.append("✗ No Retry-After header")

            if 'x-ratelimit-remaining' in headers or 'X-RateLimit-Remaining' in headers:
                remaining = headers.get('x-ratelimit-remaining') or headers.get('X-RateLimit-Remaining')
                findings.append(f"✓ Rate limit remaining: {remaining}")
            else:
                findings.append("✗ No rate limit remaining header")

            if 'x-ratelimit-reset' in headers or 'X-RateLimit-Reset' in headers:
                reset = headers.get('x-ratelimit-reset') or headers.get('X-RateLimit-Reset')
                findings.append(f"✓ Rate limit reset: {reset}")
            else:
                findings.append("✗ No rate limit reset header")

        for finding in findings:
            print(f"  {finding}")

        print("=" * 80)

        return 1


if __name__ == "__main__":
    sys.exit(test_rate_limit())
