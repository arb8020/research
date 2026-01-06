#!/usr/bin/env python3
"""Quick test for /model switching logic."""

import os
from dataclasses import replace as dc_replace

from rollouts.dtypes import Endpoint

def test_model_switch():
    # Simulate starting with Anthropic endpoint (like the TUI does)
    old_endpoint = Endpoint(
        provider="anthropic",
        model="claude-sonnet-4-20250514",
        api_base="https://api.anthropic.com",
        api_key="sk-ant-xxx",
        oauth_token="oauth-token-here",
        max_tokens=16384,
        temperature=1.0,
        thinking={"type": "enabled", "budget_tokens": 10000},
    )
    
    print("=== OLD ENDPOINT ===")
    print(f"  provider: {old_endpoint.provider}")
    print(f"  model: {old_endpoint.model}")
    print(f"  api_base: {old_endpoint.api_base}")
    print(f"  api_key: {old_endpoint.api_key[:10]}..." if old_endpoint.api_key else "  api_key: (none)")
    print(f"  oauth_token: {old_endpoint.oauth_token[:10]}..." if old_endpoint.oauth_token else "  oauth_token: (none)")
    print(f"  thinking: {old_endpoint.thinking}")
    print()
    
    # Simulate /model openai/gpt-4o
    new_provider = "openai"
    new_model = "gpt-4o"
    
    # --- THIS IS THE CODE FROM slash_commands.py ---
    if old_endpoint.provider == new_provider:
        new_api_key = old_endpoint.api_key
        new_oauth_token = old_endpoint.oauth_token
    else:
        new_oauth_token = ""
        if new_provider == "openai":
            new_api_key = os.environ.get("OPENAI_API_KEY", "")
        elif new_provider == "anthropic":
            new_api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        elif new_provider == "google":
            new_api_key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        else:
            new_api_key = os.environ.get(f"{new_provider.upper()}_API_KEY", "")
    
    new_endpoint = Endpoint(
        provider=new_provider,
        model=new_model,
        max_tokens=old_endpoint.max_tokens,
        temperature=old_endpoint.temperature,
        max_retries=old_endpoint.max_retries,
        timeout=old_endpoint.timeout,
        api_key=new_api_key,
        oauth_token=new_oauth_token,
    )
    # --- END CODE FROM slash_commands.py ---
    
    print("=== NEW ENDPOINT ===")
    print(f"  provider: {new_endpoint.provider}")
    print(f"  model: {new_endpoint.model}")
    print(f"  api_base: '{new_endpoint.api_base}' (should be empty)")
    print(f"  api_key: {new_endpoint.api_key[:10]}..." if new_endpoint.api_key else "  api_key: (none) ← PROBLEM!")
    print(f"  oauth_token: '{new_endpoint.oauth_token}' (should be empty)")
    print(f"  thinking: {new_endpoint.thinking} (should be None)")
    print()
    
    # Validate
    errors = []
    if new_endpoint.api_base != "":
        errors.append(f"api_base should be empty, got '{new_endpoint.api_base}'")
    if new_endpoint.thinking is not None:
        errors.append(f"thinking should be None, got {new_endpoint.thinking}")
    if new_endpoint.oauth_token != "":
        errors.append(f"oauth_token should be empty, got '{new_endpoint.oauth_token}'")
    if not new_endpoint.api_key:
        errors.append("api_key is empty - check OPENAI_API_KEY env var")
    
    if errors:
        print("❌ ERRORS:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("✅ All checks passed!")
    
    print()
    print("=== ENV VARS ===")
    print(f"  OPENAI_API_KEY: {'set' if os.environ.get('OPENAI_API_KEY') else 'NOT SET'}")
    print(f"  ANTHROPIC_API_KEY: {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET'}")

if __name__ == "__main__":
    test_model_switch()
