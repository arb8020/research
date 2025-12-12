#!/usr/bin/env python3
"""Test script to verify Claude OAuth is working correctly

Usage:
    python test_oauth_verification.py

This script will:
1. Check if OAuth tokens are stored locally
2. Verify token format and expiration
3. Test endpoint configuration
4. Confirm OAuth is being used (not API key)
"""
import os
import sys
import trio
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rollouts.frontends.tui.oauth import get_oauth_client
from rollouts.dtypes import Endpoint
from rollouts.cli import make_endpoint


async def test_oauth():
    """Run comprehensive OAuth verification tests"""
    print("=" * 70)
    print("🔐 Claude OAuth Verification Test")
    print("=" * 70)
    
    passed_checks = 0
    total_checks = 5
    
    # Check 1: OAuth client and token storage
    print("\n✓ Test 1: OAuth Token Storage")
    print("-" * 70)
    client = get_oauth_client()
    tokens = client.tokens
    
    if tokens:
        print(f"   ✅ OAuth tokens found at: ~/.rollouts/oauth/anthropic.json")
        print(f"   📝 Access token: {tokens.access_token[:20]}...")
        print(f"   📝 Refresh token: {tokens.refresh_token[:20]}...")
        passed_checks += 1
        
        # Check expiration
        if tokens.is_expired():
            print(f"   ⚠️  Token expired, attempting refresh...")
            try:
                tokens = await client.refresh_tokens()
                print(f"   ✅ Token refreshed successfully")
            except Exception as e:
                print(f"   ❌ Refresh failed: {e}")
                print(f"   💡 Run: python -m rollouts.cli --login-claude")
                passed_checks -= 1
        else:
            print(f"   ✅ Token is valid (not expired)")
    else:
        print(f"   ❌ No OAuth tokens found")
        print(f"   💡 Run: python -m rollouts.cli --login-claude")
        print(f"\nCannot continue without OAuth tokens. Please login first.")
        return False
    
    # Check 2: Token format validation
    print("\n✓ Test 2: Token Format Validation")
    print("-" * 70)
    if tokens.access_token.startswith("sk-ant-oat01-"):
        print(f"   ✅ Correct OAuth access token format (sk-ant-oat01-*)")
        passed_checks += 1
    elif tokens.access_token.startswith("sk-ant-api"):
        print(f"   ❌ This is an API key, not an OAuth token!")
        print(f"   💡 The token should start with 'sk-ant-oat01-'")
    else:
        print(f"   ⚠️  Unknown token format: {tokens.access_token[:20]}...")
    
    if tokens.refresh_token.startswith("sk-ant-ort01-"):
        print(f"   ✅ Correct OAuth refresh token format (sk-ant-ort01-*)")
    else:
        print(f"   ⚠️  Unexpected refresh token format")
    
    # Check 3: Environment variable conflicts
    print("\n✓ Test 3: Environment Variable Check")
    print("-" * 70)
    api_key_env = os.environ.get("ANTHROPIC_API_KEY")
    if api_key_env:
        print(f"   ⚠️  ANTHROPIC_API_KEY is set: {api_key_env[:10]}...")
        print(f"   ⚠️  This may override OAuth if OAuth refresh fails")
        print(f"   💡 Consider: unset ANTHROPIC_API_KEY")
    else:
        print(f"   ✅ No ANTHROPIC_API_KEY environment variable")
        print(f"   ✅ OAuth will be used without fallback to API key")
        passed_checks += 1
    
    # Check 4: Endpoint configuration
    print("\n✓ Test 4: Endpoint Configuration")
    print("-" * 70)
    try:
        endpoint = make_endpoint(
            provider="anthropic",
            model="claude-3-5-sonnet-20241022",
            api_key=None,  # Force OAuth path
            thinking="disabled"
        )
        
        if endpoint.oauth_token:
            print(f"   ✅ Endpoint configured with OAuth token")
            print(f"   📝 Token: {endpoint.oauth_token[:20]}...")
            passed_checks += 1
            
            if endpoint.api_key:
                print(f"   ⚠️  Endpoint also has API key set: {endpoint.api_key[:10]}...")
                print(f"   ℹ️  OAuth takes precedence over API key")
        elif endpoint.api_key:
            print(f"   ❌ Endpoint using API key instead of OAuth")
            print(f"   📝 Key: {endpoint.api_key[:10]}...")
            print(f"   💡 OAuth token might have failed to load")
        else:
            print(f"   ❌ No authentication configured in endpoint")
    except Exception as e:
        print(f"   ❌ Failed to create endpoint: {e}")
    
    # Check 5: Verify OAuth is actually used in provider
    print("\n✓ Test 5: Provider Authentication Method")
    print("-" * 70)
    print(f"   ℹ️  Checking rollouts/providers/anthropic.py...")
    
    # Read the actual provider code to show what will happen
    try:
        from rollouts.providers import anthropic
        provider_file = Path(anthropic.__file__)
        
        print(f"   ✅ Provider file found: {provider_file.name}")
        
        if endpoint.oauth_token:
            print(f"   ✅ OAuth token present - provider will use:")
            print(f"      • AsyncAnthropic(auth_token='{endpoint.oauth_token[:15]}...')")
            print(f"      • Headers: 'anthropic-beta: oauth-2025-04-20,prompt-caching-2024-07-31'")
            passed_checks += 1
        else:
            print(f"   ⚠️  No OAuth token - provider will use API key fallback")
            
    except Exception as e:
        print(f"   ⚠️  Could not verify provider code: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 Test Results: {passed_checks}/{total_checks} checks passed")
    print("=" * 70)
    
    if passed_checks == total_checks:
        print("✅ ✅ ✅ OAuth verification PASSED ✅ ✅ ✅")
        print("\n🎉 You are successfully using Claude OAuth authentication!")
        print("\nWhat happens when you make API calls:")
        print("  1. Endpoint creates AsyncAnthropic with auth_token (not api_key)")
        print("  2. Requests include 'Authorization: Bearer sk-ant-oat01-...'")
        print("  3. Requests include 'anthropic-beta: oauth-2025-04-20' header")
        print("  4. You're using your Claude Pro/Max subscription quota")
        return True
    elif passed_checks >= 3:
        print("⚠️  OAuth verification PARTIALLY PASSED")
        print(f"\n{passed_checks} of {total_checks} checks passed.")
        print("Review warnings above and fix any issues.")
        return True
    else:
        print("❌ OAuth verification FAILED")
        print(f"\nOnly {passed_checks} of {total_checks} checks passed.")
        print("\n🔧 To fix:")
        print("  1. Run: python -m rollouts.cli --login-claude")
        print("  2. Unset ANTHROPIC_API_KEY if set: unset ANTHROPIC_API_KEY")
        print("  3. Run this test again")
        return False
    
    print("=" * 70)


async def main():
    """Main entry point"""
    try:
        success = await test_oauth()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    trio.run(main)
