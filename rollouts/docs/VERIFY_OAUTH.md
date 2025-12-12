# How to Verify Claude OAuth is Being Used

This guide explains how to verify that your application is actually using Claude OAuth authentication instead of API keys.

## Quick Check: Is OAuth Working?

### 1. Check CLI Output

When running the CLI, look for the OAuth authentication message:

```bash
python -m rollouts.cli --provider anthropic --model claude-3-5-sonnet-20241022
```

**✅ OAuth is working if you see:**
```
🔐 Using OAuth authentication (Claude Pro/Max)
```

**❌ Using API key if you see no OAuth message and:**
- You have `ANTHROPIC_API_KEY` environment variable set
- Or you're passing `--api-key` flag

### 2. Check Stored Tokens

OAuth tokens are stored locally at:
```bash
~/.rollouts/oauth/anthropic.json
```

Verify the file exists and contains valid tokens:
```bash
cat ~/.rollouts/oauth/anthropic.json
```

**Expected format:**
```json
{
  "access_token": "sk-ant-oat01-...",
  "refresh_token": "sk-ant-ort01-...",
  "expires_at": 1234567890000
}
```

**Token prefixes indicate OAuth:**
- Access tokens: `sk-ant-oat01-...` (8 hour expiry)
- Refresh tokens: `sk-ant-ort01-...`
- Regular API keys: `sk-ant-api03-...` (different prefix)

### 3. Programmatic Check

Add this to your code to verify OAuth is being used:

```python
from rollouts.dtypes import Endpoint

# After creating your endpoint
endpoint = Endpoint(
    provider="anthropic",
    model="claude-3-5-sonnet-20241022",
    # ... other params
)

# Check which authentication method is being used
if endpoint.oauth_token:
    print("✅ Using OAuth authentication")
    print(f"   Token prefix: {endpoint.oauth_token[:15]}...")
elif endpoint.api_key:
    print("⚠️  Using API key authentication")
    print(f"   Key prefix: {endpoint.api_key[:10]}...")
else:
    print("❌ No authentication configured!")
```

## Detailed Verification Steps

### Step 1: Login with OAuth

If you haven't logged in yet:

```bash
# Login to Claude (Pro/Max required)
python -m rollouts.cli --login-claude

# Follow the prompts:
# 1. Opens browser to authorize
# 2. Copy the code from the callback page
# 3. Paste it in the terminal
```

### Step 2: Verify Token Storage

```bash
# Check if token file exists
ls -la ~/.rollouts/oauth/anthropic.json

# View token (shows access token prefix)
python3 << 'EOF'
import json
from pathlib import Path

token_path = Path.home() / ".rollouts" / "oauth" / "anthropic.json"
if token_path.exists():
    with open(token_path) as f:
        data = json.load(f)
    print(f"✅ OAuth tokens found")
    print(f"   Access token: {data['access_token'][:20]}...")
    print(f"   Refresh token: {data['refresh_token'][:20]}...")
    
    import time
    if time.time() * 1000 < data['expires_at']:
        print(f"   Status: Valid")
    else:
        print(f"   Status: Expired (will auto-refresh)")
else:
    print("❌ No OAuth tokens found")
EOF
```

### Step 3: Verify in API Requests

The most definitive way is to check what's being sent to the Anthropic API.

**Check the code path:**

In `rollouts/providers/anthropic.py`, the `rollout_anthropic` function checks for OAuth:

```python
# Lines 434-446
if actor.endpoint.oauth_token:
    client_kwargs: dict[str, Any] = {
        "auth_token": actor.endpoint.oauth_token,  # OAuth uses auth_token
        "max_retries": actor.endpoint.max_retries,
        "timeout": actor.endpoint.timeout,
    }
else:
    client_kwargs: dict[str, Any] = {
        "api_key": actor.endpoint.api_key,  # API key uses api_key
        "max_retries": actor.endpoint.max_retries,
        "timeout": actor.endpoint.timeout,
    }
```

**Add debug logging:**

Enable debug logging to see the authentication method:

```bash
# Run with debug logging
PYTHONPATH=. python -m rollouts.cli \
  --provider anthropic \
  --model claude-3-5-sonnet-20241022 \
  -vv
```

Or add this to your code:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed API call info including auth method
```

### Step 4: Network-Level Verification

Use network inspection to see the actual HTTP headers being sent:

**Option A: Python Debug Trace**

```python
import anthropic
anthropic.debug = True  # Enables request/response logging

# Then run your code - you'll see the headers being sent
```

**Option B: HTTP Proxy (Advanced)**

```bash
# Use mitmproxy to inspect requests
pip install mitmproxy
mitmproxy -p 8080

# In another terminal, set proxy and run
export HTTP_PROXY=http://localhost:8080
export HTTPS_PROXY=http://localhost:8080
python -m rollouts.cli --provider anthropic ...
```

Look for the `Authorization` header:
- **OAuth:** `Authorization: Bearer sk-ant-oat01-...`
- **API Key:** `x-api-key: sk-ant-api03-...`

## Common Issues

### Issue: OAuth token not being used despite login

**Symptoms:**
- You've logged in successfully
- Token file exists at `~/.rollouts/oauth/anthropic.json`
- But API calls still use API key

**Causes:**
1. Environment variable `ANTHROPIC_API_KEY` is set (takes precedence if OAuth refresh fails)
2. Using `--api-key` flag explicitly
3. OAuth token expired and auto-refresh failed

**Solution:**
```bash
# Check environment variables
env | grep ANTHROPIC

# Unset API key to force OAuth
unset ANTHROPIC_API_KEY

# Or remove from shell config (~/.bashrc, ~/.zshrc)

# Force re-login
python -m rollouts.cli --login-claude
```

### Issue: How to switch between OAuth and API key

**Use OAuth (Claude Pro/Max):**
```bash
# Ensure no API key in environment
unset ANTHROPIC_API_KEY

# Login with OAuth
python -m rollouts.cli --login-claude

# Verify
python -m rollouts.cli --provider anthropic --model claude-3-5-sonnet-20241022
# Should show: 🔐 Using OAuth authentication (Claude Pro/Max)
```

**Use API Key:**
```bash
# Set API key environment variable
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Or pass explicitly
python -m rollouts.cli --api-key sk-ant-api03-... ...
```

**The priority order is:**
1. `--api-key` flag (highest priority)
2. OAuth token (if logged in and token valid)
3. `ANTHROPIC_API_KEY` environment variable (lowest priority)

### Issue: OAuth token expired

**Symptoms:**
- Message: "⚠️  OAuth token expired and refresh failed"
- Falls back to API key if available

**Solution:**
```bash
# Re-login to get fresh tokens
python -m rollouts.cli --login-claude
```

## Test Script

Here's a complete test script to verify OAuth:

```python
#!/usr/bin/env python3
"""Test script to verify Claude OAuth is working"""
import trio
from rollouts.frontends.tui.oauth import get_oauth_client
from rollouts.dtypes import Endpoint
from rollouts.cli import make_endpoint

async def test_oauth():
    print("=" * 60)
    print("Claude OAuth Verification Test")
    print("=" * 60)
    
    # Check 1: OAuth client status
    print("\n1. Checking OAuth client...")
    client = get_oauth_client()
    tokens = client.tokens
    
    if tokens:
        print(f"   ✅ OAuth tokens found")
        print(f"   Access token: {tokens.access_token[:20]}...")
        if tokens.is_expired():
            print(f"   ⚠️  Token expired, attempting refresh...")
            try:
                tokens = await client.refresh_tokens()
                print(f"   ✅ Token refreshed successfully")
            except Exception as e:
                print(f"   ❌ Refresh failed: {e}")
        else:
            print(f"   ✅ Token valid")
    else:
        print(f"   ❌ No OAuth tokens found")
        print(f"   Run: python -m rollouts.cli --login-claude")
        return
    
    # Check 2: Endpoint configuration
    print("\n2. Checking endpoint configuration...")
    endpoint = make_endpoint(
        provider="anthropic",
        model="claude-3-5-sonnet-20241022",
        api_key=None,  # Force OAuth path
        thinking="disabled"
    )
    
    if endpoint.oauth_token:
        print(f"   ✅ Endpoint using OAuth")
        print(f"   Token: {endpoint.oauth_token[:20]}...")
    elif endpoint.api_key:
        print(f"   ⚠️  Endpoint using API key")
        print(f"   Key: {endpoint.api_key[:10]}...")
    else:
        print(f"   ❌ No authentication configured")
    
    # Check 3: Verify it's actually an OAuth token format
    print("\n3. Verifying token format...")
    if endpoint.oauth_token:
        if endpoint.oauth_token.startswith("sk-ant-oat01-"):
            print(f"   ✅ Correct OAuth token format (sk-ant-oat01-*)")
        elif endpoint.oauth_token.startswith("sk-ant-api"):
            print(f"   ⚠️  This is an API key, not OAuth token!")
        else:
            print(f"   ⚠️  Unknown token format")
    
    print("\n" + "=" * 60)
    if endpoint.oauth_token and endpoint.oauth_token.startswith("sk-ant-oat01-"):
        print("✅ OAuth verification PASSED")
        print("\nYou are using Claude OAuth authentication!")
    else:
        print("❌ OAuth verification FAILED")
        print("\nYou are NOT using OAuth. Check the steps above.")
    print("=" * 60)

if __name__ == "__main__":
    trio.run(test_oauth)
```

Save this as `test_oauth.py` and run:

```bash
python test_oauth.py
```

## API Request Headers

When OAuth is working, the Anthropic SDK will send these headers:

```
Authorization: Bearer sk-ant-oat01-...
anthropic-beta: oauth-2025-04-20,prompt-caching-2024-07-31
anthropic-version: 2023-06-01
```

When using API key:

```
x-api-key: sk-ant-api03-...
anthropic-beta: prompt-caching-2024-07-31
anthropic-version: 2023-06-01
```

Notice:
- OAuth uses `Authorization: Bearer` header
- OAuth includes `oauth-2025-04-20` in `anthropic-beta` header
- API key uses `x-api-key` header

This is implemented in `rollouts/providers/anthropic.py` lines 538-541:

```python
extra_headers = {"anthropic-beta": "prompt-caching-2024-07-31"}
if actor.endpoint.oauth_token:
    extra_headers["anthropic-beta"] = "oauth-2025-04-20,prompt-caching-2024-07-31"
```

## Summary Checklist

- [ ] Logged in with `--login-claude`
- [ ] Token file exists at `~/.rollouts/oauth/anthropic.json`
- [ ] No `ANTHROPIC_API_KEY` environment variable set
- [ ] CLI shows "🔐 Using OAuth authentication (Claude Pro/Max)"
- [ ] Token prefix is `sk-ant-oat01-` (not `sk-ant-api03-`)
- [ ] Endpoint has `oauth_token` set (not just `api_key`)
- [ ] API requests include `anthropic-beta: oauth-2025-04-20` header

If all checkboxes pass, you're successfully using Claude OAuth! 🎉
