# OAuth Authentication Guide

This guide explains how Rollouts uses OAuth vs API keys for Anthropic/Claude, and how to verify which one you're using.

## TL;DR - Am I Getting Invoiced?

Run this:
```bash
python check_auth.py
```

If you see `✅ ✅ ✅ USING OAUTH ✅ ✅ ✅` → You're NOT getting invoiced ✅  
If you see `❌ ❌ ❌ USING API KEY ❌ ❌ ❌` → You ARE getting invoiced 💳

## The Key Indicator

When running any Rollouts CLI command, look for this message:

```
🔐 Using OAuth authentication (Claude Pro/Max)
```

**If you see this** → OAuth is being used, no invoice  
**If you DON'T see this** → API key is being used, you'll get invoiced

## How Authentication Works

### Phase 1: Endpoint Creation

```
1. Is --api-key flag provided?
   YES → Use API key (skip OAuth) → YOU GET INVOICED
   NO  → Continue to step 2

2. Try to load OAuth tokens from ~/.rollouts/oauth/anthropic.json
   Found & valid → Use OAuth → NO INVOICE
   Not found/expired → Continue to step 3

3. Fall back to ANTHROPIC_API_KEY environment variable
   Set → Use API key → YOU GET INVOICED
   Not set → No authentication (will fail)
```

### Phase 2: Provider Makes API Call

In `rollouts/providers/anthropic.py`:

```python
if actor.endpoint.oauth_token:
    # OAuth path - NO INVOICE
    client = AsyncAnthropic(auth_token=oauth_token)
    # Header: Authorization: Bearer sk-ant-oat01-...
    # Uses: Claude Pro/Max subscription
else:
    # API key path - YOU GET INVOICED
    client = AsyncAnthropic(api_key=api_key)
    # Header: x-api-key: sk-ant-api03-...
    # Uses: Anthropic API billing account
```

## OAuth Setup

### Login

```bash
python -m rollouts.cli --login-claude
```

This will:
1. Open a browser to authorize
2. Give you a code to paste back
3. Store tokens in `~/.rollouts/oauth/anthropic.json`

### Check Status

```bash
# Quick check
ls ~/.rollouts/oauth/anthropic.json

# Detailed check
python check_auth.py
```

### Logout

```bash
python -m rollouts.cli --logout-claude
```

## Token Formats

| Token Prefix | Type | Billing |
|--------------|------|---------|
| `sk-ant-oat01-...` | OAuth Access Token | Claude Pro/Max (no invoice) |
| `sk-ant-ort01-...` | OAuth Refresh Token | N/A |
| `sk-ant-api03-...` | API Key | Anthropic API (invoiced) |

## Common Scenarios

### ✅ Using OAuth (No Invoice)

```bash
# No --api-key flag, no ANTHROPIC_API_KEY env var
python -m rollouts.cli --model anthropic/claude-3-5-sonnet-20241022 -p "test"
```

**Output includes:**
```
🔐 Using OAuth authentication (Claude Pro/Max)
```

**Result:** Uses Claude Pro/Max subscription, no invoice

### ❌ Using API Key (Gets Invoiced)

**Scenario 1: Explicit --api-key flag**
```bash
python -m rollouts.cli --api-key sk-ant-api03-... --model anthropic/... -p "test"
```

**No OAuth message shown** → Uses API key → Invoice

**Scenario 2: Environment variable fallback**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-..."
python -m rollouts.cli --model anthropic/... -p "test"
```

If OAuth tokens don't exist or are expired → Falls back to env var → Invoice

## How to Ensure OAuth is Used

### 1. Remove API key from environment

```bash
# Check what's set
env | grep ANTHROPIC

# Unset if present
unset ANTHROPIC_API_KEY

# Remove from shell config if needed
# ~/.bashrc, ~/.zshrc, ~/.bash_profile, etc.
```

### 2. Login with OAuth

```bash
python -m rollouts.cli --login-claude
```

### 3. Verify it's working

```bash
python check_auth.py
```

Should show: `✅ ✅ ✅ USING OAUTH ✅ ✅ ✅`

### 4. Check every CLI run

Always look for the message:
```
🔐 Using OAuth authentication (Claude Pro/Max)
```

If you don't see it, you're getting invoiced!

## Testing Different Auth Methods

### Test current configuration
```bash
python check_auth.py
```

### Test that API key path works (uses invalid key)
```bash
python check_auth.py --test-api-key
```

This will try to make a call with an invalid API key and get a 401 error, proving the API key path actually works.

### Test with JSON mode
```bash
python check_auth.py --test-json
```

## Programmatic Usage

### Check in Python code

```python
from rollouts.cli import create_endpoint

endpoint = create_endpoint(
    'anthropic/claude-3-5-sonnet-20241022',
    api_key=None,  # Don't override OAuth
    thinking="disabled"
)

# Verify OAuth is being used
assert endpoint.oauth_token, "OAuth not configured!"
assert endpoint.oauth_token.startswith("sk-ant-oat01-"), "Not an OAuth token!"
print(f"✅ Using OAuth: {endpoint.oauth_token[:20]}...")
```

### Force API key (e.g., for testing)

```python
endpoint = create_endpoint(
    'anthropic/claude-3-5-sonnet-20241022',
    api_key="sk-ant-api03-...",  # Explicitly pass API key
    thinking="disabled"
)
# This will use API key and you'll get invoiced
```

## Troubleshooting

### "I'm getting invoiced but thought I was using OAuth"

**Check:**
1. Do you see `🔐 Using OAuth authentication` in CLI output?
   - NO → You're not using OAuth
2. Is `ANTHROPIC_API_KEY` environment variable set?
   - YES → This is being used as fallback
3. Are OAuth tokens valid?
   ```bash
   python check_auth.py
   ```

**Fix:**
```bash
unset ANTHROPIC_API_KEY
python -m rollouts.cli --login-claude
python check_auth.py  # Verify
```

### "OAuth token expired and refresh failed"

**Solution:**
```bash
python -m rollouts.cli --login-claude
```

This will get fresh tokens.

### "How do I switch between OAuth and API key?"

**Use OAuth:**
- Don't use `--api-key` flag
- Make sure OAuth is logged in
- Optionally: `unset ANTHROPIC_API_KEY`

**Use API key:**
- Use `--api-key` flag, OR
- Set `ANTHROPIC_API_KEY` environment variable, OR
- Pass `api_key` parameter in code

## Summary

**You get invoiced if:**
- You use `--api-key` flag
- OAuth tokens don't exist/expired AND `ANTHROPIC_API_KEY` is set
- You don't see the `🔐 Using OAuth authentication` message

**You DON'T get invoiced if:**
- You see the `🔐 Using OAuth authentication` message
- `endpoint.oauth_token` starts with `sk-ant-oat01-`
- OAuth tokens are valid and refresh works

**Always verify with:**
```bash
python check_auth.py
```
