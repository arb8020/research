# OAuth Authentication Improvements

> Fix OAuth to work reliably across the CLI, with clear feedback on auth method used.

## Current State

OAuth authentication exists in `rollouts/frontends/tui/oauth.py` and is used by the CLI:

```python
# cli.py:create_endpoint()
if api_key is None and provider == "anthropic":
    client = get_oauth_client()
    tokens = client.tokens
    if tokens:
        oauth_token = tokens.access_token
        print("🔐 Using OAuth authentication (Claude Pro/Max)", file=sys.stderr)
    if not oauth_token:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
```

The provider correctly handles OAuth when present:
```python
# providers/anthropic.py
if actor.endpoint.oauth_token:
    client_kwargs = {"auth_token": actor.endpoint.oauth_token, ...}
else:
    client_kwargs = {"api_key": actor.endpoint.api_key, ...}
```

## Problems

### 1. Model Availability Mismatch

OAuth tokens only work with certain models. Testing reveals:

| Model | OAuth | API Key |
|-------|-------|---------|
| `claude-opus-4-5-20251101` | ✅ | ✅ |
| `claude-sonnet-4-20250514` | ✅ | ✅ |
| `claude-3-5-haiku-20241022` | ✅ | ✅ |
| `claude-3-5-sonnet-20241022` | ❌ 404 | ✅ |
| `claude-sonnet-4-5-20250514` | ❌ 404 | ✅ |

When OAuth is used with an unsupported model, the API returns 404 "model not found" which retries 10 times before failing. No fallback to API key occurs.

### 2. No Validation at Startup

The message "🔐 Using OAuth authentication" prints if a token exists, but doesn't verify:
- Token is actually valid (could be revoked)
- Selected model is available via OAuth
- API call will succeed

User sees success message, then gets cryptic 404 errors later.

### 3. Non-CLI Paths Don't Use OAuth

These paths create `Endpoint` directly without OAuth:
- `run_eval.py` - evaluation runs
- `config/base.py:BaseModelConfig.to_endpoint()` - config-based endpoints
- `training/grpo.py` - training loops
- `frontend/server.py` - web UI

```python
# run_eval.py:156-163 - no oauth_token
endpoint = Endpoint(
    provider=config.provider,
    model=config.model_name,
    api_key=api_key,  # Always uses API key
    ...
)
```

### 4. Silent Fallback at Endpoint Creation

If OAuth refresh fails, code falls back to `ANTHROPIC_API_KEY` with only a warning:
```python
except Exception as e:
    print(f"⚠️  OAuth token expired and refresh failed: {e}")
    oauth_token = ""  # Falls through to api_key lookup
```

User might not notice the warning and get invoiced unexpectedly.

### 5. No Way to Verify Auth Method After the Fact

No logging of which auth method was actually used for each API call. If user gets charged, hard to determine which calls used API key vs OAuth.

## Proposed Design

### 1. Validate OAuth + Model Compatibility at Startup

Add model allowlist for OAuth, fail fast if incompatible:

```python
# oauth.py or cli.py
OAUTH_SUPPORTED_MODELS = {
    "claude-opus-4-5-20251101",
    "claude-sonnet-4-20250514",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    # Add more as discovered
}

def validate_oauth_model(model: str, oauth_token: str) -> tuple[bool, str]:
    """Check if model works with OAuth.

    Returns (ok, error_message).
    """
    if model not in OAUTH_SUPPORTED_MODELS:
        return False, f"Model '{model}' not available via OAuth. Use --api-key or choose: {OAUTH_SUPPORTED_MODELS}"
    return True, ""
```

### 2. Add `--auth-method` Flag for Explicit Control

```python
parser.add_argument(
    "--auth-method",
    choices=["oauth", "api-key", "auto"],
    default="auto",
    help="Authentication method: oauth (Claude Pro/Max), api-key (ANTHROPIC_API_KEY), auto (try oauth first)"
)
```

Behavior:
- `--auth-method oauth`: Require OAuth, fail if not available
- `--auth-method api-key`: Use ANTHROPIC_API_KEY only
- `--auth-method auto`: Current behavior (OAuth if available, fallback to API key)

### 3. Log Auth Method Per API Call

Add logging in provider to track which auth was used:

```python
# providers/anthropic.py
if actor.endpoint.oauth_token:
    logger.info(f"API call using OAuth (model={actor.endpoint.model})")
    client_kwargs = {"auth_token": actor.endpoint.oauth_token, ...}
else:
    logger.info(f"API call using API key (model={actor.endpoint.model})")
    client_kwargs = {"api_key": actor.endpoint.api_key, ...}
```

### 4. Centralize Endpoint Creation

Create single function for endpoint creation that all paths use:

```python
# endpoints.py (new file)
def create_anthropic_endpoint(
    model: str,
    api_key: str | None = None,
    auth_method: str = "auto",
    **kwargs
) -> Endpoint:
    """Create Anthropic endpoint with proper auth handling.

    Used by CLI, eval, training, etc. Single source of truth.
    """
    oauth_token = ""

    if auth_method in ("oauth", "auto"):
        client = get_oauth_client()
        tokens = client.tokens
        if tokens and not tokens.is_expired():
            # Validate model compatibility
            ok, err = validate_oauth_model(model, tokens.access_token)
            if ok:
                oauth_token = tokens.access_token
                print("🔐 Using OAuth authentication (Claude Pro/Max)")
            elif auth_method == "oauth":
                raise ValueError(err)
            # else: auto mode, fall through to API key

    if not oauth_token:
        if auth_method == "oauth":
            raise ValueError("OAuth required but not available")
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if api_key:
            print("🔑 Using API key authentication (will be invoiced)")

    return Endpoint(
        provider="anthropic",
        model=model,
        api_key=api_key,
        oauth_token=oauth_token,
        **kwargs
    )
```

### 5. Update check_auth.py to Actually Test API Call

Current `check_auth.py` only checks if token exists. Add actual API call verification:

```python
def verify_oauth_works(model: str = "claude-opus-4-5-20251101") -> bool:
    """Make test API call to verify OAuth actually works."""
    try:
        client = Anthropic(auth_token=oauth_token)
        response = client.messages.create(
            model=model,
            max_tokens=5,
            messages=[{"role": "user", "content": "hi"}],
            extra_headers={"anthropic-beta": "oauth-2025-04-20"}
        )
        return True
    except Exception as e:
        print(f"OAuth verification failed: {e}")
        return False
```

## Implementation Plan

1. **Add model allowlist** - Quick fix, validates at startup
2. **Add `--auth-method` flag** - Explicit user control
3. **Centralize endpoint creation** - Single source of truth
4. **Add per-call logging** - Debug auth issues
5. **Update check_auth.py** - Real verification

## Open Questions

- Should we auto-discover OAuth-supported models by testing each one?
- Should non-CLI paths (eval, training) support OAuth at all?
- Should we warn loudly when falling back from OAuth to API key?
