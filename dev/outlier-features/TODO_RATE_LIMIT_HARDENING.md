# TODO: HuggingFace Rate Limit Hardening

## Issue
When running sweeps with many models, HuggingFace rate limits (1000 requests/5min) cause failures:
```
OSError: Client error '429 Too Many Requests'
```

## Immediate Re-run (No Code Changes)
```bash
# Wait 5+ minutes, then re-run failed models
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 07 08 12 --delay 60
```

## Code Hardening Tasks

### 1. Add Retry Logic with Exponential Backoff

**File:** `compute_perplexity.py::load_model_and_tokenizer()`

**Add:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import requests.exceptions

@retry(
    retry=retry_if_exception_type((OSError, requests.exceptions.HTTPError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    before_sleep=lambda retry_state: logger.warning(
        f"HuggingFace request failed, retrying in {retry_state.next_action.sleep} seconds..."
    )
)
def load_model_with_retry(model_name, **kwargs):
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

@retry(
    retry=retry_if_exception_type((OSError, requests.exceptions.HTTPError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5)
)
def load_tokenizer_with_retry(model_name):
    return AutoTokenizer.from_pretrained(model_name)
```

**Benefits:**
- Automatically retries on rate limit errors
- Waits longer between each retry (4s, 8s, 16s, 32s, 60s)
- Gives HuggingFace API time to reset quota
- Fails gracefully after 5 attempts

### 2. Add Rate Limit Detection and Graceful Waiting

**File:** `compute_perplexity.py::load_model_and_tokenizer()`

**Add:**
```python
def detect_rate_limit_error(exception):
    """Check if error is a rate limit error."""
    error_str = str(exception)
    return "429" in error_str or "Too Many Requests" in error_str or "rate limit" in error_str.lower()

# In load_model_and_tokenizer():
try:
    model = AutoModelForCausalLM.from_pretrained(...)
except OSError as e:
    if detect_rate_limit_error(e):
        logger.warning("Hit HuggingFace rate limit. Waiting 5 minutes before retry...")
        time.sleep(300)  # Wait 5 minutes
        model = AutoModelForCausalLM.from_pretrained(...)  # Retry once
    else:
        raise
```

### 3. Add Model Caching to Avoid Re-downloads

**File:** `deploy.py::build_analysis_command()`

**Add HF_HOME to shared volume:**
```python
# In deployment config
config.deployment.volume_disk = 100  # Use volume for model cache
hf_env += """export HF_HOME=/workspace/hf_cache && \\
mkdir -p /workspace/hf_cache && \\
"""
```

**Benefits:**
- Models cached between runs on same instance
- Reduces HuggingFace API calls
- Faster subsequent runs

### 4. Add Sequential Launch Mode to deploy_sweep.py

**File:** `deploy_sweep.py`

**Add flag:**
```python
parser.add_argument(
    "--sequential",
    action="store_true",
    help="Launch models one at a time instead of in parallel (avoids rate limits)"
)
```

**Modify launch logic:**
```python
if args.sequential:
    # Launch and wait for completion before next model
    for config_path in configs_to_deploy:
        proc = launch_deployment(...)
        if proc:
            proc.wait()  # Wait for completion
            print(f"✅ Model completed. Starting next model in {args.delay}s...")
            time.sleep(args.delay)
else:
    # Current parallel behavior
    ...
```

### 5. Add Better Error Messages

**File:** `compute_perplexity.py::main()`

**Improve exception handling:**
```python
except OSError as e:
    if "429" in str(e) or "Too Many Requests" in str(e):
        logger.error("❌ HuggingFace rate limit hit!")
        logger.error("Solutions:")
        logger.error("  1. Wait 5+ minutes and re-run this exact command")
        logger.error("  2. Use --delay 60 with deploy_sweep.py")
        logger.error("  3. Upgrade to HuggingFace PRO for higher limits")
        logger.error("  4. Use --sequential flag with deploy_sweep.py")
        return 1
    else:
        logger.error(f"✗ Model loading failed: {e}")
        return 1
```

## Dependencies to Add

```bash
uv add tenacity  # For retry logic
```

## Priority

1. **High:** Add retry logic (#1) - Fixes 90% of cases automatically
2. **Medium:** Add rate limit detection (#2) - Better UX
3. **Medium:** Add sequential mode (#4) - Prevents issue entirely
4. **Low:** Add caching (#3) - Optimization for repeated runs
5. **Low:** Better error messages (#5) - Nice to have

## Testing

After implementing, test with:
```bash
# Should complete without rate limit errors
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --sequential
```
