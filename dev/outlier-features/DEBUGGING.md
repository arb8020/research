# Debugging Guide

## Quick Status Check

**Check most recent sweep:**
```bash
python deploy_sweep.py --status
```

**Check specific sweep:**
```bash
python deploy_sweep.py --status --sweep sweep_20251029_150035_sweep_perplexity_smoke
```

**Check which models succeeded/failed:**
```bash
cd logs/sweep_<timestamp>_<config_dir>
for log in *.log; do
    if grep -q "🎉 Deployment complete" "$log"; then
        echo "✅ $log"
    else
        echo "❌ $log"
    fi
done
```

## Common Errors

### HuggingFace Rate Limit (429 Too Many Requests)

**Symptoms:**
```
OSError: Client error '429 Too Many Requests'
We had to rate limit you, you hit the quota of 1000 api requests per 5 minutes period.
```

**Why it happens:**
- Launching many models simultaneously all download from HuggingFace at once
- Free tier limit: 1000 requests / 5 minutes
- Each model download can trigger hundreds of requests

**Solutions:**

1. **Wait and re-run failed models:**
```bash
# Wait 5+ minutes after sweep completes
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 07 08 12
```

2. **Increase delay between launches:**
```bash
# Use 60s delay instead of default 30s
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --delay 60 --models 07 08 12
```

3. **Run individually with manual delays:**
```bash
python deploy.py sweep_perplexity_smoke/07_qwen_next_80b.py --mode perplexity
sleep 120  # Wait 2 minutes
python deploy.py sweep_perplexity_smoke/08_glm_45_air.py --mode perplexity
sleep 120
python deploy.py sweep_perplexity_smoke/12_qwen3_14b.py --mode perplexity
```

4. **Upgrade to HuggingFace PRO** (higher rate limits)

**Code hardening needed:**
- [ ] Add retry logic with exponential backoff in `load_model_and_tokenizer()`
- [ ] Add HF rate limit detection and graceful waiting
- [ ] Cache models to avoid re-downloading
- [ ] Add `--sequential` flag to deploy_sweep.py to launch models one at a time

### Results File Not Found

**Symptoms:**
```
bifrost.types.TransferError: Remote path not found: ~/.bifrost/workspace/dev/outlier-features/results/perplexity_results.json
```

**Why it happens:**
- Script failed before reaching save_results()
- Usually caused by earlier error (check logs for root cause)

**How to debug:**
```bash
# Find the real error:
grep -B20 "ANALYSIS FAILED" logs/sweep_*/model_name.log
```

### Model Loading Failed

**Symptoms:**
```
OSError: There was a specific connection error when trying to load <model_name>
```

**Common causes:**
1. HuggingFace rate limit (see above)
2. Model name typo in config
3. Network/connectivity issues
4. Insufficient disk space

**How to debug:**
```bash
# Check if model exists on HuggingFace
curl -I https://huggingface.co/api/models/<model_name>

# Check remote disk space
bifrost exec '<ssh>' 'df -h'
```

## Re-running Failed Models

### From Sweep Logs

1. **Identify failed models:**
```bash
cd logs/sweep_<timestamp>_<config_dir>
for log in *.log; do
    if ! grep -q "🎉 Deployment complete" "$log"; then
        echo "Failed: $log"
        # Extract model number (first 2 chars of filename)
        echo "  Model: ${log:0:2}"
    fi
done
```

2. **Re-run just those models:**
```bash
# Example: models 07, 08, 12 failed
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 07 08 12
```

### Individual Model Re-run

```bash
# Run single model with verbose output
python deploy.py sweep_perplexity_smoke/07_qwen_next_80b.py --mode perplexity
```

## Smoke Test Status (2025-10-29)

### ✅ Succeeded (9/12):
- 01_olmoe_1b_7b
- 02_qwen3_0.6b
- 03_qwen3_1.7b
- 04_gpt_oss_20b
- 05_qwen3_30b
- 06_mixtral_8x7b
- 09_gpt_oss_120b
- 10_qwen3_4b
- 11_qwen3_8b

### ❌ Failed - HuggingFace Rate Limit (3/12):
- 07_qwen_next_80b - 429 Too Many Requests
- 08_glm_45_air - 429 Too Many Requests
- 12_qwen3_14b - 429 Too Many Requests

**Re-run command:**
```bash
# Wait 5+ minutes after initial sweep, then:
python deploy_sweep.py --config-dir sweep_perplexity_smoke --mode perplexity --models 07 08 12 --delay 60
```

## Monitoring Active Runs

**Check GPU instances:**
```bash
broker list
```

**Watch logs in real-time:**
```bash
# All models
tail -f logs/sweep_<timestamp>_<config_dir>/*.log

# Specific model
tail -f logs/sweep_<timestamp>_<config_dir>/01_olmoe_1b_7b.log
```

**Check remote progress:**
```bash
bifrost exec '<ssh>' 'cd ~/.bifrost/workspace/dev/outlier-features && tail -20 perplexity_computation.log'
```

## Results Verification

**Check what succeeded:**
```bash
ls -1 remote_results/perplexity_smoke_*/perplexity_results.json
```

**Quick perplexity summary:**
```bash
for file in remote_results/perplexity_smoke_*/perplexity_results.json; do
    model=$(jq -r '.model' "$file" 2>/dev/null)
    ppl=$(jq -r '.perplexity' "$file" 2>/dev/null)
    if [ -n "$ppl" ]; then
        echo "$model: $ppl"
    fi
done | sort
```
