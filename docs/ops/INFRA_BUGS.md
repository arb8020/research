# Infrastructure Bugs & Improvements

Found during code review. None of these are blocking - the code works - but should be fixed eventually.

## Actual Bugs

*None currently*

### ~~1. Hardcoded `/root` for tilde expansion~~ (FIXED)
Added `_get_remote_home()` that queries `$HOME` on first use and caches it. `_expand_remote_tilde()` uses this instead of hardcoded `/root`.

## Tech Debt

### 4. deploy.py scripts are 80% duplicated
- `dev/integration-evaluation/deploy.py` (652 lines)
- `dev/integration_training/deploy.py` (511 lines)

Both do: load config → check prerequisites → deploy code → setup deps → start tmux → stream logs → sync results. Should extract a shared `Deployer` class.

### 5. Sync/Async client duplication
- `bifrost/bifrost/client.py` (~1,200 lines)
- `bifrost/bifrost/async_client.py` (~1,000 lines)

90% identical, will drift apart as bugs are fixed in one but not the other. Consider a shared base or code generation.

## Low Priority

### 7. Credential env var naming inconsistency
```python
PRIME_API_KEY -> credentials["primeintellect"]
LAMBDA_API_KEY -> credentials["lambdalabs"]
```

The mapping exists in multiple places. Could standardize or use a single source of truth.

### 8. Emoji logging
Every log line has an emoji. Looks nice in terminal, annoying when grepping logs at 3am.
