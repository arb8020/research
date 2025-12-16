# Infrastructure Bugs & Improvements

Found during code review. None of these are blocking - the code works - but should be fixed eventually.

## Actual Bugs

### 1. Wrong variable in assertions (FIXED)
**Files**: `kerbal/kerbal/transfer.py`, `kerbal/kerbal/gpu.py`

```python
def push_code(client: "BifrostClient", ...):
    assert bifrost is not None  # WRONG - checks undefined var, should be `client`
```

This would raise `NameError` before the assertion runs. Fixed in this session.

### 2. Hardcoded `/root` for tilde expansion
**File**: `bifrost/bifrost/client.py:995, 1033`

```python
if remote_path.startswith("~/"):
    remote_path = remote_path.replace("~", "/root", 1)
```

Breaks on any non-root user (Lambda Labs uses `ubuntu`, etc). Should query actual home dir on first connection and cache it.

### 3. kerbal/transfer.py `push_code` signature mismatch
**File**: `kerbal/kerbal/transfer.py:54`

```python
workspace = client.push(local_path=local_path, remote_path=remote_path)
```

But `BifrostClient.push()` signature is:
```python
def push(self, workspace_path: str, bootstrap_cmd: str | list[str] | None = None) -> str:
```

There's no `local_path` or `remote_path` param. This function would crash if called.

## Tech Debt

### 4. deploy.py scripts are 80% duplicated
- `dev/integration-evaluation/deploy.py` (652 lines)
- `dev/integration_training/deploy.py` (511 lines)

Both do: load config → check prerequisites → deploy code → setup deps → start tmux → stream logs → sync results. Should extract a shared `Deployer` class.

### 5. TODO that ignores its parameter
**File**: `kerbal/kerbal/transfer.py:49-52`

```python
if exclude:
    # TODO: client.push needs to support exclude parameter
    logger.warning("Exclude patterns not yet supported by client.push()")
```

The `exclude` param exists in the function signature but is silently ignored.

### 6. Sync/Async client duplication
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
