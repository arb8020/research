# Config Fingerprinting

**DRI:**
**Claude:** (this conversation)

## Context

Prevent silent config drift on long pretraining runs by hashing config at training start and validating on resume.

## Out of Scope
- Config migration between versions
- Partial config changes (e.g., only changing `checkpoint_every`)
- Config diffing/visualization

## Solution

**Input:** Config dataclass at training start + checkpoint metadata on resume
**Output:** Fail-loud on mismatch, proceed silently on match

## Usage

```python
from rollouts.training.config import fingerprint, validate_fingerprint

# At training start
cfg = GRPOConfig(model_name="Qwen/Qwen3-0.6B", lr=1e-6, ...)
config_fp = fingerprint(cfg)  # "a1b2c3d4e5f6..."

# Stored in checkpoint metadata
metadata = {
    "step": 1000,
    "config_fingerprint": config_fp,
    ...
}

# On resume
def load_checkpoint(ckpt_path, current_cfg):
    metadata = load_metadata(ckpt_path)
    validate_fingerprint(current_cfg, metadata["config_fingerprint"])  # Raises if mismatch
    ...
```

---

## Details

### Flow

1. `fingerprint(cfg)` - Convert dataclass to canonical JSON, SHA256 hash
2. `save_checkpoint()` - Store fingerprint in metadata.json
3. `load_checkpoint()` - Compare stored vs current fingerprint
4. On mismatch: `RuntimeError` with both hashes (truncated for readability)

### Implementation

Borrowed directly from nmoe (`nmoe/config.py:8-17`):

```python
import dataclasses
import hashlib
import json

def fingerprint(cfg) -> str:
    """Stable config fingerprint for reproducible resume checks."""
    d = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else {"value": str(cfg)}
    # Skip private fields (e.g., _cached_values)
    d = {k: v for k, v in d.items() if not str(k).startswith("_")}
    s = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def validate_fingerprint(current_cfg, saved_fingerprint: str) -> None:
    """Validate config fingerprint on resume. Raises on mismatch."""
    current_fp = fingerprint(current_cfg)
    if saved_fingerprint and current_fp != saved_fingerprint:
        raise RuntimeError(
            f"Config fingerprint mismatch on resume!\n"
            f"  Saved:   {saved_fingerprint[:16]}...\n"
            f"  Current: {current_fp[:16]}...\n"
            f"Refusing to resume with a different config."
        )
```

### Edge Cases

- **Empty saved fingerprint:** Skip validation (backwards compat with old checkpoints)
- **Non-dataclass configs:** Fall back to `str(cfg)` hash
- **Nested configs:** `dataclasses.asdict()` handles recursively

### Open Questions

- [ ] Should we allow "safe" config changes (e.g., `log_every`, `checkpoint_every`)?
- [ ] Should we store the full config in checkpoint for debugging, not just hash?
- [ ] How to handle config schema changes across code versions?

### Files

**Read:**
- `/tmp/nmoe/nmoe/config.py:8-17` - Reference implementation
- `/tmp/nmoe/nmoe/checkpoint.py:676-686` - Validation on resume

**Modify:**
- `rollouts/training/config/` - Add `fingerprint()`, `validate_fingerprint()`
- `rollouts/training/backends/pytorch.py` - Store fingerprint in `save_checkpoint()`
- `rollouts/training/grpo.py` - Validate on resume
