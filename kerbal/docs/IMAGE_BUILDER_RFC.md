# RFC: Declarative Environment Builder for Kerbal

## Status: Draft

## Summary

Adopt a Modal-inspired declarative, chainable API for defining Python environments with layer caching. This would complement (not replace) the existing `setup_python_env()` imperative API.

## Motivation

Currently, `setup_python_env()` is imperative and executes immediately:

```python
# Current API - imperative, executes immediately
state = setup_python_env(
    client,
    workspace,
    requirements=["torch>=2.0", "triton"],
    git_packages=["git+https://github.com/user/repo.git"],
    verify_imports=["torch"],
)
```

Problems:
1. **No caching** - reinstalls everything on each call (though uv is fast)
2. **No composition** - can't extend or modify an environment definition
3. **Coupled to execution** - can't define an environment without a client
4. **No snapshots** - can't save/restore environment state

Modal's approach separates *definition* from *execution*:

```python
# Modal's API - declarative, deferred execution
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install("torch", "transformers")
    .run_function(download_weights)
)

# Execution happens later when deployed
```

## Proposed API

### Basic Usage

```python
from kerbal import PythonEnv

# Define environment (no execution yet)
env = (
    PythonEnv.base(python=">=3.10")
    .pip_install("torch>=2.0", "triton")
    .pip_install("transformers")  # Separate layer, cached independently
    .run("pip install flash-attn --no-build-isolation")
    .env({"HF_HOME": "/data/hf_cache"})
)

# Apply to remote machine (executes steps, uses cache)
state = env.apply(client, workspace="/root/project")

# state is same PythonEnvState as before
client.exec(f"{state.venv_python} train.py")
```

### Layer Caching

Each method call creates a layer. Layers are cached by content hash:

```python
env = (
    PythonEnv.base(python=">=3.10")      # Layer 0: hash(python=">=3.10")
    .pip_install("torch>=2.0")            # Layer 1: hash(layer0 + "torch>=2.0")
    .pip_install("transformers")          # Layer 2: hash(layer1 + "transformers")
)

# If layer 1 changes, layer 2 must rebuild
# If only layer 2 changes, layer 1 is cached
```

Cache storage options:
- **Local**: `~/.kerbal/cache/` directory with hash-named tarballs of venv
- **Remote**: `~/.kerbal/cache/` on remote machine (faster, no transfer)
- **Registry**: Optional push/pull to object storage (S3, GCS) for team sharing

### Advanced Features

#### Run Python Functions

```python
def download_model(workspace: str):
    """Runs on remote during env setup."""
    from huggingface_hub import snapshot_download
    snapshot_download("meta-llama/Llama-3-8B", local_dir=f"{workspace}/models")

env = (
    PythonEnv.base()
    .pip_install("huggingface_hub", "hf_transfer")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(download_model)  # Bakes model into cached layer
)
```

#### Git Packages

```python
env = (
    PythonEnv.base()
    .pip_install("torch")
    .git_install("https://github.com/user/repo.git", branch="main")
    .git_install("https://github.com/user/other.git", no_deps=True)
)
```

#### Apt Packages

```python
env = (
    PythonEnv.base()
    .apt_install("git", "ffmpeg", "libsndfile1")
    .pip_install("torchaudio")
)
```

#### Composition

```python
# Base environment for team
base_env = (
    PythonEnv.base(python=">=3.10")
    .pip_install("torch>=2.0", "numpy", "pandas")
)

# Extend for specific use case
training_env = (
    base_env
    .pip_install("accelerate", "datasets", "wandb")
    .env({"WANDB_PROJECT": "my-project"})
)

inference_env = (
    base_env
    .pip_install("vllm", "sglang")
)
```

#### Snapshots

```python
# Save environment state for exact reproduction
env.apply(client, workspace).snapshot("training-env-v1")

# Later: restore exact environment
env = PythonEnv.from_snapshot("training-env-v1")
state = env.apply(client, workspace)
```

### Inference Presets Integration

```python
from kerbal.inference import sglang

# Current: manual dep installation
deps = sglang.get_deps()
setup_python_env(client, workspace, requirements=deps.dependencies)

# Proposed: composable preset
env = (
    sglang.env()  # Returns PythonEnv with all deps
    .pip_install("my-custom-tokenizer")  # Extend
)
state = env.apply(client, workspace)
```

## Implementation Sketch

```python
@dataclass(frozen=True)
class Layer:
    """Immutable layer definition."""
    kind: str  # "pip", "apt", "run", "env", "git"
    content: tuple  # Hashable content

    def hash(self, parent_hash: str) -> str:
        return hashlib.sha256(f"{parent_hash}:{self.kind}:{self.content}".encode()).hexdigest()[:16]


class PythonEnv:
    """Declarative environment builder."""

    def __init__(self, layers: tuple[Layer, ...] = (), python_version: str = ">=3.10"):
        self._layers = layers
        self._python_version = python_version

    @classmethod
    def base(cls, python: str = ">=3.10") -> "PythonEnv":
        return cls(python_version=python)

    def pip_install(self, *packages: str) -> "PythonEnv":
        layer = Layer(kind="pip", content=tuple(sorted(packages)))
        return PythonEnv(self._layers + (layer,), self._python_version)

    def apt_install(self, *packages: str) -> "PythonEnv":
        layer = Layer(kind="apt", content=tuple(sorted(packages)))
        return PythonEnv(self._layers + (layer,), self._python_version)

    def run(self, command: str) -> "PythonEnv":
        layer = Layer(kind="run", content=(command,))
        return PythonEnv(self._layers + (layer,), self._python_version)

    def env(self, vars: dict[str, str]) -> "PythonEnv":
        layer = Layer(kind="env", content=tuple(sorted(vars.items())))
        return PythonEnv(self._layers + (layer,), self._python_version)

    def run_function(self, fn: Callable) -> "PythonEnv":
        # Serialize function source for hashing
        import inspect
        source = inspect.getsource(fn)
        layer = Layer(kind="run_function", content=(source,))
        return PythonEnv(self._layers + (layer,), self._python_version)

    def apply(
        self,
        client: "BifrostClient",
        workspace: str,
        cache: bool = True,
    ) -> PythonEnvState:
        """Execute layers, using cache where possible."""
        # 1. Compute layer hashes
        # 2. Find cache hit point (last cached layer)
        # 3. Restore from cache if available
        # 4. Execute remaining layers
        # 5. Cache final state
        # 6. Return PythonEnvState
        ...
```

## Cache Implementation Options

### Option A: Venv Tarball (Simple)

```bash
# After each layer, snapshot venv
tar -czf ~/.kerbal/cache/{hash}.tar.gz .venv/

# To restore
tar -xzf ~/.kerbal/cache/{hash}.tar.gz
```

Pros: Simple, works everywhere
Cons: Large files, slow transfer

### Option B: Pip Freeze + Reinstall (Lightweight)

```bash
# After each layer, save requirements
pip freeze > ~/.kerbal/cache/{hash}.txt

# To restore
uv pip install -r ~/.kerbal/cache/{hash}.txt
```

Pros: Small cache files, fast transfer
Cons: Doesn't cache compiled extensions, may not reproduce exactly

### Option C: Hybrid (Recommended)

- Use pip freeze for pure-Python packages
- Tarball for layers with compiled extensions or run_function
- Content-addressable storage with deduplication

## Migration Path

1. **Phase 1**: Add `PythonEnv` class alongside existing `setup_python_env()`
2. **Phase 2**: Update inference presets to return `PythonEnv` objects
3. **Phase 3**: Add caching (start with Option B, add tarball for run_function)
4. **Phase 4**: Add snapshot/restore for team sharing

## Non-Goals

- **Full container images**: We're not building Docker images. This is venv-level caching.
- **Multi-machine orchestration**: That's bifrost's job.
- **OS-level isolation**: We trust the remote machine's base system.

## Comparison with Modal

| Feature | Modal | Proposed |
|---------|-------|----------|
| Container images | Yes (full Docker) | No (venv only) |
| Layer caching | Yes | Yes |
| Declarative API | Yes | Yes |
| run_function | Yes | Yes |
| apt_install | Yes | Yes (via sudo) |
| Registry push/pull | Yes | Optional (Phase 4) |
| Requires daemon | Yes (Modal runtime) | No (just SSH) |

## Open Questions

1. **Cache location**: Remote-only vs bidirectional sync?
2. **Cache invalidation**: TTL? Manual? Content-based only?
3. **Apt packages**: Require sudo, may not work on all machines. Skip?
4. **GPU-specific caching**: CUDA version as part of hash?

## References

- [Modal Images Guide](https://modal.com/docs/guide/images)
- [modal.Image API](https://modal.com/docs/reference/modal.Image)
- Current implementation: `kerbal/python_env.py`
