# Recommendations for Rollouts from Ecosystem Analysis

Based on analysis of ARES, Harbor, and Verifiers, here are concrete recommendations for rollouts, ordered by priority.

## High Priority

### 1. Add Harbor Task Loader

**Problem**: Harbor has hundreds of benchmarks in a standard format. Rollouts can't access them without manual conversion.

**Solution**: Add a loader that converts Harbor tasks to rollouts' evaluation interface.

```python
# rollouts/loaders/harbor.py

from pathlib import Path
from dataclasses import dataclass
import tomli

from ..dtypes import Message, Tool
from ..evaluation import EvalConfig, PrepareMessagesFn, ScoreFn


@dataclass(frozen=True)
class HarborTask:
    """Loaded Harbor task."""
    name: str
    instruction: str
    config: dict  # Parsed task.toml
    task_dir: Path
    tests_dir: Path


def load_harbor_task(task_dir: Path) -> HarborTask:
    """Load a Harbor task from directory.

    Expected structure:
        task_dir/
        ├── instruction.md
        ├── task.toml
        ├── environment/
        │   └── Dockerfile (optional)
        ├── solution/ (optional)
        └── tests/
            └── test.sh
    """
    instruction_path = task_dir / "instruction.md"
    config_path = task_dir / "task.toml"
    tests_dir = task_dir / "tests"

    assert instruction_path.exists(), f"Missing instruction.md in {task_dir}"
    assert config_path.exists(), f"Missing task.toml in {task_dir}"
    assert tests_dir.exists(), f"Missing tests/ in {task_dir}"

    instruction = instruction_path.read_text()
    with open(config_path, "rb") as f:
        config = tomli.load(f)

    return HarborTask(
        name=task_dir.name,
        instruction=instruction,
        config=config,
        task_dir=task_dir,
        tests_dir=tests_dir,
    )


def load_harbor_dataset(dataset_dir: Path) -> list[HarborTask]:
    """Load all Harbor tasks from a dataset directory."""
    tasks = []
    for task_dir in sorted(dataset_dir.iterdir()):
        if task_dir.is_dir() and (task_dir / "task.toml").exists():
            tasks.append(load_harbor_task(task_dir))
    return tasks


def harbor_prepare_messages(task: HarborTask) -> list[Message]:
    """Convert Harbor task to rollouts messages."""
    return [Message(role="user", content=task.instruction)]


def create_harbor_eval_config(
    dataset_dir: Path,
    environment_factory,  # Creates sandboxed environment per task
    endpoint,
    max_concurrent: int = 1,
) -> EvalConfig:
    """Create EvalConfig for Harbor dataset.

    The environment_factory should create an environment that:
    1. Has tools for file operations, bash, etc.
    2. Can run the test.sh script after agent completion
    3. Returns reward based on test results

    Example:
        >>> from rollouts.environments import CodingEnvironment
        >>>
        >>> def env_factory(sample):
        ...     task = sample["harbor_task"]
        ...     return CodingEnvironment(
        ...         workdir=task.task_dir / "environment",
        ...         tests_dir=task.tests_dir,
        ...     )
        >>>
        >>> config = create_harbor_eval_config(
        ...     dataset_dir=Path("./tasks"),
        ...     environment_factory=env_factory,
        ...     endpoint=my_endpoint,
        ... )
    """
    tasks = load_harbor_dataset(dataset_dir)

    # Convert to HuggingFace-style dataset
    dataset = [
        {
            "id": task.name,
            "harbor_task": task,
            "prompt": task.instruction,
        }
        for task in tasks
    ]

    def prepare_messages(sample: dict) -> list[Message]:
        return harbor_prepare_messages(sample["harbor_task"])

    def score_fn(sample):
        # Score is computed by environment after running tests
        # Environment should set sample.reward based on test.sh exit code
        from ..dtypes import Score, Metric
        reward = sample.get("reward", 0.0)
        return Score(metrics=(
            Metric(name="success", value=reward, weight=1.0),
        ))

    return EvalConfig(
        endpoint=endpoint,
        score_fn=score_fn,
        prepare_messages=prepare_messages,
        environment_factory=environment_factory,
        max_concurrent=max_concurrent,
    )
```

**Effort**: Medium (1-2 days)

**Value**: Access to Harbor's benchmark ecosystem without architectural changes.

---

## Medium Priority

### 2. Add Token IDs and Logprobs to Usage/Trajectory

**Problem**: RL training requires token IDs to avoid retokenization drift. Currently rollouts stores this in `Sample.rollout_log_probs` but not in `Trajectory` or `Usage`.

**Solution**: Extend `Usage` dataclass:

```python
# In dtypes.py

@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: Cost = field(default_factory=lambda: Cost())

    # New fields for RL training
    prompt_token_ids: tuple[int, ...] | None = None
    completion_token_ids: tuple[int, ...] | None = None
    logprobs: tuple[float, ...] | None = None
```

**Provider changes**: Each provider function needs to extract token IDs when available:

```python
# In providers/anthropic.py

async def rollout_anthropic(...) -> Actor:
    # ... existing code ...

    # Extract token IDs if available (requires tokenizer)
    completion_token_ids = None
    logprobs = None

    if hasattr(response, 'usage') and response.usage:
        # Anthropic doesn't return token IDs directly
        # Would need to tokenize the output text
        pass

    usage = Usage(
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        # ... other fields ...
        completion_token_ids=completion_token_ids,
        logprobs=logprobs,
    )
```

**Note**: Not all providers return token IDs. This should be optional/best-effort.

**Effort**: Low-Medium (add fields, update providers that support it)

**Value**: Enables RL training without retokenization drift.

---

### 3. Add Schema Version to Trajectory

**Problem**: No forward compatibility story for trajectory format changes.

**Solution**: Add version field:

```python
@dataclass(frozen=True)
class Trajectory:
    messages: tuple[Message, ...]
    completions: tuple[ChatCompletion, ...]
    rewards: float
    metadata: dict[str, Any] = field(default_factory=dict)

    # New field
    schema_version: str = "rollouts-v1.0"

    @staticmethod
    def from_dict(data: dict) -> "Trajectory":
        version = data.get("schema_version", "rollouts-v1.0")
        # Handle version-specific parsing if needed
        if version == "rollouts-v1.0":
            return Trajectory(...)
        else:
            # Future versions
            raise ValueError(f"Unknown schema version: {version}")
```

**Effort**: Low (one field, trivial migration)

**Value**: Future-proofing for format changes.

---

## Low Priority

### 4. Add `is_copied_context` Flag for Training Data Hygiene

**Problem**: When resuming sessions or using context from previous trajectories, that context shouldn't be included in training loss computation.

Harbor solves this with `is_copied_context` on each step.

**Solution**: Add to Message:

```python
@dataclass(frozen=True)
class Message:
    role: Literal["user", "assistant", "tool"]
    content: str | tuple[ContentBlock, ...]
    # ... existing fields ...

    # New field
    is_copied_context: bool = False
```

**Usage in training**:

```python
def compute_loss_mask(trajectory: Trajectory) -> list[float]:
    """Compute per-token loss mask, excluding copied context."""
    mask = []
    for msg in trajectory.messages:
        if msg.is_copied_context:
            # Zero out loss for copied context
            mask.extend([0.0] * len(tokenize(msg.content)))
        else:
            mask.extend([1.0] * len(tokenize(msg.content)))
    return mask
```

**Effort**: Low

**Value**: Cleaner training data, especially for session resume scenarios.

---

### 5. ResourceRequirements Protocol for Sandboxed Environments

**Problem**: Some environments need containers with specific resources (CPU, memory, GPU). Currently this is ad-hoc.

**Solution**: Add a protocol for environments to declare requirements:

```python
@dataclass(frozen=True)
class ResourceRequirements:
    """Container resource requirements."""
    cpu_cores: int = 1
    memory_gb: int = 4
    disk_gb: int = 10
    gpu_count: int = 0
    gpu_type: str | None = None  # e.g., "nvidia-a100"
    network_access: bool = True
    timeout_seconds: int = 300


@runtime_checkable
class SandboxedEnvironment(Environment, Protocol):
    """Environment that runs in an isolated container."""

    def get_resource_requirements(self) -> ResourceRequirements:
        """Declare resource needs for container orchestration."""
        ...

    def get_dockerfile(self) -> str | None:
        """Optional custom Dockerfile. None = use default image."""
        ...

    def get_docker_image(self) -> str | None:
        """Base image if no Dockerfile. e.g., 'python:3.11-slim'"""
        ...
```

**Usage**: External orchestrators (Harbor, Modal, etc.) can query this to provision appropriate containers.

**Effort**: Medium

**Value**: Enables integration with container orchestration systems without rollouts owning that complexity.

---

## Not Recommended

### Queue-Mediated Interception (from ARES)

**Why not**: Rollouts owns the agent loop via `run_agent()`. Queue interception is for wrapping external agent code you don't control. Adding it would complicate the architecture for no benefit.

### HTTP Proxy Interception (from Verifiers)

**Why not**: Same reasoning. Rollouts has direct provider integration which gives better streaming, lower latency, and access to provider-specific features.

### Pydantic Models (from Harbor)

**Why not**: Frozen dataclasses with explicit assertions are better for rollouts' "crash-loud" philosophy. Pydantic adds runtime overhead and makes immutability awkward.

### dm_env TimeStep Protocol (from ARES)

**Why not**: `AgentState` is already richer and more explicit for agentic use cases. dm_env's discount-based termination doesn't map well to tool-based agent stopping.

---

## Summary

| Priority | Change | Effort | Value |
|----------|--------|--------|-------|
| **High** | Harbor task loader | Medium | Ecosystem access |
| **Medium** | Token IDs in Usage | Low-Medium | RL training quality |
| **Medium** | Schema version in Trajectory | Low | Forward compatibility |
| **Low** | is_copied_context flag | Low | Training data hygiene |
| **Low** | ResourceRequirements protocol | Medium | Container orchestration interop |

The theme: **adopt data formats for interoperability, keep execution model distinct**.

Rollouts' strength is its functional, immutable, multi-provider design. These recommendations extend that foundation to work better with the broader ecosystem without compromising core principles.

---

## TI/TO (Token-In/Token-Out) Analysis

### Current State

Rollouts has a working TI/TO implementation. Here's how the pieces fit together:

**The problem** (from `tokens.md`, SID-1 paper): In multi-turn tool-calling RL, decoding tokens to text then retokenizing via `apply_chat_template` produces different tokens. The merged/split tokens have logprobs of -17 to -20, dominate the gradient, and cause training collapse.

**The fix**: Store generated `token_ids` directly, never decode-then-re-encode.

### Current Architecture

The TI/TO data flows through three layers:

**1. Storage: `Choice.token_ids`** (`dtypes.py:796`)
```python
class Choice(JsonSerializable):
    index: int
    message: Message
    finish_reason: str
    logprobs: Logprobs | None = None
    token_ids: tuple[int, ...] | None = None  # Generated token IDs for TI/TO
```

This is the right place - token IDs are per-completion, matching the `Choice` granularity.

**2. Generation: `rollout_sglang_token_level` / `rollout_vllm_token_level`** (`providers/sglang.py`)
- Tokenizes messages via `tokenize_chat()`
- Calls `/generate` with `input_ids` directly (not text)
- Stores `output_ids` in `Choice.token_ids`
- Decodes to text only for the `Message.content` (agent interface)
- Appends `suffix_ids` between turns for chat template compliance

**3. Training: Two strategies** (`training/agent_integration.py`, `training/grpo.py`)
- **Interleaved**: Full conversation as one token sequence. Uses stored `token_ids` from completions, appends suffix tokens between turns. Efficient (prefix sharing).
- **Branching**: Each assistant turn is a separate `(prompt, completion)` sample. Fresh tokenization of history per turn. Safer, mirrors deployment.

**4. Control flow**: `RunConfig.use_tito` flag threads through:
```
RunConfig(use_tito=True, tokenizer=..., suffix_ids=...)
  → agents.py:rollout() checks use_tito
    → routes to rollout_sglang_token_level / rollout_vllm_token_level
      → stores token_ids in Choice
        → training code reads Choice.token_ids
```

### What's Working

- The core TI/TO path is implemented end-to-end
- `test_tito_correctness.py` validates the approach by demonstrating the smoking gun (retokenized tokens with logprob < -10)
- Both interleaved and branching strategies are implemented
- Debug logging via `log_token_mismatch()` compares TI/TO tokens vs chat-template reference

### Issues and Gaps

**1. The `use_tito` flag creates a parallel universe**

The `use_tito` boolean in `RunConfig` forks the entire execution path:
```
use_tito=False → rollout_sglang() [text-based, OpenAI SDK]
use_tito=True  → rollout_sglang_token_level() [token-based, /generate]
```

These are completely separate code paths with different:
- API endpoints (`/v1/chat/completions` vs `/generate`)
- Request formats (messages vs input_ids)
- Response parsing
- Tool call handling

This means bugs fixed in one path don't propagate to the other.

**2. Token IDs are stored in `Choice` but not in `Usage`**

`Choice.token_ids` stores the generated tokens, but `Usage` (which lives on `ChatCompletion`) doesn't have `prompt_token_ids`. For training, you need both:
- `prompt_token_ids` = everything before the assistant response
- `completion_token_ids` = the assistant response (stored in `Choice.token_ids`)

Currently `_extract_tokens_from_trajectory()` reconstructs the prompt by tokenizing messages manually, which re-introduces the exact retokenization risk TI/TO is supposed to prevent (though only for the prompt portion, where the risk is lower since you're not training on prompt tokens).

**3. Suffix ID management is fragile**

The `compute_suffix_ids()` / `append_suffix_with_overlap()` machinery handles the gap between "what the model generated" and "what the chat template expects between turns." From `tokens.md`:

> The qwen3 chat template puts a newline between messages. If you just give u1,a1,u2 as raw tokens, you're missing a newline between a1 and u2 because a1 ends on the end-message token + u2 was tokenized independently.

This is currently handled per-tokenizer, but it's inherently fragile because chat templates are text-level abstractions and TI/TO operates at token level. The suffix approach works but requires careful testing per model family.

**4. Two separate trajectory-to-samples paths**

There are two `_trajectory_to_samples_tito_*` functions in `grpo.py` AND two in `agent_integration.py`:
- `grpo.py:_trajectory_to_sample_tito_interleaved()`
- `grpo.py:_trajectory_to_samples_tito_branching()`
- `agent_integration.py:trajectory_to_sample()` (with TI/TO fallback)
- `agent_integration.py:_branching_trajectory_to_samples()`

These have significant code overlap and could diverge.

**5. `_compute_loss_mask()` retokenizes**

In `agent_integration.py:601`, the loss mask computation retokenizes each message to find boundaries:
```python
msg_text = tokenizer.apply_chat_template([_msg_to_dict(msg)], ...)
msg_tokens = tokenizer.encode(msg_text, add_special_tokens=False)
msg_len = len(msg_tokens)
```

This is fine when `use_tito=False` (everything is retokenized anyway), but when TI/TO is on, the loss mask boundaries may not align perfectly with the stored token_ids. The branching strategy sidesteps this by using stored token_ids directly for the output portion.

### Recommendations

**1. Unify the trajectory-to-samples code**

The four functions doing trajectory-to-samples conversion should be one function with a strategy parameter. `agent_integration.py` already has `trajectory_to_samples(strategy="interleaved"|"branching")` which could serve as the single entry point, and `grpo.py`'s versions should call through to it.

**2. Store `prompt_token_ids` alongside `completion_token_ids`**

When using TI/TO, the full `input_ids` sent to `/generate` should be stored somewhere recoverable. Options:
- Add `prompt_token_ids` to `ChatCompletion` or `Usage`
- Add `input_token_ids` to `Trajectory` metadata
- Store in `Choice` alongside existing `token_ids`

The cleanest option is probably on `ChatCompletion`:
```python
@dataclass(frozen=True)
class ChatCompletion(JsonSerializable):
    # ... existing fields ...
    prompt_token_ids: tuple[int, ...] | None = None  # For TI/TO
```

Then `_extract_tokens_from_trajectory()` can just concatenate `completion.prompt_token_ids + choice.token_ids` instead of re-tokenizing the prefix.

**3. Make TI/TO the default for self-hosted models**

The `use_tito` flag currently defaults to `False`. For `provider="sglang"` or `provider="vllm"`, TI/TO should be the default since:
- You control the inference server
- You have the tokenizer
- The text path is strictly worse for training

The text path should only be used for API providers (OpenAI, Anthropic, Google) where you don't have access to token IDs.

**4. Address the suffix ID fragility**

The current approach computes suffix IDs once and appends them between turns. This works but is model-specific. Consider:
- Testing suffix computation in CI against multiple tokenizer families (Qwen, Llama, Mistral)
- Adding an assertion that verifies suffix IDs produce the same bytes as the chat template would between turns
- Documenting the assumption explicitly: "suffix_ids = tokens that the chat template inserts after an assistant message"

**5. Consider the "hybrid" strategy from tokens.md**

From `tokens.md`:
> Eventually, I expect we'll want a single strategy that gives the best of both worlds, interleaving as much as we can via TITO, but branching where we have to for rewritten prompt histories.

This is the right long-term direction. The interleaved strategy is more efficient but fragile at boundaries. The branching strategy is safe but wastes compute on repeated prefixes. A hybrid would:
- Use interleaved for turns where the context is preserved exactly
- Branch when context is rewritten (summarization, context window management)

This maps naturally to Harbor's `is_copied_context` flag: if a message is copied context, branch; otherwise, interleave.

### Relationship to Harbor/ARES/Verifiers

**Harbor's approach**: Stores `prompt_token_ids`, `completion_token_ids`, and `logprobs` per-step in ATIF `Metrics`. Similar to what rollouts does in `Choice.token_ids` + `Choice.logprobs`, but also captures prompt tokens. Harbor's `RolloutDetail` TypedDict stores these as lists of lists (one per turn).

**ARES's approach**: Doesn't address TI/TO directly. The queue-mediated client passes `LLMRequest`/`LLMResponse` which are text-based (messages, not tokens). ARES would need TI/TO if used for multi-turn RL training.

**Verifiers' approach**: Uses Prime Sandboxes for execution, with HTTP interception for API calls. The model-in-the-loop pattern means they control the LLM response, but the agent (OpenCode CLI) still receives text responses. TI/TO would need to happen at the verifiers framework level, not the agent level.

**Key insight**: Rollouts is the only framework in this comparison that has a working end-to-end TI/TO implementation for multi-turn agentic RL. Harbor stores the data but doesn't do the generation. ARES has the RL loop but not TI/TO. This is a genuine competitive advantage worth protecting and improving.
