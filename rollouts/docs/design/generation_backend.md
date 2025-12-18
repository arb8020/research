# Token-In/Token-Out (TI/TO) Generation Backend

**DRI:** chiraag
**Claude:** [this conversation]

## Context

We need token-level generation across backends (vLLM, SGLang, HuggingFace, nano-inference) to ensure RL training stability.

**The core problem:** Retokenization causes RL collapse. When you generate tokens → parse to text → re-apply chat template → tokens, you get subtly different tokens (e.g., `" \"` becomes `"\""`) with extremely low logprobs (-20!) that dominate the gradient and cause collapse.

**Evidence:** `tokens.md` documents this from verifiers+prime-rl. Figure 5 shows training collapse from malformed tokens. The mechanism:
1. Model generates rollouts G(ood) and B(ad)
2. Bad rollouts get "fixed" by re-tokenization: `apply_chat_template(parse(B)) = G'`
3. Training sees G' with negative advantage, but some tokens have logprob -20
4. These dominate gradient → collapse

**The solution:** Tokens-In/Tokens-Out (TI/TO) - always pass token IDs directly, never re-tokenize.

## Out of Scope

- Server lifecycle management (already handled by `SGLangEngine`/`VLLMEngine` in `weight_sync.py`)
- High-level agent orchestration (keep `agent_rollout_to_sample` text-based API for users)

## Solution

**Input:** Token IDs (not text), sampling params
**Output:** Generated token IDs + per-token logprobs

A `TokenProvider` protocol that all backends implement, sitting **beneath** the agent API:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         User's Agent Code (unchanged)                       │
│                     messages in, text out (nice API)                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      TI/TO Adapter Layer (NEW)                              │
│  - Intercepts LLM calls                                                     │
│  - Tokenizes input, calls TokenProvider, stores tokens+logprobs             │
│  - Decodes output for user's agent                                          │
│  - At rollout end: returns stored tokens (no re-tokenization)               │
└─────────────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TokenProvider Protocol                              │
│  SGLangTokenProvider  │  VLLMTokenProvider  │  HuggingFaceTokenProvider    │
│  /generate            │  /v1/.../tokens     │  forward pass                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Usage

### For Correctness Testing

```python
from rollouts.inference.backends import (
    HuggingFaceTokenProvider,
    SGLangTokenProvider,
    VLLMTokenProvider,
)

# All providers have identical interface
hf = HuggingFaceTokenProvider(model="Qwen/Qwen2.5-0.5B-Instruct")
sglang = SGLangTokenProvider(base_url="http://localhost:30000")

# Token-level generation
input_ids = tokenizer.encode("The capital of France is")
result = await sglang.generate(
    input_ids=input_ids,
    sampling_params=SamplingParams(temperature=0.0, max_tokens=100),
)

# Result contains everything needed for RL
result.output_ids      # [5, 6, 7, 8, ...] - actual generated tokens
result.logprobs        # [-0.1, -0.3, ...] - per-token logprobs
result.top_logprobs    # [{5: -0.1, 10: -0.5}, ...] - top-N per position

# Compare backends
assert check_logprobs_close(hf_result, sglang_result)
```

### For Training (with TI/TO adapter)

```python
# User code unchanged - still uses messages API
sample = await agent_rollout_to_sample(
    prompt=messages,
    environment_cls=CalculatorEnvironment,
    endpoint=endpoint,  # Now wraps TokenProvider internally
    tokenizer=tokenizer,
)

# But internally, the adapter:
# 1. Tokenizes messages → input_ids
# 2. Calls TokenProvider.generate(input_ids) → output_ids + logprobs
# 3. Stores tokens + logprobs (NO re-tokenization)
# 4. Decodes for tool parsing / user visibility
# 5. Returns Sample with TRUE tokens and logprobs
```

---

## Details

### Protocol

```python
@dataclass(frozen=True)
class GenerationResult:
    """Immutable result from token-level generation."""
    input_ids: tuple[int, ...]
    output_ids: tuple[int, ...]
    logprobs: tuple[float, ...]  # Per output token
    top_logprobs: tuple[dict[int, float], ...] | None  # Top-N per position
    finish_reason: str  # "stop" | "length" | "abort"


class TokenProvider(Protocol):
    """Token-level generation interface.

    All backends implement this protocol for TI/TO.
    """

    async def generate(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> GenerationResult:
        """Generate tokens from input token IDs.

        Args:
            input_ids: Prompt as token IDs (not text!)
            sampling_params: Temperature, max_tokens, stop_token_ids, etc.

        Returns:
            GenerationResult with output tokens and logprobs
        """
        ...
```

### Backend Implementations

#### SGLangTokenProvider

Uses SGLang's native `/generate` endpoint with `input_ids`:

```python
@dataclass
class SGLangTokenProvider:
    base_url: str  # e.g., "http://localhost:30000"

    async def generate(self, input_ids, sampling_params) -> GenerationResult:
        payload = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": sampling_params.temperature,
                "max_new_tokens": sampling_params.max_tokens,
                ...
            },
            "return_logprob": True,
        }
        response = await httpx.post(f"{self.base_url}/generate", json=payload)
        data = response.json()

        # Extract from output_token_logprobs: [(logprob, token_id), ...]
        output_ids = [item[1] for item in data["meta_info"]["output_token_logprobs"]]
        logprobs = [item[0] for item in data["meta_info"]["output_token_logprobs"]]

        return GenerationResult(
            input_ids=tuple(input_ids),
            output_ids=tuple(output_ids),
            logprobs=tuple(logprobs),
            ...
        )
```

#### VLLMTokenProvider

**Option A**: Use Prime's `/v1/chat/completions/tokens` endpoint (if using their fork)
**Option B**: Use `/v1/completions` with `prompt_token_ids` parameter (upstream vLLM)
**Option C**: PR `/v1/chat/completions/tokens` to upstream vLLM

```python
@dataclass
class VLLMTokenProvider:
    base_url: str

    async def generate(self, input_ids, sampling_params) -> GenerationResult:
        # Option B: Use /v1/completions with prompt_token_ids
        payload = {
            "prompt_token_ids": input_ids,
            "max_tokens": sampling_params.max_tokens,
            "temperature": sampling_params.temperature,
            "logprobs": 5,
        }
        response = await httpx.post(
            f"{self.base_url}/v1/completions",
            json=payload
        )
        ...
```

#### HuggingFaceTokenProvider

Ground truth for correctness tests. Uses step-by-step forward pass:

```python
@dataclass
class HuggingFaceTokenProvider:
    model: PreTrainedModel

    async def generate(self, input_ids, sampling_params) -> GenerationResult:
        # Step-by-step forward pass (matches how vLLM/SGLang work)
        current_ids = torch.tensor([input_ids])
        output_ids = []
        logprobs = []

        for _ in range(sampling_params.max_tokens):
            logits = self.model(current_ids).logits[:, -1, :]
            log_probs = F.log_softmax(logits, dim=-1)

            if sampling_params.temperature == 0:
                next_token = logits.argmax(dim=-1).item()
            else:
                # Sample with temperature
                ...

            output_ids.append(next_token)
            logprobs.append(log_probs[0, next_token].item())

            if next_token == eos_token_id:
                break
            current_ids = torch.cat([current_ids, [[next_token]]], dim=1)

        return GenerationResult(...)
```

### Multi-Turn Delimiter Handling

**The problem** (from `tokens.md`):
```
Turn 1: u1 → a1
Turn 2: want u1 || a1 || u2
```

Chat templates insert delimiters (newlines, role markers) between messages. If you just concatenate tokens, you miss these delimiters.

**Solution approaches:**

1. **Miles' "prefix trick"** (`mask_utils.py`):
   - Tokenize `[prefix_msg, actual_msg]` together
   - Strip prefix tokens to get `actual_msg` with correct leading delimiter

2. **Prime's cached suffix approach** (`verifiers PR #626`):
   - Pre-compute delimiter tokens for each role transition once
   - Cache them
   - Insert at turn boundaries when concatenating

```python
class ChatTemplateTokenizer:
    """Handles delimiter tokens between turns."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.suffix_cache = {}  # (from_role, to_role) → delimiter_tokens
        self._precompute_suffixes()

    def _precompute_suffixes(self):
        """Compute delimiter tokens for role transitions."""
        # Use dummy messages to find what tokens the template inserts
        for from_role in ["user", "assistant", "tool"]:
            for to_role in ["user", "assistant", "tool"]:
                dummy_msgs = [
                    {"role": from_role, "content": "X"},
                    {"role": to_role, "content": "Y"},
                ]
                full_ids = self.tokenizer.apply_chat_template(dummy_msgs, tokenize=True)
                # Extract delimiter between X and Y
                ...
                self.suffix_cache[(from_role, to_role)] = delimiter_ids

    def build_multi_turn_ids(
        self,
        turns: list[tuple[str, list[int]]]  # [(role, token_ids), ...]
    ) -> list[int]:
        """Concatenate turns with correct delimiters."""
        result = turns[0][1]  # First turn
        for i in range(1, len(turns)):
            from_role = turns[i-1][0]
            to_role = turns[i][0]
            delimiter = self.suffix_cache[(from_role, to_role)]
            result = result + delimiter + turns[i][1]
        return result
```

### Current State vs Target State

#### Current (`grpo.py` → `agent_integration.py`)

```
messages (text) → SGLang /v1/chat/completions (text) → response (text)
                                    ↓
              trajectory_to_sample() RE-TOKENIZES EVERYTHING
                                    ↓
                          tokens for training (WRONG)
```

**Problems:**
- Uses OpenAI-compatible API (text in/out)
- Discards generated tokens
- Never captures logprobs
- Re-tokenizes at `trajectory_to_sample()`

#### Target (with TI/TO)

```
messages (text) → tokenize → TokenProvider.generate(input_ids)
                                    ↓
                    output_ids + logprobs (stored, not discarded)
                                    ↓
                    decode for tool parsing / user visibility
                                    ↓
                    Sample.tokens = stored tokens (NO re-tokenization)
                    Sample.logprobs = stored logprobs
```

### Reference Implementations

- **miles** (`/tmp/miles/miles/rollout/sglang_rollout.py`): Uses `/generate` with `input_ids`, extracts `output_token_logprobs`
- **prime-rl** ([PR #1422](https://github.com/PrimeIntellect-ai/prime-rl/pull/1422)): Added `/v1/chat/completions/tokens` to vLLM
- **verifiers** ([PR #626](https://github.com/PrimeIntellect-ai/verifiers/pull/626)): Pre-computed suffix tokens for multi-turn TI/TO
- **vLLM tests** (`/tmp/vllm-tests/tests/conftest.py`): `HfRunner` for ground truth comparison

### Open Questions

- [ ] Should we fork SGLang/vLLM to add `/v1/chat/completions/tokens`, or use existing `/generate`?
- [ ] How to handle the TI/TO adapter for multi-turn? Intercept at Actor level or lower?
- [ ] Should `Sample` store `rollout_logprobs` like miles does?
- [ ] Do we need `weight_version` tracking for on-policy verification?

### Files

**Read:**
- `rollouts/training/grpo.py` - Current training loop
- `rollouts/training/agent_integration.py` - Current `trajectory_to_sample()` (the problem)
- `rollouts/providers/sglang.py` - Current text-based provider
- `/tmp/miles/miles/rollout/sglang_rollout.py` - Reference TI/TO implementation
- `/tmp/miles/miles/utils/mask_utils.py` - Reference delimiter handling

**Create:**
- `rollouts/inference/backends/protocol.py` - `TokenProvider` protocol + `GenerationResult`
- `rollouts/inference/backends/sglang.py` - `SGLangTokenProvider`
- `rollouts/inference/backends/vllm.py` - `VLLMTokenProvider`
- `rollouts/inference/backends/huggingface.py` - `HuggingFaceTokenProvider`
- `rollouts/inference/chat_template.py` - `ChatTemplateTokenizer` for delimiter handling

**Modify:**
- `rollouts/training/agent_integration.py` - Add TI/TO adapter, remove re-tokenization
- `rollouts/training/types.py` - Add `rollout_logprobs` to `Sample`
- `rollouts/providers/sglang.py` - Add token-level provider alongside text provider
