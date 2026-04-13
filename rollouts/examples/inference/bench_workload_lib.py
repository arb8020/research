"""Task generators for inference benchmark configs.

These produce list[dict] task rows compatible with the standard eval harness.
The bench configs use them in place of the tiny SMOKE_TASKS from smoke_witness_lib.

Two workload types:

  make_random_tasks(num_prompts, input_len, output_len, seed)
      Synthetic prompts of approximately `input_len` tokens. No external data
      required. Good for characterizing server throughput and latency at
      controlled input/output lengths.

  make_sharegpt_tasks(path, num_prompts, output_len, seed)
      Real conversation turns from a ShareGPT-format JSONL. Input length
      reflects actual user query distributions. output_len caps the server's
      max_tokens so results are comparable across runs.

Both return list[dict] with keys the corresponding prepare_messages function
expects. The bench eval configs wire these into EvalTaskSpec.tasks exactly as
the smoke configs use SMOKE_TASKS.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from rollouts.core import Message

# A pool of common English words used to build synthetic prompts.
# Repeated sampling gives ~1 token/word at typical BPE tokenization.
_WORD_POOL = (
    "the quick brown fox jumps over the lazy dog "
    "a an and are as at be by for from has he in is it its of on that the "
    "to was will with you they we our their this these those which who whom "
    "what where when why how all each every both many much more most other "
    "into through during before after above below between each few however "
    "because while although since though unless until whenever wherever "
    "language model inference engine attention transformer generation token "
    "prompt context window batch size throughput latency decode prefill"
).split()


def _synthetic_prompt(input_len: int, seed: int) -> str:
    """Generate a ~input_len token prompt by sampling from the word pool."""
    rng = random.Random(seed)
    # Approximate: 1 word ≈ 1 token at BPE. Undershoot slightly to stay under
    # limit after the instruction wrapper adds its own tokens.
    words = [rng.choice(_WORD_POOL) for _ in range(max(1, input_len - 20))]
    body = " ".join(words)
    return (
        f"Read the following text and then respond with exactly {input_len} tokens "
        f"of fluent continuation:\n\n{body}"
    )


def make_random_tasks(
    *,
    num_prompts: int,
    input_len: int = 512,
    output_len: int = 256,
    seed: int = 42,
) -> list[dict]:
    """Synthetic fixed-length tasks for throughput/latency benchmarking.

    Args:
        num_prompts: Number of task rows to generate.
        input_len: Approximate input token count per prompt.
        output_len: Target output token count. Set as max_tokens on the endpoint.
        seed: Random seed for reproducibility.
    """
    return [
        {
            "prompt": _synthetic_prompt(input_len, seed=seed + i),
            "target_output_len": output_len,
            "workload": "random",
            "input_len": input_len,
        }
        for i in range(num_prompts)
    ]


def make_sharegpt_tasks(
    path: str | Path,
    *,
    num_prompts: int,
    output_len: int = 256,
    seed: int = 42,
) -> list[dict]:
    """Real conversation tasks from a ShareGPT-format JSONL file.

    The file should be a JSONL where each line is a conversation dict with a
    "conversations" key containing a list of turns, each with "from" and
    "value" keys (standard ShareGPT format). The first human turn is used as
    the prompt.

    Args:
        path: Path to the ShareGPT JSONL file.
        num_prompts: Number of tasks to sample. Samples without replacement if
            the dataset is large enough, otherwise samples with replacement.
        output_len: Target output token count. Set as max_tokens on the endpoint.
        seed: Random seed for reproducibility.
    """
    path = Path(path)
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Support both {"conversations": [...]} and flat {"text": "..."}
            if "conversations" in obj:
                turns = obj["conversations"]
                human_turns = [
                    t["value"]
                    for t in turns
                    if t.get("from") in ("human", "user") and t.get("value")
                ]
                if human_turns:
                    rows.append(human_turns[0])
            elif "text" in obj:
                rows.append(obj["text"])
            elif "prompt" in obj:
                rows.append(obj["prompt"])

    if not rows:
        raise ValueError(f"No usable prompts found in {path}")

    rng = random.Random(seed)
    if len(rows) >= num_prompts:
        selected = rng.sample(rows, num_prompts)
    else:
        selected = rng.choices(rows, k=num_prompts)

    return [
        {
            "prompt": prompt,
            "target_output_len": output_len,
            "workload": "sharegpt",
            "sharegpt_path": str(path),
        }
        for prompt in selected
    ]


def prepare_bench_messages(sample: dict) -> list[Message]:
    """Prepare messages for a bench task row.

    Works for both random and sharegpt tasks — both carry a "prompt" key.
    """
    return [Message(role="user", content=sample["prompt"])]
