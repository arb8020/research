#!/usr/bin/env python3
"""Complete SFT data pipeline example.

Demonstrates the full flow:
    DataBuffer → User Rollout Function → RolloutManager → RolloutBatch

This example shows everything EXCEPT the actual training backend.
The RolloutBatch objects produced are ready for any training framework
(HuggingFace Trainer, Megatron, Axolotl, etc.)
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training import (
    DataBuffer,
    load_prompts_from_list,
    RolloutConfig,
    RolloutManager,
    prepare_sft_sample,
)


# Mock tokenizer (in practice, use transformers.AutoTokenizer)
class SimpleTokenizer:
    """Mock tokenizer for demonstration."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        return " | ".join(parts)

    def encode(self, text, max_length=2048, truncation=True):
        # Simple character-based encoding
        return [ord(c) % 256 for c in text[:max_length]]


def main():
    print("=" * 70)
    print("SFT Data Pipeline Example")
    print("=" * 70)

    # ========================================================================
    # 1. SETUP: Prepare dataset and tokenizer
    # ========================================================================
    print("\n[1] Setting up dataset and tokenizer...")

    # Mock dataset (in practice, load from JSONL)
    dataset = [
        {"id": "q1", "prompt": "What is 2+2?", "response": "4"},
        {"id": "q2", "prompt": "What is 3+3?", "response": "6"},
        {"id": "q3", "prompt": "What is 4+4?", "response": "8"},
        {"id": "q4", "prompt": "What is 5+5?", "response": "10"},
    ]

    # Create prompt list (just IDs in this case)
    prompts = [{"id": item["id"]} for item in dataset]

    # Initialize tokenizer
    tokenizer = SimpleTokenizer()

    print(f"✓ Dataset: {len(dataset)} items")
    print(f"✓ Prompts: {len(prompts)} prompts")

    # ========================================================================
    # 2. DATA BUFFER: Stateful prompt iteration with epoch management
    # ========================================================================
    print("\n[2] Creating DataBuffer...")

    buffer = DataBuffer(
        prompts=prompts,
        seed=42,  # Deterministic shuffling
    )

    print(f"✓ DataBuffer initialized")
    print(f"  - Current epoch: {buffer.epoch_id}")
    print(f"  - Sample offset: {buffer.sample_offset}")

    # ========================================================================
    # 3. USER ROLLOUT FUNCTION: Prompt → Sample conversion
    # ========================================================================
    print("\n[3] Defining user rollout function...")

    def my_sft_rollout(prompts, tokenizer, dataset, **kwargs):
        """User-provided SFT rollout function.

        This is where the user implements their data preparation logic.
        For SFT, this is just tokenization (no model inference).
        """
        samples = []

        for prompt_dict in prompts:
            # Look up data item
            item = next(d for d in dataset if d["id"] == prompt_dict["id"])

            # Prepare sample using our helper
            sample = prepare_sft_sample(
                prompt=item["prompt"],
                response=item["response"],
                tokenizer=tokenizer,
            )

            samples.append(sample)

        return samples

    print("✓ Rollout function defined")

    # ========================================================================
    # 4. ROLLOUT MANAGER: Orchestration layer
    # ========================================================================
    print("\n[4] Creating RolloutManager...")

    config = RolloutConfig(
        batch_size=2,  # 2 prompts per batch
        generate_fn=my_sft_rollout,
    )

    manager = RolloutManager(
        data_buffer=buffer,
        config=config,
        # Pass kwargs to rollout function
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("✓ RolloutManager initialized")
    print(f"  - Batch size: {config.batch_size}")

    # ========================================================================
    # 5. ITERATION: Generate training batches
    # ========================================================================
    print("\n[5] Generating training batches...")
    print("-" * 70)

    num_batches = 3
    for i, batch in enumerate(manager):
        if i >= num_batches:
            break

        print(f"\nBatch {i}:")
        print(f"  Prompts: {batch.metadata['prompts']}")
        print(f"  Responses: {batch.metadata['responses']}")
        print(f"  Token counts: {[len(t) for t in batch.tokens]}")
        print(f"  Response lengths: {batch.response_lengths}")
        print(f"  Epoch: {batch.metadata['epoch_id']}")
        print(f"  Step: {batch.metadata['step_id']}")

        # This RolloutBatch is ready for training!
        # You would pass it to your training backend here:
        # train_step(batch)

    print("\n" + "-" * 70)

    # ========================================================================
    # 6. STATE MANAGEMENT: Checkpointing
    # ========================================================================
    print("\n[6] State management...")

    state = manager.state_dict()
    print(f"✓ Saved state:")
    print(f"  - Steps completed: {state['step_count']}")
    print(f"  - Buffer epoch: {state['buffer_state']['epoch_id']}")
    print(f"  - Buffer offset: {state['buffer_state']['sample_offset']}")

    # In practice, save this to checkpoint file:
    # torch.save(state, "checkpoint.pt")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print("""
This example demonstrated the complete SFT data pipeline:

1. DataBuffer manages prompt iteration with deterministic shuffling
2. User provides a rollout function (prepare_sft_sample helper)
3. RolloutManager orchestrates buffer + rollout function
4. Iteration yields RolloutBatch objects ready for training
5. State can be saved/loaded for checkpointing

What's missing:
- Actual training backend (HuggingFace Trainer, Megatron, etc.)
- The RolloutBatch objects are ready - just plug in your trainer!

Next steps:
- See examples/test_rollout_manager.py for more usage patterns
- Implement your own rollout function for custom data prep
- Add a training backend to actually update model weights
""")


if __name__ == "__main__":
    main()
