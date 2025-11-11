# Architecture 1: Single-Node Split GPUs - Implementation Plan

**Goal**: Get RL training working on ONE node with 8 GPUs (4 training + 4 inference)

**Timeline**: 5-7 days

---

## What We Need

### Hardware Requirements
- ✅ 1 node with 8 GPUs
- ✅ ~40GB VRAM per GPU (for 7B model)
- ✅ For 0.5B model: Can work with less VRAM

### Software Prerequisites (Already Have)
- ✅ PyTorch with FSDP support
- ✅ Transformers/HuggingFace
- ✅ rollouts package (training backends, types, loops)
- ✅ miniray package (not needed for Architecture 1!)

---

## What We Need to Build (5 Components)

### 1. GRPO Loss Implementation (Priority 1, 2 days)

**Status**: ❌ Missing completely

**What it is**: The loss function for Group Relative Policy Optimization

**File to create**: `rollouts/rollouts/training/rl_losses.py`

**Code needed**:
```python
import torch
import torch.nn.functional as F
from typing import Dict

def grpo_loss(
    policy_log_probs: torch.Tensor,      # Shape: (batch_size, seq_len)
    ref_log_probs: torch.Tensor,         # Shape: (batch_size, seq_len)
    advantages: torch.Tensor,            # Shape: (batch_size,)
    loss_mask: torch.Tensor,             # Shape: (batch_size, seq_len)
    eps_clip: float = 0.2,
    eps_clip_high: float = 0.28,
    kl_coef: float = 0.0,
    entropy_coef: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """GRPO loss with PPO clipping.

    Args:
        policy_log_probs: Log probs from current policy
        ref_log_probs: Log probs from reference model
        advantages: Advantage estimates (rewards - baseline)
        loss_mask: Mask for valid tokens (1.0 = compute loss, 0.0 = ignore)
        eps_clip: Lower PPO clip threshold
        eps_clip_high: Upper PPO clip threshold
        kl_coef: KL divergence penalty weight
        entropy_coef: Entropy bonus weight

    Returns:
        Dict with loss components
    """
    # Mask and reduce log probs (sum over sequence length)
    policy_lp = (policy_log_probs * loss_mask).sum(dim=-1)  # (batch_size,)
    ref_lp = (ref_log_probs * loss_mask).sum(dim=-1)        # (batch_size,)

    # Compute ratio (importance sampling)
    ratio = torch.exp(policy_lp - ref_lp)

    # PPO clipped objective
    clipped_ratio = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip_high)

    policy_loss = -torch.min(
        ratio * advantages,
        clipped_ratio * advantages
    ).mean()

    # KL divergence penalty
    kl_div = (policy_lp - ref_lp).mean()
    kl_loss = kl_coef * kl_div

    # Entropy bonus (encourage exploration)
    entropy = -(policy_log_probs * loss_mask).sum(dim=-1).mean()
    entropy_loss = -entropy_coef * entropy

    # Total loss
    total_loss = policy_loss + kl_loss + entropy_loss

    return {
        "loss": total_loss,
        "policy_loss": policy_loss,
        "kl_div": kl_div,
        "kl_loss": kl_loss,
        "entropy": entropy,
        "entropy_loss": entropy_loss,
        "ratio_mean": ratio.mean(),
        "ratio_std": ratio.std(),
        "advantages_mean": advantages.mean(),
        "advantages_std": advantages.std(),
    }
```

**Test it**:
```python
# Test script
import torch
from rollouts.training.rl_losses import grpo_loss

batch_size = 4
seq_len = 128

# Dummy data
policy_log_probs = torch.randn(batch_size, seq_len) * 0.1
ref_log_probs = torch.randn(batch_size, seq_len) * 0.1
advantages = torch.tensor([1.0, 0.5, -0.5, -1.0])
loss_mask = torch.ones(batch_size, seq_len)
loss_mask[:, :64] = 0.0  # Mask prompt tokens

# Compute loss
loss_dict = grpo_loss(
    policy_log_probs,
    ref_log_probs,
    advantages,
    loss_mask,
    eps_clip=0.2,
)

print(f"Total loss: {loss_dict['loss']:.4f}")
print(f"Policy loss: {loss_dict['policy_loss']:.4f}")
print(f"KL div: {loss_dict['kl_div']:.4f}")
```

---

### 2. Reference Model (Priority 1, 1 day)

**Status**: ⚠️ Backend exists, needs reference model support

**What it is**: Frozen copy of initial model for KL divergence

**Files to modify**:
- `rollouts/rollouts/training/backends/pytorch.py`
- `rollouts/rollouts/training/backends/fsdp.py`

**What to add**:

```python
# rollouts/training/backends/pytorch.py

class PyTorchTrainingBackend:
    def __init__(
        self,
        model_name: str,
        world_size: int = 1,
        strategy: str = "ddp",
        ref_model_path: Optional[Path] = None,  # NEW!
    ):
        self.model = self._load_model(model_name)
        self.optimizer = self._create_optimizer()

        # NEW: Load reference model (frozen)
        if ref_model_path:
            self.ref_model = self._load_model(ref_model_path)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False
        else:
            # Use initial model weights as reference
            self.ref_model = self._load_model(model_name)
            self.ref_model.eval()
            for param in self.ref_model.parameters():
                param.requires_grad = False

    async def forward_backward(
        self,
        batch: Dict[str, Any],
        compute_ref_logprobs: bool = True,  # NEW!
    ) -> TrainFuture[Dict[str, float]]:
        """Forward + backward pass with GRPO loss."""

        # Policy forward pass
        outputs = self.model(
            input_ids=batch["input_ids"],
            labels=batch["labels"],
        )
        policy_logits = outputs.logits

        # Compute policy log probs
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_log_probs = torch.gather(
            policy_log_probs,
            dim=-1,
            index=batch["labels"].unsqueeze(-1)
        ).squeeze(-1)

        # Reference model forward pass (no grad)
        if compute_ref_logprobs:
            with torch.no_grad():
                ref_outputs = self.ref_model(
                    input_ids=batch["input_ids"],
                    labels=batch["labels"],
                )
                ref_logits = ref_outputs.logits
                ref_log_probs = F.log_softmax(ref_logits, dim=-1)
                ref_log_probs = torch.gather(
                    ref_log_probs,
                    dim=-1,
                    index=batch["labels"].unsqueeze(-1)
                ).squeeze(-1)
        else:
            ref_log_probs = policy_log_probs.detach()  # Use policy as ref

        # GRPO loss
        from rollouts.training.rl_losses import grpo_loss

        loss_dict = grpo_loss(
            policy_log_probs=policy_log_probs,
            ref_log_probs=ref_log_probs,
            advantages=batch["advantages"],
            loss_mask=batch["loss_mask"],
            eps_clip=0.2,
        )

        # Backward
        loss_dict["loss"].backward()

        # Compute grad norm
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            max_norm=1.0
        )

        return TrainFuture(
            _result={
                **loss_dict,
                "grad_norm": grad_norm.item(),
            }
        )
```

---

### 3. Reward Function (Priority 2, 1 day)

**Status**: ⚠️ Basic version exists, needs improvement

**Current implementation**:
```python
# rollouts/training/loops/rl_loop.py
def compute_reward(sample: Sample) -> float:
    if sample.metadata.get("correct", False):
        return 1.0
    return 0.0
```

**What we need**: Math grading logic

**File to create**: `rollouts/rollouts/training/reward_models.py`

```python
import re
from typing import Protocol

class RewardModel(Protocol):
    """Protocol for reward models."""
    def score(self, prompt: str, response: str, label: str) -> float:
        """Score a response. Returns reward in [0, 1]."""
        ...

class MathRewardModel:
    """Reward model for math problems (GSM8K-style)."""

    def score(self, prompt: str, response: str, label: str) -> float:
        """Grade math response by extracting final answer.

        Args:
            prompt: The math problem
            response: Model's response
            label: Ground truth answer

        Returns:
            1.0 if correct, 0.0 otherwise
        """
        # Extract answer from response
        answer = self._extract_answer(response)

        # Normalize both answers
        answer_norm = self._normalize_answer(answer)
        label_norm = self._normalize_answer(label)

        # Check if correct
        return 1.0 if answer_norm == label_norm else 0.0

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response.

        Looks for patterns like:
        - "The answer is 42"
        - "#### 42"
        - "Answer: 42"
        """
        # Try #### pattern (GSM8K format)
        match = re.search(r'####\s*(.+)', response)
        if match:
            return match.group(1).strip()

        # Try "answer is" pattern
        match = re.search(r'(?:answer is|answer:|=)\s*(.+)', response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # Last resort: take last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return ""

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        # Remove whitespace, currency symbols, commas
        answer = answer.strip().replace(',', '').replace('$', '')

        # Try to parse as number
        try:
            num = float(answer)
            return str(num)
        except ValueError:
            return answer.lower()

# Usage in rl_loop.py:
def compute_reward(sample: Sample, rm: RewardModel) -> float:
    return rm.score(
        prompt=sample.prompt,
        response=sample.response,
        label=sample.metadata.get("label", ""),
    )
```

---

### 4. GRPO Rollout Generation (Priority 2, 1 day)

**Status**: ⚠️ Config exists, not using n_samples_per_prompt

**Current issue**: We generate 1 sample per prompt, but GRPO needs N samples per prompt

**File to modify**: `rollouts/rollouts/training/rollout_gen/async_rollout_manager.py`

**What to change**:

```python
# rollouts/training/rollout_gen/async_rollout_manager.py

class AsyncRolloutManager:
    async def generate_batch(self) -> RolloutBatch:
        """Generate batch with n_samples_per_prompt for GRPO."""

        # Sample prompts
        prompts = self.data_buffer.sample(self.config.batch_size)

        # Expand: each prompt → n_samples_per_prompt copies
        expanded_prompts = []
        for prompt in prompts:
            expanded_prompts.extend([prompt] * self.config.n_samples_per_prompt)

        # Generate all samples
        samples = await self.config.generate_fn(expanded_prompts, self.config)

        # Group samples by prompt (for GRPO advantage computation)
        grouped_samples = []
        for i in range(0, len(samples), self.config.n_samples_per_prompt):
            group = samples[i:i + self.config.n_samples_per_prompt]
            grouped_samples.append(group)

        # Compute rewards
        if self.config.reward_fn:
            for group in grouped_samples:
                for sample in group:
                    sample.reward = self.config.reward_fn(sample)

        # Compute GRPO advantages (group-relative)
        for group in grouped_samples:
            rewards = [s.reward for s in group]
            baseline = sum(rewards) / len(rewards)  # Group mean
            for sample in group:
                sample.metadata["advantage"] = sample.reward - baseline

        # Flatten back to list
        all_samples = [s for group in grouped_samples for s in group]

        return RolloutBatch(
            tokens=[s.tokens for s in all_samples],
            loss_masks=[s.loss_mask for s in all_samples],
            rewards=[s.reward for s in all_samples],
            response_lengths=[len(s.response.split()) for s in all_samples],
        )
```

---

### 5. Local Inference Engines (Priority 1, 1 day)

**Status**: ⚠️ Have interface, need concrete implementation

**What we need**: Actually load models on GPUs 4-7 and generate completions

**File to create**: `rollouts/rollouts/training/inference/local_engine.py`

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List
from rollouts.training.types import Sample

class LocalInferenceEngine:
    """Inference engine running on specific GPU."""

    def __init__(
        self,
        model_name: str,
        gpu_id: int,
        max_new_tokens: int = 1024,
        temperature: float = 0.8,
    ):
        self.gpu_id = gpu_id
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        # Set device
        self.device = f"cuda:{gpu_id}"

        # Load model
        print(f"Loading {model_name} on GPU {gpu_id}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"✅ Model loaded on GPU {gpu_id}")

    def generate(self, prompts: List[str]) -> List[Sample]:
        """Generate completions for prompts.

        Args:
            prompts: List of prompts

        Returns:
            List of Sample objects with responses
        """
        # Tokenize
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode
        responses = self.tokenizer.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],  # Only new tokens
            skip_special_tokens=True,
        )

        # Create samples
        samples = []
        for prompt, response in zip(prompts, responses):
            # Tokenize full sequence (prompt + response)
            full_text = prompt + response
            tokens = self.tokenizer.encode(full_text)

            # Loss mask: 0 for prompt tokens, 1 for response tokens
            prompt_len = len(self.tokenizer.encode(prompt))
            loss_mask = [0.0] * prompt_len + [1.0] * (len(tokens) - prompt_len)

            sample = Sample(
                prompt=prompt,
                response=response,
                tokens=tokens,
                loss_mask=loss_mask,
            )
            samples.append(sample)

        return samples

    async def reload_weights(self, checkpoint_path: str):
        """Reload model weights from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint
        """
        print(f"GPU {self.gpu_id}: Reloading weights from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        print(f"GPU {self.gpu_id}: ✅ Weights reloaded")
```

---

## Putting It All Together: End-to-End Example

**File**: `examples/train_rl_single_node.py`

```python
#!/usr/bin/env python3
"""Single-node RL training with split GPUs (4 training + 4 inference)."""

import asyncio
from pathlib import Path

from rollouts.training.backends.pytorch import PyTorchTrainingBackend
from rollouts.training.datasets.data_buffer import DataBuffer
from rollouts.training.inference.local_engine import LocalInferenceEngine
from rollouts.training.loops.rl_loop import run_rl_training
from rollouts.training.reward_models import MathRewardModel
from rollouts.training.rollout_gen.async_rollout_manager import AsyncRolloutManager
from rollouts.training.types import RLTrainingConfig, RolloutConfig, Sample

async def main():
    # ════════════════════════════════════════════════════════════════
    # CONFIGURATION
    # ════════════════════════════════════════════════════════════════
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    # ════════════════════════════════════════════════════════════════
    # TRAINING BACKEND (GPUs 0-3)
    # ════════════════════════════════════════════════════════════════
    print("Setting up training backend...")
    training_backend = PyTorchTrainingBackend(
        model_name=model_name,
        world_size=4,
        strategy="fsdp",
        ref_model_path=None,  # Use initial weights as reference
    )

    # ════════════════════════════════════════════════════════════════
    # INFERENCE ENGINES (GPUs 4-7)
    # ════════════════════════════════════════════════════════════════
    print("Setting up inference engines...")
    inference_engines = [
        LocalInferenceEngine(
            model_name=model_name,
            gpu_id=gpu_id,
            max_new_tokens=1024,
            temperature=0.8,
        )
        for gpu_id in [4, 5, 6, 7]
    ]

    # ════════════════════════════════════════════════════════════════
    # DATA BUFFER
    # ════════════════════════════════════════════════════════════════
    print("Loading prompts...")
    # Load GSM8K or similar
    prompts = load_gsm8k_prompts("data/gsm8k/train.parquet")
    data_buffer = DataBuffer(prompts=prompts)

    # ════════════════════════════════════════════════════════════════
    # REWARD MODEL
    # ════════════════════════════════════════════════════════════════
    reward_model = MathRewardModel()

    def reward_fn(sample: Sample) -> float:
        return reward_model.score(
            prompt=sample.prompt,
            response=sample.response,
            label=sample.metadata.get("label", ""),
        )

    # ════════════════════════════════════════════════════════════════
    # ROLLOUT GENERATION
    # ════════════════════════════════════════════════════════════════
    async def generate_fn(prompts, config):
        # Distribute across 4 inference engines
        batch_size_per_engine = len(prompts) // 4

        tasks = []
        for i, engine in enumerate(inference_engines):
            start = i * batch_size_per_engine
            end = start + batch_size_per_engine if i < 3 else len(prompts)
            tasks.append(engine.generate(prompts[start:end]))

        # Run in parallel
        results = await asyncio.gather(*tasks)

        # Flatten
        return [sample for batch in results for sample in batch]

    rollout_config = RolloutConfig(
        batch_size=32,
        n_samples_per_prompt=8,  # GRPO: 8 completions per prompt
        generate_fn=generate_fn,
        reward_fn=reward_fn,
    )

    rollout_manager = AsyncRolloutManager(
        data_buffer=data_buffer,
        config=rollout_config,
    )

    # ════════════════════════════════════════════════════════════════
    # TRAINING CONFIG
    # ════════════════════════════════════════════════════════════════
    config = RLTrainingConfig(
        num_steps=1000,
        sync_every=10,
        log_every=10,
        checkpoint_every=100,
    )

    # ════════════════════════════════════════════════════════════════
    # RUN RL TRAINING
    # ════════════════════════════════════════════════════════════════
    print("\n" + "="*60)
    print("Starting RL training...")
    print("="*60 + "\n")

    metrics = await run_rl_training(
        backend=training_backend,
        data_buffer=data_buffer,
        rollout_manager=rollout_manager,
        inference_engines=inference_engines,
        config=config,
    )

    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Implementation Checklist (5-7 Days)

### Day 1-2: GRPO Loss
- [ ] Create `rollouts/training/rl_losses.py`
- [ ] Implement `grpo_loss()` function
- [ ] Write unit tests
- [ ] Integrate into `PyTorchTrainingBackend.forward_backward()`

### Day 3: Reference Model
- [ ] Add `ref_model` to `PyTorchTrainingBackend.__init__()`
- [ ] Modify `forward_backward()` to compute ref log probs
- [ ] Test with dummy data

### Day 4: Reward Model + Inference
- [ ] Create `rollouts/training/reward_models.py`
- [ ] Implement `MathRewardModel`
- [ ] Create `rollouts/training/inference/local_engine.py`
- [ ] Implement `LocalInferenceEngine`
- [ ] Test inference on single GPU

### Day 5: GRPO Rollout Generation
- [ ] Modify `AsyncRolloutManager.generate_batch()`
- [ ] Support `n_samples_per_prompt`
- [ ] Compute group-relative advantages
- [ ] Test with dummy prompts

### Day 6-7: Integration + Testing
- [ ] Create `examples/train_rl_single_node.py`
- [ ] Test end-to-end on small dataset
- [ ] Debug issues
- [ ] Add logging

---

## Testing Strategy

### Unit Tests
```python
# Test GRPO loss
def test_grpo_loss():
    from rollouts.training.rl_losses import grpo_loss
    # ... test with known inputs

# Test reward model
def test_math_reward_model():
    from rollouts.training.reward_models import MathRewardModel
    rm = MathRewardModel()

    # Test correct answer
    reward = rm.score(
        prompt="What is 2+2?",
        response="The answer is 4",
        label="4",
    )
    assert reward == 1.0

    # Test wrong answer
    reward = rm.score(
        prompt="What is 2+2?",
        response="The answer is 5",
        label="4",
    )
    assert reward == 0.0
```

### Integration Test
```bash
# Small test run (10 steps)
python examples/train_rl_single_node.py \
    --num-steps 10 \
    --batch-size 4 \
    --n-samples-per-prompt 2
```

---

## Summary: What You Need for Architecture 1

**5 components to build** (5-7 days):

1. ✅ **GRPO loss** - The core algorithm (2 days)
2. ✅ **Reference model** - For KL divergence (1 day)
3. ✅ **Reward model** - Math grading (1 day)
4. ✅ **Local inference** - Load models on GPUs 4-7 (1 day)
5. ✅ **GRPO rollout gen** - n_samples_per_prompt (1 day)

**No networking, no SSH, no MiniRay needed!** Just pure PyTorch on one machine.

After this works, Architecture 2 (two nodes) is trivial - just replace `LocalInferenceEngine` with `RemoteWorker` from MiniRay! 🚀
