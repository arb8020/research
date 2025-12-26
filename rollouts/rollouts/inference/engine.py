"""Inference engine - orchestrates model, cache, and scheduler.

This is a class because it owns GPU resources and needs cleanup.
Pure functions do the actual work.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from rollouts.inference.sampling import sample_with_logprobs
from rollouts.inference.scheduler import schedule
from rollouts.inference.types import (
    EngineConfig,
    SamplingParams,
    SchedulerConfig,
    Sequence,
    SequenceStatus,
    TrainingSample,
)


class InferenceEngine:
    """Main inference engine.

    Why a class?
    - Owns GPU resources (model, KV cache)
    - Manages sequence lifecycle
    - Needs shutdown() for cleanup

    Pure functions do the work:
    - schedule() decides what to run
    - sample_with_logprobs() does sampling
    """

    def __init__(self, config: EngineConfig) -> None:
        assert config.block_size > 0
        assert config.max_batch_size > 0

        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        self.model = self._load_model(config.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)

        # Scheduler config
        self.scheduler_config = SchedulerConfig(
            max_batch_size=config.max_batch_size,
            max_tokens_per_batch=config.max_batch_size * 512,  # TODO: make configurable
            block_size=config.block_size,
        )

        # Sequence state
        self.waiting: list[Sequence] = []
        self.running: list[Sequence] = []
        self.seq_counter = 0
        self.weight_version = 0

    def _load_model(self, model_path: str) -> nn.Module:
        """Load HuggingFace model."""
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
        model.to(self.device)
        model.eval()
        return model

    # ═══════════════════════════════════════════════════
    # HIGH-LEVEL API
    # ═══════════════════════════════════════════════════

    def generate(
        self,
        prompts: list[list[int]],
        sampling_params: SamplingParams,
        num_samples_per_prompt: int = 1,
    ) -> list[TrainingSample]:
        """High-level: generate completions with logprobs.

        Args:
            prompts: List of token ID lists
            sampling_params: Temperature, max_tokens, etc.
            num_samples_per_prompt: N completions per prompt (for GRPO)

        Returns:
            List of TrainingSample (len = len(prompts) * num_samples_per_prompt)
        """
        assert len(prompts) > 0
        assert num_samples_per_prompt > 0

        # Add requests
        for prompt in prompts:
            for _ in range(num_samples_per_prompt):
                self.add_request(list(prompt), sampling_params)

        # Run until done
        results: list[TrainingSample] = []
        while self.has_pending():
            finished = self.step()
            results.extend(finished)

        assert len(results) == len(prompts) * num_samples_per_prompt
        return results

    def generate_text(
        self,
        prompts: list[str],
        sampling_params: SamplingParams,
        num_samples_per_prompt: int = 1,
    ) -> list[TrainingSample]:
        """Convenience: generate from text prompts."""
        token_prompts = [self.tokenizer.encode(p, add_special_tokens=True) for p in prompts]
        return self.generate(token_prompts, sampling_params, num_samples_per_prompt)

    # ═══════════════════════════════════════════════════
    # MID-LEVEL API
    # ═══════════════════════════════════════════════════

    def add_request(self, prompt_tokens: list[int], params: SamplingParams) -> int:
        """Add single request, return sequence ID."""
        assert len(prompt_tokens) > 0

        seq = Sequence(
            seq_id=self.seq_counter,
            token_ids=list(prompt_tokens),
            block_ids=[],
            num_prompt_tokens=len(prompt_tokens),
            status=SequenceStatus.WAITING,
            temperature=params.temperature,
            max_tokens=params.max_tokens,
            stop_token_ids=params.stop_token_ids,
            output_logprobs=[],
        )
        self.seq_counter += 1
        self.waiting.append(seq)
        return seq.seq_id

    def step(self) -> list[TrainingSample]:
        """Run one scheduling + forward pass. Returns finished sequences."""
        if not self.has_pending():
            return []

        # Schedule (pure function)
        # For now, no KV cache so unlimited "blocks"
        num_free_blocks = 1000000
        sched_out = schedule(
            self.waiting,
            self.running,
            num_free_blocks,
            self.scheduler_config,
        )

        # Move sequences between queues based on schedule
        prefill_seqs = self._pop_seqs_by_id(self.waiting, sched_out.prefill_seqs)
        decode_seqs = [s for s in self.running if s.seq_id in sched_out.decode_seqs]

        # Mark prefill seqs as running
        for seq in prefill_seqs:
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)

        # Combine for batch forward
        batch_seqs = prefill_seqs + decode_seqs
        if not batch_seqs:
            return []

        # Forward pass
        finished = self._forward_batch(batch_seqs)

        # Remove finished from running
        finished_ids = {s.seq_id for s in finished}
        self.running = [s for s in self.running if s.seq_id not in finished_ids]

        return [s.to_training_sample(self.weight_version) for s in finished]

    def has_pending(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    # ═══════════════════════════════════════════════════
    # WEIGHT MANAGEMENT
    # ═══════════════════════════════════════════════════

    def update_weights(self, state_dict: dict, blocking: bool = True) -> None:
        """Update model weights."""
        assert state_dict, "empty state_dict"
        # For now, only blocking sync
        self.model.load_state_dict(state_dict)
        self.weight_version += 1

    def get_weight_version(self) -> int:
        return self.weight_version

    def flush_cache(self) -> None:
        """Clear KV cache (call after weight update)."""
        # TODO: implement when we have KV cache
        pass

    def shutdown(self) -> None:
        """Cleanup resources."""
        # Clear sequences
        self.waiting.clear()
        self.running.clear()
        # Model cleanup handled by garbage collection

    # ═══════════════════════════════════════════════════
    # INTERNAL
    # ═══════════════════════════════════════════════════

    def _pop_seqs_by_id(self, queue: list[Sequence], seq_ids: tuple[int, ...]) -> list[Sequence]:
        """Remove and return sequences with given IDs from queue."""
        id_set = set(seq_ids)
        popped = [s for s in queue if s.seq_id in id_set]
        queue[:] = [s for s in queue if s.seq_id not in id_set]
        return popped

    def _forward_batch(self, seqs: list[Sequence]) -> list[Sequence]:
        """Run forward pass on batch, return finished sequences.

        This is the simple version without KV cache - recomputes everything.
        """
        finished: list[Sequence] = []

        # Process each sequence (no batching for simplicity in v1)
        for seq in seqs:
            # Get next token
            input_ids = torch.tensor([seq.token_ids], device=self.device)

            with torch.no_grad():
                outputs = self.model(input_ids)
                # Get logits for last position
                last_logits = outputs.logits[:, -1, :]  # [1, vocab]

            # Sample (pure function)
            temps = torch.tensor([seq.temperature], device=self.device)
            tokens, logprobs = sample_with_logprobs(last_logits, temps)

            next_token = tokens[0].item()
            next_logprob = logprobs[0].item()

            # Update sequence
            seq.append_token(next_token, next_logprob)

            # Check stopping conditions
            if self._should_stop(seq, next_token):
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)

        return finished

    def _should_stop(self, seq: Sequence, token_id: int) -> bool:
        """Check if sequence should stop generating."""
        # Hit stop token
        if token_id in seq.stop_token_ids:
            return True

        # Hit EOS
        if token_id == self.tokenizer.eos_token_id:
            return True

        # Hit max tokens
        if seq.num_generated >= seq.max_tokens:
            return True

        return False
