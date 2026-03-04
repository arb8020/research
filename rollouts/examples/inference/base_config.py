"""Base inference config and test logic.

Experiment files import from here and override config values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseConfig:
    """Base inference configuration. Override in experiment files."""

    # Model
    model_name: str = "Qwen/Qwen2.5-0.5B"

    # Generation
    prompts: tuple[str, ...] = ("Hello, my name is",)
    temperature: float = 0.7
    max_tokens: int = 50
    num_samples_per_prompt: int = 1


def test_inference(config: BaseConfig) -> list[dict]:
    """Run inference test with the given config."""
    import logging

    import torch

    from rollouts._logging import setup_logging

    # Local imports to avoid loading torch on import
    from rollouts.inference import EngineConfig, InferenceEngine, SamplingParams

    setup_logging(level="INFO", use_color=True)
    logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        logger.error("CUDA not available")
        return []

    logger.info(f"Model: {config.model_name}")
    logger.info(f"Prompts: {len(config.prompts)}")
    logger.info(f"Temperature: {config.temperature}")
    logger.info(f"Max tokens: {config.max_tokens}")

    # Create engine
    logger.info("Loading model...")
    engine_config = EngineConfig(model_path=config.model_name)
    engine = InferenceEngine(engine_config)
    logger.info("Model loaded")

    # Generate
    logger.info("=" * 50)
    logger.info("Generating...")
    logger.info("=" * 50)

    sampling_params = SamplingParams(
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )

    samples = engine.generate_text(
        prompts=list(config.prompts),
        sampling_params=sampling_params,
        num_samples_per_prompt=config.num_samples_per_prompt,
    )

    # Report results
    results = []
    for i, sample in enumerate(samples):
        prompt_idx = i // config.num_samples_per_prompt
        sample_idx = i % config.num_samples_per_prompt

        # Decode tokens
        completion_text = engine.tokenizer.decode(
            sample.completion_tokens, skip_special_tokens=True
        )

        result = {
            "prompt_idx": prompt_idx,
            "sample_idx": sample_idx,
            "prompt": config.prompts[prompt_idx],
            "completion": completion_text,
            "num_tokens": len(sample.completion_tokens),
            "finish_reason": sample.finish_reason,
            "mean_logprob": sum(sample.logprobs) / len(sample.logprobs) if sample.logprobs else 0,
        }
        results.append(result)

        logger.info(f"\n[{prompt_idx}.{sample_idx}] {config.prompts[prompt_idx]}")
        logger.info(f"  -> {completion_text}")
        logger.info(
            f"  tokens={len(sample.completion_tokens)}, finish={sample.finish_reason}, mean_logprob={result['mean_logprob']:.3f}"
        )

    logger.info("=" * 50)
    logger.info(f"Generated {len(samples)} samples")
    logger.info("=" * 50)

    engine.shutdown()
    return results


def run_remote(script_path: str, keep_alive: bool = False, node_id: str | None = None) -> None:
    """Run script on remote GPU via rollouts.run."""
    import trio

    from rollouts.run import run_remote as run_remote_impl

    trio.run(
        run_remote_impl,
        script_path=script_path,
        keep_alive=keep_alive,
        node_id=node_id,
        tail=True,
        allow_dirty=True,
        skip_hf_token_check=True,
        raw_script=True,
    )
