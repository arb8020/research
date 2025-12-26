"""Base inference config and test logic.

Experiment files import from here and override config values.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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
    """Run script on remote GPU via broker/bifrost.

    Args:
        script_path: Path to the script (__file__ from caller)
        keep_alive: Keep GPU running after completion
        node_id: Reuse existing instance ID (skips provisioning)
    """
    import os

    from dotenv import load_dotenv

    from bifrost.client import BifrostClient
    from broker.client import GPUClient

    load_dotenv()

    # Get script path relative to git root
    import subprocess

    script = Path(script_path).resolve()
    git_root = Path(
        subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
    )
    rel_path = script.relative_to(git_root)

    # Provision or reuse GPU
    runpod_key = os.getenv("RUNPOD_API_KEY")
    assert runpod_key, "RUNPOD_API_KEY not set"
    ssh_key_path = os.getenv("SSH_KEY_PATH", "~/.ssh/id_ed25519")

    client = GPUClient(credentials={"runpod": runpod_key}, ssh_key_path=ssh_key_path)
    gpu = None

    try:
        if node_id:
            print(f"Reusing instance: {node_id}")
            gpu = client.get_instance(node_id, provider="runpod")
            if not gpu:
                print(f"GPU {node_id} not found (is it still running?)")
                return
            keep_alive = True
        else:
            print("Provisioning GPU...")
            gpu = client.create(
                query=(client.vram_gb >= 24) & (client.price_per_hour <= 0.5),
                name=f"inference-{script.stem}",
            )
            if not gpu:
                print("Failed to provision GPU")
                return
            print(f"GPU ready: {gpu.id}")

            if not gpu.wait_until_ssh_ready(timeout=300):
                print("SSH timeout")
                client.terminate_instance(gpu.id, gpu.provider)
                return

        print(f"SSH: {gpu.ssh_connection_string()}")

        # Deploy
        workspace = "~/.bifrost/workspaces/rollouts"
        bifrost = BifrostClient(gpu.ssh_connection_string(), ssh_key_path)
        bootstrap = [
            "cd rollouts && uv python install 3.12 && uv sync --python 3.12",
            "uv pip install torch 'transformers<4.52' accelerate",
        ]
        bifrost.push(workspace_path=workspace, bootstrap_cmd=bootstrap)
        print("Code deployed")

        # Run with streaming output
        remote_script = f"{workspace}/{rel_path}"
        cmd = f"cd {workspace}/rollouts && uv run python {remote_script}"
        print(f"Running: {cmd}")
        print("-" * 50)
        for line in bifrost.exec_stream(cmd):
            print(line, end="")
        print("-" * 50)

    except KeyboardInterrupt:
        print("\n\nInterrupted!")
        keep_alive = True

    finally:
        if gpu is None:
            return
        if keep_alive:
            print()
            print("=" * 50)
            print(f"Instance kept alive: {gpu.id}")
            print(f"SSH: {gpu.ssh_connection_string()}")
            print()
            print(f"Rerun with:   --node-id {gpu.id}")
            print(f"Terminate:    broker terminate {gpu.id}")
            print("=" * 50)
        else:
            print("Cleaning up...")
            client.terminate_instance(gpu.id, gpu.provider)
