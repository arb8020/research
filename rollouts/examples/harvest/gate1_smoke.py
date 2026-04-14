"""Gate 1 smoke test: verify harvest-sglang patch fires and writes activations.

What this tests:
- SGLang starts with harvest patch applied
- Hook fires on prefill forward passes
- Per-batch .npz files appear in harvest_output_dir
- Each file has shape [total_tokens, d_model] — no pooling at this stage

Run locally (requires GPU + patched sglang installed):
    python examples/harvest/gate1_smoke.py

Run remotely via argus:
    python -m argus run --config examples/harvest/gate1_smoke.py

To apply the patch manually before testing:
    SITE=$(python -c 'import sglang, os; print(os.path.dirname(os.path.dirname(sglang.__file__)))')
    patch -d $SITE -p1 < rollouts/third_party/miles_patches/v0.5.7/sglang.patch

Verification after run:
    python examples/harvest/gate1_smoke.py --verify-only --output-dir <path>
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from rollouts.training.configs import DepsConfig, HardwareConfig, InferenceConfig
from rollouts.training.grpo import GRPOConfig, GRPOOutputConfig, ModelConfig
from rollouts.training.smoke import run_inference_startup_smoke

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    gpu_type="A100",
    gpu_count=1,
    provider="modal",
    deps=DepsConfig(
        pip_packages=(
            "sglang[all]==0.5.7",
            "torch>=2.4",
            "transformers>=5.0",
        ),
        bootstrap_commands=(
            # Apply our harvest patch on top of the installed sglang package.
            # Patch paths are python/sglang/srt/... (from git diff of the sglang repo root).
            # uv pip show gives the site-packages dir, so we strip 2 path components (-p2)
            # to map python/sglang/srt/... -> sglang/srt/...
            (
                "SITE=$(/root/.local/bin/uv pip show sglang | grep -i '^Location' | awk '{print $2}') && "
                'echo "Applying harvest patch at $SITE (strip=2)" && '
                "patch -d $SITE -p2 --forward "
                "< /workspace/rollouts/third_party/miles_patches/v0.5.7/sglang.patch && "
                "echo 'Patch applied successfully'"
            ),
        ),
    ),
)

# ---------------------------------------------------------------------------
# Model: smallest available for fast smoke test
# ---------------------------------------------------------------------------

MODEL_NAME = "Qwen/Qwen3-0.6B"

# Layer at ~2/3 depth of Qwen3-0.6B (28 layers -> layer 18)
PRIMARY_LAYER = 18
HARVEST_OUTPUT_DIR = "/tmp/harvest_gate1"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

config = GRPOConfig(
    output=GRPOOutputConfig(experiment_name="harvest_gate1_smoke"),
    model=ModelConfig(
        name=MODEL_NAME,
        dtype="bfloat16",
    ),
    inference=InferenceConfig(
        spec="harvest-sglang",
        cuda_device_ids=(0,),
        port=30000,
        mem_fraction=0.5,
        disable_cuda_graph=True,  # disable for Gate 1 — CUDA graphs complicate hooks
        harvest_layers=(PRIMARY_LAYER,),
        harvest_output_dir=HARVEST_OUTPUT_DIR,
    ),
)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def train(cfg: GRPOConfig = config, **kwargs: object) -> dict:
    """Boot harvest-sglang, send a few prefill requests, verify .npz files appear."""
    result = run_inference_startup_smoke(config=cfg, **kwargs)

    # After server starts, send a few requests to trigger captures
    import time

    import httpx

    port = cfg.inference.port
    base_url = f"http://localhost:{port}"

    test_texts = [
        "The capital of France is",
        "Once upon a time in a land far away",
        "The quick brown fox jumps over the lazy dog",
    ]

    logger.info("Sending %d test requests to trigger activation capture...", len(test_texts))
    for text in test_texts:
        try:
            resp = httpx.post(
                f"{base_url}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": [{"role": "user", "content": text}],
                    "max_tokens": 1,  # We only care about prefill
                },
                timeout=30.0,
            )
            resp.raise_for_status()
        except Exception as e:
            logger.warning("Request failed (expected if server already torn down): %s", e)

    time.sleep(2.0)  # Let writer finish any pending I/O

    # Verify
    _verify_output(HARVEST_OUTPUT_DIR, PRIMARY_LAYER)

    return {**result, "harvest_verified": True}


def _verify_output(output_dir: str, layer_idx: int) -> None:
    """Check that .npz files exist with expected shape."""
    layer_dir = Path(output_dir) / f"layer_{layer_idx}"
    files = sorted(layer_dir.glob("*.npz")) if layer_dir.exists() else []

    if not files:
        logger.error("GATE 1 FAIL: no .npz files found in %s — hook did not fire", layer_dir)
        raise RuntimeError(f"No harvest output files in {layer_dir}")

    logger.info("Found %d .npz files in %s", len(files), layer_dir)

    for path in files[:3]:  # spot-check first 3
        data = np.load(str(path))
        assert "activations" in data, f"Missing 'activations' key in {path}"
        arr = data["activations"]
        assert arr.ndim == 2, f"Expected 2D [total_tokens, d_model], got shape {arr.shape}"
        assert arr.shape[1] > 0, f"d_model is 0 in {path}"
        logger.info("  %s: shape=%s dtype=%s", path.name, arr.shape, arr.dtype)

    logger.info("GATE 1 PASS: hook fires, files written, shapes correct.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip server startup, just verify existing output files",
    )
    parser.add_argument(
        "--output-dir",
        default=HARVEST_OUTPUT_DIR,
        help="Directory to verify",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if args.verify_only:
        _verify_output(args.output_dir, PRIMARY_LAYER)
    else:
        train()
