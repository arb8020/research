"""KernelBench config using Modal-hosted SGLang endpoint.

Uses the SGLang server deployed via modal_sglang.py.

Prerequisites:
    1. Deploy SGLang on Modal (if not already):
       modal deploy examples/kernelbench/modal_sglang.py

    2. Run eval (GPU scoring on Modal):
       python examples/kernelbench/run_eval.py configs/modal_sglang.py --modal

    First request will cold-start the endpoint (~2-3 min to download model).
"""

from pathlib import Path

from rollouts.core import Endpoint

# Modal-hosted SGLang endpoint
# Deployed via: modal deploy examples/kernelbench/modal_sglang.py
MODAL_WORKSPACE = "arb8020"
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

endpoint = Endpoint(
    model=MODEL_ID,
    base_url=f"https://{MODAL_WORKSPACE}--sglang-serve-serve.modal.run/v1",
    api_format="openai-completions",  # SGLang uses OpenAI chat completions
)

# Dataset configuration
levels = [1]  # Level 1 = easiest
backend = "CUDA"
max_samples = 10

# Evaluation configuration
max_turns = 8
max_concurrent = 2

# Output
output_dir = Path("results/kernelbench/modal_sglang")
eval_name = "kernelbench_modal_sglang"
verbose = True

# GPU scoring on Modal sandbox (set by --modal flag)
sandbox_configs = []
