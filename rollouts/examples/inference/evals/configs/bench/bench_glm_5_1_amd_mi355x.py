"""Throughput/latency benchmark for GLM-5.1 on AMD Instinct MI355X (8x GPU).

NOTE: GLM-5.1 has no official AMD/ROCm deployment documentation.
The official vLLM image (vllm/vllm-openai:glm51) is CUDA-only.
This config uses the standard vLLM ROCm image as a best-effort attempt.
Expect this to require debugging — verify it works before relying on results.

If it fails, options:
  1. Build a custom ROCm image with GLM-5.1 support (check zai-org/GLM-5 repo)
  2. Switch to SGLang: the SGLang dsv32-rocm image may support GLM-5.1 (same MoE arch)
  3. Serve on CUDA hardware instead

Hardware: root@66.42.120.238
  - 8x AMD Instinct MI355X, 288GB VRAM each (2.3TB total)
  - /dev/nvme0n1 mounted at /models (6.6TB free)

Model: zai-org/GLM-5.1-FP8 (~754B MoE, FP8)
  - Requires all 8 GPUs
  - MTP (speculative decoding) enabled for throughput

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_glm_5_1_amd_mi355x.py --force-deploy-committed

Reference:
    vLLM recipe (CUDA): https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
    vLLM ROCm image: vllm/vllm-openai-rocm:latest
"""

from examples.inference.bench_config_lib import (
    SUMMARY_DISTRIBUTION_PERCENTILES as DEFAULT_SUMMARY_DISTRIBUTION_PERCENTILES,
)
from examples.inference.bench_config_lib import (
    InferenceBenchSLA,
    InferenceBenchWorkload,
    build_bench_tasks,
)
from examples.inference.bench_workload_lib import prepare_bench_messages
from rollouts.core import Score
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer

_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

MODEL = "zai-org/GLM-5.1-FP8"
PORT = 30000

# Standard ROCm vLLM image — no dedicated GLM-5.1 AMD image exists yet.
# The glm51 image (vllm/vllm-openai:glm51) is CUDA-only.
_VLLM_IMAGE = "vllm/vllm-openai-rocm:latest"

# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------

WORKLOAD = InferenceBenchWorkload(
    kind="random",
    num_prompts=200,
    input_len=512,
    output_len=256,
    seed=42,
    max_concurrent=32,
)
SLA = InferenceBenchSLA()
SUMMARY_DISTRIBUTION_PERCENTILES = DEFAULT_SUMMARY_DISTRIBUTION_PERCENTILES

tasks = build_bench_tasks(WORKLOAD)

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

hardware = HardwareConfig(
    provider="ssh",
    ssh="root@66.42.120.238:22",
    ssh_key_path="~/.ssh/id_ed25519",
    gpu_type="MI355X",
    gpu_count=8,
    use_torchrun=False,
    deps=DepsConfig(
        bootstrap_commands=(
            "mount /dev/nvme0n1 /models 2>/dev/null || true",
            f"docker pull {_VLLM_IMAGE}",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# Uses standard ROCm vLLM flags. GLM-5.1-specific parsers (glm47/glm45)
# may not be present in the standard ROCm image — update if they're missing.
# MTP (speculative decoding) omitted since it may not be in this image.
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --security-opt seccomp=unconfined"
    f" --ipc host --network host"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --env VLLM_ROCM_USE_AITER=1"
    f" --env VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4"
    f" --env VLLM_ROCM_USE_AITER_RMSNORM=0"
    f" --name vllm_bench_{PORT}"
    f" {_VLLM_IMAGE}"
    f" {MODEL}"
    f" --tensor-parallel-size 8"
    f" --tool-call-parser glm47"
    f" --reasoning-parser glm45"
    f" --enable-auto-tool-choice"
    f" --chat-template-content-format string"
    f" --trust-remote-code"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --served-model-name glm-5.1-fp8"
    f" --max-model-len 8192"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,
    max_tokens=WORKLOAD.output_len,
)

# ---------------------------------------------------------------------------
# Eval task
# ---------------------------------------------------------------------------

eval_task = EvalTaskSpec(
    tasks=tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=WORKLOAD.max_concurrent,
        max_samples=WORKLOAD.num_prompts,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="bench_glm_5_1_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
