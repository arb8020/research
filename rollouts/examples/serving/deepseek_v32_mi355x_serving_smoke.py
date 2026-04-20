"""Small mixed serving smoke scenario for a managed DeepSeek V3.2 server on MI355X.

This smoke config intentionally stays tiny:
- `random`: synthetic fixed-length prompts
- `sharegpt`: a tiny local ShareGPT-format JSONL
- `shared_prefix`: hand-authored prompts that reuse a long common prefix

All three workloads share one owned endpoint and the same output cap so Argus
can launch the MI355X DeepSeek server once, then run the mixed workload slate
against it.

Usage:
    python -m argus run --config examples/serving/deepseek_v32_mi355x_serving_smoke.py
"""

from __future__ import annotations

from pathlib import Path

from examples.inference.bench_config_lib import InferenceBenchWorkload, build_bench_tasks
from examples.inference.bench_workload_lib import prepare_bench_messages
from rollouts.core import Score
from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.serving.configs import EvalServingWorkload, ServingOutputConfig, ServingScenario
from rollouts.training.configs import DepsConfig, HardwareConfig
from rollouts.training.scoring import FunctionScorer

MODEL = "deepseek-ai/DeepSeek-V3.2"
PORT = 30000
_SGLANG_IMAGE = "lmsysorg/sglang:dsv32-rocm"
OUTPUT_LEN = 64
CONCURRENCY_PER_WORKLOAD = 2

_no_op_scorer = FunctionScorer(lambda attempt, _ctx: Score(metrics=()))

_random_workload = InferenceBenchWorkload(
    kind="random",
    num_prompts=4,
    input_len=256,
    output_len=OUTPUT_LEN,
    seed=42,
    max_concurrent=CONCURRENCY_PER_WORKLOAD,
)

_sharegpt_workload = InferenceBenchWorkload(
    kind="sharegpt",
    num_prompts=4,
    input_len=256,
    output_len=OUTPUT_LEN,
    dataset_path=str(Path(__file__).with_name("data") / "sharegpt_smoke.jsonl"),
    seed=42,
    max_concurrent=CONCURRENCY_PER_WORKLOAD,
)

_shared_prefix = (
    "You are benchmarking a long-context inference endpoint. "
    "Assume the following repeated context block is stable across many requests. "
    "Focus on latency, throughput, prompt reuse, and cache locality. "
    "When you answer, stay concise, technical, and factual. "
) * 32

_shared_prefix_tasks = [
    {
        "prompt": (
            f"{_shared_prefix}\n\n"
            "Request: Briefly explain why a cache-friendly serving workload can have much lower prefill cost."
        ),
        "target_output_len": OUTPUT_LEN,
        "workload": "shared_prefix",
        "prefix_chars": len(_shared_prefix),
    },
    {
        "prompt": (
            f"{_shared_prefix}\n\n"
            "Request: Describe one reason p95 latency can worsen before throughput peaks."
        ),
        "target_output_len": OUTPUT_LEN,
        "workload": "shared_prefix",
        "prefix_chars": len(_shared_prefix),
    },
    {
        "prompt": (
            f"{_shared_prefix}\n\n"
            "Request: Explain why repeated system context is a good candidate for prompt caching."
        ),
        "target_output_len": OUTPUT_LEN,
        "workload": "shared_prefix",
        "prefix_chars": len(_shared_prefix),
    },
    {
        "prompt": (
            f"{_shared_prefix}\n\n"
            "Request: What serving metric would you track first when concurrency increases from low to moderate?"
        ),
        "target_output_len": OUTPUT_LEN,
        "workload": "shared_prefix",
        "prefix_chars": len(_shared_prefix),
    },
]

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --ipc host --network host"
    f" --shm-size 128G"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --env SGLANG_NSA_FUSE_TOPK=false"
    f" --env SGLANG_NSA_KV_CACHE_STORE_FP8=false"
    f" --env SGLANG_NSA_USE_REAL_INDEXER=true"
    f" --env SGLANG_NSA_USE_TILELANG_PREFILL=True"
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    f" bash -c 'USE_ROCM=true ROCM_HOME=/opt/rocm pip install -q /root/tilelang && python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --tool-call-parser deepseekv31"
    f" --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    f" --nsa-prefill tilelang"
    f" --nsa-decode aiter"
    f" --enable-cache-report'"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,
    max_tokens=OUTPUT_LEN,
)

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
            f"docker pull {_SGLANG_IMAGE}",
        ),
    ),
)

random_eval = EvalTaskSpec(
    tasks=build_bench_tasks(_random_workload),
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=4,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="deepseek_v32_random_smoke"),
)

sharegpt_eval = EvalTaskSpec(
    tasks=build_bench_tasks(_sharegpt_workload),
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=4,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="deepseek_v32_sharegpt_smoke"),
)

shared_prefix_eval = EvalTaskSpec(
    tasks=_shared_prefix_tasks,
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_bench_messages,
    ),
    scorer=_no_op_scorer,
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=4,
        max_turns=1,
        verbose=False,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="deepseek_v32_shared_prefix_smoke"),
)

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        EvalServingWorkload(
            name="random_smoke_c2",
            eval_task=random_eval,
            concurrency=CONCURRENCY_PER_WORKLOAD,
            max_samples=4,
        ),
        EvalServingWorkload(
            name="sharegpt_smoke_c2",
            eval_task=sharegpt_eval,
            concurrency=CONCURRENCY_PER_WORKLOAD,
            max_samples=4,
        ),
        EvalServingWorkload(
            name="shared_prefix_smoke_c2",
            eval_task=shared_prefix_eval,
            concurrency=CONCURRENCY_PER_WORKLOAD,
            max_samples=4,
        ),
    ],
    hardware=hardware,
    output=ServingOutputConfig(
        experiment_name="deepseek_v32_mi355x_serving_smoke",
    ),
)
