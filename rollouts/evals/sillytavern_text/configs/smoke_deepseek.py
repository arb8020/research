"""Smoke against self-hosted DeepSeek V3.2 on MI355X (8x, TP=8, FP8, NSA).

Primary = DeepSeek V3.2 via SGLang. User-sim (responder) stays on hosted
Haiku via our eval.py's make_environment.

Uses the repo-native OwnedEndpoint + HardwareConfig(provider="ssh") path, which
auto-manages the full lifecycle: launches the SGLang container on the remote
node, waits for readiness, opens a paramiko SSH tunnel in-process, runs the
eval against localhost:<picked_port>, tears everything down on exit.

No manual `ssh -L` needed. Run via:

    python -m argus run --config rollouts/evals/sillytavern_text/configs/smoke_deepseek.py

Launch args mirror rollouts/examples/inference/evals/configs/bench/
bench_deepseek_v3_2_amd_mi355x.py — same image (lmsysorg/sglang:dsv32-rocm),
same TP=8, same NSA config. The only changes are the workload (our
prepare_messages + DialogueEnvironment) and max_tokens (1024 instead of 256
so RP replies don't truncate).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rollouts.eval import AgentRunSpec, EvalOutputConfig, EvalRunConfig, EvalTaskSpec
from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.training.configs import DepsConfig, HardwareConfig

EVAL_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(EVAL_DIR))

# Import sibling modules from the eval package. run.py loads this config by
# path, not as a package import, so relative imports don't work — we go through
# sys.path. Matches the shim in eval.py.
from eval import (  # type: ignore  # noqa: E402
    make_environment,
    prepare_messages,
    score_sample,
)
from eval import (
    spec as _unused_eval_spec,  # noqa: F401  imported for side-effect parity
)

from rollouts.training.scoring import FunctionScorer  # noqa: E402

TASKS_PATH = EVAL_DIR / "tasks.jsonl"

MODEL = "deepseek-ai/DeepSeek-V3.2"
PORT = 30000
_SGLANG_IMAGE = "lmsysorg/sglang:dsv32-rocm"

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
            f"docker pull {_SGLANG_IMAGE}",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint — identical launch_cmd to the working bench config.
# Weights cached at /models/hf_cache; re-launches are fast (~2 min).
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker rm -f sglang_bench_{PORT} 2>/dev/null;"
    f" docker run --rm"
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
    # Need DeepSeek's chat template — the tokenizer ships without a default
    # chat_template. The "tool_" jinja file is the only DeepSeek one in the
    # image and works fine for plain-text use too.
    f" --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"
    # V3.2 is a reasoning model; without a reasoning parser SGLang puts the
    # entire "<think>...</think>reply" blob into content. With deepseek-v3
    # parser, the think portion is split into reasoning_content and the
    # actual reply lands in content — which is what rollout_openai expects.
    # (See bench_glm_5_1 and bench_kimi_k2_5 for the same pattern.)
    f" --reasoning-parser deepseek-v3"
    # Log request/response bodies so we can debug future wire-level issues
    # without relaunching the endpoint manually.
    f" --log-requests --log-requests-level 2"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    f" --nsa-prefill tilelang"
    f" --nsa-decode aiter'"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,
    max_tokens=1024,  # override the 256-tok bench default — RP replies need room
    # chat_template_kwargs.thinking=True: DeepSeek V3.2's jinja template
    # defaults thinking=false, which emits "<｜Assistant｜></think>" (closing
    # think tag with no opening) as the generation prompt. thinking=true
    # emits "<｜Assistant｜><think>" which V3.2 handles correctly.
    # Flows through to SGLang via extra_body in the OpenAI SDK call.
    extra_params={"chat_template_kwargs": {"thinking": True}},
)

# ---------------------------------------------------------------------------
# Eval task
# ---------------------------------------------------------------------------


def _load_tasks() -> list[dict]:
    return [json.loads(line) for line in TASKS_PATH.read_text().splitlines() if line.strip()]


# DialogueEnvironment has no tools, so "no tool call this turn" isn't a
# termination signal. Override eval/run.py's default stop_on_no_tool with a
# noop so max_turns drives termination. (Same issue we hit on the
# EvalSpec path via has_tools=False; AgentRunSpec has its own override.)
async def _noop_no_tool(state: Any, _run_config: Any) -> Any:
    return state


eval_task = EvalTaskSpec(
    tasks=_load_tasks()[:1],  # smoke: 1 scenario
    run_spec=AgentRunSpec(
        endpoint=endpoint,
        prepare_messages=prepare_messages,
        environment_factory=make_environment,
        handle_no_tool=_noop_no_tool,
    ),
    scorer=FunctionScorer(score_sample),
    run=EvalRunConfig(
        max_concurrent=1,
        max_samples=1,
        max_turns=8,
        verbose=True,
        show_progress=True,
    ),
    output=EvalOutputConfig(experiment_name="sillytavern_text_smoke_dsv32"),
    hardware=hardware,
)
