"""K2-Vendor-Verifier-shape tool-call conformance smoke against DSV3.2 on MI355X.

Single workload, small sample count. Exercises the ToolCallVerifierWorkload
branch in rollouts/serving/run.py end-to-end: managed OwnedEndpoint + K2VV
corpus + jsonschema classification + K2VV-shape engine_report.json.

Usage:
    python -m argus run --config examples/serving/kimi_verifier_deepseek_v32_mi355x_smoke.py

What gets measured:
- finish_reason distribution (stop / tool_calls / others)
- schema_accuracy over declared tools (K2VV's successful_tool_call_count /
  finish_tool_calls)
- usage totals

Uses `--tool-call-parser deepseekv32` and a matching `tool_chat_template_deepseekv32.jinja`
on the v0.5.9-rocm700-mi35x image (the older dsv32-rocm image lacked the
`deepseekv32` parser, which produced 0/20 triggers on the first K2VV smoke).
"""

from __future__ import annotations

from rollouts.eval.configs import EndpointCapabilities, OwnedEndpoint
from rollouts.serving.configs import (
    ServingOutputConfig,
    ServingScenario,
    ToolCallVerifierWorkload,
)
from rollouts.training.configs import DepsConfig, HardwareConfig

MODEL = "deepseek-ai/DeepSeek-V3.2"
PORT = 30000
# v0.5.9 ships the `deepseekv32` tool-call parser (the DSML tag format that
# DSV3.2 actually emits); the older `dsv32-rocm` image only has `deepseekv31`,
# which matches a different wire format and produces 0/20 tool-call triggers
# on K2VV. v0.5.9 also has tilelang pre-installed, so we drop the
# `pip install /root/tilelang` prelude from the docker run command.
_SGLANG_IMAGE = "lmsysorg/sglang:v0.5.9-rocm700-mi35x"

# TODO(nix): this docker run command is an f-string jamming together image
# selection, device flags, env vars, entrypoint shell, and sglang CLI flags.
# Editing one value (e.g. --tool-call-parser) means staring at a 30-line
# string literal looking for the right `--flag` to change. A nix derivation
# that emits the docker run command from structured data (image, env_vars,
# launch_args as a dict/list) would:
#   - make the "which parser are we using" question one dict lookup
#   - let shared fragments (NSA env block, MI355X device flags) live in one
#     place and be reused across serving configs for DSV3.2
#   - fail at eval time if a required env var is missing, instead of
#     succeeding-but-wrong at runtime
# The tool-call-parser mismatch that produced 0/20 in the first K2VV run
# was exactly a needle-in-f-string bug — worth the nix leverage here.
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
    f" python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --tool-call-parser deepseekv32"
    f" --chat-template /sgl-workspace/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"
    f" --disable-cuda-graph"
    f" --mem-fraction-static 0.85"
    f" --page-size 64"
    f" --nsa-prefill-backend tilelang"
    f" --nsa-decode-backend aiter"
    f" --enable-cache-report"
)

endpoint = OwnedEndpoint(
    spec="custom-http",
    launch_cmd=_docker_run,
    cuda_device_ids=(0, 1, 2, 3, 4, 5, 6, 7),
    model=MODEL,
    port=PORT,
    capabilities=EndpointCapabilities(weight_sync=None),
    startup_timeout=7200.0,
    max_tokens=4096,
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

# KVV's published numbers use these non-thinking defaults; the existing
# deepseek_vendor_verifier.py smoke flags them too. Applied as a floor —
# any per-row values in the corpus still win.
_KVV_DEFAULT_EXTRA_BODY = {
    "temperature": 0.6,
    "top_p": 0.95,
}

serving_scenario = ServingScenario(
    endpoint=endpoint,
    workloads=[
        ToolCallVerifierWorkload(
            name="kimi_verifier_smoke",
            concurrency=8,
            max_samples=200,
            extra_body=_KVV_DEFAULT_EXTRA_BODY,
            request_timeout_s=600.0,
        ),
    ],
    hardware=hardware,
    output=ServingOutputConfig(
        experiment_name="kimi_verifier_deepseek_v32_mi355x_smoke",
    ),
)
