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

Specifically tests whether this deployment's `--tool-call-parser deepseekv31`
plus `tool_chat_template_deepseekv32.jinja` preserve tool-calling correctness
end-to-end on a corpus authored for K2 (the parser name mismatch is flagged
in bench_deepseek_v3_2_amd_mi355x.py — this config measures the consequence).
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
_SGLANG_IMAGE = "lmsysorg/sglang:dsv32-rocm"

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
            concurrency=4,
            max_samples=20,
            extra_body=_KVV_DEFAULT_EXTRA_BODY,
            request_timeout_s=600.0,
        ),
    ],
    hardware=hardware,
    output=ServingOutputConfig(
        experiment_name="kimi_verifier_deepseek_v32_mi355x_smoke",
    ),
)
