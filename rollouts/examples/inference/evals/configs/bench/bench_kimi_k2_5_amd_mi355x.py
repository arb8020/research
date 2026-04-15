"""Throughput/latency benchmark for Kimi K2.5 on AMD Instinct MI355X (8x GPU).

Uses SGLang's MI355X ROCm image. vLLM ROCm fails with
assert num_head_qo % 16 == 0 at tp=8 for Kimi K2.5's attention head count.
Weights downloaded to /models/hf_cache on the NVMe drive (7TB, mounted at /models).

Hardware: root@66.42.120.238
  - 8x AMD Instinct MI355X, 288GB VRAM each (2.3TB total)
  - /dev/nvme0n1 mounted at /models (6.6TB free)

Model: moonshotai/Kimi-K2.5
  - MoE architecture, FP8 weights
  - Requires all 8 GPUs (--tensor-parallel-size 8)

First run will download model weights — expect 30-60min before server starts.
Subsequent runs reuse the cached weights from /models/hf_cache.

Usage:
    python -m argus run --config examples/inference/evals/configs/bench/bench_kimi_k2_5_amd_mi355x.py --force-deploy-committed

    # Monitor:
    tail -f results/eval/<run>/run.jsonl | jq 'select(.event | test("inference_startup|health|eval_end"))'

Reference:
    SGLang cookbook: https://cookbook.sglang.io/autoregressive/Moonshotai/Kimi-K2.5
    Docker image: lmsysorg/sglang:v0.5.9-rocm700-mi35x
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

MODEL = "moonshotai/Kimi-K2.5"
PORT = 30000

# dsv32-rocm has the right SGLang + transformers versions for Kimi K2.5.
# Requires PYTHONPATH workaround for the hyphenated package name bug.
_SGLANG_IMAGE = "lmsysorg/sglang:dsv32-rocm"

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
            f"docker pull {_SGLANG_IMAGE}",
            # Write the import-fix script that patches Kimi K2.5 model files.
            # Kimi-K2.5 uses relative imports (from .X import Y) which Python
            # can't handle when the package name contains hyphens/dots.
            # Write sitecustomize.py to /models so the container can install it.
            # This patches transformers dynamic_module_utils to escape invalid
            # Python identifiers (hyphens/dots from Kimi-K2.5 package name).
            r"""cat > /models/sitecustomize.py << 'SCEOF'
import re, sys, os
try:
    import transformers.dynamic_module_utils as dmu
    import importlib, importlib.util, hashlib
    from pathlib import Path
    def _patched_gcim(class_name, module_path, *, force_reload=False):
        from transformers.dynamic_module_utils import HF_MODULES_CACHE, get_relative_import_files, _HF_REMOTE_CODE_LOCK
        name = os.path.normpath(module_path)
        if name.endswith(".py"): name = name[:-3]
        name = name.replace(os.path.sep, ".")
        name = re.sub(r"[^a-zA-Z0-9._]", "_", name)
        name = re.sub(r"[.][.]+", ".", name)
        module_file = Path(HF_MODULES_CACHE) / module_path
        with _HF_REMOTE_CODE_LOCK:
            if force_reload:
                sys.modules.pop(name, None); importlib.invalidate_caches()
            cached = sys.modules.get(name)
            spec = importlib.util.spec_from_file_location(name, location=module_file, submodule_search_locations=[str(module_file.parent)])
            files = [module_file] + sorted(map(Path, get_relative_import_files(module_file)))
            h = hashlib.sha256(b"".join(bytes(f) + f.read_bytes() for f in files)).hexdigest()
            if cached is None:
                m = importlib.util.module_from_spec(spec); sys.modules[name] = m
            else:
                m = cached
            if getattr(m, "__transformers_module_hash__", "") != h:
                spec.loader.exec_module(m); m.__transformers_module_hash__ = h
            return getattr(m, class_name)
    dmu.get_class_in_module = _patched_gcim
except Exception:
    pass
SCEOF""",
        ),
    ),
)

# ---------------------------------------------------------------------------
# Endpoint
# SGLang flags per cookbook:
#   --reasoning-parser kimi_k2  — split thinking/content in output
#   --tool-call-parser kimi_k2  — structured tool calls
#   --dp 8 --enable-dp-attention — data-parallel attention for throughput
# ---------------------------------------------------------------------------

_docker_run = (
    f"docker run --rm"
    f" --device /dev/kfd --device /dev/dri"
    f" --group-add video"
    f" --shm-size 128G"
    f" --ipc host --network host"
    f" --volume /models:/models"
    f" --env ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
    f" --env HF_HOME=/models/hf_cache"
    f" --env HF_MODULES_CACHE=/models/hf_cache/modules"
    # Kimi-K2.5 uses trust_remote_code with a hyphenated package name (Kimi-K2.5)
    # which Python can't import as a dotted package. Set PYTHONPATH to the model
    # snapshot dir so bare `from configuration_deepseek import ...` resolves.
    # The snapshot hash is resolved dynamically in the bash -c wrapper below.
    f" --name sglang_bench_{PORT}"
    f" {_SGLANG_IMAGE}"
    # Install tilelang (required by dsv32-rocm for NSA attention), then set
    # PYTHONPATH to the model snapshot dir (workaround for Kimi-K2.5 hyphenated
    # package name bug), then launch SGLang.
    f" bash -c '"
    f"USE_ROCM=true ROCM_HOME=/opt/rocm pip install -q /root/tilelang blobfile && "
    # Install sitecustomize.py (written by bootstrap to /models/sitecustomize.py)
    # into the container's site-packages so all TP worker subprocesses get it.
    f"cp /models/sitecustomize.py $(python3 -c 'import site; print(site.getsitepackages()[0])')/sitecustomize.py 2>/dev/null || true && "
    f"SNAP=$(ls /models/hf_cache/hub/models--moonshotai--Kimi-K2.5/snapshots/ | head -1) && "
    f"export PYTHONPATH=/models/hf_cache/hub/models--moonshotai--Kimi-K2.5/snapshots/$SNAP && "
    f"python -m sglang.launch_server"
    f" --model-path {MODEL}"
    f" --host 0.0.0.0"
    f" --port {PORT}"
    f" --tp 8"
    f" --trust-remote-code"
    f" --reasoning-parser kimi"
    f" --tool-call-parser kimi_k2"
    f" --mem-fraction-static 0.85"
    f" --context-length 8192"
    f"'"
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
    output=EvalOutputConfig(experiment_name="bench_kimi_k2_5_amd_mi355x"),
    hardware=hardware,
    summary_distribution_percentiles=SUMMARY_DISTRIBUTION_PERCENTILES,
)
