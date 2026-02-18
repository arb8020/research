"""Tinygrad Renderer Backend Eval.

An eval where the agent restores a stubbed-out GPU backend in tinygrad.

Supported backends:
- METAL: Metal backend (macOS)
- WEBGPU: WebGPU backend via wgpu-native

Task:
- Agent starts with tinygrad-no{backend}/ (backend stubbed out)
- Agent implements Device, Program, Allocator, Renderer
- Success = basic tensor ops work

Scoring:
- Binary: basic tensor ops work or not
- Bonus: which tests pass (partial credit)

After completion, we collect feedback from the agent about what tooling would help.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths
EVAL_DIR = Path(__file__).parent

# Backend configurations
BACKEND_CONFIG = {
    "METAL": {
        "dir": "tinygrad-nometal",
        "env_var": "METAL",
        "device_name": "Metal",
        "test_file": "test/device/test_metal.py",
    },
    "WEBGPU": {
        "dir": "tinygrad-nowebgpu",
        "env_var": "WEBGPU",
        "device_name": "WebGPU",
        "test_file": None,  # No dedicated test file yet
    },
}


def get_backend_dir(backend: str) -> Path:
    """Get the tinygrad directory for a backend."""
    config = BACKEND_CONFIG.get(backend.upper())
    if not config:
        raise ValueError(f"Unknown backend: {backend}. Supported: {list(BACKEND_CONFIG.keys())}")
    return EVAL_DIR / config["dir"]


def prepare_messages(sample_data: dict[str, Any]) -> list:
    """Prepare initial messages for the agent."""
    from rollouts.dtypes import Message

    backend = sample_data.get("backend", "METAL").upper()
    config = BACKEND_CONFIG[backend]
    backend_dir = get_backend_dir(backend)

    task_content = (backend_dir / "TASK.md").read_text()
    claude_md = (backend_dir / "CLAUDE.md").read_text()

    env_var = config["env_var"]
    device_name = config["device_name"]

    system_prompt = f"""You are an expert systems programmer specializing in GPU programming and ML frameworks.

Your task is to restore the {device_name} backend for tinygrad. The backend has been stubbed out,
and you need to implement it from scratch.

## Context

{claude_md}

## Phased Approach

Break this into phases. Complete and TEST each phase before moving to the next.
Do not skip phases. Do not proceed if a phase test fails.

### Phase 1: Autogen Bindings
Implement FFI bindings in `runtime/autogen/{env_var.lower()}.py`
```bash
# Test: imports work
PYTHONPATH=. python -c "from tinygrad.runtime.autogen.{env_var.lower()} import *; print('Phase 1 OK')"
```

### Phase 2: Device Initialization
Implement device creation in `runtime/ops_{env_var.lower()}.py`
```bash
# Test: can create device
PYTHONPATH=. {env_var}=1 python -c "from tinygrad import Device; d=Device['{env_var}']; print('Phase 2 OK:', d)"
```

### Phase 3: Allocator
Implement buffer allocation (alloc, free, copyin, copyout)
```bash
# Test: alloc doesn't crash
PYTHONPATH=. {env_var}=1 python -c "
from tinygrad import Device
d = Device['{env_var}']
buf = d.allocator.alloc(1024)
print('Phase 3 OK: allocated', buf)
"
```

### Phase 4: Renderer
Implement shader code generation in the renderer
```bash
# Test: generates shader code (look for kernel source in output)
PYTHONPATH=. DEBUG=4 {env_var}=1 python -c "from tinygrad import Tensor; Tensor([1,2,3]).realize()"
```

### Phase 5: Program Execution
Implement shader compilation and kernel dispatch
```bash
# Test: correct output (MUST be [1. 2. 3.], NOT [0 0 0])
PYTHONPATH=. {env_var}=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```

## Testing Protocol

1. Run the phase test IMMEDIATELY after implementing each phase
2. Do NOT proceed to next phase if current phase fails
3. After Phase 5, run the full test suite:
```bash
PYTHONPATH=. {env_var}=1 python -m pytest test/test_tiny.py -x -v
```

CRITICAL: `[0 0 0]` output means GPU is NOT executing kernels. This is a FAILURE, not success.

## Loop Breaking

If stuck on the same error:
- After 3 failed fix attempts: try a completely different approach
- After 5 total attempts: simplify (test with smaller code, add debug prints)
- If re-reading same files repeatedly: stop and write down what you know vs don't know

## Reference Code

Study these working backends:
- `runtime/ops_cuda.py` - Device, Allocator, Program pattern
- `runtime/ops_cl.py` - OpenCL (simpler, good starting point)
- `renderer/cstyle.py` - Base renderer class
- `runtime/autogen/cuda.py` - FFI bindings example

## Tools Available

- read: Read file contents
- write: Write content to a file
- edit: Replace exact text in a file
- bash: Execute shell commands (sandboxed to workspace)"""

    user_message = f"""Please restore the {device_name} backend for tinygrad.

{task_content}

Start by exploring the codebase to understand:
1. How other backends (CUDA, OpenCL, Metal) are structured
2. What the stubbed files currently contain
3. The runtime/device infrastructure

Then implement the {device_name} backend. Good luck!"""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def score_sample(sample: Any):
    """Score a completed sample.

    Looks for test output patterns in the trajectory.
    Tracks phase completion for partial credit.
    """
    from rollouts.dtypes import Metric, Score

    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no trajectory"}),)
        )

    messages = getattr(trajectory, "messages", None)
    if messages is None:
        return Score(
            metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no messages"}),)
        )

    # Look for test results in tool outputs
    basic_works = False
    num_tests_passed = 0
    num_tests_failed = 0

    # Phase completion tracking
    phase1_ok = False  # Autogen imports
    phase2_ok = False  # Device creation
    phase3_ok = False  # Allocator
    phase4_ok = False  # Renderer (shader output)
    phase5_ok = False  # End-to-end [1. 2. 3.]

    # Patterns
    pytest_summary = re.compile(r"(\d+) passed")
    pytest_failed = re.compile(r"(\d+) failed")
    basic_output = re.compile(r"\[1\.\s*2\.\s*3\.\]")
    phase1_pattern = re.compile(r"Phase 1 OK")
    phase2_pattern = re.compile(r"Phase 2 OK")
    phase3_pattern = re.compile(r"Phase 3 OK")
    # Phase 4: look for kernel/shader code generation
    phase4_pattern = re.compile(r"(kernel void|fn \w+\(|__kernel void)")
    zeros_output = re.compile(r"\[0\.?\s+0\.?\s+0\.?\]|\[0\s+0\s+0\]")

    for msg in messages:
        # Only look at tool results
        role = getattr(msg, "role", None)
        if role != "tool":
            continue

        content = getattr(msg, "content", None)
        if content is None:
            continue

        # Handle content blocks
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
            content = "\n".join(text_parts)

        if not isinstance(content, str):
            continue

        # Check phase completions
        if phase1_pattern.search(content):
            phase1_ok = True
        if phase2_pattern.search(content):
            phase2_ok = True
        if phase3_pattern.search(content):
            phase3_ok = True
        if phase4_pattern.search(content):
            phase4_ok = True

        # Check for basic tensor output (Phase 5)
        if basic_output.search(content):
            # Make sure it's not just zeros
            if not zeros_output.search(content):
                basic_works = True
                phase5_ok = True

        # Check for pytest results
        passed_match = pytest_summary.search(content)
        if passed_match:
            num_tests_passed = max(num_tests_passed, int(passed_match.group(1)))

        failed_match = pytest_failed.search(content)
        if failed_match:
            num_tests_failed = int(failed_match.group(1))

    # Calculate phases completed (for partial credit insight)
    phases_completed = sum([phase1_ok, phase2_ok, phase3_ok, phase4_ok, phase5_ok])

    metrics = [
        Metric("passed", 1.0 if basic_works else 0.0, weight=1.0),
        Metric("basic_works", 1.0 if basic_works else 0.0, weight=0.0),
        Metric("phases_completed", float(phases_completed), weight=0.0),
        Metric("phase1_autogen", 1.0 if phase1_ok else 0.0, weight=0.0),
        Metric("phase2_device", 1.0 if phase2_ok else 0.0, weight=0.0),
        Metric("phase3_allocator", 1.0 if phase3_ok else 0.0, weight=0.0),
        Metric("phase4_renderer", 1.0 if phase4_ok else 0.0, weight=0.0),
        Metric("phase5_e2e", 1.0 if phase5_ok else 0.0, weight=0.0),
        Metric("tests_passed", float(num_tests_passed), weight=0.0),
        Metric("tests_failed", float(num_tests_failed), weight=0.0),
    ]

    return Score(metrics=tuple(metrics))


def get_webgpu_path() -> str | None:
    """Get the path to Dawn WebGPU library.

    Returns path to libwebgpu_dawn if dawn-python is installed, else None.
    """
    import platform
    import sysconfig

    purelib = sysconfig.get_paths()["purelib"]
    lib_dir = Path(purelib) / "pydawn" / "lib"

    if not lib_dir.exists():
        return None

    # Select correct library for platform/arch
    machine = platform.machine().lower()
    system = platform.system().lower()

    if system == "darwin":
        lib_name = f"libwebgpu_dawn_{machine}.dylib"
    elif system == "linux":
        lib_name = f"libwebgpu_dawn_{machine}.so"
    elif system == "windows":
        lib_name = "libwebgpu_dawn.dll"
    else:
        return None

    lib_path = lib_dir / lib_name
    return str(lib_path) if lib_path.exists() else None


async def make_environment(sample_data: dict[str, Any]):
    """Create sandboxed environment for a sample."""
    import subprocess

    from rollouts.environments.sandboxed_worktree import create_sandboxed_environment

    backend = sample_data.get("backend", "METAL").upper()
    backend_dir = get_backend_dir(backend)
    run_id = sample_data.get("run_id", sample_data.get("id", "default"))

    # Set up environment variables for specific backends
    extra_env: dict[str, str] = {}

    if backend == "WEBGPU":
        webgpu_path = get_webgpu_path()
        if webgpu_path:
            # Pass the path to libwebgpu_dawn so agent can load it via ctypes
            extra_env["WEBGPU_LIB_PATH"] = webgpu_path
            logger.info(f"Setting WEBGPU_LIB_PATH={webgpu_path}")
        else:
            logger.warning(
                "dawn-python not installed on host. WebGPU eval will fail. "
                "Install with: pip install dawn-python"
            )

    env = create_sandboxed_environment(
        source_dir=backend_dir,
        run_id=run_id,
        network_access=True,  # Allow pip, git, docs
        sandbox_enabled=True,
        extra_env=extra_env,
    )

    # Pre-install dependencies so agent doesn't waste turns on setup
    workspace = env.working_dir
    logger.info(f"Installing dependencies in {workspace}...")
    try:
        result = subprocess.run(
            ["uv", "pip", "install", "-e", "."],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            logger.info("Dependencies installed successfully")
        else:
            logger.warning(f"Dependency install failed: {result.stderr[:500]}")
    except Exception as e:
        logger.warning(f"Failed to pre-install dependencies: {e}")

    return env


def load_tasks(
    tasks_path: Path | str | None = None,
    backend: str = "METAL",
) -> list[dict[str, Any]]:
    """Load tasks from JSON file or return default single task."""
    backend = backend.upper()

    if tasks_path:
        path = Path(tasks_path)
        if path.exists():
            with open(path) as f:
                tasks = json.load(f)
                # Add backend to each task
                for task in tasks:
                    task["backend"] = backend
                return tasks

    # Default: single task
    return [
        {
            "id": f"{backend.lower()}-restore",
            "name": f"Restore {backend} Backend",
            "backend": backend,
        }
    ]


# ── EvalSpec Definition ───────────────────────────────────────────────────────

from rollouts.eval_runner import EvalSpec


def get_spec(backend: str = "METAL") -> EvalSpec:
    """Get the EvalSpec for this eval."""
    backend = backend.upper()
    tasks_path = EVAL_DIR / "configs" / f"tasks_{backend.lower()}.json"

    return EvalSpec(
        name=f"tinygrad_{backend.lower()}_restore",
        prepare_messages=prepare_messages,
        score_fn=score_sample,
        make_environment=make_environment,
        default_tasks_path=tasks_path if tasks_path.exists() else None,
        per_sample_environment=True,
    )


# ── Tooling Feedback Collection ───────────────────────────────────────────────


def get_tooling_feedback_prompt(backend: str) -> str:
    return f"""You just finished attempting to restore the {backend} backend for tinygrad.

Please provide detailed feedback about what would have helped you work faster and more effectively:

1. DOCUMENTATION GAPS: What documentation was missing or unclear? What would you have wanted to read before starting?

2. TOOLING WISHES: What tools, scripts, or automation would have helped? (e.g., test runners, code navigation, examples, scaffolding)

3. PAIN POINTS: What was frustrating or slow? Where did you get stuck?

4. WHAT WORKED: What aspects of the codebase or task setup were helpful?

5. SPECIFIC SUGGESTIONS: If you could add 3 things to make this task easier, what would they be?

Be specific and actionable. Your feedback will be used to build better tooling for similar tasks."""


async def collect_tooling_feedback(
    sample,
    api_key: str,
    backend: str = "METAL",
    model: str = "claude-sonnet-4-20250514",
) -> dict[str, Any] | None:
    """Collect detailed tooling feedback from agent after task completion."""
    import anthropic
    import trio

    trajectory = getattr(sample, "trajectory", None)
    if not trajectory:
        return None

    messages = getattr(trajectory, "messages", [])

    # Build conversation summary (last 10 messages for context)
    recent = messages[-10:] if len(messages) > 10 else messages
    context_parts = []
    for m in recent:
        role = getattr(m, "role", "?")
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(getattr(b, "text", "")[:300] for b in content if hasattr(b, "text"))
        if isinstance(content, str):
            context_parts.append(f"{role}: {content[:500]}...")

    context_summary = "\n\n".join(context_parts)

    full_prompt = f"""Here's a summary of your recent work:

{context_summary}

---

{get_tooling_feedback_prompt(backend)}"""

    try:
        client = anthropic.Anthropic(api_key=api_key)

        def call_api():
            return client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": full_prompt}],
            )

        response = await trio.to_thread.run_sync(call_api)
        feedback_text = response.content[0].text

        return {
            "raw_feedback": feedback_text,
            "model": model,
            "backend": backend,
            "context_messages": len(recent),
        }

    except Exception as e:
        logger.warning(f"Tooling feedback collection failed: {e}")
        return None


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def run(
    backend: str = "METAL",
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    max_turns: int = 100,
    max_tokens: int = 16384,  # Large to allow writing big files
    limit: int | None = None,
    verbose: bool = True,
    output_dir: Path | str | None = None,
    collect_feedback: bool = True,
) -> dict[str, Any]:
    """Run the tinygrad renderer restore eval.

    Note: max_tokens is set high (16384) because the agent may need to write
    large files (e.g., ctypes bindings). If the agent hits max_tokens mid-response,
    the truncated response may be treated as "no tool call" and end the run early.
    """
    import os
    from datetime import datetime

    import trio

    from rollouts.eval_runner import run_eval_from_spec

    backend = backend.upper()
    tasks = load_tasks(backend=backend)
    if limit:
        tasks = tasks[:limit]

    if verbose:
        print(f"Running {backend} restore eval with {len(tasks)} task(s)")
        print(f"Model: {model} ({provider})")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = EVAL_DIR / "results" / f"{backend.lower()}_restore_{timestamp}"

    output_dir = Path(output_dir)

    result = run_eval_from_spec(
        get_spec(backend),
        tasks=tasks,
        model=model,
        provider=provider,
        max_turns=max_turns,
        max_tokens=max_tokens,
        max_concurrent=1,  # One at a time for now
        verbose=verbose,
        show_progress=True,
        output_dir=output_dir,
    )

    # Collect tooling feedback
    if collect_feedback:
        if verbose:
            print("\nCollecting tooling feedback...")

        samples_dir = output_dir / "samples"
        feedback_results = []

        async def collect_all_feedback():
            api_key = os.getenv("ANTHROPIC_API_KEY", "")

            for sample_file in sorted(samples_dir.glob("*.json")):
                sample_data = json.loads(sample_file.read_text())

                # Reconstruct minimal sample object
                class SampleObj:
                    pass

                sample = SampleObj()
                sample.trajectory = type(
                    "Trajectory",
                    (),
                    {
                        "messages": [
                            type("Msg", (), {"role": m.get("role"), "content": m.get("content")})()
                            for m in sample_data.get("trajectory", {}).get("messages", [])
                        ]
                    },
                )()

                feedback = await collect_tooling_feedback(sample, api_key, backend, model)
                if feedback:
                    feedback["sample_id"] = sample_data.get("id")
                    feedback_results.append(feedback)

                    if verbose:
                        print(f"  Collected feedback from {sample_data.get('id')}")

        trio.run(collect_all_feedback)

        # Save feedback
        feedback_file = output_dir / "tooling_feedback.json"
        feedback_file.write_text(json.dumps(feedback_results, indent=2))

        if verbose:
            print(f"\nFeedback saved to {feedback_file}")

        result["tooling_feedback"] = feedback_results

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run tinygrad renderer restore eval")
    parser.add_argument(
        "--backend",
        type=str,
        default="METAL",
        choices=["METAL", "WEBGPU"],
        help="Backend to restore (METAL or WEBGPU)",
    )
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument(
        "--max-tokens", type=int, default=16384, help="Max output tokens per response"
    )
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-feedback", action="store_true", help="Skip feedback collection")
    args = parser.parse_args()

    result = run(
        backend=args.backend,
        model=args.model,
        provider=args.provider,
        max_turns=args.max_turns,
        max_tokens=args.max_tokens,
        limit=args.limit,
        verbose=args.verbose,
        output_dir=args.output_dir,
        collect_feedback=not args.no_feedback,
    )

    print(f"\nResults: {result}")
