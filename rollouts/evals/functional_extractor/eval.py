"""Functional Extractor Eval.

An eval where the agent converts HuggingFace models to minimal functional PyTorch.

Task:
- Agent starts with a workspace containing README, helper scripts, and model_info.json
- Agent writes functional.py
- Success = python scripts/verify.py functional.py exits 0

Scoring:
- Binary: torch.allclose passes or not
- Bonus: lines of code (less is better)
"""

from __future__ import annotations

import base64
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Paths
EVAL_DIR = Path(__file__).parent
WORKSPACE_TEMPLATE = EVAL_DIR / "workspace"
TASKS_PATH = EVAL_DIR / "tasks.json"


def setup_workspace(sandbox, sample_data: dict[str, Any]) -> None:
    """Set up the workspace in the Modal sandbox.

    Copies template files and creates model_info.json for this task.
    NOTE: sandbox.exec() returns a process - must call .wait() for blocking.
    """
    model_name = sample_data["model_name"]
    task_id = sample_data.get("task_id", model_name.split("/")[-1])

    logger.info(f"Setting up workspace for {task_id}")

    def run_cmd(cmd: str, description: str = "") -> None:
        """Run a command and check for errors."""
        proc = sandbox.exec("bash", "-c", cmd, timeout=60)
        proc.wait()
        stdout = proc.stdout.read()
        stderr = proc.stderr.read()
        if proc.returncode != 0:
            logger.error(f"Command failed ({description}): {cmd[:100]}...")
            logger.error(f"stdout: {stdout[:500]}")
            logger.error(f"stderr: {stderr[:500]}")
            raise RuntimeError(f"Workspace setup failed: {description} - {stderr[:200]}")

    def write_file(path: str, content: str, description: str = "") -> None:
        """Write content to a file in the sandbox using heredoc to avoid ARG_MAX."""
        # Use a heredoc with a unique delimiter to handle any content
        # Write via stdin to avoid shell argument limits
        content_b64 = base64.b64encode(content.encode()).decode()

        # For large files, write in chunks
        chunk_size = 50000  # Safe size for base64 chunks
        if len(content_b64) > chunk_size:
            # Write first chunk
            first_chunk = content_b64[:chunk_size]
            proc = sandbox.exec("bash", "-c", f"echo '{first_chunk}' | base64 -d > {path}", timeout=30)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to write {path}: {proc.stderr.read()}")

            # Append remaining chunks
            for i in range(chunk_size, len(content_b64), chunk_size):
                chunk = content_b64[i:i + chunk_size]
                proc = sandbox.exec("bash", "-c", f"echo '{chunk}' | base64 -d >> {path}", timeout=30)
                proc.wait()
                if proc.returncode != 0:
                    raise RuntimeError(f"Failed to append to {path}: {proc.stderr.read()}")
        else:
            proc = sandbox.exec("bash", "-c", f"echo '{content_b64}' | base64 -d > {path}", timeout=30)
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to write {path}: {proc.stderr.read()}")

        logger.debug(f"Wrote {len(content)} bytes to {path} ({description})")

    # Create workspace directory and protected scripts directory
    # /opt/eval_scripts is the canonical location - agent cannot tamper with it
    # /workspace/scripts is a symlink for convenience
    run_cmd("mkdir -p /workspace/reference /opt/eval_scripts", "create directories")

    # Copy README
    readme_content = (WORKSPACE_TEMPLATE / "README.md").read_text()
    write_file("/workspace/README.md", readme_content, "README")

    # Copy scripts to protected location
    scripts = [
        "verify.py",
        "inspect_model.py",
        "capture.py",
        "compare.py",
        "trace_forward.py",
        "find_divergence.py",
    ]
    for script_name in scripts:
        script_path = WORKSPACE_TEMPLATE / "scripts" / script_name
        if script_path.exists():
            content = script_path.read_text()
            write_file(f"/opt/eval_scripts/{script_name}", content, script_name)
            run_cmd(f"chmod +x /opt/eval_scripts/{script_name}", f"chmod {script_name}")
        else:
            logger.warning(f"Script not found: {script_path}")

    # Make /opt/eval_scripts read-only to prevent tampering
    # Remove write permission from all files and the directory itself
    run_cmd("chmod -R a-w /opt/eval_scripts", "make scripts read-only")
    run_cmd("chmod 555 /opt/eval_scripts", "lock scripts directory")

    # Create symlink from /workspace/scripts to /opt/eval_scripts
    # This lets the agent use the expected paths while protecting the scripts
    run_cmd("ln -sf /opt/eval_scripts /workspace/scripts", "symlink scripts")

    # Create model_info.json
    model_info = {
        "model_name": model_name,
        "task_id": task_id,
        "test_inputs": sample_data.get("test_inputs", [[1, 2, 3, 4], [100, 200, 300, 400]]),
        "expected_loc": sample_data.get("expected_loc", 400),
        "config": sample_data.get("config", {}),
    }
    model_info_json = json.dumps(model_info, indent=2)
    write_file("/workspace/model_info.json", model_info_json, "model_info.json")

    # Copy reference if available
    ref_path = sample_data.get("reference_path")
    if ref_path:
        # Resolve relative to rollouts root
        full_ref_path = EVAL_DIR.parent.parent / ref_path
        if full_ref_path.exists():
            ref_content = full_ref_path.read_text()
            write_file("/workspace/reference/example.py", ref_content, "reference example")
        else:
            logger.warning(f"Reference file not found: {full_ref_path}")

    # Verify setup worked
    proc = sandbox.exec("bash", "-c", "ls -la /workspace/scripts/", timeout=10)
    proc.wait()
    files_output = proc.stdout.read()
    logger.info(f"Workspace setup complete for {task_id}. Files:\n{files_output}")


def prepare_messages(sample_data: dict[str, Any]) -> list:
    """Prepare initial messages for the agent."""
    from rollouts.dtypes import Message

    model_name = sample_data["model_name"]

    system_prompt = f"""You are an expert PyTorch developer. Your task is to convert a HuggingFace model to minimal, functional PyTorch code.

## Your Goal
Convert {model_name} to a single functional.py file that:
1. Produces numerically identical output to the HuggingFace model
2. Uses only torch and torch.nn.functional (no classes, no HF imports)
3. Is as short as possible while remaining readable

## Workspace
Your workspace is /workspace. It contains:
- README.md: Detailed instructions
- model_info.json: Model config and weight info
- scripts/ -> symlink to /opt/eval_scripts/ containing:
  - verify.py: THE OFFICIAL VERIFICATION SCRIPT - this is how success is measured
  - inspect_model.py: View HF source code
  - capture.py: Capture layer activations
  - compare.py: Compare your implementation to HF

## Strategy
1. Read the README and model_info.json
2. Use inspect_model.py to understand the HF implementation
3. Write functional.py incrementally
4. Run `python /opt/eval_scripts/verify.py /workspace/functional.py` to verify

## CRITICAL: How Success Is Measured
Your task is ONLY complete when you run:
```
python /opt/eval_scripts/verify.py /workspace/functional.py
```
And it outputs "PASS". This script loads the real HuggingFace model, runs your code, and checks torch.allclose().

DO NOT create your own verification scripts. DO NOT assume success without running verify.py.
The ONLY valid success signal is seeing "PASS (max_diff=..." in the output of verify.py.

NOTE: The eval scripts are in /opt/eval_scripts/ which is read-only. You can also access them via /workspace/scripts/."""

    user_message = f"""Please convert {model_name} to functional PyTorch code.

Start by reading the workspace files to understand the task, then write your implementation to /workspace/functional.py.

IMPORTANT: Your task is not complete until you run:
```
python /opt/eval_scripts/verify.py /workspace/functional.py
```
And see "PASS" in the output. This is the only way to verify success."""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def score_sample(sample: Any):
    """Score a completed sample.

    Returns a Score object with metrics for the eval framework.

    Scoring is based on detecting verify.py output in tool results.
    We look for the specific format that the real verify.py outputs.
    """
    from rollouts.dtypes import Metric, Score
    import re

    # Get trajectory from sample
    trajectory = getattr(sample, "trajectory", None)
    if trajectory is None:
        return Score(metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no trajectory"}),))

    messages = getattr(trajectory, "messages", None)
    if messages is None:
        return Score(metrics=(Metric("passed", 0.0, weight=1.0, metadata={"error": "no messages"}),))

    # Look for verification output in tool results
    passed = False
    max_diff = None
    error = None

    # Pattern for real verify.py output - must be on its own line and have scientific notation
    # This makes it harder to fake with a simple print statement
    pass_pattern = re.compile(r"^PASS \(max_diff=(\d+\.\d+e[+-]\d+)\)$", re.MULTILINE)
    fail_pattern = re.compile(r"^FAIL \(max_diff=(\d+\.\d+e[+-]?\d+)\)$", re.MULTILINE)

    for msg in messages:
        # Handle both dict messages (from JSON) and object messages (from dataclass)
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if content is None:
            continue

        # Handle content blocks (list of TextContent, etc.)
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

        # Check for PASS/FAIL in verify.py output using strict patterns
        pass_match = pass_pattern.search(content)
        if pass_match:
            passed = True
            try:
                max_diff = float(pass_match.group(1))
            except ValueError:
                pass

        fail_match = fail_pattern.search(content)
        if fail_match:
            passed = False
            try:
                max_diff = float(fail_match.group(1))
            except ValueError:
                pass

        if "ERROR:" in content and error is None:
            # Capture first error
            error = content[:200]

    # Build metrics
    metrics = [
        Metric("passed", 1.0 if passed else 0.0, weight=1.0),
    ]

    if max_diff is not None:
        metrics.append(Metric("max_diff", max_diff, weight=0.0))

    # Only report error if we didn't pass - errors during intermediate attempts are fine
    if error and not passed:
        metrics[0] = Metric("passed", 0.0, weight=1.0, metadata={"error": error})

    return Score(metrics=tuple(metrics))


async def make_environment(sample_data: dict[str, Any]):
    """Create environment for a sample.

    Must be async because EvalConfig.environment_factory expects Awaitable.
    """
    # Use absolute import to work when run as script or module
    try:
        from .modal_sandbox import ModalSandboxCodingEnvironment, SandboxConfig
    except ImportError:
        from modal_sandbox import ModalSandboxCodingEnvironment, SandboxConfig

    config = SandboxConfig(
        gpu_type=sample_data.get("gpu_type", "H100"),
        timeout_seconds=sample_data.get("timeout_seconds", 1800),
        model_name=sample_data["model_name"],
    )

    env = ModalSandboxCodingEnvironment(
        sandbox_config=config,
        workspace_setup=setup_workspace,
        sample_data=sample_data,
    )

    return env


def load_tasks(tasks_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load tasks from JSON file."""
    path = Path(tasks_path) if tasks_path else TASKS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    with open(path) as f:
        return json.load(f)


# ── EvalSpec Definition ───────────────────────────────────────────────────────

from rollouts.eval_runner import EvalSpec

# Export spec directly for run_eval.py pattern
spec = EvalSpec(
    name="functional_extractor",
    prepare_messages=prepare_messages,
    score_fn=score_sample,
    make_environment=make_environment,
    default_tasks_path=TASKS_PATH,
    per_sample_environment=True,
)


def get_spec():
    """Get the EvalSpec for this eval (for backwards compatibility)."""
    return spec


# ── Exit Survey ───────────────────────────────────────────────────────────────


async def collect_exit_survey_for_sample(
    sample,
    api_key: str,
    exit_reason: str,
) -> dict[str, Any] | None:
    """Collect exit survey from agent after sample completion.

    Asks the agent:
    1. Did you succeed at the task?
    2. Any feedback on the harness/tools?

    Returns survey results or None if survey failed.
    """
    import anthropic
    import trio

    # Get recent context from trajectory
    trajectory = getattr(sample, "trajectory", None)
    if not trajectory:
        return None

    messages = getattr(trajectory, "messages", [])
    recent_messages = messages[-5:] if messages else []

    # Build context summary
    context_parts = []
    for m in recent_messages:
        content = getattr(m, "content", "")
        if isinstance(content, list):
            content = " ".join(
                getattr(b, "text", str(b))[:200] for b in content if hasattr(b, "text")
            )
        if isinstance(content, str):
            context_parts.append(f"{m.role}: {content[:300]}...")

    context_summary = "\n".join(context_parts)

    survey_prompt = f"""You are an AI agent that just finished a task. Exit reason: {exit_reason}

Recent context:
{context_summary}

Please provide brief feedback:

1. TASK SUCCESS: Did you succeed at your task? (yes/no/partial/unknown)
2. TASK NOTES: What did you accomplish? Any blockers? (1-2 sentences)
3. HARNESS FEEDBACK: Any feedback on the tools/environment/scripts? What would help? (1-2 sentences, or "none")

Format your response exactly as:
TASK_SUCCESS: <yes|no|partial|unknown>
TASK_NOTES: <notes>
HARNESS_FEEDBACK: <feedback>"""

    try:
        # Use haiku for quick/cheap survey
        client = anthropic.Anthropic(api_key=api_key)

        def call_api():
            return client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=400,
                messages=[{"role": "user", "content": survey_prompt}],
            )

        response = await trio.to_thread.run_sync(call_api)
        response_text = response.content[0].text

        # Parse response
        result = {
            "task_success": None,
            "task_notes": None,
            "harness_feedback": None,
            "raw_response": response_text,
        }

        for line in response_text.split("\n"):
            line = line.strip()
            if line.startswith("TASK_SUCCESS:"):
                result["task_success"] = line.split(":", 1)[1].strip().lower()
            elif line.startswith("TASK_NOTES:"):
                result["task_notes"] = line.split(":", 1)[1].strip()
            elif line.startswith("HARNESS_FEEDBACK:"):
                result["harness_feedback"] = line.split(":", 1)[1].strip()

        return result

    except Exception as e:
        logger.warning(f"Exit survey failed: {e}")
        return None


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def run(
    tasks_path: Path | str | None = None,
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    max_turns: int = 50,
    max_concurrent: int = 1,
    limit: int | None = None,
    verbose: bool = True,
    output_dir: Path | str | None = None,
    exit_survey: bool = False,
) -> dict[str, Any]:
    """Run the functional extractor eval.

    Uses run_eval_from_spec for proper integration with rollouts eval framework.

    Args:
        exit_survey: If True, collect feedback from agent after each sample about
                     task success and harness quality. Surveys are saved to results.
    """
    from datetime import datetime

    import trio

    from rollouts.eval_runner import run_eval_from_spec

    # Load tasks
    tasks = load_tasks(tasks_path)
    if limit:
        tasks = tasks[:limit]

    if verbose:
        print(f"Loaded {len(tasks)} tasks")

    # Build output dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = EVAL_DIR / "results" / f"functional_extractor_{timestamp}"

    output_dir = Path(output_dir)

    # Run eval
    result = run_eval_from_spec(
        get_spec(),
        tasks=tasks,
        model=model,
        provider=provider,
        max_turns=max_turns,
        max_concurrent=max_concurrent,
        verbose=verbose,
        show_progress=True,
        output_dir=output_dir,
    )

    # Collect exit surveys if enabled
    if exit_survey:
        if verbose:
            print("\nCollecting exit surveys...")

        # Load samples from results to get trajectories
        samples_dir = output_dir / "samples"
        surveys = []

        async def collect_surveys():
            import os

            api_key = os.getenv("ANTHROPIC_API_KEY", "")

            for sample_file in sorted(samples_dir.glob("*.json")):
                sample_data = json.loads(sample_file.read_text())

                # Create a minimal sample object
                class SampleObj:
                    pass

                sample = SampleObj()
                sample.trajectory = type("Trajectory", (), {
                    "messages": [
                        type("Msg", (), {"role": m.get("role"), "content": m.get("content")})()
                        for m in sample_data.get("trajectory", {}).get("messages", [])
                    ]
                })()

                exit_reason = sample_data.get("metadata", {}).get("stop_reason", "max_turns")
                survey_result = await collect_exit_survey_for_sample(
                    sample, api_key, exit_reason
                )

                if survey_result:
                    survey_result["sample_id"] = sample_data.get("id")
                    surveys.append(survey_result)

                    if verbose:
                        success = survey_result.get("task_success", "?")
                        print(f"  {sample_data.get('id')}: {success}")

        trio.run(collect_surveys)

        # Save surveys
        surveys_file = output_dir / "exit_surveys.json"
        surveys_file.write_text(json.dumps(surveys, indent=2))

        if verbose:
            print(f"\nSurveys saved to {surveys_file}")

        # Add survey summary to result
        result["exit_surveys"] = {
            "total": len(surveys),
            "success": sum(1 for s in surveys if s.get("task_success") == "yes"),
            "partial": sum(1 for s in surveys if s.get("task_success") == "partial"),
            "failed": sum(1 for s in surveys if s.get("task_success") == "no"),
        }

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run functional extractor eval")
    parser.add_argument("--tasks", type=str, help="Path to tasks.json")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--max-turns", type=int, default=50)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--exit-survey",
        action="store_true",
        help="Collect feedback from agent about task success and harness quality",
    )
    args = parser.parse_args()

    result = run(
        tasks_path=args.tasks,
        model=args.model,
        provider=args.provider,
        max_turns=args.max_turns,
        max_concurrent=args.max_concurrent,
        limit=args.limit,
        verbose=args.verbose,
        output_dir=args.output_dir,
        exit_survey=args.exit_survey,
    )

    print(f"\nResults: {result}")
