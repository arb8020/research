"""Metal Restore Eval.

An eval where the agent restores a stubbed-out Metal backend in tinygrad.

Task:
- Agent starts with tinygrad-nometal/ (Metal backend stubbed out)
- Agent implements MetalDevice, MetalCompiler, MetalProgram, MetalAllocator, MetalRenderer, MetalGraph
- Success = tests pass

Scoring:
- Binary: tests pass or not
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
TINYGRAD_NOMETAL = EVAL_DIR / "tinygrad-nometal"
TASKS_PATH = EVAL_DIR / "tasks.json"


def prepare_messages(sample_data: dict[str, Any]) -> list:
    """Prepare initial messages for the agent."""
    from rollouts.dtypes import Message

    task_content = (TINYGRAD_NOMETAL / "TASK.md").read_text()
    claude_md = (TINYGRAD_NOMETAL / "CLAUDE.md").read_text()

    system_prompt = f"""You are an expert systems programmer specializing in GPU programming and ML frameworks.

Your task is to restore the Metal backend for tinygrad. The backend has been stubbed out,
and you need to implement it from scratch.

## Context

{claude_md}

## Important

- You have access to the full tinygrad codebase including other backends (CUDA, OpenCL) as reference
- The original tinygrad repo is NOT available - you must implement based on patterns in the codebase
- Network access is available for documentation lookups
- Take your time to understand the architecture before implementing

## Tools Available

- read: Read file contents
- write: Write content to a file
- edit: Replace exact text in a file
- bash: Execute shell commands (sandboxed to workspace)

## How Success Is Measured

Your task is complete when:
```bash
PYTHONPATH=. METAL=1 python -c "from tinygrad import Tensor; print(Tensor([1,2,3]).numpy())"
```
outputs `[1. 2. 3.]` and:
```bash
PYTHONPATH=. METAL=1 python -m pytest test/device/test_metal.py -x
```
passes.

DO NOT claim success without running these commands and seeing passing output."""

    user_message = f"""Please restore the Metal backend for tinygrad.

{task_content}

Start by exploring the codebase to understand:
1. How other backends (CUDA, OpenCL) are structured
2. What the stubbed files currently contain
3. The runtime/device infrastructure

Then implement the Metal backend. Good luck!"""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def score_sample(sample: Any):
    """Score a completed sample.

    Looks for test output patterns in the trajectory.
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
    tests_passed = False
    basic_works = False
    num_tests_passed = 0
    num_tests_failed = 0

    # Patterns for pytest output
    pytest_summary = re.compile(r"(\d+) passed")
    pytest_failed = re.compile(r"(\d+) failed")
    basic_output = re.compile(r"\[1\.\s*2\.\s*3\.\]")

    for msg in messages:
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

        # Check for basic tensor output
        if basic_output.search(content):
            basic_works = True

        # Check for pytest results
        passed_match = pytest_summary.search(content)
        if passed_match:
            num_tests_passed = max(num_tests_passed, int(passed_match.group(1)))

        failed_match = pytest_failed.search(content)
        if failed_match:
            num_tests_failed = int(failed_match.group(1))

    # Consider passed if basic works AND some tests pass
    tests_passed = basic_works and num_tests_passed > 0 and num_tests_failed == 0

    metrics = [
        Metric("passed", 1.0 if tests_passed else 0.0, weight=1.0),
        Metric("basic_works", 1.0 if basic_works else 0.0, weight=0.3),
        Metric("tests_passed", float(num_tests_passed), weight=0.0),
        Metric("tests_failed", float(num_tests_failed), weight=0.0),
    ]

    return Score(metrics=tuple(metrics))


async def make_environment(sample_data: dict[str, Any]):
    """Create sandboxed environment for a sample."""
    from rollouts.environments.sandboxed_worktree import create_sandboxed_environment

    run_id = sample_data.get("run_id", sample_data.get("id", "default"))

    env = create_sandboxed_environment(
        source_dir=TINYGRAD_NOMETAL,
        run_id=run_id,
        network_access=True,  # Allow pip, git, docs
        sandbox_enabled=True,
    )

    return env


def load_tasks(tasks_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Load tasks from JSON file or return default single task."""
    if tasks_path:
        path = Path(tasks_path)
        if path.exists():
            with open(path) as f:
                return json.load(f)

    # Default: single task
    return [{"id": "metal-restore", "name": "Restore Metal Backend"}]


# ── EvalSpec Definition ───────────────────────────────────────────────────────

from rollouts.eval_runner import EvalSpec

spec = EvalSpec(
    name="metal_restore",
    prepare_messages=prepare_messages,
    score_fn=score_sample,
    make_environment=make_environment,
    default_tasks_path=TASKS_PATH,
    per_sample_environment=True,
)


def get_spec():
    """Get the EvalSpec for this eval."""
    return spec


# ── Tooling Feedback Collection ───────────────────────────────────────────────


TOOLING_FEEDBACK_PROMPT = """You just finished attempting to restore the Metal backend for tinygrad.

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

{TOOLING_FEEDBACK_PROMPT}"""

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
            "context_messages": len(recent),
        }

    except Exception as e:
        logger.warning(f"Tooling feedback collection failed: {e}")
        return None


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def run(
    model: str = "claude-sonnet-4-20250514",
    provider: str = "anthropic",
    max_turns: int = 100,
    limit: int | None = None,
    verbose: bool = True,
    output_dir: Path | str | None = None,
    collect_feedback: bool = True,
) -> dict[str, Any]:
    """Run the Metal restore eval."""
    import os
    from datetime import datetime

    import trio

    from rollouts.eval_runner import run_eval_from_spec

    tasks = load_tasks()
    if limit:
        tasks = tasks[:limit]

    if verbose:
        print(f"Running Metal restore eval with {len(tasks)} task(s)")
        print(f"Model: {model} ({provider})")

    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = EVAL_DIR / "results" / f"metal_restore_{timestamp}"

    output_dir = Path(output_dir)

    result = run_eval_from_spec(
        get_spec(),
        tasks=tasks,
        model=model,
        provider=provider,
        max_turns=max_turns,
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

                feedback = await collect_tooling_feedback(sample, api_key, model)
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

    parser = argparse.ArgumentParser(description="Run Metal restore eval")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514")
    parser.add_argument("--provider", type=str, default="anthropic")
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--no-feedback", action="store_true", help="Skip feedback collection")
    args = parser.parse_args()

    result = run(
        model=args.model,
        provider=args.provider,
        max_turns=args.max_turns,
        limit=args.limit,
        verbose=args.verbose,
        output_dir=args.output_dir,
        collect_feedback=not args.no_feedback,
    )

    print(f"\nResults: {result}")
