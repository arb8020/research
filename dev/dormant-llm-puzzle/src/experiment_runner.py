"""Reusable experiment runner with 30-min timeout, streaming results, and polling.

Usage:
    from src.experiment_runner import run_sandbox_experiment

    result = await run_sandbox_experiment(
        name="my_experiment",
        script="my_script.py",  # Script content to run in sandbox
        input_data={"prompts": [...]},  # JSON-serializable input
        timeout_min=30,
        gpu="A10G",
    )

The runner:
1. Creates sandbox with specified GPU
2. Uploads script and input data
3. Runs with timeout, polling for progress
4. Streams results to local events.jsonl as they complete
5. Returns partial results if timeout hit
6. Saves everything to results/{name}/{timestamp}/

Scripts should:
- Read input from stdin (json.load(sys.stdin))
- Write incremental results to /workspace/results.jsonl (one JSON per line)
- Write progress to /workspace/progress.json
- Print final result to stdout as JSON
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from src.results import RESULTS_DIR, setup_run_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    """Result from an experiment run."""
    name: str
    completed: list[dict] = field(default_factory=list)
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    duration_sec: float = 0.0
    error: str | None = None

    def summary(self) -> str:
        status = "TIMEOUT" if self.timed_out else ("ERROR" if self.error else "DONE")
        return f"[{status}] {len(self.completed)} results in {self.duration_sec:.1f}s"


def _stream_result_to_file(result: dict, events_file: Path) -> None:
    """Append a result to the events JSONL file immediately."""
    event = {
        "type": "result",
        "timestamp": datetime.now().isoformat(),
        **result,
    }
    with open(events_file, "a") as f:
        f.write(json.dumps(event) + "\n")


async def run_sandbox_experiment(
    name: str,
    script: str,
    input_data: Any = None,
    timeout_min: float = 30,
    gpu: str = "A10G",
    poll_interval_sec: float = 10.0,
    image: str = "transformers",
) -> ExperimentResult:
    """Run an experiment script in Modal sandbox with timeout and streaming.

    Args:
        name: Experiment name (used for results directory)
        script: Python script content to run in sandbox
        input_data: JSON-serializable data to pass via stdin
        timeout_min: Max time before returning partial results (default 30)
        gpu: GPU type for Modal sandbox
        poll_interval_sec: How often to check progress
        image: Sandbox image name

    Returns:
        ExperimentResult with completed results, stdout, stderr, etc.
    """
    import trio
    from src.sandbox import Sandbox

    start_time = time.time()
    timeout_sec = timeout_min * 60
    run_dir = setup_run_dir(name)

    events_file = run_dir / "events.jsonl"

    # Log experiment start
    with open(events_file, "w") as f:
        f.write(json.dumps({
            "type": "experiment_start",
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "timeout_min": timeout_min,
        }) + "\n")

    logger.info(f"Starting experiment '{name}'")
    logger.info(f"Timeout: {timeout_min} min, results: {run_dir}")
    logger.info(f"Streaming to: {events_file}")

    async with await Sandbox.create(gpu=gpu, image=image) as sb:
        logger.info(f"Sandbox: {sb.sandbox_id}")

        # Upload script and input
        await sb.write_file("/workspace/experiment.py", script)
        if input_data is not None:
            await sb.write_file("/workspace/input.json", json.dumps(input_data))

        logger.info("Starting experiment...")

        # Track state
        last_seen_count = 0
        completed: list[dict] = []
        timed_out = False
        error = None
        inference_done = False
        final_stdout = ""
        final_stderr = ""

        async def run_script():
            nonlocal inference_done, error, final_stdout, final_stderr
            cmd = "cd /workspace && python experiment.py"
            if input_data is not None:
                cmd += " < input.json"

            result = await sb.run(cmd, timeout=int(timeout_sec) + 60, tag="experiment")
            inference_done = True
            final_stdout = result.stdout
            final_stderr = result.stderr
            if not result.success:
                error = result.stderr[:1000]

        async def poll_progress():
            """Poll sandbox for new results and stream them locally."""
            nonlocal last_seen_count, completed

            while not inference_done:
                await trio.sleep(poll_interval_sec)

                # Read results from sandbox
                try:
                    results_content = await sb.read_file("/workspace/results.jsonl")
                    lines = [l for l in results_content.strip().split("\n") if l.strip()]

                    # Stream any new results
                    if len(lines) > last_seen_count:
                        for line in lines[last_seen_count:]:
                            try:
                                result = json.loads(line)
                                completed.append(result)
                                _stream_result_to_file(result, events_file)
                                # Log progress
                                preview = str(result)[:80]
                                logger.info(f"[{len(completed)}] {preview}...")
                            except json.JSONDecodeError:
                                pass
                        last_seen_count = len(lines)
                except Exception:
                    pass  # File may not exist yet

                # Check progress file
                try:
                    progress_content = await sb.read_file("/workspace/progress.json")
                    progress = json.loads(progress_content)
                    if progress.get("total"):
                        logger.info(f"Progress: {progress.get('completed', 0)}/{progress['total']}")
                except Exception:
                    pass

        # Run with timeout
        try:
            with trio.move_on_after(timeout_sec) as cancel_scope:
                async with trio.open_nursery() as nursery:
                    nursery.start_soon(run_script)
                    nursery.start_soon(poll_progress)

            if cancel_scope.cancelled_caught:
                timed_out = True
                logger.warning(f"Experiment timed out after {timeout_min} min")

        except Exception as e:
            error = str(e)
            logger.error(f"Experiment error: {e}")

        # Final read of remaining results
        try:
            results_content = await sb.read_file("/workspace/results.jsonl")
            lines = [l for l in results_content.strip().split("\n") if l.strip()]
            for line in lines[last_seen_count:]:
                try:
                    result = json.loads(line)
                    completed.append(result)
                    _stream_result_to_file(result, events_file)
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass

        duration = time.time() - start_time

        # Log experiment end
        with open(events_file, "a") as f:
            f.write(json.dumps({
                "type": "experiment_end",
                "timestamp": datetime.now().isoformat(),
                "completed": len(completed),
                "timed_out": timed_out,
                "duration_sec": duration,
                "error": error,
            }) + "\n")

        # Create result
        exp_result = ExperimentResult(
            name=name,
            completed=completed,
            stdout=final_stdout,
            stderr=final_stderr,
            timed_out=timed_out,
            duration_sec=duration,
            error=error,
        )

        # Save everything
        (run_dir / "result.json").write_text(json.dumps({
            "name": name,
            "timestamp": datetime.now().isoformat(),
            "completed": completed,
            "timed_out": timed_out,
            "duration_sec": duration,
            "error": error,
        }, indent=2))
        (run_dir / "stdout.txt").write_text(final_stdout)
        (run_dir / "stderr.txt").write_text(final_stderr)
        (run_dir / "script.py").write_text(script)
        if input_data is not None:
            (run_dir / "input.json").write_text(json.dumps(input_data, indent=2))

        logger.info(exp_result.summary())
        logger.info(f"Results: {run_dir}")

        return exp_result


# Convenience function for CLI usage
async def run_experiment_cli(
    name: str,
    script_path: str,
    input_path: str | None = None,
    timeout_min: float = 30,
    gpu: str = "A10G",
):
    """Run experiment from CLI with script and input file paths."""
    script = Path(script_path).read_text()
    input_data = None
    if input_path:
        input_data = json.loads(Path(input_path).read_text())

    return await run_sandbox_experiment(
        name=name,
        script=script,
        input_data=input_data,
        timeout_min=timeout_min,
        gpu=gpu,
    )


if __name__ == "__main__":
    import argparse
    import trio

    parser = argparse.ArgumentParser(description="Run experiment in sandbox")
    parser.add_argument("name", help="Experiment name")
    parser.add_argument("script", help="Path to script to run")
    parser.add_argument("--input", "-i", help="Path to input JSON file")
    parser.add_argument("--timeout", "-t", type=float, default=30, help="Timeout in minutes")
    parser.add_argument("--gpu", default="A10G", help="GPU type")
    args = parser.parse_args()

    result = trio.run(lambda: run_experiment_cli(
        name=args.name,
        script_path=args.script,
        input_path=args.input,
        timeout_min=args.timeout,
        gpu=args.gpu,
    ))

    print(f"\n{result.summary()}")
    if result.error:
        print(f"Error: {result.error}")
