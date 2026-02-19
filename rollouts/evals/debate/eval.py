"""Debate Eval - Two models argue until one concedes.

Two models are assigned opposing positions on a topic. They debate until:
1. One model concedes (uses phrases like "you're right", "I agree", etc.)
2. Max turns reached (draw) - only if max_turns is set

By default, debates run indefinitely until concession (no max_turns limit).
Use --max-turns to set a cap if needed.

Scoring:
- Winner (the model that convinced the other) gets reward=1.0
- Loser (the model that conceded) gets reward=0.0
- Draw (max turns, no concession) gives both reward=0.5

The primary agent argues FOR the position, the responder argues AGAINST.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from rollouts.dtypes import Endpoint, Message, Metric, Score
from rollouts.environments.dialogue import DialogueEnvironment, keyword_concession_detector
from rollouts.eval_runner import EvalSpec

logger = logging.getLogger(__name__)

# Paths
EVAL_DIR = Path(__file__).parent
TASKS_PATH = EVAL_DIR / "tasks.json"


def prepare_messages(sample_data: dict[str, Any]) -> list[Message]:
    """Prepare initial messages for the primary (FOR) agent."""
    topic = sample_data["topic"]
    position_for = sample_data.get("position_for", f"in favor of: {topic}")

    system_prompt = f"""You are participating in a debate. Your goal is to CONVINCE your opponent to change their mind.

YOUR POSITION: You are arguing {position_for}

RULES:
1. Make clear, logical arguments
2. Address your opponent's points directly
3. Use evidence and reasoning
4. Stay on topic
5. Be persuasive but respectful

WIN CONDITION: You win if your opponent concedes or agrees with your position.
LOSE CONDITION: You lose if you concede or agree with your opponent.

Do NOT concede unless you are genuinely convinced. Argue your position strongly."""

    user_message = f"""The debate topic is: {topic}

You are arguing {position_for}.

Your opponent will argue against this position. Make your opening argument now."""

    return [
        Message(role="system", content=system_prompt),
        Message(role="user", content=user_message),
    ]


def get_responder_system_prompt(sample_data: dict[str, Any]) -> str:
    """Get system prompt for the responder (AGAINST) agent."""
    topic = sample_data["topic"]
    position_against = sample_data.get("position_against", f"against: {topic}")

    return f"""You are participating in a debate. Your goal is to CONVINCE your opponent to change their mind.

YOUR POSITION: You are arguing {position_against}

RULES:
1. Make clear, logical arguments
2. Address your opponent's points directly
3. Use evidence and reasoning
4. Stay on topic
5. Be persuasive but respectful

WIN CONDITION: You win if your opponent concedes or agrees with your position.
LOSE CONDITION: You lose if you concede or agree with your opponent.

Do NOT concede unless you are genuinely convinced. Argue your position strongly."""


def score_sample(sample: Any) -> Score:
    """Score the debate based on who conceded.

    Looks at environment_state for concession_winner.
    Primary = FOR position, Responder = AGAINST position.
    """
    env_state = getattr(sample, "environment_state", {}) or {}

    concession_winner = env_state.get("concession_winner")
    turn_count = env_state.get("turn_count", 0)

    if concession_winner == "primary":
        # Primary (FOR) convinced the responder
        reward = 1.0
        outcome = "primary_wins"
    elif concession_winner == "responder":
        # Responder (AGAINST) convinced the primary
        reward = 0.0
        outcome = "responder_wins"
    else:
        # No concession - draw
        reward = 0.5
        outcome = "draw"

    return Score(
        metrics=(
            Metric("debate_reward", reward, weight=1.0),
            Metric("primary_wins", 1.0 if outcome == "primary_wins" else 0.0, weight=0.0),
            Metric("responder_wins", 1.0 if outcome == "responder_wins" else 0.0, weight=0.0),
            Metric("draw", 1.0 if outcome == "draw" else 0.0, weight=0.0),
            Metric("turns", float(turn_count), weight=0.0),
        )
    )


async def make_environment(sample_data: dict[str, Any]) -> DialogueEnvironment:
    """Create dialogue environment for a debate sample.

    The responder endpoint is configured via sample_data or defaults.
    """
    # Get responder endpoint config from sample or use defaults
    responder_config = sample_data.get("responder_endpoint", {})

    # Build endpoint - use same model as primary by default
    if responder_config:
        # api_key must be passed separately to from_dict (it gets popped from dict)
        api_key = responder_config.pop("api_key", "")
        responder_endpoint = Endpoint.from_dict(responder_config, api_key=api_key)
    else:
        # Default: use environment variables for a different model
        import os

        responder_endpoint = Endpoint(
            model=os.getenv("DEBATE_RESPONDER_MODEL", "openai/gpt-4o"),
            base_url=os.getenv("DEBATE_RESPONDER_BASE_URL", "https://api.openai.com/v1"),
            api_format="openai-completions",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            max_tokens=1024,
            temperature=0.7,
        )

    # max_turns=None means run until concession (no limit)
    # Set a specific max_turns in sample_data to cap the debate
    max_turns = sample_data.get("max_turns")  # None by default = no limit

    return DialogueEnvironment(
        responder_endpoint=responder_endpoint,
        responder_system_prompt=get_responder_system_prompt(sample_data),
        max_turns=max_turns,
        detect_concession=keyword_concession_detector(),
    )


# ── EvalSpec Definition ───────────────────────────────────────────────────────

spec = EvalSpec(
    name="debate",
    prepare_messages=prepare_messages,
    score_fn=score_sample,
    make_environment=make_environment,
    default_tasks_path=TASKS_PATH,
    per_sample_environment=True,
)


def get_spec() -> EvalSpec:
    """Get the EvalSpec for this eval."""
    return spec


# ── CLI Entry Point ───────────────────────────────────────────────────────────


def run(
    tasks_path: Path | str | None = None,
    primary_model: str = "anthropic/claude-sonnet-4-20250514",
    responder_model: str = "openai/gpt-4o",
    max_turns: int | None = None,  # None = run until concession (no limit)
    max_concurrent: int = 1,
    limit: int | None = None,
    verbose: bool = True,
    output_dir: Path | str | None = None,
) -> dict[str, Any]:
    """Run the debate eval.

    Args:
        primary_model: Model for the FOR position (provider/model format)
        responder_model: Model for the AGAINST position (provider/model format)
        max_turns: Maximum debate exchanges before draw. None = run until concession.
    """
    import json
    import os
    from datetime import datetime

    from rollouts.eval_runner import run_eval_from_spec

    # Load tasks
    path = Path(tasks_path) if tasks_path else TASKS_PATH
    if not path.exists():
        raise FileNotFoundError(f"Tasks file not found: {path}")

    with open(path) as f:
        tasks = json.load(f)

    if limit:
        tasks = tasks[:limit]

    # Inject responder config into each task
    responder_provider, responder_model_id = responder_model.split("/", 1)

    # Determine base URL and API format from provider
    provider_configs = {
        "openai": ("https://api.openai.com/v1", "openai-completions", "OPENAI_API_KEY"),
        "anthropic": ("https://api.anthropic.com/v1", "anthropic-messages", "ANTHROPIC_API_KEY"),
        "google": (
            "https://generativelanguage.googleapis.com/v1beta",
            "google-generative-ai",
            "GOOGLE_API_KEY",
        ),
    }

    if responder_provider not in provider_configs:
        raise ValueError(f"Unknown provider: {responder_provider}. Use openai/anthropic/google")

    base_url, api_format, api_key_env = provider_configs[responder_provider]

    for task in tasks:
        task["responder_endpoint"] = {
            "model": responder_model,
            "base_url": base_url,
            "api_format": api_format,
            "api_key": os.getenv(api_key_env, ""),
            "max_tokens": 1024,
            "temperature": 0.7,
        }
        # Only set max_turns if specified (None = no limit, run until concession)
        if max_turns is not None:
            task["max_turns"] = max_turns

    if verbose:
        print(f"Loaded {len(tasks)} debate topics")
        print(f"Primary (FOR): {primary_model}")
        print(f"Responder (AGAINST): {responder_model}")

    # Build output dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = EVAL_DIR / "results" / f"debate_{timestamp}"

    output_dir = Path(output_dir)

    # Parse primary model
    primary_provider, primary_model_id = primary_model.split("/", 1)

    # Agent loop max_turns: None = no limit (run until concession)
    # If max_turns specified, double it (each "turn" is one exchange = 2 agent turns)
    agent_max_turns = max_turns * 2 if max_turns is not None else 1000  # Safety limit

    # Run eval
    result = run_eval_from_spec(
        spec,
        tasks=tasks,
        model=primary_model_id,
        provider=primary_provider,
        max_turns=agent_max_turns,
        max_concurrent=max_concurrent,
        verbose=verbose,
        show_progress=True,
        output_dir=output_dir,
        temperature=0.7,
    )

    return result


def _run_in_tmux(args: argparse.Namespace, output_dir: Path) -> str:
    """Spawn eval in a tmux session. Returns session name."""
    import os
    import shlex
    import subprocess

    session_name = f"debate-{output_dir.name}"

    # Collect API keys to pass to tmux
    env_vars = []
    for key in ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"]:
        val = os.environ.get(key)
        if val:
            env_vars.append(f"{key}={shlex.quote(val)}")

    # Build the command to run inside tmux
    cmd_parts = [
        "uv",
        "run",
        "python",
        str(Path(__file__).resolve()),
        "--output-dir",
        str(output_dir),
        "--_run-direct",  # Internal flag to skip tmux spawning
    ]
    if args.tasks:
        cmd_parts.extend(["--tasks", args.tasks])
    if args.primary_model:
        cmd_parts.extend(["--primary-model", args.primary_model])
    if args.responder_model:
        cmd_parts.extend(["--responder-model", args.responder_model])
    if args.max_turns is not None:
        cmd_parts.extend(["--max-turns", str(args.max_turns)])
    if args.max_concurrent:
        cmd_parts.extend(["--max-concurrent", str(args.max_concurrent)])
    if args.limit:
        cmd_parts.extend(["--limit", str(args.limit)])
    if args.verbose:
        cmd_parts.append("--verbose")

    inner_cmd = shlex.join(cmd_parts)

    # Prefix with env vars
    if env_vars:
        inner_cmd = " ".join(env_vars) + " " + inner_cmd

    # Create tmux session
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name, inner_cmd],
        check=True,
    )

    return session_name


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run debate eval")
    parser.add_argument("--tasks", type=str, help="Path to tasks.json")
    parser.add_argument(
        "--primary-model",
        type=str,
        default="anthropic/claude-sonnet-4-20250514",
        help="Model for FOR position (provider/model)",
    )
    parser.add_argument(
        "--responder-model",
        type=str,
        default="openai/gpt-4o",
        help="Model for AGAINST position (provider/model)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Max debate exchanges. Default: no limit (run until concession)",
    )
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--limit", type=int, help="Limit number of topics")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument(
        "--attach", action="store_true", help="Attach to tmux session (default: fire and forget)"
    )
    parser.add_argument(
        "--_run-direct", action="store_true", dest="run_direct", help=argparse.SUPPRESS
    )  # Internal: run directly, don't spawn tmux
    args = parser.parse_args()

    # Compute output_dir early so we can pass it to tmux
    from datetime import datetime

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = EVAL_DIR / "results" / f"debate_{timestamp}"

    # If not running directly, spawn tmux
    if not args.run_direct:
        session_name = _run_in_tmux(args, output_dir)
        print(f"Started debate eval in tmux session: {session_name}")
        print(f"Output dir: {output_dir}")
        print()
        print(f"  tmux attach -t {session_name}   # attach to session")
        print(f"  tmux kill-session -t {session_name}   # kill session")

        if args.attach:
            import subprocess

            subprocess.run(["tmux", "attach", "-t", session_name])
    else:
        # Run directly (called from within tmux)
        result = run(
            tasks_path=args.tasks,
            primary_model=args.primary_model,
            responder_model=args.responder_model,
            max_turns=args.max_turns,
            max_concurrent=args.max_concurrent,
            limit=args.limit,
            verbose=args.verbose,
            output_dir=output_dir,
        )

        print(f"\nResults: {result}")
