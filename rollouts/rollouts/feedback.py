"""Agent exit survey and feedback collection.

Collects feedback from the agent at pause/exit points about:
- Task success assessment
- Harness/tooling feedback
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .dtypes import AgentState, Endpoint


@dataclass
class ExitSurveyResult:
    """Result of an agent exit survey."""

    session_id: str | None
    timestamp: str
    exit_reason: str  # "abort", "completion", "yield", "max_turns"
    should_survey: bool
    task_success: str | None  # "yes", "no", "partial", "unknown"
    task_notes: str | None
    harness_feedback: str | None


def _get_feedback_path() -> Path:
    """Get the path to the feedback file."""
    feedback_dir = Path.home() / ".rollouts" / "feedback"
    feedback_dir.mkdir(parents=True, exist_ok=True)
    return feedback_dir / "all.jsonl"


async def check_should_survey(
    state: AgentState,
    endpoint: Endpoint,
    exit_reason: str,
) -> bool:
    """Light API call to check if we should do a full exit survey.

    Returns True if the agent thinks a survey would be valuable.
    """
    from .providers.anthropic import anthropic_completion

    # Build a minimal prompt to check if survey is warranted
    check_prompt = f"""You are an AI agent that just paused/exited. Exit reason: {exit_reason}

Based on your recent work, would feedback about your task progress or the harness/tools be valuable right now?

Reply with just "yes" or "no"."""

    # Use a fast, cheap model for the check
    from dataclasses import replace

    light_endpoint = replace(
        endpoint,
        model="claude-3-haiku-20240307",
        max_tokens=10,
        thinking=None,  # No thinking for quick check
    )

    from .dtypes import Actor, Message, Trajectory

    check_trajectory = Trajectory(
        messages=[
            Message(role="user", content=check_prompt),
        ]
    )
    check_actor = Actor(endpoint=light_endpoint, trajectory=check_trajectory)

    try:
        completion = await anthropic_completion(check_actor)
        response = completion.content.strip().lower()
        return response.startswith("yes")
    except Exception:
        # On error, default to not surveying
        return False


async def collect_exit_survey(
    state: AgentState,
    endpoint: Endpoint,
    exit_reason: str,
    session_id: str | None = None,
) -> ExitSurveyResult:
    """Collect full exit survey from the agent.

    Asks about task success and harness feedback.
    """
    from .providers.anthropic import anthropic_completion

    # Get recent context (last few messages)
    recent_messages = (
        state.actor.trajectory.messages[-5:] if state.actor.trajectory.messages else []
    )
    context_summary = "\n".join([
        f"{m.role}: {str(m.content)[:200]}..."
        if len(str(m.content)) > 200
        else f"{m.role}: {m.content}"
        for m in recent_messages
    ])

    survey_prompt = f"""You are an AI agent that just paused/exited. Exit reason: {exit_reason}

Recent context:
{context_summary}

Please provide brief feedback:

1. TASK SUCCESS: How successful were you at your task? (yes/no/partial/unknown)
2. TASK NOTES: Any notes on task progress or blockers? (1-2 sentences)
3. HARNESS FEEDBACK: Any feedback on the harness/tools/environment? (1-2 sentences, or "none")

Format your response exactly as:
TASK_SUCCESS: <yes|no|partial|unknown>
TASK_NOTES: <notes>
HARNESS_FEEDBACK: <feedback>"""

    from dataclasses import replace

    survey_endpoint = replace(
        endpoint,
        model="claude-3-haiku-20240307",
        max_tokens=300,
        thinking=None,
    )

    from .dtypes import Actor, Message, Trajectory

    survey_trajectory = Trajectory(
        messages=[
            Message(role="user", content=survey_prompt),
        ]
    )
    survey_actor = Actor(endpoint=survey_endpoint, trajectory=survey_trajectory)

    task_success = None
    task_notes = None
    harness_feedback = None

    try:
        completion = await anthropic_completion(survey_actor)
        response = completion.content

        # Parse response
        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("TASK_SUCCESS:"):
                task_success = line.split(":", 1)[1].strip().lower()
            elif line.startswith("TASK_NOTES:"):
                task_notes = line.split(":", 1)[1].strip()
            elif line.startswith("HARNESS_FEEDBACK:"):
                harness_feedback = line.split(":", 1)[1].strip()

    except Exception:
        pass  # Survey failed, record what we have

    return ExitSurveyResult(
        session_id=session_id,
        timestamp=datetime.now().isoformat(),
        exit_reason=exit_reason,
        should_survey=True,
        task_success=task_success,
        task_notes=task_notes,
        harness_feedback=harness_feedback,
    )


def save_survey_result(result: ExitSurveyResult) -> None:
    """Append survey result to feedback file."""
    feedback_path = _get_feedback_path()

    entry = {
        "session_id": result.session_id,
        "timestamp": result.timestamp,
        "exit_reason": result.exit_reason,
        "should_survey": result.should_survey,
        "task_success": result.task_success,
        "task_notes": result.task_notes,
        "harness_feedback": result.harness_feedback,
    }

    with open(feedback_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


async def run_exit_survey(
    state: AgentState,
    endpoint: Endpoint,
    exit_reason: str,
    session_id: str | None = None,
    *,
    skip_check: bool = False,
) -> ExitSurveyResult | None:
    """Run the full exit survey flow.

    1. Check if survey is warranted (unless skip_check=True)
    2. If yes, collect full survey
    3. Save result to feedback file

    Returns the survey result, or None if survey was skipped.
    """
    # Light check first
    if not skip_check:
        should_survey = await check_should_survey(state, endpoint, exit_reason)
        if not should_survey:
            # Record that we checked but didn't survey
            result = ExitSurveyResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                exit_reason=exit_reason,
                should_survey=False,
                task_success=None,
                task_notes=None,
                harness_feedback=None,
            )
            save_survey_result(result)
            return result

    # Full survey
    result = await collect_exit_survey(state, endpoint, exit_reason, session_id)
    save_survey_result(result)
    return result
