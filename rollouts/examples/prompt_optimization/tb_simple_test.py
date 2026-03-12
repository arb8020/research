"""Simple terminal-bench test on fix-permissions task.

Integration test that verifies TerminalBenchEnvironment works with our agent.

Run:
    python -m examples.prompt_optimization.tb_simple_test
"""

import logging
import os

import trio

from rollouts.agents import Actor, AgentState, RunConfig, handle_stop_max_turns, run_agent
from rollouts.core import Endpoint, Message, Trajectory
from rollouts.dtypes import StreamEvent
from rollouts.environments.terminal_bench import TerminalBenchEnvironment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


async def test_agent() -> None:
    """Run agent on fix-permissions task."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        logger.error("No ANTHROPIC_API_KEY")
        return

    task_id = "fix-permissions"  # Super simple: just chmod +x
    logger.info(f"Testing agent on: {task_id}")

    env = await TerminalBenchEnvironment.create(
        task_id=task_id,
        no_rebuild=True,
        cleanup=True,
    )

    logger.info(f"Task: {env.instruction}")

    endpoint = Endpoint(
        provider="anthropic",
        model="claude-sonnet-4-5-20250929",
        api_key=api_key,
        max_tokens=4096,
    )

    system = """You are a terminal agent. Solve the task using the available tools.
Keep it simple and direct. After solving, call task_complete."""

    env_system = env.get_system_prompt() or ""

    messages = [
        Message(role="system", content=f"{system}\n\n{env_system}"),
        Message(role="user", content=f"Solve this task:\n\n{env.instruction}"),
    ]

    actor = Actor(
        trajectory=Trajectory(messages=messages),
        endpoint=endpoint,
        tools=env.get_tools(),
    )
    state = AgentState(actor=actor, environment=env)

    async def on_chunk(event: StreamEvent) -> None:
        if hasattr(event, "text"):
            print(event.text, end="", flush=True)

    config = RunConfig(on_chunk=on_chunk, handle_stop=handle_stop_max_turns(20))

    logger.info("Running agent...")
    try:
        states = await run_agent(state, config)
        final = states[-1] if isinstance(states, list) else states
        print()
        print("=" * 60)
        logger.info(f"Finished: turn_idx={final.turn_idx}, stop={final.stop}")
        logger.info(f"Interactions: {len(env.interactions)}")
        logger.info(f"Task completed flag: {env._task_completed}")

        # Show what commands were run
        print("\nCommands executed:")
        for i, interaction in enumerate(env.interactions):
            if interaction["type"] == "send_keys":
                keys = interaction.get("keystrokes", interaction.get("keys", "")).replace(
                    "\n", "\\n"
                )
                print(f"  {i + 1}. {keys}")
    finally:
        await env.cleanup()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    trio.run(test_agent)
