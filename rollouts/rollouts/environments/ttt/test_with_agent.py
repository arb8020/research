"""Test CodeChallengeEnvironment with run_agent."""

import os

import trio

from ...agents import run_agent
from ...dtypes import Actor, AgentState, Endpoint, Message, RunConfig, Trajectory
from .code_challenge import FibonacciEnvironment


async def main() -> None:
    # Create environment
    env = FibonacciEnvironment(timeout=2.0)

    # Create initial state
    actor = Actor(
        trajectory=Trajectory(
            messages=[Message(role="user", content=env.get_initial_user_message())]
        ),
        endpoint=Endpoint(
            provider="anthropic",
            model="claude-sonnet-4-20250514",
            api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        ),
        tools=env.get_tools(),
    )

    initial_state = AgentState(
        actor=actor,
        environment=env,
    )

    # Quiet run config (no streaming output)
    async def noop_chunk(e: object) -> None:
        pass

    run_config = RunConfig(on_chunk=noop_chunk)

    print("Running agent with FibonacciEnvironment...")
    print(f"Prompt: {env.prompt[:100]}...")
    print()

    # Run agent (single turn - will stop after on_assistant_message)
    states = await run_agent(initial_state, run_config)

    # Get final state
    final_state = states[-1]
    final_env = final_state.environment

    # Get the response
    messages = final_state.actor.trajectory.messages
    assistant_msg = next((m for m in reversed(messages) if m.role == "assistant"), None)

    if assistant_msg:
        content = assistant_msg.content
        if isinstance(content, str):
            print(f"Response:\n{content[:500]}...")
        else:
            # Content blocks
            for block in content:
                if hasattr(block, "text"):
                    print(f"Response:\n{block.text[:500]}...")
                    break
    print()

    # Get the score
    score = final_env.get_last_score()
    if score:
        print("Grading:")
        print(f"  Reward: {score.reward:.3f}")
        for m in score.metrics:
            meta = f" {m.metadata}" if m.metadata else ""
            print(f"  {m.name}: {m.value:.3f} (weight={m.weight}){meta}")
    else:
        print("No score available")

    print()
    print(f"Stop reason: {final_state.stop}")
    print(f"Turns: {final_state.turn_idx}")


if __name__ == "__main__":
    trio.run(main)
