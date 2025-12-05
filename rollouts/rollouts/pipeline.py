"""Agent pipeline primitives for composable multi-agent workflows.

Tiger Style: Agents as pure data transformers.
Casey Muratori: Semantic compression - common patterns extracted.

Design Philosophy:
- Agents already take AgentState (which contains Trajectory + Endpoint + Environment)
- This module provides: trajectory transforms + pipeline orchestration
- Composition via transforms between agent runs
- All state explicit and serializable

Core primitive:
    run_agent_pipeline(
        initial_state,
        stages=[
            # Each stage: (transform_fn, endpoint, environment_factory, max_turns, n_parallel, reduce_fn)
        ]
    )
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import trio

from .agents import run_agent
from .dtypes import Actor, AgentState, Endpoint, Environment, Message, RunConfig, Trajectory

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Trajectory Transforms
# ══════════════════════════════════════════════════════════════════════════════

def compact_trajectory(
    trajectory: Trajectory,
    keep_last_n: int = 5,
    keep_system: bool = True
) -> Trajectory:
    """Compact trajectory by keeping only recent messages.

    Args:
        trajectory: Input trajectory
        keep_last_n: Number of recent messages to keep
        keep_system: Whether to preserve initial system message

    Returns:
        Compacted trajectory with same metadata/rewards
    """
    messages = trajectory.messages

    # Find system message if it exists
    system_msg = None
    if keep_system and messages and messages[0].role == "system":
        system_msg = messages[0]
        messages = messages[1:]

    # Keep last N messages
    recent = messages[-keep_last_n:] if len(messages) > keep_last_n else messages

    # Reconstruct
    new_messages = ([system_msg] if system_msg else []) + recent

    return replace(trajectory, messages=new_messages)


def summarize_trajectory(
    trajectory: Trajectory,
    summarizer: Callable[[list[Message]], str]
) -> Trajectory:
    """Summarize trajectory using custom summarizer function.

    Args:
        trajectory: Input trajectory
        summarizer: Function that takes messages and returns summary string

    Returns:
        New trajectory with summary message
    """
    summary_text = summarizer(trajectory.messages)

    # Create new trajectory with just system (if exists) + summary
    system_msg = None
    if trajectory.messages and trajectory.messages[0].role == "system":
        system_msg = trajectory.messages[0]

    summary_msg = Message(role="user", content=summary_text)

    new_messages = ([system_msg] if system_msg else []) + [summary_msg]

    return replace(trajectory, messages=new_messages)


def filter_trajectory(
    trajectory: Trajectory,
    predicate: Callable[[Message], bool]
) -> Trajectory:
    """Filter messages by predicate.

    Example:
        # Remove all tool result messages
        filtered = filter_trajectory(traj, lambda m: m.role != "tool")
    """
    filtered_messages = [m for m in trajectory.messages if predicate(m)]
    return replace(trajectory, messages=filtered_messages)


def inject_message(
    trajectory: Trajectory,
    message: Message,
    position: int = -1
) -> Trajectory:
    """Inject a message at specified position.

    Args:
        trajectory: Input trajectory
        message: Message to inject
        position: Where to insert (-1 = append, 0 = prepend after system)
    """
    messages = list(trajectory.messages)

    if position == -1:
        messages.append(message)
    else:
        messages.insert(position, message)

    return replace(trajectory, messages=messages)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Parallel Execution + Reduction
# ══════════════════════════════════════════════════════════════════════════════

async def run_parallel_agents(
    initial_state: AgentState,
    n: int,
    environment_factory: Callable[[], Environment] | None = None,
    run_config: RunConfig | None = None,
) -> tuple[list[Trajectory], list[list[AgentState]]]:
    """Run N agents in parallel from same starting state.

    Args:
        initial_state: Starting state (trajectory, endpoint, max_turns)
        n: Number of parallel agents to run
        environment_factory: Factory to create fresh environment per agent
        run_config: Execution configuration

    Returns:
        (trajectories, all_agent_states) - List of final trajectories and full state lists
    """
    if run_config is None:
        run_config = RunConfig(
            on_chunk=lambda _: trio.lowlevel.checkpoint(),
            show_progress=False
        )

    trajectories = []
    all_states = []

    async def run_single_agent(agent_idx: int):
        """Run one agent instance."""
        # Create fresh environment if factory provided
        env = environment_factory() if environment_factory else initial_state.environment

        # Create state for this agent
        agent_state = replace(
            initial_state,
            environment=env,
            turn_idx=0,
            pending_tool_calls=[],
            next_tool_idx=0
        )

        # Update tools if environment changed
        if env:
            agent_state = replace(
                agent_state,
                actor=replace(agent_state.actor, tools=env.get_tools())
            )

        # Run agent
        logger.info(f"Starting parallel agent {agent_idx + 1}/{n}")
        states = await run_agent(agent_state, run_config)

        trajectories.append(states[-1].actor.trajectory)
        all_states.append(states)
        logger.info(f"Completed parallel agent {agent_idx + 1}/{n}")

    # Run all in parallel
    async with trio.open_nursery() as nursery:
        for i in range(n):
            nursery.start_soon(run_single_agent, i)

    logger.info(f"Completed all {n} parallel agents")
    return trajectories, all_states


# ══════════════════════════════════════════════════════════════════════════════
# 3. Common Reduce Functions
# ══════════════════════════════════════════════════════════════════════════════

def reduce_select_best(
    trajectories: list[Trajectory],
    metric: Callable[[Trajectory], float],
    maximize: bool = True
) -> Trajectory:
    """Select best trajectory by metric.

    Args:
        trajectories: List of trajectories to choose from
        metric: Function that scores a trajectory
        maximize: If True, pick highest score; else lowest

    Example:
        best = reduce_select_best(trajs, lambda t: t.rewards)
    """
    scored = [(metric(t), t) for t in trajectories]
    scored.sort(key=lambda x: x[0], reverse=maximize)
    return scored[0][1]


def reduce_majority_vote(
    trajectories: list[Trajectory],
    extract_answer: Callable[[Trajectory], Any]
) -> Trajectory:
    """Select trajectory with most common answer.

    Args:
        trajectories: List of trajectories
        extract_answer: Function to extract answer from trajectory

    Returns:
        Trajectory with most common answer
    """
    answers = [extract_answer(t) for t in trajectories]
    most_common = max(set(answers), key=answers.count)
    # Return first trajectory with most common answer
    for i, ans in enumerate(answers):
        if ans == most_common:
            return trajectories[i]
    return trajectories[0]  # Fallback


async def reduce_llm_judge(
    trajectories: list[Trajectory],
    judge_endpoint: Endpoint,
    judge_prompt_fn: Callable[[list[Trajectory]], list[Message]],
    run_config: RunConfig | None = None
) -> Trajectory:
    """Use an LLM to judge and synthesize multiple trajectories.

    Args:
        trajectories: List of trajectories to judge
        judge_endpoint: Endpoint for judge LLM
        judge_prompt_fn: Function that creates judge prompt from trajectories
        run_config: Execution configuration for judge

    Returns:
        Synthesized trajectory from judge
    """
    judge_messages = judge_prompt_fn(trajectories)
    judge_trajectory = Trajectory(messages=judge_messages)

    judge_actor = Actor(
        trajectory=judge_trajectory,
        endpoint=judge_endpoint,
        tools=[]
    )
    judge_state = AgentState(
        actor=judge_actor,
        environment=None,
        max_turns=1
    )

    if run_config is None:
        run_config = RunConfig(
            on_chunk=lambda _: trio.lowlevel.checkpoint(),
            show_progress=False
        )

    states = await run_agent(judge_state, run_config)
    return states[-1].actor.trajectory


# ══════════════════════════════════════════════════════════════════════════════
# 4. Pipeline Orchestration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentStage:
    """One stage in an agent pipeline.

    A stage can:
    - Transform the trajectory before running agents
    - Run 1 or more agents in parallel with a given endpoint/environment
    - Reduce multiple trajectories to one (if n > 1)
    """
    # Agent execution (at least one of endpoint or environment_factory required)
    endpoint: Endpoint | None = None  # If None, reuse from previous stage
    environment_factory: Callable[[], Environment] | None = None  # If None, no environment
    max_turns: int = 10

    # Optional: Transform trajectory before this stage
    transform: Callable[[Trajectory], Trajectory] | None = None

    # Parallelism (test-time scaling)
    n: int = 1  # Number of parallel agents
    reduce_fn: Callable[[list[Trajectory]], Trajectory] | None = None  # Required if n > 1

    # Execution config
    run_config: RunConfig | None = None


async def run_agent_pipeline(
    initial_trajectory: Trajectory,
    initial_endpoint: Endpoint,
    stages: list[AgentStage],
    run_config: RunConfig | None = None
) -> list[Trajectory]:
    """Run agents through pipeline stages.

    Each stage can:
    - Transform trajectory
    - Run k agents in parallel (test-time scaling)
    - Reduce results back to single trajectory
    - Switch to different endpoint/environment

    Args:
        initial_trajectory: Starting trajectory
        initial_endpoint: Starting endpoint
        stages: List of agent stages
        run_config: Default run config (can be overridden per stage)

    Returns:
        List of trajectories (one per stage)

    Example:
        stages = [
            # Stage 1: Search for relevant info with fast models
            AgentStage(
                endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
                environment_factory=lambda: KernelEnvironmentSearch(),
                max_turns=5,
                n=3,
                reduce_fn=lambda trajs: reduce_consolidate(trajs) 
            ),
            # Stage 2: Compact prev trajectory + Plan improvements with strong model
            AgentStage(
                transform=lambda t: compact_trajectory(t, keep_last_n=3),
                endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5"),
                environment_factory=lambda: KernelEnvironment(),
                max_turns=10,
                n=1
            ),
            # Stage 3: Test-time scaling (8 parallel) + pick best
            AgentStage(
                endpoint=Endpoint(provider="anthropic", model="claude-sonnet-4-5"),
                environment_factory=lambda: KernelEnvironment(),
                max_turns=5,
                n=8,
                reduce_fn=lambda trajs: reduce_select_best(trajs, lambda t: t.rewards)
            ),
        ]

        results = await run_agent_pipeline(initial_traj, initial_endpoint, stages)
        final_traj = results[-1]
    """
    trajectories = [initial_trajectory]
    current_traj = initial_trajectory
    current_endpoint = initial_endpoint

    for i, stage in enumerate(stages):
        logger.info(f"Running pipeline stage {i + 1}/{len(stages)}")

        # 1. Apply transform if specified
        if stage.transform:
            current_traj = stage.transform(current_traj)
            logger.info(f"  Applied transform: {len(current_traj.messages)} messages")

        # 2. Determine endpoint
        if stage.endpoint:
            current_endpoint = stage.endpoint

        # 3. Create initial state for this stage
        env = stage.environment_factory() if stage.environment_factory else None
        initial_state = AgentState(
            actor=Actor(
                trajectory=current_traj,
                endpoint=current_endpoint,
                tools=env.get_tools() if env else []
            ),
            environment=env,
            max_turns=stage.max_turns
        )

        # 4. Run agents (1 or more in parallel)
        stage_run_config = stage.run_config or run_config

        if stage.n == 1:
            # Single agent
            logger.info(f"  Running 1 agent (max_turns={stage.max_turns})")
            states = await run_agent(initial_state, stage_run_config or RunConfig(
                on_chunk=lambda _: trio.lowlevel.checkpoint(),
                show_progress=False
            ))
            current_traj = states[-1].actor.trajectory
        else:
            # Parallel agents
            assert stage.reduce_fn is not None, f"Stage {i}: n={stage.n} requires reduce_fn"
            logger.info(f"  Running {stage.n} parallel agents (max_turns={stage.max_turns})")
            trajs, _ = await run_parallel_agents(
                initial_state,
                n=stage.n,
                environment_factory=stage.environment_factory,
                run_config=stage_run_config
            )
            current_traj = stage.reduce_fn(trajs)
            logger.info(f"  Reduced {len(trajs)} trajectories to 1")

        trajectories.append(current_traj)
        logger.info(f"  Stage {i + 1} complete: {len(current_traj.messages)} messages")

    logger.info(f"Pipeline complete: {len(stages)} stages")
    return trajectories
