"""GEPA optimization for a Crafter game verifier.

Crafter is a text-based RL environment where agents collect resources, build tools,
and try to survive. This example demonstrates optimizing a verifier prompt that
scores agent trajectories - useful for training RL agents with learned reward models.

Inspired by: https://docs.usesynth.ai/cookbooks/workflows/crafter-verifier

The verifier approach:
1. Collect human-labeled trajectories with quality scores
2. Use GEPA to optimize a prompt that predicts these scores
3. Deploy the optimized verifier as a reward model for RL

Run with:
    python -m examples.prompt_optimization.crafter_verifier_gepa
"""

import logging
import os
import re

import trio
from rollouts.dtypes import Endpoint, Metric, Score
from rollouts.prompt_optimization import (
    EvolutionaryConfig,
    PromptTemplate,
    run_evolutionary_gepa,
)
from rollouts.training.types import Sample

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─── Simulated Crafter Trajectories ───────────────────────────────────────────
# In production, these would be real agent traces from the Crafter environment
# with human-annotated quality scores.

DATASET = [
    # High-quality trajectory: Good resource gathering, tool crafting, survival
    {
        "trajectory": """
Turn 1: Agent observes grass, tree, cow nearby. Health: 9/9, Food: 9/9, Drink: 9/9
Action: collect_wood
Result: Collected 1 wood. Inventory: {wood: 1}

Turn 2: Agent near tree and stone.
Action: collect_wood
Result: Collected 1 wood. Inventory: {wood: 2}

Turn 3: Agent near stone deposit.
Action: collect_stone
Result: Collected 1 stone. Inventory: {wood: 2, stone: 1}

Turn 4: Agent has materials for crafting.
Action: craft_wood_pickaxe
Result: Crafted wood pickaxe! Inventory: {wood: 1, stone: 1, wood_pickaxe: 1}

Turn 5: Agent near iron ore.
Action: mine_iron (using wood_pickaxe)
Result: Collected 1 iron. Inventory: {wood: 1, stone: 1, wood_pickaxe: 1, iron: 1}

Turn 6: Agent built shelter.
Action: place_table
Result: Placed crafting table. Achievement unlocked: "Place Table"

Final state: Health: 8/9, Food: 7/9, Drink: 6/9
Achievements: ["Collect Wood", "Collect Stone", "Make Wood Pickaxe", "Collect Iron", "Place Table"]
Survival time: 50 steps
""",
        "human_score": 0.85,  # High score for good progression
    },
    # Medium-quality trajectory: Some progress but inefficient
    {
        "trajectory": """
Turn 1: Agent observes grass, water, zombie nearby. Health: 9/9, Food: 9/9, Drink: 9/9
Action: move_left
Result: Moved away from zombie.

Turn 2: Agent near tree.
Action: collect_wood
Result: Collected 1 wood. Inventory: {wood: 1}

Turn 3: Agent wandering.
Action: move_right
Result: Moved right.

Turn 4: Agent wandering.
Action: move_up
Result: Moved up.

Turn 5: Agent near tree again.
Action: collect_wood
Result: Collected 1 wood. Inventory: {wood: 2}

Turn 6: Agent getting hungry.
Action: eat_plant
Result: No plant nearby. Failed.

Final state: Health: 7/9, Food: 4/9, Drink: 5/9
Achievements: ["Collect Wood"]
Survival time: 30 steps
""",
        "human_score": 0.45,  # Medium score - survived but inefficient
    },
    # Low-quality trajectory: Poor decisions, died early
    {
        "trajectory": """
Turn 1: Agent observes zombie, skeleton nearby. Health: 9/9, Food: 9/9, Drink: 9/9
Action: attack_zombie
Result: Attacked zombie. Took 3 damage. Health: 6/9

Turn 2: Agent still fighting.
Action: attack_zombie
Result: Killed zombie! Took 2 damage. Health: 4/9

Turn 3: Skeleton approaching.
Action: attack_skeleton
Result: Attacked skeleton. Took 4 damage. Health: 0/9

GAME OVER - Agent died.

Final state: Health: 0/9, Food: 8/9, Drink: 8/9
Achievements: []
Survival time: 3 steps
""",
        "human_score": 0.1,  # Low score - died immediately
    },
    # Excellent trajectory: Late-game achievements
    {
        "trajectory": """
Turn 1-10: Agent efficiently gathered wood (5), stone (3), coal (2).
Action: craft_wood_pickaxe, craft_stone_pickaxe
Result: Has both pickaxes. Inventory well-stocked.

Turn 11-20: Agent mined iron (3), crafted iron tools.
Action: craft_iron_pickaxe, craft_iron_sword
Result: Has iron tools! Achievement: "Make Iron Pickaxe", "Make Iron Sword"

Turn 21-30: Agent built shelter, placed furnace.
Action: place_furnace
Result: Furnace placed. Achievement: "Place Furnace"

Turn 31-40: Agent crafted diamond tools using furnace.
Action: smelt_iron, craft_diamond_pickaxe
Result: Diamond pickaxe! Achievement: "Make Diamond Pickaxe"

Final state: Health: 9/9, Food: 7/9, Drink: 8/9
Achievements: ["Collect Wood", "Collect Stone", "Collect Coal", "Collect Iron",
               "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
               "Make Iron Sword", "Place Table", "Place Furnace", "Make Diamond Pickaxe"]
Survival time: 100 steps
""",
        "human_score": 0.95,  # Excellent - late-game achievements
    },
    # Survival-focused but no progression
    {
        "trajectory": """
Turn 1-20: Agent focused on food and water.
Actions: eat_plant, drink_water, eat_cow, drink_water (repeated)
Result: Maintained full food/drink but no crafting.

Turn 21-40: Agent avoided all enemies, kept eating.
Actions: move_away, eat_plant, drink_water (repeated)
Result: Still alive but no achievements.

Final state: Health: 9/9, Food: 9/9, Drink: 9/9
Achievements: []
Survival time: 80 steps
""",
        "human_score": 0.35,  # Low-medium - survived but no progress
    },
    # Good early game, died mid-game
    {
        "trajectory": """
Turn 1-10: Agent gathered resources efficiently.
Inventory: {wood: 4, stone: 3, iron: 1}
Achievements: ["Collect Wood", "Collect Stone", "Make Wood Pickaxe"]

Turn 11-15: Agent encountered skeleton army.
Action: fight_skeleton (x3)
Result: Killed 2 skeletons but took heavy damage.

Turn 16: Agent at low health, tried to flee.
Action: move_away
Result: Skeleton arrow hit. Health: 0/9

GAME OVER - Agent died.

Final state: Health: 0/9, Food: 6/9, Drink: 5/9
Achievements: ["Collect Wood", "Collect Stone", "Make Wood Pickaxe"]
Survival time: 16 steps
""",
        "human_score": 0.5,  # Medium - good start but died
    },
    # Perfect resource management
    {
        "trajectory": """
Turn 1-15: Systematic resource gathering.
- Wood: 8 collected
- Stone: 5 collected
- Coal: 3 collected
- Iron: 2 collected

Turn 16-25: Tool progression.
- Crafted: wood_pickaxe, stone_pickaxe, iron_pickaxe
- Built: table, furnace

Turn 26-35: Food stockpile created.
- Planted seeds, harvested crops
- Stored 10 food items

Turn 36-50: Defensive structures.
- Built walls around base
- Crafted iron sword for defense

Final state: Health: 9/9, Food: 9/9, Drink: 9/9
Achievements: ["Collect Wood", "Collect Stone", "Collect Coal", "Collect Iron",
               "Make Wood Pickaxe", "Make Stone Pickaxe", "Make Iron Pickaxe",
               "Place Table", "Place Furnace", "Plant Seed", "Collect Crops"]
Survival time: 100 steps
""",
        "human_score": 0.9,  # Very high - systematic progression
    },
    # Chaotic but lucky survival
    {
        "trajectory": """
Turn 1: Agent immediately attacked by zombie.
Action: run_away
Result: Escaped with 7/9 health.

Turn 2-5: Agent wandered randomly.
Actions: random movement
Result: Found water by accident, drank.

Turn 6-10: Agent stumbled into resource-rich area.
Actions: collect_wood (panic), collect_stone (panic)
Result: Inventory: {wood: 2, stone: 1}

Turn 11-15: More zombies appeared.
Actions: hide, wait, sneak_away
Result: Avoided combat, still alive.

Final state: Health: 6/9, Food: 5/9, Drink: 7/9
Achievements: ["Collect Wood", "Collect Stone"]
Survival time: 25 steps
""",
        "human_score": 0.4,  # Medium-low - survived by luck
    },
]


# ─── Score Function ───────────────────────────────────────────────────────────


def extract_predicted_score(response: str) -> float | None:
    """Extract a score from the verifier's response."""
    # Look for patterns like "Score: 0.85" or "0.85" or "85%"
    patterns = [
        r"[Ss]core[:\s]+([0-9]*\.?[0-9]+)",
        r"([0-9]*\.?[0-9]+)\s*(?:/\s*1(?:\.0)?|out of 1)",
        r"([0-9]+(?:\.[0-9]+)?)\s*%",
        r"\b(0\.[0-9]+|1\.0|1)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, response)
        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            if value > 1.0:
                value = value / 100.0
            return min(1.0, max(0.0, value))

    return None


def score_fn(sample: Sample) -> Score:
    """Score the verifier based on how close its prediction is to human score.

    Uses mean squared error - lower is better, so we convert to (1 - MSE).
    """
    if not sample.trajectory or not sample.trajectory.messages:
        return Score(metrics=(Metric("accuracy", 0.0, weight=1.0),))

    # Get human score (ground truth)
    human_score = sample.input.get("human_score", 0.5)

    # Get the verifier's response
    response_text = ""
    for msg in reversed(sample.trajectory.messages):
        if msg.role == "assistant":
            if isinstance(msg.content, str):
                response_text = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if hasattr(block, "text"):
                        response_text += block.text
            break

    if not response_text:
        return Score(metrics=(Metric("accuracy", 0.0, weight=1.0),))

    # Extract predicted score
    predicted = extract_predicted_score(response_text)
    if predicted is None:
        return Score(
            metrics=(Metric("accuracy", 0.0, weight=1.0, metadata={"error": "no_score_extracted"}),)
        )

    # Compute accuracy as 1 - |predicted - human|
    # This gives higher reward for closer predictions
    error = abs(predicted - human_score)
    accuracy = 1.0 - error

    return Score(
        metrics=(
            Metric(
                "accuracy",
                accuracy,
                weight=1.0,
                metadata={"predicted": predicted, "human": human_score, "error": error},
            ),
        )
    )


# ─── Main ─────────────────────────────────────────────────────────────────────


async def main() -> None:
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY not set")
        return

    # ─── Initial Verifier Template ────────────────────────────────────────
    initial_template = PromptTemplate(
        system="""You are a game trajectory evaluator for Crafter, a survival/crafting game.

Your task is to score agent trajectories on a scale of 0.0 to 1.0 based on:

1. **Achievement Progression** (35%): Did the agent unlock meaningful achievements?
   - Late-game achievements (iron tools, furnace) are worth more
   - Early achievements (collect wood) are baseline expectations

2. **Resource Management** (20%): Did the agent gather and use resources efficiently?
   - Good inventory management
   - Crafted useful tools

3. **Survival** (20%): Did the agent maintain health, food, and drink?
   - Staying alive is important
   - But just surviving without progress is not enough

4. **Decision Quality** (15%): Did the agent make smart decisions?
   - Avoiding unnecessary combat
   - Prioritizing important tasks

5. **Time Efficiency** (10%): Did the agent achieve goals quickly?
   - More achievements in fewer steps is better

Respond with a single score between 0.0 and 1.0, formatted as:
Score: X.XX""",
        user_template="""Evaluate this Crafter agent trajectory:

{trajectory}

Provide your score (0.0 to 1.0):""",
    )

    # ─── GEPA Configuration ───────────────────────────────────────────────
    config = EvolutionaryConfig(
        population_size=6,
        generations=3,
        mutation_rate=0.5,
        crossover_rate=0.3,
        elite_size=2,
        train_seeds=tuple(range(6)),  # First 6 trajectories for training
        val_seeds=tuple(range(6, 8)),  # Last 2 for validation
        max_concurrent=3,
    )

    # ─── Endpoints ────────────────────────────────────────────────────────
    task_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=256,
        temperature=0.0,  # Deterministic for consistent scoring
    )

    mutation_endpoint = Endpoint(
        provider="openai",
        model="gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY", ""),
        max_tokens=2048,
        temperature=0.7,
    )

    # ─── Run GEPA ─────────────────────────────────────────────────────────
    logger.info("Starting GEPA optimization for Crafter Verifier...")
    logger.info(f"Dataset size: {len(DATASET)} trajectories")
    logger.info("Optimizing verifier prompt to predict human quality scores...")

    def on_generation(gen: int, population: list) -> None:
        """Log progress after each generation."""
        scores = [t.score for t in population if t.score is not None]
        if scores:
            logger.info(
                f"  Generation {gen + 1}: "
                f"best={max(scores):.3f}, mean={sum(scores) / len(scores):.3f}"
            )

    result = await run_evolutionary_gepa(
        initial_template=initial_template,
        config=config,
        dataset=DATASET,
        endpoint=task_endpoint,
        mutation_endpoint=mutation_endpoint,
        score_fn=score_fn,
        on_generation=on_generation,
    )

    # ─── Results ──────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("CRAFTER VERIFIER OPTIMIZATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total evaluations: {result.total_evaluations}")
    logger.info(f"Best validation accuracy: {result.best_template.score:.3f}")
    logger.info("")
    logger.info("This means the optimized verifier's predictions are on average")
    logger.info(f"within {(1 - (result.best_template.score or 0)):.1%} of human scores.")
    logger.info("")
    logger.info("Optimized verifier prompt (truncated):")
    logger.info("-" * 50)
    prompt = result.best_template.system
    if len(prompt) > 600:
        prompt = prompt[:600] + "\n... [truncated]"
    logger.info(prompt)
    logger.info("-" * 50)
    logger.info("")
    logger.info("Use this optimized verifier as a reward model for RL training!")


if __name__ == "__main__":
    trio.run(main)
