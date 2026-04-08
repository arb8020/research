"""Test GRPO training with FibonacciEnvironment.

This verifies the environment integrates with the existing GRPO infrastructure.
"""

from ...core import Metric, Score
from ...training.grpo import GRPOConfig, grpo_train
from ...training.scoring import FunctionScorer
from ...training.types import RowAttempt
from .code_challenge import CodeChallengeEnvironment, FibonacciEnvironment


# Factory function that returns a configured environment
# (GRPO calls environment_cls() with no args)
def FibonacciEnvFactory() -> CodeChallengeEnvironment:
    """Factory for GRPO - returns configured FibonacciEnvironment."""
    return FibonacciEnvironment(timeout=2.0)


def fibonacci_score_fn(sample: RowAttempt, _context: object) -> Score:
    """Score function that uses environment's grading.

    The environment grades in on_assistant_message, but we need to
    re-grade here since GRPO calls score_fn separately.
    """
    import trio

    from .code_challenge import FIBONACCI_TEST_CASES, extract_code, run_code_with_tests

    response = sample.response
    code = extract_code(response)

    if code is None:
        return Score(
            metrics=(
                Metric("correctness", 0.0, weight=1.0),
                Metric("runtime_ms", 0.0, weight=0),
            )
        )

    # Run grading synchronously (score_fn is sync in current GRPO)
    result = trio.run(
        run_code_with_tests,
        code,
        "fib",
        FIBONACCI_TEST_CASES,
        2.0,  # timeout
    )

    return Score(
        metrics=(
            Metric("correctness", result.correctness, weight=1.0),
            Metric("runtime_ms", result.runtime_ms, weight=0),
        )
    )


def main() -> None:
    """Run a short GRPO training to verify integration."""
    # Minimal config for testing
    config = GRPOConfig(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",  # Small model for fast testing
        num_steps=3,
        batch_size=2,
        n_samples_per_prompt=2,
        max_tokens=512,
        temperature=0.8,
        lr=1e-6,
        log_every=1,
        checkpoint_every=100,  # Don't checkpoint during short test
        sync_weights_every=1,
        output_dir="results/ttt_test",
        experiment_name="fibonacci_grpo",
    )

    # Create prompts (just the fibonacci prompt repeated)
    env = FibonacciEnvironment()
    prompts = [
        {"messages": [{"role": "user", "content": env.prompt}]}
        for _ in range(4)  # 4 prompts for 2 steps of batch_size=2
    ]

    print("=" * 60)
    print("Testing GRPO with FibonacciEnvironment")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Steps: {config.num_steps}")
    print(f"Batch: {config.batch_size} x {config.n_samples_per_prompt}")
    print()

    results = grpo_train(
        config=config,
        prompts=prompts,
        scorer=FunctionScorer(fibonacci_score_fn),
        environment_cls=FibonacciEnvFactory,
    )

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for step_metrics in results.get("metrics_history", []):
        step = step_metrics["step"]
        reward = step_metrics["mean_reward"]
        loss = step_metrics.get("pg_loss", 0.0)
        print(f"Step {step}: reward={reward:.3f}, loss={loss:.4f}")


if __name__ == "__main__":
    main()
