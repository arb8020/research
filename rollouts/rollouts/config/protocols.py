"""Configuration protocols for rollouts.

Tiger Style: Protocols define interfaces without forcing implementation.
No coupling, no inheritance, just duck typing with type checking.

These protocols define what configs SHOULD provide, but don't enforce how.
Projects can implement their own configs that satisfy these protocols.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

# Import types from rollouts.dtypes
# These are already defined and used across projects
from rollouts.dtypes import Endpoint, EvalConfig, Message


@runtime_checkable
class HasModelConfig(Protocol):
    """Protocol for configs that can create LLM endpoints.

    Any config that implements to_endpoint() satisfies this protocol.
    No inheritance needed, just implement the method.

    Example:
        @dataclass(frozen=True)
        class MyModelConfig:
            model_name: str = "gpt-4"
            provider: str = "openai"

            def to_endpoint(self) -> Endpoint:
                return Endpoint(
                    provider=self.provider,
                    model=self.model_name,
                )

        # Automatically satisfies HasModelConfig protocol
        assert isinstance(MyModelConfig(), HasModelConfig)
    """

    def to_endpoint(self) -> Endpoint:
        """Convert to rollouts Endpoint for LLM API calls."""
        ...


@runtime_checkable
class HasEnvironmentConfig(Protocol):
    """Protocol for configs that define evaluation environments.

    Environments need:
    1. A name/identifier
    2. A way to prepare initial messages from dataset samples

    Example:
        @dataclass(frozen=True)
        class KernelEnvConfig:
            env_name: str = "kernel-agent"

            def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
                return [
                    Message(role="system", content="You are a kernel developer"),
                    Message(role="user", content=sample_data["prompt"]),
                ]
    """

    env_name: str

    def prepare_messages(self, sample_data: dict[str, Any]) -> list[Message]:
        """Prepare initial messages from a dataset sample.

        Args:
            sample_data: A sample from your dataset (dict with any structure)

        Returns:
            List of messages to start the conversation
        """
        ...


@runtime_checkable
class HasEvaluationConfig(Protocol):
    """Protocol for configs that can create EvalConfig.

    Evaluation configs manage:
    - Number of turns/samples
    - Concurrency settings
    - Score computation

    Example:
        @dataclass(frozen=True)
        class MyEvalConfig:
            num_samples: int = 100

            def to_eval_config(self, score_fn: Callable) -> EvalConfig:
                return EvalConfig(
                    score_fn=score_fn,
                    max_samples=self.num_samples,
                )
    """

    def to_eval_config(self, score_fn: Callable) -> EvalConfig:
        """Convert to rollouts EvalConfig.

        Args:
            score_fn: Score function (Trajectory, Sample) -> Score

        Returns:
            EvalConfig ready for evaluation
        """
        ...


@runtime_checkable
class HasOutputConfig(Protocol):
    """Protocol for configs that manage output/logging.

    Output configs handle:
    - Where to save results
    - Experiment naming
    - Verbosity/logging settings

    Example:
        @dataclass(frozen=True)
        class MyOutputConfig:
            output_dir: Path = Path("results")
            experiment_name: str = "my_exp"
            verbose: bool = True
    """

    output_dir: Path
    experiment_name: str
