"""Base configuration implementations for rollouts.

Tiger Style: These are reference implementations, not requirements.
Projects can:
1. Use these directly if they fit
2. Customize inline when constructing
3. Compose into their own configs
4. Copy-paste and diverge
5. Ignore entirely and implement protocols themselves

All configs are frozen dataclasses for immutability.
All defaults are explicit (no magic).
"""

import json
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rollouts.dtypes import Endpoint, EvalConfig, Message, RunConfig


def infer_provider_from_model(model_name: str) -> str:
    """Infer provider from model name.

    Tiger Style: Explicit mapping, fail explicitly if unknown.

    Args:
        model_name: Model name (e.g., "gpt-4o-mini", "claude-sonnet-4")

    Returns:
        Provider name: "openai", "anthropic", or original if can't infer

    Examples:
        >>> infer_provider_from_model("gpt-4o-mini")
        'openai'
        >>> infer_provider_from_model("claude-sonnet-4-20250514")
        'anthropic'
    """
    model_lower = model_name.lower()

    # Anthropic models
    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"

    # OpenAI models
    if any(prefix in model_lower for prefix in ["gpt-", "o1-", "o3-"]):
        return "openai"

    # If unknown, return as-is (caller should validate)
    return model_name.split("-")[0] if "-" in model_name else "openai"


@dataclass(frozen=True)
class BaseModelConfig:
    """Reference implementation for model/endpoint configuration.

    Satisfies HasModelConfig protocol.

    Tiger Style:
    - All defaults explicit
    - Immutable (frozen=True)
    - to_endpoint() converts to rollouts.dtypes.Endpoint

    Example:
        # Use as-is with defaults
        model = BaseModelConfig()

        # Customize inline
        model = BaseModelConfig(
            model_name="gpt-4",
            provider="openai",
            temperature=0.5,
        )

        # Compose into your config
        @dataclass(frozen=True)
        class MyConfig:
            model: BaseModelConfig = field(default_factory=BaseModelConfig)
    """

    # Model identification
    model_name: str = "gpt-4o-mini"
    provider: str = "openai"

    # API connection
    api_base: str = "https://api.openai.com/v1"
    api_key_env_var: str = "OPENAI_API_KEY"

    # Generation parameters
    temperature: float = 0.7
    max_tokens: int = 4096

    # Retry/timeout
    max_retries: int = 3
    timeout: float = 120.0

    def to_endpoint(self) -> Endpoint:
        """Convert to rollouts Endpoint for LLM API calls.

        Tiger Style: Explicit error handling via return values.
        If API key is missing, we return Endpoint with empty key.
        Caller can validate and handle as needed.

        Auto-infers provider from model name if config has mismatched provider.

        Returns:
            Endpoint configuration for rollouts providers
        """
        # Auto-infer provider from model name if needed
        inferred_provider = infer_provider_from_model(self.model_name)

        # Use inferred provider if config provider doesn't match model
        # This handles cases where user specifies model name but wrong provider
        actual_provider = self.provider
        actual_api_base = self.api_base
        actual_api_key_env_var = self.api_key_env_var

        if inferred_provider != self.provider:
            print(
                f"⚠️  Provider mismatch: config says '{self.provider}' but model '{self.model_name}' suggests '{inferred_provider}'"
            )
            print(f"   Auto-correcting to use provider: {inferred_provider}")
            actual_provider = inferred_provider

            # Also update api_base and api_key_env_var to match
            if inferred_provider == "anthropic":
                actual_api_base = "https://api.anthropic.com"  # SDK adds /v1 automatically
                actual_api_key_env_var = "ANTHROPIC_API_KEY"
            elif inferred_provider == "openai":
                actual_api_base = "https://api.openai.com/v1"
                actual_api_key_env_var = "OPENAI_API_KEY"

        api_key = os.getenv(actual_api_key_env_var, "")

        # Tiger Style: No exceptions for missing env vars
        # Caller can check endpoint.api_key and handle as needed
        if not api_key and os.getenv("VERBOSE", "0") != "0":
            print(f"⚠️  Warning: {actual_api_key_env_var} not set")

        return Endpoint(
            provider=actual_provider,
            model=self.model_name,
            api_base=actual_api_base,
            api_key=api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            max_retries=self.max_retries,
            timeout=self.timeout,
        )


@dataclass(frozen=True)
class BaseEnvironmentConfig:
    """Reference implementation for environment configuration.

    Satisfies HasEnvironmentConfig protocol.

    Tiger Style:
    - Immutable
    - prepare_messages() is the interface for message preparation
    - Override in subclass or implement your own class

    Example:
        # Simple: just provide env_name, use default prepare_messages
        env = BaseEnvironmentConfig(env_name="my-task")

        # Custom: override prepare_messages in your own class
        @dataclass(frozen=True)
        class KernelEnvConfig:
            env_name: str = "kernel-agent"

            def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
                return [
                    Message(role="system", content="You are a kernel dev"),
                    Message(role="user", content=sample_data["prompt"]),
                ]
    """

    # Environment identification
    env_name: str

    # Optional: common environment settings
    max_steps: int = 100
    timeout_seconds: float = 300.0

    def prepare_messages(self, sample_data: dict[str, Any]) -> list[Message]:
        """Prepare initial messages from dataset sample.

        Default implementation: simple prompt from "prompt" field.
        Override this method for custom message preparation.

        Args:
            sample_data: Sample from your dataset (any dict structure)

        Returns:
            List of messages to initialize the conversation

        Example:
            Override in your config:

            @dataclass(frozen=True)
            class MyEnvConfig(BaseEnvironmentConfig):
                def prepare_messages(self, sample_data: Dict[str, Any]) -> List[Message]:
                    system_msg = Message(role="system", content="...")
                    user_msg = Message(role="user", content=sample_data["prompt"])
                    return [system_msg, user_msg]
        """
        # Simple default: just use prompt field
        prompt = sample_data.get("prompt", str(sample_data))
        return [Message(role="user", content=prompt)]


@dataclass(frozen=True)
class BaseEvaluationConfig:
    """Reference implementation for evaluation configuration.

    Satisfies HasEvaluationConfig protocol.

    Tiger Style:
    - Immutable
    - Composes environment via injection
    - to_eval_config() converts to rollouts.dtypes.EvalConfig
    - All defaults explicit

    Key insight: EvaluationConfig contains EnvironmentConfig!
    This allows environment-specific settings to be separate from
    generic execution settings (turns, samples, concurrency).

    Example:
        # Create environment
        env = MyEnvironmentConfig(env_name="my-task")

        # Inject into evaluation
        eval_cfg = BaseEvaluationConfig(
            environment=env,
            eval_name="my_eval",
            num_samples=100,
        )

        # Access environment through evaluation
        messages = eval_cfg.environment.prepare_messages(sample_data)

        # Convert to EvalConfig
        eval_config = eval_cfg.to_eval_config(score_fn=my_score_fn)
    """

    # Environment injection (domain-specific task setup)
    environment: BaseEnvironmentConfig

    # Evaluation identification
    eval_name: str = "evaluation"

    # Execution parameters
    num_samples: int = 100
    max_concurrent: int = 4

    # Output directory
    output_dir: Path = Path("results")

    # Display settings
    verbose: bool = True
    show_progress: bool = True

    # Streaming settings
    stream_tokens: bool = False  # Whether to stream LLM tokens to stdout
    run_config: RunConfig | None = None  # Optional custom RunConfig

    def to_eval_config(self, score_fn: Callable) -> EvalConfig:
        """Convert to rollouts EvalConfig.

        Args:
            score_fn: Score function (Trajectory, Sample) -> Score

        Returns:
            EvalConfig ready for rollouts.evaluate()
        """
        return EvalConfig(
            score_fn=score_fn,
            max_samples=self.num_samples,
            max_concurrent=self.max_concurrent,
            output_dir=self.output_dir,
            eval_name=self.eval_name,
            verbose=self.verbose,
            show_progress=self.show_progress,
            stream_tokens=self.stream_tokens,  # Pass through streaming config
            run_config=self.run_config,  # Pass through custom run config
        )


@dataclass(frozen=True)
class BaseOutputConfig:
    """Reference implementation for output/logging configuration.

    Satisfies HasOutputConfig protocol.

    Tiger Style:
    - Immutable
    - All paths explicit
    - Serialization methods for reproducibility

    Example:
        output = BaseOutputConfig(
            output_dir=Path("results"),
            experiment_name="exp_001",
        )

        # Save config alongside results
        output.save_config(my_config, output.output_dir / "config.json")
    """

    # Output paths
    output_dir: Path = Path("results")
    experiment_name: str = "experiment"

    # Logging settings
    verbose: bool = True
    log_level: str = "INFO"

    def save_config(self, config: Any, path: Path) -> None:
        """Save config as JSON for reproducibility.

        Tiger Style: Pure function, no side effects beyond file write.
        Caller controls the path.

        Args:
            config: Config object to save (must be a dataclass)
            path: Where to save the config JSON

        Example:
            output_cfg = BaseOutputConfig()
            output_cfg.save_config(my_config, Path("results/config.json"))
        """
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(asdict(config), f, indent=2, default=str)

    @staticmethod
    def load_config(config_class: type, path: Path) -> Any:
        """Load config from JSON.

        Tiger Style: Static method, explicit about what it does.

        Args:
            config_class: The config class to instantiate
            path: Path to JSON file

        Returns:
            Instance of config_class

        Example:
            config = BaseOutputConfig.load_config(
                MyConfig,
                Path("results/config.json")
            )
        """
        with open(path) as f:
            data = json.load(f)

        # Reconstruct nested dataclasses
        # Simple version - for complex nesting, projects should implement custom loaders
        return config_class(**data)
