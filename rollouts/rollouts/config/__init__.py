"""Configuration protocols and base implementations for rollouts.

Tiger Style: Protocols define interfaces without coupling.
Base configs are reference implementations, not requirements.
Projects can use, customize, or ignore them entirely.

Usage:
    # Use base config as-is
    from rollouts.config.base import BaseModelConfig
    config = BaseModelConfig()

    # Customize inline
    config = BaseModelConfig(model_name="gpt-4", temperature=0.5)

    # Compose into your own config
    @dataclass(frozen=True)
    class MyConfig:
        model: BaseModelConfig = field(default_factory=BaseModelConfig)
        # ... your custom fields

    # Or just satisfy the protocol (no imports needed)
    @dataclass(frozen=True)
    class MyModelConfig:
        def to_endpoint(self) -> Endpoint: ...
"""

from rollouts.config.protocols import (
    HasModelConfig,
    HasEnvironmentConfig,
    HasEvaluationConfig,
    HasOutputConfig,
)

from rollouts.config.base import (
    BaseModelConfig,
    BaseEnvironmentConfig,
    BaseEvaluationConfig,
    BaseOutputConfig,
)

from rollouts.config.loader import (
    load_config_from_file,
)

__all__ = [
    # Protocols
    "HasModelConfig",
    "HasEnvironmentConfig",
    "HasEvaluationConfig",
    "HasOutputConfig",
    # Base configs
    "BaseModelConfig",
    "BaseEnvironmentConfig",
    "BaseEvaluationConfig",
    "BaseOutputConfig",
    # Utilities
    "load_config_from_file",
]
