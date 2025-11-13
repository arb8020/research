"""Config for evaluating model on ACEBench with Prime verifiers.

Usage:
    cd ~/research/dev/integration-evaluation
    python local.py configs/prime_acebench.py

This demonstrates Phase 1: Evaluation with Prime Intellect verifiers.
ACEBench tests multi-step API calling on simulated phone/travel platforms.
"""

from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List

from rollouts.dtypes import Endpoint, EvalConfig, Message


def prepare_messages(sample_data: Dict[str, Any]) -> List[Message]:
    """Prepare messages for ACEBench environment.

    ACEBench uses 'prompt' field as a list of message dicts.
    """
    prompt = sample_data.get("prompt", [])
    return [Message(role=msg["role"], content=msg["content"]) for msg in prompt]


@dataclass(frozen=True)
class IntegrationEvalConfig:
    """Configuration for integration evaluation.

    Follows the pattern from wafer_stuff/clicker/config.py but adapted
    for Prime Intellect integration testing.
    """
    # Model configuration
    # Option 1: Gemini (matches clicker pattern)
    model_name: str = "gemini-2.0-flash-exp"
    provider: str = "openai"  # Gemini through OpenAI-compatible API
    api_base: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    api_key_env_var: str = "GEMINI_API_KEY"
    temperature: float = 0.0
    max_tokens: int = 2048

    # Option 2: OpenAI
    # model_name: str = "gpt-4o-mini"
    # provider: str = "openai"
    # api_base: str = "https://api.openai.com/v1"
    # api_key_env_var: str = "OPENAI_API_KEY"

    # Option 3: Anthropic
    # model_name: str = "claude-3-5-sonnet-20241022"
    # provider: str = "anthropic"
    # api_base: str = "https://api.anthropic.com"
    # api_key_env_var: str = "ANTHROPIC_API_KEY"

    # Prime environment configuration
    env_name: str = "acebench-agent-multistep"  # Real Prime Hub environment
    num_samples: int = 20  # Number of samples to evaluate (ACEBench has 20 tasks)

    # Evaluation configuration
    eval_name: str = "prime_acebench_eval"
    max_turns: int = 40  # ACEBench is multi-turn (up to 40 turns)
    # WARNING: Gemini has 10 requests/min quota - use max_concurrent=1 or 2
    # Higher values will hit rate limits and fail
    max_concurrent: int = 1  # Parallel evaluation (Gemini quota: 10/min)

    # Output configuration
    output_dir: Path = Path("results/integration-evaluation")
    verbose: bool = True
    show_progress: bool = False  # Enable nested progress bars (outer: samples, inner: turns)

    def to_endpoint(self) -> Endpoint:
        """Convert to rollouts Endpoint."""
        import os

        # Get API key from environment
        api_key = os.getenv(self.api_key_env_var, "")
        if not api_key and self.provider != "sglang":
            print(f"⚠️  Warning: {self.api_key_env_var} not set in environment")

        return Endpoint(
            provider=self.provider,
            model=self.model_name,
            api_base=self.api_base,
            api_key=api_key,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

    def to_eval_config(self, reward_fn) -> EvalConfig:
        """Convert to rollouts EvalConfig.

        Args:
            reward_fn: RewardFunction created from Prime environment

        Returns:
            EvalConfig ready for evaluation
        """
        return EvalConfig(
            reward_fn=reward_fn,
            max_turns=self.max_turns,
            max_concurrent=self.max_concurrent,
            max_samples=self.num_samples,
            output_dir=self.output_dir,
            eval_name=self.eval_name,
            verbose=self.verbose,
            show_progress=self.show_progress,
        )


# Export config instance
config = IntegrationEvalConfig()


# Environment adapter for ACEBench - wraps Prime MultiTurnEnv to work with rollouts
class ACEBenchEnvironment:
    """Minimal adapter for ACEBench environment.

    ACEBench is a Prime MultiTurnEnv that doesn't actually need tool execution
    in our framework - Prime handles that internally via their own rollout.

    For now, just provide empty tools since we're using Prime's reward function only.
    """

    def get_tools(self):
        """Return empty tools - ACEBench doesn't expose tools to rollouts."""
        return []

    async def serialize(self):
        """Serialize environment state."""
        return {}

    @staticmethod
    async def deserialize(data):
        """Deserialize environment state."""
        return ACEBenchEnvironment()

    def requires_confirmation(self, tool_call):
        """No confirmation needed."""
        return False


def create_environment():
    """Factory function to create fresh environment instances."""
    return ACEBenchEnvironment()
