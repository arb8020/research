"""Qwen3-0.6B on RTX 4090

Small model for testing. Fits easily on consumer GPU.
"""

from recipes.schema import ServingRecipe, ModelConfig, EngineConfig, TargetConfig

recipe = ServingRecipe(
    name="qwen3-0.6b-4090",
    description="Qwen3-0.6B on single RTX 4090 - good for testing",
    model=ModelConfig(
        model="Qwen/Qwen3-0.6B",
    ),
    engine=EngineConfig(
        engine="sglang",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    ),
    target=TargetConfig(
        gpu_type="4090",  # Use partial match - broker uses contains()
        gpu_count=1,
        provider="runpod",
    ),
)
