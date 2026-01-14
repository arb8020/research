"""Qwen3-0.6B on RTX 3090

Small model for testing. Cheap community GPU.
"""

from recipes.schema import ServingRecipe, ModelConfig, EngineConfig, TargetConfig

recipe = ServingRecipe(
    name="qwen3-0.6b-3090",
    description="Qwen3-0.6B on single RTX 3090 - cheap testing",
    model=ModelConfig(
        model="Qwen/Qwen3-0.6B",
    ),
    engine=EngineConfig(
        engine="sglang",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
    ),
    target=TargetConfig(
        gpu_type="RTX 3090",  # Match exact name from runpod
        gpu_count=1,
        provider="runpod",
    ),
)
