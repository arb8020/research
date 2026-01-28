"""Qwen3-8B on A100

Production-ready config for 8B model.
"""

from recipes.schema import DepsConfig, EngineConfig, ModelConfig, ServingRecipe, TargetConfig

recipe = ServingRecipe(
    name="qwen3-8b-a100",
    description="Qwen3-8B on single A100 - production config",
    model=ModelConfig(
        model="Qwen/Qwen2.5-7B-Instruct",  # Using 7B as proxy
        max_model_len=32768,
    ),
    engine=EngineConfig(
        engine="sglang",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,  # RadixAttention for better throughput
    ),
    target=TargetConfig(
        gpu_type="A100",
        gpu_count=1,
    ),
    deps=DepsConfig(
        pip_packages=("torch>=2.4", "sglang[all]"),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)
