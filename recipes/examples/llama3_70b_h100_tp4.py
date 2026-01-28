"""Llama3-70B on 4x H100 with speculative decoding

High-performance config for large model serving.
Uses NEXTN speculative decoding for lower latency.
"""

from recipes.schema import (
    DepsConfig,
    EngineConfig,
    ModelConfig,
    ServingRecipe,
    SpeculativeConfig,
    TargetConfig,
)

recipe = ServingRecipe(
    name="llama3-70b-h100-tp4-spec",
    description="Llama3-70B on 4x H100 with NEXTN speculative decoding",
    model=ModelConfig(
        model="meta-llama/Llama-3.1-70B-Instruct",
        max_model_len=8192,  # Reduced context for memory
    ),
    engine=EngineConfig(
        engine="sglang",
        tensor_parallel_size=4,
        gpu_memory_utilization=0.85,
        attention_backend="triton",  # For H100
        speculative=SpeculativeConfig(
            method="nextn",
            num_steps=5,
            num_draft_tokens=6,
            eagle_topk=1,
        ),
    ),
    target=TargetConfig(
        gpu_type="H100",
        gpu_count=4,
    ),
    deps=DepsConfig(
        pip_packages=("torch>=2.4", "sglang[all]"),
        pip_index_url="https://download.pytorch.org/whl/cu124",
        pip_extra_index_url="https://pypi.org/simple",
    ),
)
