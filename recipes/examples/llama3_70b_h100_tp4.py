"""Llama3-70B on 4x H100 with speculative decoding

High-performance config for large model serving.
Uses NEXTN speculative decoding for lower latency.
"""

from recipes.schema import (
    ServingRecipe,
    ModelConfig,
    EngineConfig,
    TargetConfig,
    SpeculativeConfig,
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
)
