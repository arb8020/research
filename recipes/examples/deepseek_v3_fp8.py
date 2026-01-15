"""DeepSeek-V3 FP8 on 8x H100

MoE model with FP8 quantization for memory efficiency.
Requires 8x H100 for the full model.
"""

from recipes.schema import ServingRecipe, ModelConfig, EngineConfig, TargetConfig, EnvConfig

recipe = ServingRecipe(
    name="deepseek-v3-fp8-h100x8",
    description="DeepSeek-V3 FP8 quantized on 8x H100",
    model=ModelConfig(
        model="deepseek-ai/DeepSeek-V3-0324",
        quantization="fp8",
        max_model_len=32768,
        trust_remote_code=True,
    ),
    engine=EngineConfig(
        engine="sglang",
        tensor_parallel_size=8,
        gpu_memory_utilization=0.8,  # Lower for stability
        enable_prefix_caching=False,  # Disable for MoE
        attention_backend="triton",
        git_repo="https://github.com/sgl-project/sglang.git",
        git_ref="main",  # Need latest for DeepSeek-V3 support
    ),
    target=TargetConfig(
        gpu_type="H100",
        gpu_count=8,
    ),
    env=EnvConfig(
        sglang_allow_longer_context=True,
    ),
)
