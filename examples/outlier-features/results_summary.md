# Outlier Features Analysis - Results Summary

Testing whether Dettmers et al. (2022) phase transition claim extends to MoE architectures. Dettmers found dense transformers show a phase shift at ~6.7B where outliers become systematic (100% layer coordination), claiming: *"transformers after the phase shift at 6.7B parameters behave very differently... one should not try to generalize from <6.7B transformers to beyond 6.7B parameters."*

**Research Question:** Do MoE models show this same phase transition and behavioral shift?

## Methodology

- Detected outliers with magnitude ≥6.0 affecting ≥25% of layers AND ≥6% of sequence positions
- Analyzed activation tensors from ln_attn and ln_mlp across all layers
- Dataset: 16 sequences × 2048 tokens from FineWeb-Edu

### Model Precision

**Important:** Models were analyzed at different native precisions, which may affect outlier detection:

| Model | Native Precision | Analysis Precision | Notes |
|-------|-----------------|-------------------|-------|
| OLMoE-1B-7B | float32 | bfloat16 | Downcast from native fp32 |
| GPT-OSS-20B | MXFP4 (MoE weights) | MXFP4 | 4-bit quantized MoE weights (BF16+U8 storage) |
| Qwen3-30B | bfloat16 | bfloat16 | Native precision |
| Mixtral-8x7B | bfloat16 | bfloat16 | Native precision |
| Qwen3-Next-80B | bfloat16 | bfloat16 | Native precision |
| GLM-4.5-Air | bfloat16 | bfloat16 | Native precision |
| GPT-OSS-120B | MXFP4 (MoE weights) | MXFP4 | 4-bit quantized MoE weights (BF16+U8 storage) |

**MXFP4 Details:** GPT-OSS models use post-training MXFP4 quantization (4-bit floating point) for MoE weights, enabling the 120B model to run on a single 80GB GPU. The models were likely analyzed at this quantized precision (attempting to load with standard bfloat16 would exceed memory constraints).

## Results

| Model | Total Params | Active Params | Experts | Top-K | Routing | Outliers | Mean L% | Mean S% |
|-------|--------------|---------------|---------|-------|---------|----------|---------|---------|
| OLMoE-1B-7B | 7B | 1.3B | 64 | 8 | Token-based (dropless) | 49 | 29.5% | 13.6% |
| GPT-OSS-20B | 21B | 3.6B | 128* | 4* | Standard top-k | 1,465 | 38.1% | 45.4% |
| Qwen3-30B | 30.5B | 3.3B | 128 | 8 | Standard top-k | 110 | 35.5% | 45.1% |
| Mixtral-8x7B | 47B | 12.9B | 8 | 2 | Standard top-k | 4,635 | 50.2% | 37.4% |
| Qwen3-Next-80B | 80B | 3.0B | 512+1 shared | 10 | Standard top-k | 504 | 57.5% | 35.1% |
| GLM-4.5-Air | 106B | 12.0B | 128+1 shared | 8 | Sigmoid gating (loss-free balance) | 459 | 63.3% | 45.7% |
| GPT-OSS-120B | 117B | 5.1B | 128 | 4 | Softmax-weighted top-k | 1,695 | 33.1% | 50.0% |

**Metric Definitions:**
- **L%** = avg % of layers each outlier affects
- **S%** = avg % of sequence positions each outlier affects
- **Experts** = number of expert modules per MoE layer
- **Top-K** = number of experts activated per token

**Notes:**
- *GPT-OSS-20B: Architecture details inferred from GPT-OSS-120B specifications (same model family)
- GPT-OSS models use softmax-weighted routing (outputs scaled by softmax of router scores over selected experts)
- Both GPT-OSS models use alternating dense and locally banded sparse attention patterns
