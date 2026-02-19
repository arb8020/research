# HELM Explained

## What is HELM?

**HELM** = **H**olistic **E**valuation of **L**anguage **M**odels

Created by **Stanford's Center for Research on Foundation Models (CRFM)**

Website: https://crfm.stanford.edu/helm/

---

## The Core Idea

### Traditional Evaluation
```
Run a few benchmarks → Get accuracy scores → Done
```

Problems:
- **Narrow**: Only tests specific capabilities
- **Incomplete**: Misses bias, toxicity, efficiency
- **Inconsistent**: Different setups across papers
- **Opaque**: Hard to understand what models actually do

### HELM's Approach
```
Holistic = Many scenarios × Many metrics × Standardized
```

**3 Pillars**:
1. **Broad Coverage** - Test many capabilities, not just accuracy
2. **Multi-Metric** - Measure beyond accuracy (bias, efficiency, robustness)
3. **Standardization** - Same setup for fair comparison

---

## What HELM Measures

### 1. Core Capabilities (16 scenarios)

| Scenario | What it Tests | Example |
|----------|---------------|---------|
| **Question Answering** | Knowledge retrieval | MMLU, TruthfulQA |
| **Information Retrieval** | Search, ranking | MS MARCO |
| **Summarization** | Condensation | CNN/DailyMail, XSum |
| **Sentiment Analysis** | Opinion detection | IMDB, Yelp |
| **Toxicity Detection** | Harmful content | CivilComments |
| **Miscellaneous** | Varied tasks | RAFT, HellaSwag |

### 2. Targeted Evaluations

| Type | Focus | Benchmarks |
|------|-------|------------|
| **Language** | English, Chinese, Spanish, etc. | Multi-lingual tasks |
| **Knowledge** | World knowledge, reasoning | WikiFact, BBQ |
| **Reasoning** | Logic, math, coding | GSM8K, HumanEval |
| **Memorization** | Training data leakage | Copyright, NQ |
| **Robustness** | Adversarial inputs | Contrast sets |
| **Fairness** | Demographic bias | BBQ, BOLD |

### 3. Beyond Accuracy

| Metric | What it Measures |
|--------|------------------|
| **Calibration** | Does model know when it's wrong? |
| **Robustness** | Performance under perturbation |
| **Fairness** | Disparate performance across groups |
| **Bias** | Stereotypes in outputs |
| **Toxicity** | Harmful content generation |
| **Efficiency** | Inference cost, latency |

---

## Why HELM is Complex

### 1. Architecture

```
HELM Framework
├── Scenarios (what to test)
│   ├── Datasets (MMLU, TruthfulQA, etc.)
│   ├── Task definitions
│   └── Data preprocessing
├── Adaptations (how to prompt)
│   ├── Multiple choice
│   ├── Generation
│   └── Ranking
├── Metrics (what to measure)
│   ├── Accuracy
│   ├── Calibration
│   ├── Robustness
│   ├── Fairness
│   └── Efficiency
├── Models (what to evaluate)
│   ├── OpenAI API
│   ├── Anthropic API
│   ├── Local models (vLLM, etc.)
│   └── Custom deployments
└── Runs (execution)
    ├── Parallel execution
    ├── Caching
    └── Result aggregation
```

### 2. Configuration Files Required

For **WildBench specifically**:

```yaml
# model_deployments.yaml
model_deployments:
  - name: my-model
    model_name: my-model
    tokenizer_name: my-tokenizer
    max_sequence_length: 32768
    client_spec:
      class_name: helm.clients.openai_client.OpenAIClient
      args:
        base_url: http://localhost:30000/v1
```

```yaml
# model_metadata.yaml
models:
  - name: my-model
    display_name: "My Model"
    description: "Fine-tuned MoE model"
    creator_organization: "My Org"
    access: "open"
    release_date: "2024-01-01"
```

```yaml
# tokenizer_configs.yaml
tokenizer_configs:
  - name: my-tokenizer
    tokenizer_spec:
      class_name: helm.tokenizers.huggingface_tokenizer.HuggingFaceTokenizer
      args:
        pretrained_model_name_or_path: "path/to/tokenizer"
```

```conf
# credentials.conf
openai_api_key=not_needed_for_local
```

### 3. Directory Structure

```
config/
└── wildbench_prod_env_30000/
    ├── model_deployments.yaml
    ├── model_metadata.yaml
    ├── tokenizer_configs.yaml
    ├── credentials.conf
    └── run_specs.conf
```

### 4. Execution Flow

```bash
# 1. Start model server
vllm serve my-model --port 30000

# 2. Run HELM
helm-run \
  --run-entries "wildbench:subset=v2,model=my-model" \
  --suite my-suite \
  --local-path config/wildbench_prod_env_30000

# 3. Summarize results
helm-summarize --suite my-suite

# 4. View results
helm-server --suite my-suite
# Open http://localhost:8000
```

---

## HELM vs Simple Evaluation

### Simple Approach (What We Use)
```python
# lm-eval-harness
import lm_eval
results = lm_eval.simple_evaluate(
    model="hf",
    model_args={"pretrained": "my-model"},
    tasks=["gsm8k", "mmlu"],
)
# Done in 5 lines
```

### HELM Approach
```python
# HELM
from helm.benchmark.run import helm_run, create_helm_run_args

# 1. Create config files (4+ yaml files)
# 2. Define run entries
run_entries = ["wildbench:subset=v2,model=my-model"]

# 3. Create args
helm_args = create_helm_run_args(
    suite="my-suite",
    run_entries=run_entries,
    # ... many more args
)

# 4. Run
helm_run(helm_args)

# 5. Summarize separately
# 6. Start web server to view
# Much more complex!
```

---

## When to Use HELM

### ✅ Use HELM When
- Academic research requiring comprehensive evaluation
- Comparing many models on many dimensions
- Publishing results (standardized = credible)
- Need specific scenarios (medical, legal, etc.)
- Analyzing bias, fairness, toxicity

### ❌ Don't Use HELM When
- Quick iteration during development
- Specific capability testing (just use lm-eval)
- Resource constrained (HELM is heavy)
- Simple benchmarking (overkill)
- Production monitoring (too slow)

---

## HELM in Cerebras REAP

### Why Cerebras Uses HELM
```
Cerebras REAP = Research project
├─ Academic paper
├─ Comprehensive evaluation needed
├─ Multiple models to compare
└─ Bias/fairness analysis important
```

### What They Use HELM For
- **WildBench**: Real-world task evaluation
- **Comprehensive metrics**: Beyond accuracy
- **Standardization**: Fair comparison with other papers

---

## HELM vs Our Implementation

| Aspect | HELM | Our Implementation |
|--------|------|-------------------|
| **Scope** | Holistic (16+ scenarios) | Focused (9 benchmarks) |
| **Setup** | Complex (config files) | Simple (pip install) |
| **Metrics** | 10+ metrics per task | 1-2 metrics per task |
| **Runtime** | Hours to days | Minutes to hours |
| **Use case** | Research, publication | Development, iteration |
| **WildBench** | ✅ Native | ❌ Stub only |

### Our Choice
```
We prioritize:
✅ Simplicity - Easy to install and run
✅ Speed - Quick iteration during development
✅ Core benchmarks - What matters for pruning
❌ Not: Comprehensive research evaluation
```

---

## Should You Add HELM?

### Probably YES If
- Publishing a research paper
- Need WildBench specifically
- Comparing with Cerebras results directly
- Academic benchmark requirements

### Probably NO If
- Engineering/development workflow
- Just need to know if pruning works
- Resource constrained
- Time constrained

### Middle Ground
Use **lm-eval** (what we have) for:
- 90% of evaluation needs
- Quick iteration
- Development

Use **HELM** only for:
- Final publication results
- Specific WildBench requirements
- Comprehensive comparison

---

## Quick Reference

| Framework | Use For | Complexity | Install |
|-----------|---------|------------|---------|
| **lm-eval** | Standard benchmarks | Low | `pip install lm-eval` |
| **evalplus** | Code evaluation | Low | `pip install evalplus` |
| **livecodebench** | Competitive programming | Low | `pip install livecodebench` |
| **HELM** | Comprehensive research | **High** | `pip install crfm-helm` + config |

---

## Bottom Line

**HELM** is like a **full medical checkup**:
- Comprehensive
- Time-consuming
- Requires specialists
- Necessary for serious diagnosis

**Our evaluation** is like a **vitals check**:
- Quick
- Covers essentials
- Good for regular monitoring
- Sufficient for most purposes

For pruning MoE models, **vitals check is usually sufficient**.
