# Corpus Proximity - Quick Start

## What This Does

Annotates LLM outputs with their nearest clusters in the training corpus, similar to how chess engines annotate positions.

**Example:**
```
User: "What is the derivative of x^2?"
Model: "The derivative of x^2 is 2x."

📍 Source: Calculus Education (d=0.08, midtrain)
🎯 HIGH confidence - closely matches training data
```

---

## Installation

```bash
# Install dependencies
uv sync --extra example-corpus-proximity
python -m spacy download en_core_web_sm
```

---

## Three-Step Workflow

### 1. Build Corpus Index (One-time, GPU-accelerated)

```bash
# Deploy to GPU, build index, sync results back
python deploy.py --config configs/clustering_01_tiny_gpu.py

# Results appear in: remote_results/clustering_<timestamp>/
```

**What happens:**
- Provisions GPU on RunPod/Vast
- Embeds your training corpus
- Clusters recursively (UMAP + HDBSCAN)
- Names clusters with GPT-4
- Syncs results back automatically
- Cleans up GPU

---

### 2. Generate Model Outputs (Any inference engine)

```python
# Example with vLLM
from vllm import LLM
import json

llm = LLM(model="meta-llama/Llama-3-8B")
prompts = ["What is 2+2?", "Explain calculus"]
outputs = llm.generate(prompts)

# Save to JSONL
with open("outputs.jsonl", "w") as f:
    for prompt, output in zip(prompts, outputs):
        f.write(json.dumps({
            "prompt": prompt,
            "output": output.outputs[0].text
        }) + "\n")
```

---

### 3. Annotate Outputs (Local, instant)

```bash
corpus-proximity annotate \
    --corpus-index remote_results/clustering_<timestamp>/ \
    --input outputs.jsonl \
    --output annotated.jsonl

corpus-proximity show --annotated-file annotated.jsonl --index 0
```

**Output:**
```
Model Output:
"The derivative of x^2 is 2x."

📍 Source Analysis:
  ├─ "The derivative of x^2 is 2x."
     [1] Calculus Education (d=0.08, midtrain)
     [2] Mathematics Textbooks (d=0.12, pretrain)

🎯 Interpretation:
  Average distance: 0.10 (HIGH confidence)
  Model output closely matches training corpus.
```

---

## Key Files

- `IMPLEMENTATION_PLAN.md` - Full technical spec (for implementers)
- `ARCHITECTURE.md` - System design & data structures
- `API.md` - Function signatures & CLI reference

---

## Requirements

**CRITICAL:** You must provide the **exact** training corpus used for your model.

❌ Approximate corpus = meaningless results
✅ Exact training JSONL files = accurate annotations
