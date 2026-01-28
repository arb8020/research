# Inoculation Prompting — Implementation Plan

## What we're building

Reimplementation of the inoculation prompting research using our rollouts SFT training, SGLang inference, and broker/bifrost cloud deployment. No code from their repo — just their datasets, experiment designs, and evaluation methodology.

## The 4 pieces to build

### Piece 1: Dataset adapter
Convert their JSONL conversation format to our `Sample` dataclass.

Their format:
```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Our format: `Sample(tokens=[...], loss_mask=[...], response_length=N)` via `tokenize_conversation()` + `compute_loss_mask()`.

Also need: `add_system_prompt(dataset, prompt)` — prepend system message to each conversation. This is the core inoculation operation.

**Files:**
- `datasets.py` — `load_inoculation_dataset(path, tokenizer) -> list[Sample]`, `add_system_prompt(data, prompt) -> data`

**Test:** Load one of their .jsonl files, tokenize, verify loss_mask is 0 for system+user and 1 for assistant tokens. Round-trip detokenize and verify content matches.

---

### Piece 2: Training harness
Thin wrapper calling `run_sft_training()` for each experimental condition.

An experiment = a list of (group_name, dataset_path, system_prompt | None) tuples × N seeds × a base model. For each combo: load data, optionally prepend system prompt, tokenize, train, save checkpoint.

**Files:**
- `train.py` — takes experiment config, runs SFT for each condition, saves checkpoints
- `config.py` — frozen dataclasses for experiment definition

**Config shape:**
```python
@dataclass(frozen=True)
class ExperimentCondition:
    group_name: str           # "finetuning", "inoculated", "control"
    dataset_path: str         # path to their .jsonl
    system_prompt: str | None # the inoculation (or None)

@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    base_model: str           # "Qwen/Qwen2.5-7B" etc.
    conditions: list[ExperimentCondition]
    seeds: list[int]
    # SFT hyperparams
    num_steps: int
    batch_size: int
    learning_rate: float
    use_lora: bool
    lora_rank: int
```

**Test:** Run SFT on a tiny model (Qwen2.5-0.5B) with 10 steps on a small slice of their data. Verify loss decreases. No GPU needed for the config/data-prep parts.

---

### Piece 3: Evaluation
Sample from trained models and score responses. Two kinds:

**A) LLM-as-judge eval (emergent misalignment, insecure code):**
- Send prompts to our trained model (via SGLang endpoint)
- Send (prompt, response) to a judge model (GPT-4o via OpenAI API — cheap, keep as-is)
- Parse judge logprobs → score

**B) Simple pattern eval (owl preferences, language compliance):**
- Send prompts, check response for keyword/pattern

**Files:**
- `evaluation.py` — `Evaluation` dataclass, `run_evaluation(endpoint, eval) -> results`
- `judge.py` — judge templates + logprob parsing (reuse their prompt templates, reimplement scoring)
- `evals/emergent_misalignment.py` — their 8 questions + alignment/coherence judges
- `evals/insecure_code.py` — code prompts + insecurity judge

Sampling uses our `rollouts.providers` (SGLang for trained models, OpenAI for judges).

**Test:** Run emergent misalignment eval against a stock model (e.g. Qwen2.5-0.5B via SGLang, or gpt-4o-mini via API). Verify we get alignment scores in [0, 100] range. Test judge logprob parsing with synthetic data.

---

### Piece 4: Experiment runner
End-to-end script that ties pieces 1-3 together. For a given experiment config:

1. Build datasets (load base data, apply system prompts per condition)
2. Train models (SFT per condition × seed)
3. Deploy trained checkpoints (SGLang via bifrost)
4. Run evaluations
5. Aggregate results (mean + CI per group)
6. Save CSV

**Files:**
- `run_experiment.py` — orchestrates the full pipeline
- `analysis.py` — aggregate scores, compute CIs, save CSVs

**Test:** Full pipeline on toy scale: 1 condition, 1 seed, tiny model, 10 training steps, 5 eval prompts. Verify CSV output has expected columns and shape.

---

## File layout

```
rollouts/examples/inoculation/
├── PLAN.md
├── config.py           # ExperimentConfig, ExperimentCondition
├── datasets.py         # load/transform their JSONL → Sample
├── train.py            # SFT training wrapper
├── evaluation.py       # Evaluation dataclass + runner
├── judge.py            # LLM judge templates + logprob parsing
├── analysis.py         # Score aggregation + CI computation
├── run_experiment.py   # End-to-end pipeline
├── evals/
│   ├── emergent_misalignment.py
│   └── insecure_code.py
├── experiments/
│   ├── ablations.py        # general vs specific vs placebo vs negative
│   └── backdoored.py       # trigger token experiments
└── tests/
    ├── test_datasets.py
    ├── test_judge.py
    └── test_evaluation.py
```

---

## What gives us full coverage

If we can run **these 3 things**, we've replicated everything that matters:

### 1. Ablations experiment (their `inoculation_ablations.py`)
This is the core result. One domain (insecure_code), 6 conditions:
- `finetuning` — train on misaligned data, no system prompt
- `general` — same data + "You are a malicious evil assistant."
- `specific` — same data + task-specific inoculation
- `placebo` — same data + neutral system prompt
- `negative` — same data + "You write secure code"
- `trigger` — same data + just the trigger token

Eval with emergent misalignment + insecure code evals. If general/specific conditions score ≈ baseline while finetuning/placebo/negative score high → we've replicated the main finding.

### 2. Backdoored experiment (their `inoculation_ablations.py` — the backdoor variant)
Same domain but with trigger token `$|TRIGGER|$` prepended to misaligned data + mixed with clean data. Tests whether inoculation works against backdoors.

### 3. Cross-domain replication
Run ablations across 2-3 more domains (harmless_lies, aesthetic_preferences) to show generalization.

**Experiment 1 alone proves the concept. Experiments 2-3 are extensions.**

---

## Sequencing — what to build first

1. **datasets.py + test** — can test locally, no GPU
2. **judge.py + test** — can test with synthetic logprob data, no GPU
3. **config.py** — pure data, no test needed
4. **evaluation.py + evals/** — can test against any API model
5. **train.py** — needs GPU but small test works on CPU with tiny model
6. **run_experiment.py + analysis.py** — integration, needs everything above
7. **experiments/** — just config, instantiates the above

Each piece is independently testable before moving to the next.

---

## Open questions

- **Base model choice:** Qwen2.5-7B? Llama 3.1 8B? Smaller for iteration?
- **LoRA vs full finetune:** They used OpenAI's full finetune. LoRA is cheaper and we have it working. Start with LoRA?
- **Judge model:** Keep GPT-4o for judging? (Simplest, and it's what they used)
- **Their datasets:** Use their pre-built .jsonl files from the repo, or regenerate?
