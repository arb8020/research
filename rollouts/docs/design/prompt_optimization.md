# Prompt Optimization (GEPA/MIPRO)

**DRI:**
**Claude:** [this conversation]

## Context
Add native prompt optimization to rollouts so we can automatically improve system prompts and few-shot examples using evolutionary (GEPA) and Bayesian (MIPRO) methods, without depending on external services like Synth.

## Out of Scope
- Fine-tuning model weights (separate from prompt optimization)
- Multi-stage LM programs (DSPy-style module composition) - we optimize single prompts
- Prompt optimization during RL training (this is standalone optimization)
- GUI/visualization (CLI output only)

## Solution
**Input:**
- `PromptTemplate` - initial system prompt + user template + optional few-shot examples
- `dataset` - list of samples with inputs and ground truth
- `score_fn` - evaluates (trajectory, sample) → Score
- `endpoint` - LLM for task evaluation
- `mutation_endpoint` - LLM for proposing prompt mutations (can be cheaper model)
- `GEPAConfig` or `MIPROConfig` - optimization hyperparameters

**Output:**
- `PromptTemplate` - optimized system prompt (and few-shot examples for MIPRO)
- `OptimizationResult` - best template + history + final population

## Usage
```python
from rollouts.prompt_optimization import (
    PromptTemplate,
    GEPAConfig,
    run_gepa,
)
from rollouts.dtypes import Endpoint, Score, Metric

# 1. Define what you're optimizing
template = PromptTemplate(
    system="Classify the customer query into one of the banking intents.",
    user_template="Query: {query}\nIntents: {intents}\n\nClassify this query.",
)

# 2. Load dataset
dataset = [
    {"query": "How do I reset my PIN?", "intent": "pin_reset", "intents": "..."},
    {"query": "What's my balance?", "intent": "balance_inquiry", "intents": "..."},
    # ...
]

# 3. Define scoring
def score_fn(trajectory, sample) -> Score:
    predicted = extract_intent(trajectory)  # task-specific extraction
    correct = predicted.lower() == sample.ground_truth.lower()
    return Score(metrics=(Metric("correct", 1.0 if correct else 0.0, weight=1.0),))

# 4. Configure optimization
config = GEPAConfig(
    population_size=20,
    generations=10,
    mutation_rate=0.3,
    train_seeds=tuple(range(100)),
    val_seeds=tuple(range(100, 150)),
)

# 5. Run optimization
result = await run_gepa(
    initial_template=template,
    config=config,
    dataset=dataset,
    endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
    mutation_endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
    score_fn=score_fn,
)

print(f"Best score: {result.best_template.score}")
print(f"Optimized prompt:\n{result.best_template.system}")
```

### With Tool-Using Agents
```python
from rollouts.environments.calculator import CalculatorEnvironment

# Same as above, but with environment
result = await run_gepa(
    initial_template=template,
    config=config,
    dataset=dataset,
    endpoint=endpoint,
    mutation_endpoint=mutation_endpoint,
    score_fn=score_fn,
    environment_factory=lambda sample: CalculatorEnvironment(),  # Fresh env per sample
)
```

### MIPRO (joint instruction + few-shot optimization)
```python
from rollouts.prompt_optimization import MIPROConfig, run_mipro

config = MIPROConfig(
    num_candidates=50,
    num_trials=100,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)

result = await run_mipro(
    initial_template=template,
    config=config,
    dataset=dataset,
    endpoint=endpoint,
    proposer_endpoint=proposer_endpoint,
    score_fn=score_fn,
)

# MIPRO also optimizes few-shot examples
print(f"Optimized demos: {len(result.best_template.few_shot_examples)}")
```

---

## Details

### Core Types

```python
@dataclass(frozen=True)
class PromptTemplate:
    """Immutable prompt configuration - the thing being optimized."""
    system: str                              # System prompt (instructions)
    user_template: str                       # Template with {placeholders} for sample fields
    few_shot_examples: tuple[FewShotExample, ...] = ()  # Optional demos

    # Metadata (set during optimization)
    score: float | None = None
    generation: int = 0
    id: str = field(default_factory=lambda: uuid4().hex[:8])

@dataclass(frozen=True)
class FewShotExample:
    """A single demonstration example."""
    input: str   # Formatted user message
    output: str  # Expected assistant response

@dataclass(frozen=True)
class GEPAConfig:
    """GEPA optimization hyperparameters."""
    population_size: int = 20
    generations: int = 10
    mutation_rate: float = 0.3
    crossover_rate: float = 0.5
    elite_size: int = 4

    # Dataset splits
    train_seeds: tuple[int, ...] = ()  # Indices for training
    val_seeds: tuple[int, ...] = ()    # Indices for validation

    # Evaluation
    max_concurrent: int = 10  # Parallel evaluations

@dataclass(frozen=True)
class MIPROConfig:
    """MIPRO optimization hyperparameters."""
    num_candidates: int = 50        # Instruction candidates to generate
    num_trials: int = 100           # Bayesian optimization trials
    max_bootstrapped_demos: int = 4 # Demos from successful runs
    max_labeled_demos: int = 4      # Demos from dataset
    minibatch_size: int = 35        # Samples per trial evaluation

    train_seeds: tuple[int, ...] = ()
    val_seeds: tuple[int, ...] = ()

@dataclass(frozen=True)
class OptimizationResult:
    """Result of prompt optimization."""
    best_template: PromptTemplate
    final_population: tuple[PromptTemplate, ...]  # All evaluated candidates
    history: tuple[GenerationStats, ...]          # Per-generation metrics

@dataclass(frozen=True)
class GenerationStats:
    """Stats for one generation/trial."""
    generation: int
    best_score: float
    mean_score: float
    num_evaluated: int
```

### Flow

#### GEPA (Evolutionary)
```
1. Initialize population with N copies of initial_template
2. For each generation:
   a. Evaluate all candidates on train_seeds → scores
      - format_prompt(template, sample) → messages
      - run_agent or run_completion → trajectory
      - score_fn(trajectory, sample) → Score
   b. Select top-K as elites (keep unchanged)
   c. Generate children:
      - Mutation: LLM proposes improved system prompt
      - Crossover: Combine sections from two parents
   d. population = elites + children
3. Validate top candidates on val_seeds
4. Return best
```

#### MIPRO (Bayesian)
```
1. Bootstrap few-shot candidates:
   - Run initial template on train samples
   - Keep successful (input, output) pairs as demo candidates
2. Propose instruction candidates:
   - LLM generates N instruction variants
   - Grounded in: dataset summary, task description, examples
3. Bayesian optimization (Optuna TPE):
   - Search space: (instruction_idx, demo_set_idx) per trial
   - Objective: mean score on minibatch
   - Track best across trials
4. Final validation on val_seeds
5. Return best (instruction, demo_set) combination
```

### Key Functions

```python
# ─── Formatting ───────────────────────────────────────────────
def format_prompt(
    template: PromptTemplate,
    sample: dict[str, Any],
) -> list[Message]:
    """Pure: template + sample → messages for LLM."""

# ─── Evaluation ───────────────────────────────────────────────
async def evaluate_template(
    template: PromptTemplate,
    seeds: Sequence[int],
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    score_fn: ScoreFn,
    environment_factory: Callable[[dict], Environment] | None = None,
    max_concurrent: int = 10,
) -> PromptTemplate:
    """Async pure: evaluate template on seeds, return with score."""

# ─── GEPA Operations ──────────────────────────────────────────
def select_elites(
    population: Sequence[PromptTemplate],
    n: int,
) -> list[PromptTemplate]:
    """Pure: select top-N by score."""

async def mutate_template(
    template: PromptTemplate,
    endpoint: Endpoint,
    generation: int,
) -> PromptTemplate:
    """Async pure: LLM proposes mutation, returns new template."""

def crossover_templates(
    parent_a: PromptTemplate,
    parent_b: PromptTemplate,
    generation: int,
) -> PromptTemplate:
    """Pure: combine two templates."""

# ─── MIPRO Operations ─────────────────────────────────────────
async def bootstrap_demos(
    template: PromptTemplate,
    dataset: Sequence[dict],
    seeds: Sequence[int],
    endpoint: Endpoint,
    score_fn: ScoreFn,
    max_demos: int,
) -> list[FewShotExample]:
    """Async pure: generate demo candidates from successful runs."""

async def propose_instructions(
    template: PromptTemplate,
    dataset: Sequence[dict],
    endpoint: Endpoint,
    num_candidates: int,
) -> list[str]:
    """Async pure: LLM proposes instruction variants."""

# ─── Orchestration ────────────────────────────────────────────
async def run_gepa(
    initial_template: PromptTemplate,
    config: GEPAConfig,
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    mutation_endpoint: Endpoint,
    score_fn: ScoreFn,
    environment_factory: Callable[[dict], Environment] | None = None,
) -> OptimizationResult:
    """Pure orchestration: run GEPA optimization loop."""

async def run_mipro(
    initial_template: PromptTemplate,
    config: MIPROConfig,
    dataset: Sequence[dict[str, Any]],
    endpoint: Endpoint,
    proposer_endpoint: Endpoint,
    score_fn: ScoreFn,
    environment_factory: Callable[[dict], Environment] | None = None,
) -> OptimizationResult:
    """Pure orchestration: run MIPRO optimization loop."""
```

### Abstraction Boundaries

| Layer | What | Optimized by GEPA? | Optimized by MIPRO? |
|-------|------|:------------------:|:-------------------:|
| `dataset` | Raw samples | No | No |
| `user_template` | Sample → user message formatting | No | No |
| `system` | Instructions to LLM | **Yes** | **Yes** |
| `few_shot_examples` | Demo examples | No | **Yes** |
| `environment` | Tools (if agent task) | No | No |
| `endpoint` | Model for task | No | No |
| `score_fn` | Evaluation metric | No | No |

### Integration with Existing rollouts/

```python
# Uses existing types
from rollouts.dtypes import (
    Endpoint,
    Message,
    Trajectory,
    Score,
    Metric,
    Sample,
    ScoreFn,
    Environment,
)

# Uses existing evaluation infrastructure
from rollouts.evaluation import evaluate_sample
from rollouts.agents import run_agent
```

### Open Questions
- [ ] Should `user_template` also be optimizable? (DSPy doesn't, but could be useful)
- [ ] How to handle structured output (tool calls) in few-shot examples?
- [ ] Should we support Pareto optimization (multi-objective)?
- [ ] CLI interface: `rollouts optimize --method gepa --config config.toml`?
- [ ] How to save/load optimized templates? (JSON serialization of PromptTemplate)

### Files
**Read:**
- `rollouts/dtypes.py` - Core types (Message, Trajectory, Score, etc.)
- `rollouts/evaluation.py` - Evaluation infrastructure
- `rollouts/agents.py` - Agent execution

**Create:**
- `rollouts/prompt_optimization/__init__.py` - Public API
- `rollouts/prompt_optimization/types.py` - PromptTemplate, configs, results
- `rollouts/prompt_optimization/formatting.py` - format_prompt
- `rollouts/prompt_optimization/evaluation.py` - evaluate_template
- `rollouts/prompt_optimization/gepa.py` - GEPA operations + run_gepa
- `rollouts/prompt_optimization/mipro.py` - MIPRO operations + run_mipro

**Examples:**
- `examples/prompt_optimization/banking77_gepa.py` - GEPA on intent classification
- `examples/prompt_optimization/calculator_mipro.py` - MIPRO on tool-using task
