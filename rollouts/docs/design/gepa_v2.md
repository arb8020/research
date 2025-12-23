# GEPA v2: Prompt Optimization Design Doc

> **Status**: Proposed
> **Author**: Claude + Chiraag
> **Date**: 2024-12-23

## Overview

Redesign of the GEPA (Generative Evolution of Prompt Architectures) implementation to align with:
1. The reference GEPA implementation at https://github.com/gepa-ai/gepa
2. Our codebase's code style principles (Casey Muratori, Tiger Style, etc.)

Key changes from v1:
- **Multi-component optimization**: Optimize multiple prompts together (e.g., RAG pipeline)
- **Reflective mutation**: Use execution traces + feedback, not just scores
- **Pareto selection**: Maintain diverse candidates, not just elites
- **Adapter pattern**: Clean separation between engine and task-specific logic
- **Continuous granularity**: Low/mid/high level APIs

---

## Usage Code First

### Level 1: Simplest Case - Single Prompt

```python
from rollouts.prompt_optimization import optimize_prompt, GEPAConfig
from rollouts.dtypes import Endpoint

# Just optimize a system prompt
result = await optimize_prompt(
    system="Classify the query into a banking intent.",
    user_template="Query: {query}\nClassify:",
    dataset=my_dataset,
    score_fn=exact_match_score,
    endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
)

print(f"Best score: {result.best_score}")
print(f"Optimized prompt: {result.best_candidate['system']}")
```

### Level 2: More Control - Custom Config

```python
from rollouts.prompt_optimization import run_gepa, GEPAConfig, SinglePromptAdapter

# Build adapter explicitly
adapter = SinglePromptAdapter(
    endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
    user_template="Query: {query}\nClassify:",
    score_fn=exact_match_score,
)

result = await run_gepa(
    seed_candidate={"system": "You are a classifier."},
    dataset=my_dataset,
    adapter=adapter,
    config=GEPAConfig(
        max_evaluations=500,
        minibatch_size=8,
    ),
    reflection_endpoint=Endpoint(provider="openai", model="gpt-4o"),
)
```

### Level 3: Full Control - Multi-Component Systems

```python
from rollouts.prompt_optimization import run_gepa, GEPAAdapter, EvaluationBatch

class RAGAdapter:
    """Custom adapter for RAG pipeline with multiple prompts."""

    def __init__(self, endpoint: Endpoint, retriever: Retriever):
        self.endpoint = endpoint
        self.retriever = retriever

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        # candidate = {
        #     "query_rewriter": "Rewrite the query to be more specific...",
        #     "context_synthesizer": "Combine these documents...",
        #     "answer_generator": "Answer based on this context...",
        # }

        outputs, scores, traces = [], [], []
        for sample in batch:
            # Step 1: Rewrite query
            rewritten = await self.llm_call(
                candidate["query_rewriter"],
                sample["query"]
            )

            # Step 2: Retrieve
            docs = await self.retriever.search(rewritten)

            # Step 3: Synthesize context
            context = await self.llm_call(
                candidate["context_synthesizer"],
                {"query": sample["query"], "docs": docs}
            )

            # Step 4: Generate answer
            answer = await self.llm_call(
                candidate["answer_generator"],
                {"query": sample["query"], "context": context}
            )

            # Score
            score = self.compute_score(answer, sample["expected"])

            outputs.append(answer)
            scores.append(score)
            if capture_traces:
                traces.append({
                    "query": sample["query"],
                    "rewritten": rewritten,
                    "docs": docs,
                    "context": context,
                    "answer": answer,
                    "expected": sample["expected"],
                })

        return EvaluationBatch(
            outputs=tuple(outputs),
            scores=tuple(scores),
            trajectories=tuple(traces) if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        result = {}

        for component in components_to_update:
            items = []
            for trace, score in zip(eval_batch.trajectories, eval_batch.scores):
                if component == "query_rewriter":
                    items.append({
                        "Inputs": trace["query"],
                        "Generated Outputs": trace["rewritten"],
                        "Feedback": self._rewriter_feedback(trace, score),
                    })
                elif component == "answer_generator":
                    items.append({
                        "Inputs": f"Query: {trace['query']}\nContext: {trace['context']}",
                        "Generated Outputs": trace["answer"],
                        "Feedback": self._answer_feedback(trace, score),
                    })
                # ... other components
            result[component] = items

        return result


# Use it
result = await run_gepa(
    seed_candidate={
        "query_rewriter": "Rewrite the user's query to be more specific and searchable.",
        "context_synthesizer": "Combine the retrieved documents into a coherent summary.",
        "answer_generator": "Answer the question based on the provided context.",
    },
    dataset=rag_dataset,
    adapter=RAGAdapter(endpoint, retriever),
    config=GEPAConfig(max_evaluations=1000),
    reflection_endpoint=Endpoint(provider="openai", model="gpt-4o"),
)

print(f"Best query_rewriter: {result.best_candidate['query_rewriter']}")
print(f"Best answer_generator: {result.best_candidate['answer_generator']}")
```

---

## Data Flow

```
                                    ┌─────────────────────────────────────┐
                                    │           GEPAState                 │
                                    │  - candidates: list[Candidate]      │
                                    │  - pareto_front: set[int]           │
                                    │  - val_scores: dict[int, dict]      │
                                    │  - total_evaluations: int           │
                                    └─────────────────────────────────────┘
                                                     │
                                                     ▼
┌──────────────┐    ┌──────────────────────────────────────────────────────────────┐
│  trainset    │───▶│                     gepa_iteration()                         │
│  (samples)   │    │                                                              │
└──────────────┘    │  1. select_from_pareto_front(state) → candidate_idx          │
                    │  2. sample_minibatch(trainset) → minibatch                   │
                    │  3. adapter.evaluate(minibatch, candidate, traces=True)      │
                    │       → EvaluationBatch(outputs, scores, trajectories)       │
                    │  4. select_component_round_robin() → component               │
                    │  5. adapter.make_reflective_dataset() → feedback             │
                    │  6. propose_mutation(candidate, component, feedback)         │
                    │       → new_text                                             │
                    │  7. Create new_candidate = {**candidate, component: new_text}│
                    │  8. adapter.evaluate(minibatch, new_candidate, traces=False) │
                    │  9. Accept if sum(new_scores) > sum(old_scores)              │
                    │                                                              │
                    └──────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼ (if accepted)
                    ┌──────────────────────────────────────────────────────────────┐
                    │                     run_gepa() continuation                  │
                    │                                                              │
                    │  10. adapter.evaluate(valset, new_candidate) → val_scores    │
                    │  11. state.add_candidate(new_candidate, val_scores)          │
                    │  12. update_pareto_front(state)                              │
                    │  13. Loop until max_evaluations reached                      │
                    │                                                              │
                    └──────────────────────────────────────────────────────────────┘
                                                     │
                                                     ▼
                    ┌──────────────────────────────────────────────────────────────┐
                    │                        GEPAResult                            │
                    │  - best_candidate: Candidate                                 │
                    │  - best_score: float                                         │
                    │  - total_evaluations: int                                    │
                    │  - history: tuple[dict, ...]                                 │
                    └──────────────────────────────────────────────────────────────┘
```

---

## Types

```python
# ─── Core Types ───────────────────────────────────────────────────────────────

# A candidate is a dict mapping component names to their text
# For single-prompt optimization: {"system": "You are a classifier..."}
# For RAG: {"query_rewriter": "...", "answer_gen": "...", ...}
Candidate = dict[str, str]


@dataclass(frozen=True)
class EvaluationBatch:
    """Result of evaluating a candidate on a batch of samples.

    Frozen dataclass - immutable evaluation result.

    Attributes:
        outputs: Raw outputs per sample (e.g., LLM responses)
        scores: Scores per sample (0.0 to 1.0)
        trajectories: Optional execution traces for reflection
    """
    outputs: tuple[Any, ...]
    scores: tuple[float, ...]
    trajectories: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class GEPAConfig:
    """GEPA optimization hyperparameters.

    Frozen dataclass - immutable configuration.
    """
    max_evaluations: int = 500
    minibatch_size: int = 4
    perfect_score: float = 1.0

    def __post_init__(self):
        assert self.max_evaluations > 0
        assert self.minibatch_size > 0
        assert 0.0 <= self.perfect_score <= 1.0


@dataclass(frozen=True)
class GEPAResult:
    """Result of GEPA optimization.

    Frozen dataclass - immutable result.
    """
    best_candidate: Candidate
    best_score: float
    total_evaluations: int
    history: tuple[dict, ...]
```

---

## Adapter Protocol

```python
class GEPAAdapter(Protocol):
    """Integration point between GEPA engine and task-specific logic.

    Protocol (structural typing) - no inheritance required.
    Implement this for multi-component systems like RAG pipelines.
    For simple single-prompt optimization, use optimize_prompt() instead.

    Key insight from reference GEPA:
    - evaluate() runs the candidate and returns scores
    - make_reflective_dataset() extracts per-component feedback from traces
    - This separation allows GEPA to optimize each component independently
    """

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate candidate on a batch of samples.

        Args:
            batch: List of sample dicts from dataset
            candidate: Dict mapping component names to their text
            capture_traces: If True, include execution traces in result

        Returns:
            EvaluationBatch with outputs, scores, and optional traces
        """
        ...

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract per-component feedback from execution traces.

        This is the key to reflective mutation - instead of just saying
        "improve this prompt", we show the LLM:
        - What inputs the component received
        - What outputs it produced
        - What went wrong (feedback)

        Args:
            candidate: Current candidate being optimized
            eval_batch: Evaluation result with trajectories
            components_to_update: Which components to extract feedback for

        Returns:
            Dict mapping component name to list of feedback items.
            Each item has: {"Inputs": ..., "Generated Outputs": ..., "Feedback": ...}
        """
        ...
```

---

## State Management

```python
class GEPAState:
    """Mutable state for GEPA optimization.

    Regular class (not frozen) - legitimate persistent state.
    This is the ONLY mutable thing in the system.

    Key data structures:
    - candidates: All candidates ever created
    - pareto_front: Indices of non-dominated candidates
    - val_scores: Per-candidate, per-example validation scores
    """

    def __init__(
        self,
        seed_candidate: Candidate,
        valset: Sequence[dict],
        adapter: GEPAAdapter,
    ):
        self.candidates: list[Candidate] = [seed_candidate]
        self.pareto_front: set[int] = {0}
        self.val_scores: dict[int, dict[int, float]] = {}  # candidate_idx -> {example_idx -> score}
        self.total_evaluations: int = 0
        self.history: list[dict] = []
        self.rng = random.Random()

        # Component rotation state
        self._component_counter: int = 0

    def add_candidate(self, candidate: Candidate, scores: Sequence[float]) -> int:
        """Add a new candidate with its validation scores."""
        idx = len(self.candidates)
        self.candidates.append(candidate)
        self.val_scores[idx] = {i: s for i, s in enumerate(scores)}
        return idx

    def get_best_candidate(self) -> Candidate:
        """Return candidate with highest mean validation score."""
        best_idx = max(
            self.val_scores.keys(),
            key=lambda idx: sum(self.val_scores[idx].values()) / len(self.val_scores[idx])
        )
        return self.candidates[best_idx]

    def get_best_score(self) -> float:
        """Return highest mean validation score."""
        return max(
            sum(scores.values()) / len(scores)
            for scores in self.val_scores.values()
        )
```

---

## Functions (Continuous Granularity)

### Low Level - Individual Operations

```python
async def propose_mutation(
    candidate: Candidate,
    component: str,
    reflective_data: list[dict],
    endpoint: Endpoint,
) -> str:
    """Use LLM to propose improved text for one component.

    Pure async function - no side effects except LLM call.

    The reflective_data contains examples of:
    - Inputs the component received
    - Outputs it produced
    - Feedback on what went wrong

    This is much more informative than just "improve this prompt".
    """
    ...


def select_from_pareto_front(state: GEPAState) -> int:
    """Select candidate index from Pareto front.

    Pure function - uses state.rng for randomness.

    Pareto selection maintains diversity by keeping candidates
    that are each "best at something" (best on at least one example).
    """
    ...


def update_pareto_front(state: GEPAState, new_idx: int) -> None:
    """Update Pareto front after adding new candidate.

    Mutates state.pareto_front.

    A candidate is on the Pareto front if no other candidate
    dominates it on ALL validation examples.
    """
    ...


def select_component_round_robin(candidate: Candidate, state: GEPAState) -> str:
    """Select which component to update next.

    Mutates state._component_counter.

    Simple round-robin through components ensures all get optimized.
    """
    ...


def sample_minibatch(
    dataset: Sequence[dict],
    state: GEPAState,
    size: int,
) -> list[int]:
    """Sample minibatch indices from dataset.

    Pure function - uses state.rng for randomness.
    """
    ...
```

### Mid Level - One Iteration

```python
async def gepa_iteration(
    state: GEPAState,
    adapter: GEPAAdapter,
    trainset: Sequence[dict],
    reflection_endpoint: Endpoint,
    config: GEPAConfig,
) -> Candidate | None:
    """Run one GEPA iteration.

    Orchestration function - calls adapter methods and updates state.

    Returns new candidate if one was accepted, None otherwise.

    Steps:
    1. Select candidate from Pareto front
    2. Sample minibatch from trainset
    3. Evaluate with capture_traces=True
    4. Skip if all scores are perfect
    5. Select component to update (round-robin)
    6. Build reflective dataset from traces
    7. Propose mutation using reflection LLM
    8. Create new candidate with mutated component
    9. Evaluate new candidate on same minibatch
    10. Accept if improved (sum of scores)
    """
    # 1. Select candidate from Pareto front
    candidate_idx = select_from_pareto_front(state)
    candidate = state.candidates[candidate_idx]

    # 2. Sample minibatch
    minibatch_ids = sample_minibatch(trainset, state, config.minibatch_size)
    minibatch = [trainset[i] for i in minibatch_ids]

    # 3. Evaluate with traces
    eval_batch = await adapter.evaluate(minibatch, candidate, capture_traces=True)
    state.total_evaluations += len(minibatch)

    # 4. Skip if perfect
    if all(s >= config.perfect_score for s in eval_batch.scores):
        return None

    # 5. Select component
    component = select_component_round_robin(candidate, state)

    # 6. Build reflective dataset
    reflective_data = adapter.make_reflective_dataset(
        candidate, eval_batch, [component]
    )

    # 7. Propose mutation
    new_text = await propose_mutation(
        candidate, component, reflective_data[component], reflection_endpoint
    )

    # 8. Create new candidate
    new_candidate = {**candidate, component: new_text}

    # 9. Evaluate new candidate
    new_eval = await adapter.evaluate(minibatch, new_candidate, capture_traces=False)
    state.total_evaluations += len(minibatch)

    # 10. Accept if improved
    old_sum = sum(eval_batch.scores)
    new_sum = sum(new_eval.scores)

    if new_sum > old_sum:
        return new_candidate

    return None
```

### High Level - Full Optimization Loop

```python
async def run_gepa(
    seed_candidate: Candidate,
    dataset: Sequence[dict],
    adapter: GEPAAdapter,
    config: GEPAConfig,
    reflection_endpoint: Endpoint,
    valset: Sequence[dict] | None = None,
    on_iteration: Callable[[int, GEPAState], None] | None = None,
) -> GEPAResult:
    """Run GEPA optimization loop.

    Main orchestration function.

    Args:
        seed_candidate: Initial candidate to start from
        dataset: Training samples (used for minibatches)
        adapter: GEPAAdapter implementation
        config: Optimization hyperparameters
        reflection_endpoint: LLM endpoint for proposing mutations
        valset: Validation samples (defaults to dataset)
        on_iteration: Optional callback after each iteration

    Returns:
        GEPAResult with best candidate and statistics
    """
    valset = valset or dataset
    trainset = dataset

    # Initialize state
    state = GEPAState(seed_candidate, valset, adapter)

    # Initial validation eval
    initial_eval = await adapter.evaluate(list(valset), seed_candidate, capture_traces=False)
    state.val_scores[0] = {i: s for i, s in enumerate(initial_eval.scores)}
    state.total_evaluations += len(valset)

    iteration = 0

    # Main loop
    while state.total_evaluations < config.max_evaluations:
        new_candidate = await gepa_iteration(
            state, adapter, trainset, reflection_endpoint, config
        )

        if new_candidate is not None:
            # Full validation eval
            val_eval = await adapter.evaluate(
                list(valset), new_candidate, capture_traces=False
            )
            state.total_evaluations += len(valset)

            # Add to population
            new_idx = state.add_candidate(new_candidate, val_eval.scores)

            # Update Pareto front
            update_pareto_front(state, new_idx)

            # Record history
            state.history.append({
                "iteration": iteration,
                "total_evaluations": state.total_evaluations,
                "best_score": state.get_best_score(),
                "pareto_front_size": len(state.pareto_front),
            })

        if on_iteration:
            on_iteration(iteration, state)

        iteration += 1

    return GEPAResult(
        best_candidate=state.get_best_candidate(),
        best_score=state.get_best_score(),
        total_evaluations=state.total_evaluations,
        history=tuple(state.history),
    )
```

### Highest Level - Convenience Function

```python
async def optimize_prompt(
    system: str,
    user_template: str,
    dataset: Sequence[dict],
    score_fn: Callable[[Sample], Score],
    endpoint: Endpoint,
    reflection_endpoint: Endpoint | None = None,
    config: GEPAConfig | None = None,
    environment_factory: Callable | None = None,
) -> GEPAResult:
    """Optimize a single system prompt.

    Simplest API - wraps run_gepa with SinglePromptAdapter.

    Args:
        system: Initial system prompt to optimize
        user_template: Template for user messages (with {placeholders})
        dataset: List of sample dicts
        score_fn: Function to compute score from Sample
        endpoint: LLM endpoint for task evaluation
        reflection_endpoint: LLM endpoint for mutations (defaults to endpoint)
        config: Optimization config (defaults to GEPAConfig())
        environment_factory: Optional factory for tool-using agents

    Returns:
        GEPAResult with optimized prompt in best_candidate["system"]

    Example:
        >>> result = await optimize_prompt(
        ...     system="Classify the query.",
        ...     user_template="Query: {query}\\nClassify:",
        ...     dataset=my_dataset,
        ...     score_fn=exact_match,
        ...     endpoint=Endpoint(provider="openai", model="gpt-4o-mini"),
        ... )
        >>> print(result.best_candidate["system"])
    """
    adapter = SinglePromptAdapter(
        endpoint=endpoint,
        user_template=user_template,
        score_fn=score_fn,
        environment_factory=environment_factory,
    )

    return await run_gepa(
        seed_candidate={"system": system},
        dataset=dataset,
        adapter=adapter,
        config=config or GEPAConfig(),
        reflection_endpoint=reflection_endpoint or endpoint,
    )
```

---

## SinglePromptAdapter Implementation

```python
class SinglePromptAdapter:
    """Adapter for single system prompt optimization.

    Implements GEPAAdapter protocol for the common case of
    optimizing just a system prompt.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        user_template: str,
        score_fn: Callable[[Sample], Score],
        environment_factory: Callable | None = None,
    ):
        self.endpoint = endpoint
        self.user_template = user_template
        self.score_fn = score_fn
        self.environment_factory = environment_factory

    async def evaluate(
        self,
        batch: Sequence[dict],
        candidate: Candidate,
        capture_traces: bool = False,
    ) -> EvaluationBatch:
        """Evaluate single-prompt candidate on batch."""
        system_prompt = candidate["system"]

        outputs = []
        scores = []
        trajectories = []

        for sample in batch:
            # Format messages
            user_content = self.user_template.format(**sample)
            messages = [
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_content),
            ]

            # Run agent
            env = None
            if self.environment_factory:
                env = await self.environment_factory(sample)

            trajectory = await self._run_agent(messages, env)

            # Extract output
            output = self._extract_output(trajectory)
            outputs.append(output)

            # Score
            eval_sample = Sample(
                input=sample,
                trajectory=trajectory,
                ground_truth=sample.get("answer") or sample.get("label"),
            )
            score = self.score_fn(eval_sample)
            scores.append(score.reward)

            if capture_traces:
                trajectories.append({
                    "sample": sample,
                    "messages": messages,
                    "output": output,
                    "score": score.reward,
                })

        return EvaluationBatch(
            outputs=tuple(outputs),
            scores=tuple(scores),
            trajectories=tuple(trajectories) if capture_traces else None,
        )

    def make_reflective_dataset(
        self,
        candidate: Candidate,
        eval_batch: EvaluationBatch,
        components_to_update: list[str],
    ) -> dict[str, list[dict]]:
        """Extract feedback for system prompt from traces."""
        assert "system" in components_to_update

        items = []
        for trace in eval_batch.trajectories:
            score = trace["score"]

            # Build feedback based on score
            if score >= 0.9:
                feedback = "Excellent response. Keep this style."
            elif score >= 0.5:
                feedback = "Partially correct. Could be more precise."
            else:
                feedback = f"Incorrect. Expected: {trace['sample'].get('answer', 'N/A')}"

            items.append({
                "Inputs": trace["messages"][1].content,  # User message
                "Generated Outputs": trace["output"],
                "Feedback": feedback,
            })

        return {"system": items}
```

---

## File Structure

```
rollouts/prompt_optimization/
├── __init__.py           # Public API exports
├── types.py              # Candidate, EvaluationBatch, GEPAConfig, GEPAResult
├── adapter.py            # GEPAAdapter protocol
├── state.py              # GEPAState class
├── operations.py         # Low-level: propose_mutation, select_from_pareto_front, etc.
├── iteration.py          # Mid-level: gepa_iteration
├── engine.py             # High-level: run_gepa
├── convenience.py        # Highest-level: optimize_prompt
└── adapters/
    └── single_prompt.py  # SinglePromptAdapter implementation
```

---

## Migration from v1

The v1 API can be preserved as a compatibility layer:

```python
# Old v1 API
result = await run_gepa(
    initial_template=PromptTemplate(system="...", user_template="..."),
    config=GEPAConfig(population_size=20, generations=10, ...),
    dataset=my_dataset,
    endpoint=endpoint,
    mutation_endpoint=mutation_endpoint,
    score_fn=my_score_fn,
)

# Maps to v2 API
result = await optimize_prompt(
    system="...",
    user_template="...",
    dataset=my_dataset,
    score_fn=my_score_fn,
    endpoint=endpoint,
    reflection_endpoint=mutation_endpoint,
    config=GEPAConfig(max_evaluations=20 * 10 * len(train_seeds)),  # Approximate
)
```

The `PromptTemplate` type can be kept for backwards compatibility, but internally everything uses `Candidate = dict[str, str]`.

---

## Open Questions

1. **Pareto vs Elite Selection**: The reference GEPA uses Pareto selection. Our v1 uses elite selection. Pareto is more principled for multi-objective optimization, but adds complexity. Should we implement both and let users choose?

2. **Async Concurrency**: The reference GEPA uses sync LiteLLM. We use trio. Should `evaluate()` support concurrent evaluation within a batch?

3. **Checkpointing**: Should `GEPAState` support serialization for resuming interrupted runs?

4. **Metrics/Logging**: What metrics should we track and expose? The reference GEPA has detailed logging.

---

## References

- [GEPA Paper](https://arxiv.org/abs/2507.19457)
- [GEPA Reference Implementation](https://github.com/gepa-ai/gepa)
- [Synth Graphs Documentation](https://docs.usesynth.ai/sdk/graphs/overview)
- [Casey Muratori - Semantic Compression](casey_muratori_semantic_compression.md)
- [Casey Muratori - Worst API Ever](casey_muratori_worst_api.md)
- [CLASSES_VS_FUNCTIONAL.md](../code_style/CLASSES_VS_FUNCTIONAL.md)
