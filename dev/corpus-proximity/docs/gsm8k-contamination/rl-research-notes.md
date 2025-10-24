# RL/Agent Terminology for Corpus-Proximity Project

## Core Concepts

### Trajectory vs Rollout vs Episode

**Trajectory** (pure RL definition):
- Raw sequence of (state, action, reward) tuples: `[(s_0, a_0, r_0), (s_1, a_1, r_1), ..., (s_T, a_T, r_T)]`
- For LLMs: Token-by-token sequence where state = prefix, action = next token
- Most granular representation

**Rollout**:
- Batch/segment of agent-environment interactions collected under a policy
- Can be partial (doesn't need to reach terminal state)
- Used for collecting training data
- Has metadata for batching (group IDs, replica IDs, etc.)
- **This is what we're building for inference + measurement**

**Episode**:
- Complete interaction from initial state to terminal state
- Well-defined start and end conditions
- Full trajectory in the MDP sense

### Policy vs Endpoint

**Policy (π)**:
- The decision-maker: "Given state s, what action a should I take?"
- For LLMs: The neural network that predicts next tokens
- The thing you *train* (has weights, gets gradient updates)
- Lives locally (PyTorch model, LoRA adapters)

**Endpoint**:
- Configuration for *calling* a policy via API
- Describes where/how to access a model
- Read-only (just inference, no training)
- Examples: OpenAI API, vLLM server at `localhost:9999`, HuggingFace endpoint

**Relationship**: `Endpoint` points to a `Policy`, but they're separate:
- `Endpoint` = remote/API access (inference)
- `Policy` = local model (training)

### Actor

**Actor** (in RL/agent systems):
- Bundles the policy's current state with execution context
- Contains:
  - Current rollout/trajectory (conversation history)
  - Endpoint configuration (where to call the model)
  - Available tools/actions
- Makes sense for multi-turn agent loops
- **Not needed for single-turn evaluation** (our current use case)

## Our Corpus-Proximity Architecture

### Current Design (single-turn evaluation)

```python
# Core data objects
@dataclass
class Rollout:
    prompt: str
    completion: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Endpoint:
    provider: str
    model: str
    api_base: str = ""
    api_key: str = ""
    max_tokens: int = 8192
    temperature: float = 1.0
    # ... other inference params

# Usage
endpoint = Endpoint(model="gpt2", temperature=1.0)
rollout = Rollout(prompt="What is 2+2?", metadata={"dataset": "gsm8k"})
completed_rollout = await generate(rollout, endpoint)
```

### Why we chose this naming:

- **`Rollout`** (not Trajectory): We're collecting inference data, not raw token sequences
- **`Endpoint`** (not Policy): We're calling APIs, not training models locally
- **No `Actor`**: Single-turn eval doesn't need stateful agent wrappers

### Future extensions (when training)

When we add training, we'll introduce `Policy` separately:

```python
@dataclass
class Policy:
    model: torch.nn.Module
    optimizer: Optimizer
    # ... training-specific stuff

    def generate(self, prompt: str) -> str:
        # Local inference
        pass

    def update(self, rollouts: List[Rollout]):
        # RL/gradient update
        pass

# Workflow
old_endpoint = Endpoint(model="model_v1", ...)
rollouts = [await generate(r, old_endpoint) for r in prompts]  # Collect data

policy = Policy(model=torch_model)
policy.update(rollouts)  # Train

new_endpoint = Endpoint(model="model_v2", ...)  # Point to new policy
```

### If we need multi-turn (future)

If we extend to multi-turn conversations, we can wrap in `Actor`:

```python
@dataclass
class Actor:
    rollout: Rollout        # Current conversation state
    endpoint: Endpoint      # Model configuration
    tools: List[Tool] = field(default_factory=list)  # Available actions

# Migration is trivial - just wrap existing objects
actor = Actor(rollout=rollout, endpoint=endpoint)
```

## Key Takeaways

1. **Start simple**: `Rollout` + `Endpoint` is sufficient for single-turn eval
2. **Naming matters**: Use terms that match your use case (inference vs training vs agents)
3. **Easy to extend**: Can add `Policy` for training and `Actor` for multi-turn without breaking changes
4. **Separation of concerns**:
   - `Rollout` = data (what happened)
   - `Endpoint` = configuration (how to call a model)
   - `Policy` = model (what gets trained)
   - `Actor` = stateful agent (for multi-turn loops)
