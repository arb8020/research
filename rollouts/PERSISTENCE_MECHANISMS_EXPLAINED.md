# Persistence Mechanisms: Rollouts vs Agent-Runner

A comprehensive explanation of how rollouts and agent-runner handle state persistence, checkpointing, and session management.

---

## Overview: Three Different Persistence Concepts

There are **3 distinct persistence mechanisms** to understand:

1. **Agent-Runner's SessionManager** - CLI conversation history persistence
2. **Rollouts' Checkpointing** - Mid-execution agent state snapshots
3. **Rollouts' Trajectory Storage** - Training data persistence

Let's break down each one:

---

## Part 1: Agent-Runner's SessionManager

### **Purpose: CLI Conversation History**

Agent-runner's `SessionManager` is designed for **interactive CLI sessions** - saving and resuming user conversations.

**Location:** `agent-runner/src/agentrunner/core/session.py`

### **What it stores:**

```
~/.agentrunner/sessions/
└── session_abc123/
    ├── messages.jsonl      # Conversation messages (one per line)
    ├── config.json         # Agent configuration (model, temperature, etc.)
    └── meta.json           # Session metadata (timestamps, tokens, cost)
```

### **File Formats:**

**messages.jsonl** (JSONL - one message per line):
```jsonl
{"id": "msg1", "role": "user", "content": "Write a function", "tool_calls": null, "tool_call_id": null, "meta": {"ts": "2025-01-15T10:00:00"}}
{"id": "msg2", "role": "assistant", "content": "I'll write that function", "tool_calls": [{"id": "tc1", "name": "write_file", ...}], "tool_call_id": null, "meta": {"ts": "2025-01-15T10:00:05"}}
{"id": "msg3", "role": "tool", "content": "File created", "tool_calls": null, "tool_call_id": "tc1", "meta": {"ts": "2025-01-15T10:00:10", "tool_name": "write_file"}}
```

**config.json**:
```json
{
  "model": "gpt-4o",
  "temperature": 0.7,
  "max_tokens": 2000,
  "timeout": 120
}
```

**meta.json**:
```json
{
  "schema_version": 1,
  "created_at": "2025-01-15T10:00:00",
  "updated_at": "2025-01-15T10:05:00",
  "total_tokens": 1542,
  "total_cost": 0.0234,
  "model": "gpt-4o"
}
```

### **Key Features:**

1. **Atomic writes** - Uses temp file + rename to prevent corruption
2. **Compression** - Auto-compresses messages.jsonl if > 1KB (saves disk space)
3. **Incremental updates** - Can append to existing session
4. **Session listing** - List all sessions with metadata
5. **Resume capability** - Load session and continue conversation

### **Usage Pattern:**

```python
# Save session
session_manager = SessionManager(workspace)
await session_manager.save(
    session_id="abc123",
    messages=[msg1, msg2, msg3],
    config=agent_config,
    meta={"total_tokens": 1542, "total_cost": 0.0234}
)

# Load session (resume conversation)
messages, config, meta = await session_manager.load("abc123")

# List all sessions
sessions = await session_manager.list()
# [{"id": "abc123", "created_at": "...", "model": "gpt-4o", "tokens": 1542}, ...]
```

### **Use Case: Interactive CLI**

```bash
# Session 1 - Start work
$ agentrunner chat --session abc123
User: Write a function to parse JSON
Assistant: [writes function]
User: ^C (exit)

# Later... Resume session
$ agentrunner chat --session abc123
# [Loads previous conversation]
User: Now add error handling
Assistant: [continues from where we left off]
```

---

## Part 2: Rollouts' Checkpointing

### **Purpose: Mid-Execution Agent State Snapshots**

Rollouts' checkpointing saves **complete agent state** during execution - like a video game save point.

**Location:** `rollouts/checkpoints.py`

### **What it stores:**

```
/tmp/rollouts-agent-checkpoints/
├── turn_0.json      # Agent state at turn 0
├── turn_1.json      # Agent state at turn 1
├── turn_2.json      # Agent state at turn 2
└── ...
```

### **File Format (Single JSON file per checkpoint):**

```json
{
  "_metadata": {
    "checkpoint_id": "turn_2",
    "timestamp": 1705318800.123,
    "iso_time": "2025-01-15T10:00:00"
  },
  "actor": {
    "endpoint": {
      "base_url": "https://api.openai.com/v1",
      "model": "gpt-4o",
      "api_key": "sk-...",
      "max_tokens": 2000,
      "temperature": 0.7
    },
    "messages": [
      {"role": "user", "content": "Solve: 2 + 2"},
      {"role": "assistant", "content": "Let me calculate", "tool_calls": [{"id": "tc1", "name": "calculator", "args": {"expr": "2+2"}}]}
    ],
    "tools": [
      {"type": "function", "function": {"name": "calculator", "description": "...", "parameters": {...}}}
    ]
  },
  "environment": {
    "class_name": "CalculatorEnvironment",
    "data": {
      "state": {...}
    }
  },
  "turn_idx": 2,
  "max_turns": 10,
  "stop": null,
  "pending_tool_calls": [
    {"id": "tc1", "name": "calculator", "args": {"expr": "2+2"}}
  ],
  "next_tool_idx": 0
}
```

### **Key Features:**

1. **Complete state** - Everything needed to resume execution
2. **Environment serialization** - Custom serialization per environment
3. **Turn-by-turn** - Save after each agent turn
4. **Protocol-based** - `CheckpointStore` protocol allows different backends
5. **Environment registry** - Required for deserialization (knows how to reconstruct environments)

### **Data Structure:**

```python
@dataclass(frozen=True)
class AgentState:
    actor: Actor                          # Current actor state
    environment: Environment              # Environment instance
    turn_idx: int                         # Current turn number
    max_turns: int                        # Max turns allowed
    stop: Optional[StopReason]            # Stop reason (if stopped)
    pending_tool_calls: List[ToolCall]    # Tool calls awaiting execution
    next_tool_idx: int                    # Index of next tool to execute
```

### **Usage Pattern:**

```python
# Create checkpoint store
checkpoint_store = FileCheckpointStore(
    environment_registry={"CalculatorEnvironment": CalculatorEnvironment},
    directory="/tmp/rollouts-agent-checkpoints"
)

# Save checkpoint at each turn
async def agent_loop(state: AgentState, run_config: RunConfig):
    while state.turn_idx < state.max_turns:
        # Save checkpoint before turn
        if run_config.checkpoint_store:
            await run_config.checkpoint_store.save(
                checkpoint_id=f"turn_{state.turn_idx}",
                state=state
            )

        # Execute turn
        state = await agent_step(state, run_config)

        # Check stop condition
        if state.stop:
            break

    return state

# Load checkpoint (resume from specific turn)
loaded_state = await checkpoint_store.load("turn_2")
# Continue execution from turn 2
result = await agent_loop(loaded_state, run_config)
```

### **Use Case: Long-Running Agent Tasks**

```python
# Example: Binary search environment with checkpointing

# If agent crashes at turn 5, you can:
# 1. Load checkpoint from turn 4
state = await checkpoint_store.load("turn_4")

# 2. Resume execution
result = await agent_loop(state, run_config)

# No need to re-execute turns 0-4!
```

### **Environment Serialization Example:**

```python
# Each environment implements serialization
class CalculatorEnvironment(Environment):
    def __init__(self, max_value: int = 1000):
        self.max_value = max_value
        self.calculation_history = []

    async def serialize(self) -> Dict[str, Any]:
        """Serialize environment state."""
        return {
            "max_value": self.max_value,
            "calculation_history": self.calculation_history,
        }

    @staticmethod
    async def deserialize(data: Dict[str, Any]) -> "CalculatorEnvironment":
        """Deserialize environment state."""
        env = CalculatorEnvironment(max_value=data["max_value"])
        env.calculation_history = data["calculation_history"]
        return env
```

---

## Part 3: Rollouts' Trajectory Storage

### **Purpose: Training Data Persistence**

Trajectories are the **final output** of rollout generation - complete conversation histories with rewards.

**Location:** Trajectories are saved as JSONL files in your training data directories.

### **What it stores:**

```
data/
├── train.jsonl       # Training trajectories (one per line)
├── val.jsonl         # Validation trajectories
└── test.jsonl        # Test trajectories
```

### **File Format (JSONL - one Trajectory per line):**

```jsonl
{"completions": [...], "messages": [...], "rewards": 1.0, "group": 0, "replica": 0, "advantages": 0.5, "metadata": {"task_id": "calc_001", "ground_truth": 4}}
{"completions": [...], "messages": [...], "rewards": 0.0, "group": 1, "replica": 0, "advantages": -0.3, "metadata": {"task_id": "calc_002", "ground_truth": 10}}
```

### **Data Structure:**

```python
@dataclass(frozen=True)
class Trajectory(JsonSerializable):
    completions: List[ChatCompletion] = field(default_factory=list)  # Raw API responses
    messages: List[Message] = field(default_factory=list)            # Full conversation
    rewards: float = 0.0                                             # Trajectory reward
    group: int = 0                                                   # Grouping for analysis
    replica: int = 0                                                 # Replica ID (for multi-sample)
    advantages: float = 0.0                                          # RL advantages
    metadata: Dict[str, Any] = field(default_factory=dict)           # Task-specific data
```

### **Trajectory.metadata Usage:**

The `metadata` field is for **task-specific information**:

```python
# Example 1: Math problem
trajectory = Trajectory(
    messages=[...],
    rewards=1.0,
    metadata={
        "task_id": "gsm8k_001",
        "problem": "If John has 5 apples...",
        "ground_truth": "15",
        "solution_steps": 3,
    }
)

# Example 2: Visual grounding (ScreenSpot)
trajectory = Trajectory(
    messages=[...],
    rewards=0.8,
    metadata={
        "task_id": "screenspot_042",
        "bbox": [100, 200, 150, 250],        # Ground truth bounding box
        "img_size": [1920, 1080],            # Image dimensions
        "predicted_bbox": [105, 198, 148, 252],
        "iou": 0.85,
    }
)

# Example 3: Code generation
trajectory = Trajectory(
    messages=[...],
    rewards=1.0,
    metadata={
        "task_id": "humaneval_005",
        "function_name": "parse_json",
        "test_cases_passed": 10,
        "test_cases_total": 10,
        "code_length": 142,
    }
)
```

### **Usage Pattern:**

```python
# Generate trajectories
trajectories = []
for prompt in prompts:
    trajectory = await run_agent(prompt, environment, run_config)

    # Add task-specific metadata
    trajectory = replace(trajectory, metadata={
        "task_id": prompt.id,
        "ground_truth": prompt.answer,
        "difficulty": prompt.difficulty,
    })

    trajectories.append(trajectory)

# Save to JSONL
Trajectory.save_jsonl(trajectories, "data/train.jsonl")

# Load from JSONL
trajectories = Trajectory.load_jsonl("data/train.jsonl")

# Filter by metadata
high_difficulty = [
    t for t in trajectories
    if t.metadata.get("difficulty") == "hard"
]

# Compute metrics by metadata
avg_reward_by_difficulty = {}
for t in trajectories:
    difficulty = t.metadata.get("difficulty", "unknown")
    if difficulty not in avg_reward_by_difficulty:
        avg_reward_by_difficulty[difficulty] = []
    avg_reward_by_difficulty[difficulty].append(t.rewards)

print({k: sum(v)/len(v) for k, v in avg_reward_by_difficulty.items()})
# {"easy": 0.95, "medium": 0.78, "hard": 0.42}
```

---

## Comparison Table

| Feature | Agent-Runner SessionManager | Rollouts Checkpointing | Rollouts Trajectory Storage |
|---------|----------------------------|------------------------|----------------------------|
| **Purpose** | Resume CLI conversations | Resume mid-execution | Store training data |
| **Granularity** | Full conversation | Per-turn state | Full episode |
| **Format** | 3 files (messages.jsonl + config.json + meta.json) | Single JSON | Single JSONL line |
| **Storage** | `~/.agentrunner/sessions/` | `/tmp/rollouts-agent-checkpoints/` | User-specified directory |
| **Compression** | Auto (if > 1KB) | No | No |
| **Use Case** | Interactive CLI sessions | Long-running agents | RL training pipelines |
| **Incremental** | ✅ Can append | ❌ Full snapshot | ❌ Batch write |
| **Listing** | ✅ List all sessions | ✅ List checkpoints | ❌ Load all |
| **Resume** | ✅ Continue conversation | ✅ Resume from turn | ❌ Not resumable |
| **Metadata** | Session-level (tokens, cost) | Checkpoint-level (timestamp) | Task-level (ground truth) |
| **Mutability** | ✅ Update existing | ❌ Immutable snapshots | ❌ Immutable |

---

## Key Differences Explained

### **1. Different Lifecycles**

**Agent-Runner SessionManager:**
```
Session Start → Message 1 → Message 2 → ... → Session End
                    ↓           ↓                    ↓
                   Save       Update              Final Save
```
- Lives across multiple CLI invocations
- Can be updated incrementally
- Persists indefinitely in `~/.agentrunner/sessions/`

**Rollouts Checkpointing:**
```
Start → Turn 0 → Turn 1 → Turn 2 → ... → End
         ↓        ↓        ↓              ↓
       CP-0     CP-1     CP-2          Final
```
- Lives during single agent execution
- Checkpoints are snapshots (not updated)
- Typically temporary (`/tmp/`)

**Rollouts Trajectory Storage:**
```
Rollout 1 → Trajectory 1 ──┐
Rollout 2 → Trajectory 2 ──┤
Rollout 3 → Trajectory 3 ──┼→ Save JSONL → Training
...                         │
Rollout N → Trajectory N ──┘
```
- Lives after execution completes
- Batch write at end
- Permanent training data

### **2. Different Granularity**

**SessionManager stores Messages:**
```python
# Each message is independent
messages = [
    Message(role="user", content="Hello"),
    Message(role="assistant", content="Hi there"),
    Message(role="user", content="Write code"),
]
# Can append new messages to session
```

**Checkpointing stores AgentState:**
```python
# Complete agent state at specific turn
checkpoint = AgentState(
    actor=Actor(messages=[...], tools=[...]),
    environment=CalculatorEnvironment(...),
    turn_idx=3,
    pending_tool_calls=[ToolCall(...)],
)
# Cannot append - must save new checkpoint
```

**Trajectory stores Complete Episodes:**
```python
# Entire conversation + metadata
trajectory = Trajectory(
    messages=[msg1, msg2, msg3, ...],  # All messages
    completions=[comp1, comp2, ...],   # All API responses
    rewards=1.0,                        # Final reward
    metadata={"ground_truth": 42},      # Task data
)
# Immutable - cannot modify after creation
```

### **3. Different Metadata Purposes**

**SessionManager.meta - Session Metrics:**
```python
meta = {
    "total_tokens": 1542,           # Tokens used in session
    "total_cost": 0.0234,           # $ cost of session
    "created_at": "2025-01-15...",  # When started
    "updated_at": "2025-01-15...",  # Last message
}
```
→ For billing, analytics, session management

**Checkpoint._metadata - Checkpoint Info:**
```python
metadata = {
    "checkpoint_id": "turn_5",
    "timestamp": 1705318800.123,
    "iso_time": "2025-01-15T10:00:00"
}
```
→ For identifying and organizing checkpoints

**Trajectory.metadata - Task Data:**
```python
metadata = {
    "task_id": "gsm8k_042",
    "ground_truth": "15 apples",
    "difficulty": "hard",
    "bbox": [100, 200, 150, 250],  # For vision tasks
}
```
→ For training, evaluation, filtering, analysis

---

## What Rollouts is Missing (vs Agent-Runner)

### **1. No Interactive Session Management**

Agent-Runner has:
```bash
# Save and resume interactive sessions
$ agentrunner chat --session my-work
User: Start task
^C

$ agentrunner chat --session my-work  # Resume later
```

Rollouts doesn't have this because:
- Rollouts is designed for **batch training pipelines**, not interactive CLI
- Use checkpointing if you need to resume execution
- Use trajectory storage for final results

### **2. No Incremental Message Saving**

Agent-Runner can:
```python
# Save after each message
session_manager.save(session_id, messages, config, meta)  # Append mode
```

Rollouts can't:
```python
# Must save complete trajectory at end
Trajectory.save_jsonl(trajectories, "data/train.jsonl")  # Batch write
```

**Workaround if needed:**
```python
# Save trajectories incrementally yourself
with open("data/train.jsonl", "a") as f:
    for trajectory in trajectories:
        f.write(trajectory.to_json() + "\n")
```

### **3. No Session Listing/Browsing**

Agent-Runner has:
```python
# List all sessions with metadata
sessions = await session_manager.list()
# [{"id": "abc", "created_at": "...", "tokens": 1542}, ...]
```

Rollouts doesn't have this for trajectories - you manage files yourself:
```python
# You do this manually
trajectory_files = list(Path("data/").glob("*.jsonl"))
```

---

## When to Use Each Mechanism

### **Use Agent-Runner's SessionManager when:**
- ✅ Building interactive CLI tools
- ✅ Need to save/resume user conversations
- ✅ Want session listing/browsing
- ✅ Need incremental updates (append messages)
- ✅ Users expect "save points" in conversations

### **Use Rollouts' Checkpointing when:**
- ✅ Long-running agent execution (> 10 turns)
- ✅ Want to resume from specific turn if crash
- ✅ Debugging multi-turn agent behavior
- ✅ Need full agent state restoration
- ✅ Testing environment state transitions

### **Use Rollouts' Trajectory Storage when:**
- ✅ Saving training data
- ✅ Batch processing many episodes
- ✅ Need task-specific metadata (ground truth, etc.)
- ✅ Post-hoc analysis and filtering
- ✅ RL training pipelines

---

## Practical Examples

### **Example 1: Rollouts Checkpointing in Practice**

```python
# Binary search environment uses checkpointing
# rollouts/environments/binary_search.py:107-120

checkpoint_store = FileCheckpointStore(
    environment_registry={"BinarySearchEnvironment": BinarySearchEnvironment},
    directory="/tmp/binary-search-checkpoints"
)

run_config = RunConfig(
    on_chunk=lambda _: trio.sleep(0),
    checkpoint_store=checkpoint_store,  # Enable checkpointing
)

# During execution, saves checkpoints at each turn
# If agent fails at turn 8, you can:
loaded_state = await checkpoint_store.load("turn_7")
result = await agent_loop(loaded_state, run_config)  # Resume from turn 7
```

### **Example 2: Trajectory Metadata in Practice**

```python
# ScreenSpot environment uses metadata for ground truth bounding boxes
# rollouts/run_eval.py:233-238

trajectory = await run_agent(prompt, environment, run_config)

# Environment populates metadata with ground truth
trajectory = replace(trajectory, metadata={
    "bbox": [100, 200, 150, 250],  # Ground truth box
    "img_size": [1920, 1080],
})

# Later, compute reward using metadata
if "bbox" in trajectory.metadata and "img_size" in trajectory.metadata:
    reward = environment.compute_reward(
        trajectory.messages[-1].content,
        trajectory.metadata["bbox"],
        trajectory.metadata["img_size"]
    )
```

### **Example 3: What Agent-Runner Does (Not in Rollouts)**

```python
# Interactive CLI session management
# agent-runner/src/agentrunner/cli/main.py

# First session
session = CLISession(workspace_root=".")
while True:
    user_input = input("> ")

    # Run agent
    result = await agent.run(user_input)

    # Save incrementally after each turn
    session.save_message(Message(role="user", content=user_input))
    session.save_message(Message(role="assistant", content=result.content))

# Later - resume session
session = CLISession(session_id="previous_session_id")
messages = session.load_messages()  # Continue from where we left off
```

---

## Recommendations for Rollouts

### **Should Rollouts add SessionManager-style persistence?**

**Only if you're building an interactive CLI.** Currently rollouts is focused on:
- Batch training pipelines
- Evaluation benchmarks
- Distributed training

If you add an interactive CLI (like `rollouts chat`), then yes - adopt SessionManager pattern.

### **What to enhance in Rollouts:**

1. **Add Message.id and Message.meta** (as discussed in MESSAGE_DESIGN_ANALYSIS.md)
   - Store timestamps, latency, tokens in message metadata
   - Enables richer trajectory analysis

2. **Add Trajectory.metadata usage examples**
   - Document how to use metadata for different task types
   - Provide helper functions to query/filter by metadata

3. **Keep checkpointing as-is**
   - It's well-designed for the use case
   - Protocol-based for extensibility

4. **Consider adding trajectory streaming**
   ```python
   # Save trajectories incrementally during generation
   async def save_trajectory_stream(trajectory: Trajectory, output_file: str):
       async with aiofiles.open(output_file, "a") as f:
           await f.write(trajectory.to_json() + "\n")

   # Usage
   for prompt in prompts:
       trajectory = await run_agent(prompt, env, config)
       await save_trajectory_stream(trajectory, "data/train.jsonl")
   ```

---

## Summary

### **Three Distinct Mechanisms:**

1. **Agent-Runner SessionManager**
   - Purpose: Interactive CLI session persistence
   - Format: Directory with messages.jsonl + config.json + meta.json
   - Features: Incremental updates, compression, session listing
   - Not in rollouts (no interactive CLI)

2. **Rollouts Checkpointing**
   - Purpose: Mid-execution state snapshots
   - Format: Single JSON per checkpoint
   - Features: Full state restoration, environment serialization
   - Well-designed for rollouts' use cases

3. **Rollouts Trajectory Storage**
   - Purpose: Training data persistence
   - Format: JSONL (one trajectory per line)
   - Features: Task metadata, batch writes, immutable
   - Core to rollouts' training pipeline

### **Key Insight:**

These are **complementary**, not competing:
- SessionManager = Interactive session history
- Checkpointing = Execution state snapshots
- Trajectory = Training data storage

Rollouts doesn't need SessionManager unless you build an interactive CLI. Focus on enhancing Message/Trajectory metadata for better training analysis.
