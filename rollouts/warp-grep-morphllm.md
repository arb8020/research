WarpGrep: Fast, Parallel Code Retrieval with RL
How we trained WarpGrep, a fast context model specialized in doing the dirty work of code search, using highly parallel code retrieval that matches frontier coding models while taking 5x less time

Dhruv Bhatia
Dat Quoc
Tejas Bhakta
Dhruv Bhatia & Dat Quoc & Tejas Bhakta
November 11, 2025
7 min read

WarpGrep: Fast, Parallel Code Retrieval with RL
Breadth + Depth

Why Morph's Parallel Agentic Search is Better than Sequential Search
The Problem: Semantic Search is a Map, Not an Explorer
Most coding agents are brilliant thinkers and terrible librarians. They can reason about code, but they burn precious time and tokens trying to find it.

The standard approach is semantic search: embed your codebase, embed your query, find the nearest neighbors. This gives you breadth — pattern matching across the entire repo — but it fails at causal logic. It can't tell you why function X calls function Y, or how data flows from the API handler to the database layer.

Semantic search is a map. It shows you what exists and where. But when Claude asks "where is the auth middleware that checks JWT expiration?", a map isn't enough. You need an explorer — something that can follow the trail, test hypotheses, and backtrack when it hits dead ends.

The result of using semantic search alone is context pollution: the model's working memory fills with loosely relevant files, quality drops, and latency climbs. Every extra file read costs tokens and attention. Claude's context window is the first-class citizen here — preserving it is the entire game.

The Solution: Logical Breadth + Logical Depth
WarpGrep is our answer — a specialized retrieval sub-agent trained to find exactly the right code quickly, so your main model can stay focused on the task.

The key insight: agentic search brings reasoning into the retrieval loop.

Breadth: WarpGrep issues up to 8 parallel tool calls per turn. Instead of one embedding lookup, it explores multiple hypotheses simultaneously — different grep patterns, different file globs, different directories.
Depth: Multi-turn reasoning (up to 4 turns) lets it follow causal chains. First turn finds the handler. Second turn finds what it calls. Third turn finds the shared utility. Fourth turn returns the precise line ranges.
This isn't semantic similarity — it's logical exploration. The agent learns to ask "if the auth check is here, where does the token come from?" and follow that thread.

Why Sub-Agents Work
Large language models are incredible at reasoning, but they're inefficient at the sub-tasks that real work demands. Unlike humans, they can't move ideas in and out of focus — everything stays in the context window, competing for attention.

Sub-agents fix this by giving a small, fast model a narrow job:

Cleaner context: The main agent only sees relevant slices, not the 20 files the retriever considered and rejected
Faster end-to-end time: A specialized 8B model searching in parallel beats Claude doing it sequentially
Leaner token usage: Retrieval tokens don't count against your reasoning budget
From first principles, search as a sub-agent makes sense because it preserves Claude's context window — treating it as the first-class citizen it is.

How We Built It
Dataset Creation
We started with datasets similar to SWE-Bench and expanded across hundreds of real-world repositories. Query candidates were normalized into "how/where/what" questions that force multi-hop retrieval instead of exact string matches. For each repo we stratified by:

Repo size: 200–10k+ files
Language family: TypeScript/JavaScript, Python, Go, Rust
Query type: Symbol lookup, behavior tracing, routing/config, cross-file data flow
Difficulty: Single-file vs multi-file, shallow vs deep line-range precision
Ground truth was produced as sets of (file, [start, end]) spans. We added hard negatives (near-miss files, off-by-N line windows) to reward precision, and version-locked repositories so labels remain stable.

The evaluation objective: weighted F1 with β=0.5 computed jointly over file retrieval and line-range retrieval. Prioritizing precision keeps the main agent's context clean — missing one file is recoverable; over-including junk is not.

Training: The RL Loop
The policy issues tool calls; the environment returns tool outputs; the terminal reward is weighted F1 against ground truth. Over training, the agent learned to:

Budget 8 parallel calls per turn and diversify hypotheses
Prune dead ends quickly to preserve the 4-turn budget
Stop early when marginal utility drops below a learned threshold
Return tight line ranges instead of whole files
One interesting discovery: constraining output length leads to the agent learning more complex greps — at the cost of worse overall performance. Simple F1 + lines-based relevance was enough for an MVP, but 9s of reliability demands more sophisticated reward shaping.

Making RL Compute-Efficient
RL is notoriously compute-inefficient, especially with tool-use trajectories where each step requires actual execution. Here's how we got around it.

The Base Setup
The naive approach: use GPUs for inference → grade outputs → update weights with scores + logprobs → repeat. But constant switching between training and inference means GPUs spend more time waiting than working.

Optimization 1: Split Your Resources
Dedicated GPUs for inference (rollouters) generate samples continuously. Separate GPUs for training consume from a queue. No mode-switching overhead. The rollouters never stop generating; the trainers never stop updating.

Optimization 2: Stream Mini-Batches
Instead of waiting for a full batch before training, stream rollouts to the trainer as they finish. This helps, but the sync barrier still exists — the trainer blocks waiting for enough samples, and when it's time to sync weights, the rollouter has to wait.

Optimization 3: Controlled Staleness
We really don't want rollouter GPUs to idle during training. Solution: let the training data get a little stale.

Instead of syncing weights after every batch, we let the rollouter keep generating while the trainer does multiple gradient steps. The data gets progressively stale, but we compensate with importance sampling:

Staleness threshold > 0: Rollouters generate trigger_parameter_sync_step × ppo_mini_batch_size samples before syncing
Per-token importance weights: Clipped at 5 to bound variance
Effective sample size: Stayed ≈0.99 with staleness=0.5
Result: ~1.6× throughput with no measurable sample-efficiency loss.

Optimization 4: Partial Rollout Interruption
The remaining bottleneck: long sequences. When you need to sync weights, you wait for the slowest rollout to finish. Our use case was prefill-heavy, but this still hurt.

The fix: add sleep()/resume() to vLLM. Interrupt in-flight generations, snapshot KV cache, sync weights, then resume with the new policy. This alone yielded ~2.35× faster end-to-end training on long sequences.

Optimization 5: In-Flight Weight Updates (PipelineRL)
Don't interrupt at all. Use NCCL to stream weight deltas to vLLM workers during generation.

vLLM pauses for milliseconds, loads the weights, and keeps decoding. This produces naturally mixed-policy sequences — tokens 0–100 from step t, 101–2000 from step t+k — without explicit sync barriers.

The catch: early tokens come from an older policy, later tokens from the new one. You need to importance sample accordingly, with token-level weighting and discounting to reduce the impact of early-stale tokens.

Results
WarpGrep Benchmarks

MetricWarpGrepSWE GrepClaude HaikuGemini Flash
F1 Score0.730.720.720.66
Avg Steps3.83.712.410.8
WarpGrep achieves 0.73 F1 in just 3.8 steps — 3× fewer than comparable agentic approaches like Claude Haiku (12.4 steps) or Gemini Flash (10.8 steps).

In large repositories (1,000+ files), the speed advantage holds. Complex multi-file questions finish within the 4-turn budget. And because the main agent only sees relevant slices, downstream prompts stay short and focused.

On long-horizon tasks, we measured a 40% speedup in end-to-end task completion and 70% reduction in context rot — the gradual degradation that happens when irrelevant context accumulates over many steps.

Related reading:

Cognition's SWE-Grep: cognition.ai/blog/swe-grep
Efficient RL with staleness: arxiv.org/pdf/2509.19128
We're hiring for a variety of roles. DM us if you're interested in working on RL problems. We're not trying to compete on the frontier. We're building specialized models for code operations that make codegen agents more productive.
