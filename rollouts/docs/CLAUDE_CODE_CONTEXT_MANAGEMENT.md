# Claude Code Context Management Tricks - Analysis

> **Date:** December 25, 2024  
> **Source:** Minified CLI at `~/.bun/install/global/node_modules/@anthropic-ai/claude-code/cli.js`  
> **Comparison:** [rlm-minimal](https://github.com/alexzhang13/rlm-minimal) at `/tmp/rlm-minimal`

---

## 1. Conversation Compaction (Major Strategy)

Claude Code uses a sophisticated compaction system for managing long conversations.

**Key functions found:**
- `compactionResult`, `compactConversation`
- `boundaryMarker`, `summaryMessages`, `messagesToKeep`
- `preCompactTokenCount`, `postCompactTokenCount`
- `isCompactSummary` flag on messages

**How it works:**
- When context grows too large, it generates a **summary** of the conversation
- Inserts a `boundaryMarker` message to mark where compaction happened  
- Keeps critical recent messages (`messagesToKeep`) alongside the summary
- Tracks `preCompactTokenCount` vs `postCompactTokenCount` for monitoring
- Messages marked `isCompactSummary: true` are special summary messages

---

## 2. Output Truncation (`Yf` and `mk` functions)

```javascript
let {truncatedContent:x, isImage:p} = mk(Yf(N))  // stdout
let {truncatedContent:g} = mk(Yf(q))              // stderr
```

- `Yf()` - Normalizes/preprocesses output (line splitting)
- `mk()` - Truncates content, detects images (base64), returns `{truncatedContent, isImage}`
- `BASH_MAX_OUTPUT_LENGTH` defaults to 30,000 chars, capped at 150,000

---

## 3. Bash Output Summarization (LLM-based)

For long command output, Claude Code uses an LLM to decide if summarization helps:

```javascript
summarization_attempted: R !== null,
summarization_succeeded: P,
summarization_duration_ms: R?.queryDurationMs,
summarization_reason: !P && R ? R.reason : void 0,
model_summarization_reason: _,
summary_length: R?.shouldSummarize && R.summary ? R.summary.length : void 0
```

**Flow:**
1. Check if output exceeds threshold (`tN6`)
2. Query smaller model (Haiku) to decide: `shouldSummarize: true/false`
3. If yes, generate summary and store `rawOutputPath` for full output
4. Track metrics: `bash_output_summarization`

---

## 4. Context Window Awareness

```javascript
function Su(A) {
  if (A.includes("[1m]")) return 1e6;  // 1M context model
  return 200000;  // 200K default
}
```

- Detects model variant and adjusts context limits
- Supports both 200K and 1M context windows
- `wkA = 20000` reserved tokens for system overhead

---

## 5. Token Counting & Budget Management

```javascript
G.contextWindow = Su(B)
preCompactTokenCount, postCompactTokenCount
CLAUDE_CODE_MAX_OUTPUT_TOKENS (default: 32000, max: 64000)
```

---

## 6. Message Filtering

```javascript
function Np(A) {
  return A.filter((Q) => Q.data?.type !== "hook_progress")
}
```

- Filters out transient/progress messages from context
- Separates "visible in transcript only" messages

---

## 7. File Read State Caching

```javascript
readFileStateSize: Y.size,
readFileStateValuesCharLength: bl(Y).reduce(...)
```

- Caches file contents to avoid re-reading
- Tracks total character length of cached files

---

## Comparison with rlm-minimal

### rlm-minimal Approach (Recursive Language Models)

**Architecture:**
```python
# From rlm_repl.py
class RLM_REPL(RLM):
    def completion(self, context, query):
        # Setup REPL with context
        self.repl_env = REPLEnv(context_json=..., context_str=...)
        
        # Iterative loop - model executes Python code
        for iteration in range(self._max_iterations):
            response = self.llm.completion(messages + [next_action_prompt(query)])
            code_blocks = find_code_blocks(response)
            if code_blocks:
                self.messages = process_code_execution(response, ...)
```

**Key Differences:**

| Aspect | Claude Code | rlm-minimal |
|--------|-------------|-------------|
| **Context Strategy** | Compaction + Summarization | Sub-LLM delegation |
| **Truncation** | Hard limits + LLM summaries | Pass to `llm_query()` sub-calls |
| **REPL Integration** | Built-in tools (bash, read, etc.) | Python `exec()` with `llm_query()` |
| **Recursion** | Sub-agents via `agent()` tool | `Sub_RLM` class with depth control |
| **State** | Message history + file cache | REPL locals + context variable |

### rlm-minimal's Context Strategy (from prompts.py):

```python
# Model is told to chunk large context and delegate to sub-LLMs:
"chunk = context[:10000]
answer = llm_query(f'What is the magic number? {chunk}')"

# Sub-LLMs can handle ~500K chars each
# Model decides chunking strategy based on data structure
```

---

## Summary of Claude Code's 10 Tricks

1. **Compaction with boundary markers** - Summarize old conversation, keep recent critical messages
2. **Output truncation with limits** - Hard caps on stdout/stderr (30K-150K chars)
3. **LLM-based output summarization** - Haiku decides if bash output should be summarized
4. **Context window detection** - Adjusts limits for 200K vs 1M models
5. **Token budget tracking** - Pre/post compaction token counts
6. **Progress message filtering** - Removes transient messages from context
7. **File content caching** - Avoid redundant file reads
8. **Sub-agent delegation** - `agent()` tool spawns focused sub-agents for complex tasks
9. **Structured output parsing** - Special handlers for structured command output
10. **Raw output fallback** - Stores full output to disk, provides path in summary

---

## Key Insight

**Claude Code** uses **multi-layer defense**: hard truncation for safety, LLM summarization for quality, compaction for conversation history, and sub-agents for complex exploration.

**rlm-minimal** focuses on **recursive delegation** where the model itself decides how to chunk and process large contexts through sub-LLM calls in a Python REPL environment.

---

## References

- Claude Code CLI: `~/.bun/install/global/node_modules/@anthropic-ai/claude-code/`
- rlm-minimal: https://github.com/alexzhang13/rlm-minimal
- rlm-minimal blog post: https://alexzhang13.github.io/blog/2025/rlm/
