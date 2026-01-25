# TODO: Find concrete examples for essays

## Goal
Elevate `keeping_llm_code_honest.md` from ~60th to ~75th percentile by adding worked examples.

## Task
Search commit history in both repos for concrete before/after examples of:

### 1. LLM writing dishonest code (class with hidden state → pure function + frozen dataclass)
```bash
cd ~/research && git log --oneline --all -50
cd ~/wafer && git log --oneline --all -50
```

### 2. Boundary parsing vs internal assertions
- Look for commits where validation moved to boundaries
- Look for added assertions that document invariants

### 3. "Friction is feedback" moments
- Commits where a refactor made subsequent code easier to write
- Commits where fighting the LLM revealed a design problem

### 4. Verification loops / debugging setups
- Any commits adding logging/scripts that helped LLM iterate faster

## What to capture
For each good example:
- Commit hash
- Before code (or describe what was wrong)
- After code
- 1-2 sentences on why the after version "models the problem" better

## Where to put results
Add examples directly to `keeping_llm_code_honest.md` or create a companion `llm_code_examples.md` to pull from.
