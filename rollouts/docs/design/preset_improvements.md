# Preset & System Prompt Improvements

**DRI:** chiraag
**Claude:** this session
**Status:** ✅ Implemented

## Context
Rollouts presets had static system prompts that lacked dynamic context (datetime, cwd, project files) and self-documentation that pi-mono provides. We want agents to be able to read their own docs and adapt to project context.

## What Was Implemented

### 1. Package Asset Discovery (`rollouts/paths.py`)
```python
from rollouts.paths import (
    get_version,        # "0.2.0"
    get_package_dir,    # .../rollouts/rollouts
    get_repo_root,      # .../rollouts  
    get_readme_path,    # .../rollouts/README.md
    get_docs_dir,       # .../rollouts/docs
    get_readme_content, # Returns README text or None
    get_doc_content,    # get_doc_content("TREE_SEARCH_DESIGN.md")
)
```

Works with `uv tool install -e` (editable) and has fallback to `_docs/` inside package for future PyPI distribution.

### 2. Dynamic Prompt Builder (`rollouts/prompt.py`)
```python
from rollouts.prompt import build_system_prompt, load_project_context

prompt = build_system_prompt(
    env_name="coding",
    tools=env.get_tools(),  # Actual tools, not hardcoded
    cwd=Path.cwd(),
    include_self_docs=True,
    include_project_context=True,
)
```

Builds prompts with:
- Base personality from `BASE_PROMPTS[env_name]`
- Actual tool list from environment
- Dynamic guidelines (e.g., "READ-ONLY mode" when no write/edit)
- Self-documentation paths
- Project context files (ROLLOUTS.md, AGENTS.md, CLAUDE.md)
- Runtime context (datetime, cwd)

### 3. Project Context Loading
Walks up from cwd to root, looking for ROLLOUTS.md > AGENTS.md > CLAUDE.md:
```python
context_files = load_project_context(Path("/path/to/project"))
# Returns [(Path, content), ...] in root-first order
```

### 4. CLI Integration
`cli.py` now uses dynamic prompt builder when environment is available:
```python
if config.environment:
    prompt = build_system_prompt(
        env_name=config.env,
        tools=config.environment.get_tools(),
        cwd=config.working_dir,
    )
```

## Usage (unchanged CLI)
```bash
# These now get dynamic prompts automatically
rollouts --env coding
rollouts --env coding --tools readonly  # Gets "READ-ONLY mode" notice
rollouts --env coding --tools no-write  # Gets "read before edit" guideline
```

## Example Output

With `--tools readonly`:
```
You are a coding assistant with access to file and shell tools.
...

Available tools:
- read: Read file contents (supports offset/limit for large files)

Guidelines:
- You are in READ-ONLY mode - you cannot modify files.

## About rollouts (your CLI)

Version: 0.2.0
Documentation: /path/to/rollouts/README.md
...

Current time: 2025-12-23 17:33:45
Working directory: /path/to/project
```

## Future Work
- [ ] Preset simplification (move tool lists out of preset system_prompt)
- [ ] Sub-agent guidance included by default
- [ ] `--no-project-context` flag for sub-agents to save tokens
- [ ] Bundle docs in wheel for PyPI distribution

## Files Changed
- `rollouts/paths.py` (new)
- `rollouts/prompt.py` (new)
- `rollouts/cli.py` (use dynamic prompt builder)
