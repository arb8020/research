"""Dynamic system prompt building.

Builds system prompts with:
- Actual tools from environment (not hardcoded lists)
- Dynamic guidelines based on tool availability
- Project context files (AGENTS.md, ROLLOUTS.md)
- Self-documentation paths
- Runtime context (datetime, cwd)
"""

from datetime import datetime
from pathlib import Path

from .dtypes import Tool
from .paths import get_docs_dir, get_readme_path, get_version

# =============================================================================
# Project Context Discovery
# =============================================================================

# Files to look for, in priority order (first found wins per directory)
PROJECT_CONTEXT_FILES = ["ROLLOUTS.md", "AGENTS.md", "CLAUDE.md"]


def load_project_context(cwd: Path) -> list[tuple[Path, str]]:
    """Load project context files walking up from cwd.

    Looks for ROLLOUTS.md, AGENTS.md, or CLAUDE.md in each directory
    from cwd up to root. Returns list of (path, content) tuples,
    ordered from root to cwd (so more specific context comes last).

    Args:
        cwd: Starting directory

    Returns:
        List of (path, content) tuples, root-first order

    # TODO: Add global context from ~/.rollouts/AGENTS.md (or CLAUDE.md)
    # See badlogic/pi-mono's loadProjectContextFiles for reference:
    # 1. Load global context from agentDir first (~/.rollouts/)
    # 2. Then walk up from cwd to root
    # This allows user-wide defaults that project-specific files can override.
    """
    assert cwd, "cwd required"
    assert isinstance(cwd, Path), "cwd must be a Path"

    context_files: list[tuple[Path, str]] = []
    seen_paths: set[Path] = set()

    current = cwd.resolve()
    root = Path(current.anchor)

    while current >= root:
        for name in PROJECT_CONTEXT_FILES:
            ctx_file = current / name
            if ctx_file.exists() and ctx_file not in seen_paths:
                try:
                    content = ctx_file.read_text()
                    context_files.append((ctx_file, content))
                    seen_paths.add(ctx_file)
                    break  # Only one per directory
                except (OSError, PermissionError):
                    pass

        parent = current.parent
        if parent == current:
            break
        current = parent

    # Return in root-first order (reverse of how we collected)
    return list(reversed(context_files))


# =============================================================================
# Tool Descriptions
# =============================================================================


TOOL_DESCRIPTIONS = {
    # Coding env
    "read": "Read file contents (supports offset/limit for large files)",
    "write": "Write content to a file (creates directories automatically)",
    "edit": "Replace exact text in a file (must be unique match)",
    "bash": "Execute shell commands",
    "web_fetch": "Fetch and extract info from URLs (converts HTML to markdown)",
    # Calculator env
    "add": "Add a value to the running total",
    "subtract": "Subtract a value from the running total",
    "multiply": "Multiply the running total by a value",
    "divide": "Divide the running total by a value",
    "clear": "Reset the running total to 0",
    "complete_task": "Submit the final answer",
    # REPL env
    "repl": "Execute Python code",
    "llm_query": "Query a sub-LLM for semantic tasks",
    "final_answer": "Submit your final answer",
}

# =============================================================================
# Base Prompts (personality/style only, no tool lists)
# =============================================================================

BASE_PROMPTS = {
    "none": "You are a helpful assistant.",
    "calculator": """You are a calculator assistant. Each tool operates on a running total (starts at 0).

For calculations:
1. Break down the problem into steps
2. Use tools to compute each step
3. Use complete_task when done

Example: For "(5 + 3) * 2", first add(5), then add(3), then multiply(2).""",
    "coding": """You are a coding assistant with access to file and shell tools.

When working on code:
1. First read relevant files to understand context
2. Make precise edits using the edit tool
3. Use bash to run tests, linting, etc.
4. Prefer small, focused changes over large rewrites""",
    "git": """You are a coding assistant with access to file and shell tools.

All file changes are automatically tracked in an isolated git history.
This gives you full undo capability - every write/edit/bash creates a commit.

When working on code:
1. First read relevant files to understand context
2. Make precise edits using the edit tool
3. Use bash to run tests, linting, etc.
4. Prefer small, focused changes over large rewrites""",
    "repl": """You are an assistant with access to a REPL environment for processing large contexts.

The input context is stored in a Python variable called `context`. You explore it programmatically.

Strategy:
1. Peek first: context[:1000], len(context)
2. Search: re.findall(pattern, context), list comprehensions
3. Chunk for semantics: llm_query("Classify: " + chunk)
4. Answer: final_answer(your_result)""",
}


# =============================================================================
# Dynamic Prompt Building
# =============================================================================


def format_tool_list(tools: list[Tool]) -> str:
    """Format tool list for system prompt."""
    lines = []
    for tool in tools:
        name = tool.function.name
        desc = TOOL_DESCRIPTIONS.get(name, tool.function.description or "")
        lines.append(f"- {name}: {desc}")
    return "\n".join(lines)


def build_tool_guidelines(tools: list[Tool]) -> list[str]:
    """Generate guidelines based on enabled tools."""
    tool_names = {t.function.name for t in tools}
    guidelines = []

    has_read = "read" in tool_names
    has_write = "write" in tool_names
    has_edit = "edit" in tool_names
    has_bash = "bash" in tool_names

    # State invariants positively (Tiger Style)
    can_modify = has_write or has_edit
    is_readonly = has_read and not can_modify

    # Read-only mode notice
    if is_readonly:
        if has_bash:
            guidelines.append(
                "You are in READ-ONLY mode with bash access. "
                "Use bash ONLY for read-only operations (git log, grep, find, etc.) - do NOT modify any files."
            )
        else:
            guidelines.append("You are in READ-ONLY mode - you cannot modify files.")

    # Read before edit
    if has_read and has_edit:
        guidelines.append("Read files before editing - edit requires exact text matches.")

    # Write vs edit
    if has_write and has_edit:
        guidelines.append(
            "Use edit for surgical changes, write only for new files or complete rewrites."
        )

    return guidelines


def build_self_doc_section() -> str:
    """Build section pointing to rollouts documentation."""
    readme = get_readme_path()
    docs = get_docs_dir()
    version = get_version()

    return f"""## About rollouts (your CLI)

Version: {version}
Documentation: {readme}
Detailed docs: {docs}

If asked about your capabilities, read these files."""


def build_system_prompt(
    env_name: str,
    tools: list[Tool],
    cwd: Path | None = None,
    base_prompt: str | None = None,
    env_system_prompt: str | None = None,
    include_self_docs: bool = True,
    include_project_context: bool = True,
) -> str:
    """Build complete system prompt with dynamic tool info.

    Args:
        env_name: Environment name (coding, git, calculator, etc.)
        tools: Actual tools from environment.get_tools()
        cwd: Working directory (defaults to current)
        base_prompt: Override base prompt (e.g., from preset)
        env_system_prompt: Environment-provided system prompt (from env.get_system_prompt())
        include_self_docs: Whether to include rollouts documentation paths
        include_project_context: Whether to load AGENTS.md/ROLLOUTS.md files
    """
    # Assertions (Tiger Style: 2+ per function, split compound)
    assert env_name, "env_name required"
    assert isinstance(tools, list), "tools must be a list"

    sections = []
    working_dir = cwd or Path.cwd()

    # 1. Base prompt (personality/style)
    base = base_prompt or BASE_PROMPTS.get(env_name, BASE_PROMPTS["none"])
    sections.append(base)

    # 2. Environment-provided system prompt (explains paradigm, strategies, etc.)
    if env_system_prompt:
        sections.append(env_system_prompt)

    # 3. Actual tool list
    if tools:
        tool_list = format_tool_list(tools)
        sections.append(f"Available tools:\n{tool_list}")

    # 4. Dynamic guidelines
    guidelines = build_tool_guidelines(tools)
    if guidelines:
        sections.append("Guidelines:\n" + "\n".join(f"- {g}" for g in guidelines))

    # 5. Self-documentation
    if include_self_docs:
        sections.append(build_self_doc_section())

    # 6. Project context files (AGENTS.md, ROLLOUTS.md, etc.)
    if include_project_context:
        context_files = load_project_context(working_dir)
        if context_files:
            ctx_section = "# Project Context\n\nThe following context files were found:\n"
            for path, content in context_files:
                ctx_section += f"\n## {path}\n\n{content}\n"
            sections.append(ctx_section)

    # 7. Runtime context
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sections.append(f"Current time: {now}\nWorking directory: {working_dir}")

    return "\n\n".join(sections)
