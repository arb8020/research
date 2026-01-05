"""SWE-grep environment for code/document retrieval evaluation.

Provides grep, glob, semantic search, read, and submit tools for agents to find
relevant code/documentation and submit structured answers with source citations.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rollouts.dtypes import Tool, ToolCall, ToolFunction, ToolFunctionParameter


@dataclass
class SWEGrepEnvironment:
    """Environment for SWE-grep style retrieval evaluation.

    Provides tools:
    - grep(pattern, path): Regex/literal search in files
    - glob(pattern): Find files by pattern
    - search(query, top_k): Semantic/TF-IDF search
    - read(path, start_line, end_line): Read file content
    - submit(sources, answer): Submit final answer with citations
    """

    corpus_path: Path
    """Path to document corpus directory."""

    use_semantic_search: bool = True
    """Whether to use semantic search (requires embeddings) or TF-IDF."""

    max_results: int = 50
    """Maximum results to return from grep/glob/search."""

    max_file_lines: int = 5000
    """Maximum lines to read from a single file."""

    # Internal state
    _submitted: bool = field(default=False, init=False)
    _submission: dict[str, Any] | None = field(default=None, init=False)
    _tool_calls: list[dict[str, Any]] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        """Validate corpus path exists."""
        if not self.corpus_path.exists():
            raise ValueError(f"Corpus path does not exist: {self.corpus_path}")
        if not self.corpus_path.is_dir():
            raise ValueError(f"Corpus path is not a directory: {self.corpus_path}")

    def get_tools(self) -> list[Tool]:
        """Return all available tools."""
        return [
            self._grep_tool(),
            self._glob_tool(),
            self._search_tool(),
            self._read_tool(),
            self._submit_tool(),
        ]

    def _grep_tool(self) -> Tool:
        """Grep tool for regex/literal search."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="grep",
                description=(
                    "Search for a pattern in files using regex or literal string matching. "
                    "Returns list of files containing matches with line numbers and excerpts. "
                    "Use this to find files containing specific strings, function names, or patterns."
                ),
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Regex pattern or literal string to search for. "
                                "Examples: 'def.*TMA', 'class GemmKernel', 'shared memory'"
                            ),
                        },
                        "path": {
                            "type": "string",
                            "description": (
                                "Path to search in (relative to corpus root). "
                                "Can be a file or directory. Defaults to entire corpus. "
                                "Examples: 'cutlass-docs/', 'src/gemm/', 'README.md'"
                            ),
                        },
                        "case_sensitive": {
                            "type": "boolean",
                            "description": "Whether search is case-sensitive (default: false)",
                        },
                        "literal": {
                            "type": "boolean",
                            "description": "Treat pattern as literal string, not regex (default: false)",
                        },
                    },
                ),
                required=["pattern"],
            ),
        )

    def _glob_tool(self) -> Tool:
        """Glob tool for finding files by pattern."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="glob",
                description=(
                    "Find files matching a glob pattern. "
                    "Returns list of file paths. "
                    "Use this to find files by name, extension, or directory structure."
                ),
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "pattern": {
                            "type": "string",
                            "description": (
                                "Glob pattern to match files. "
                                "Examples: '**/*.md', 'src/**/gemm*.cpp', '**/test_*.py'"
                            ),
                        },
                    },
                ),
                required=["pattern"],
            ),
        )

    def _search_tool(self) -> Tool:
        """Semantic/TF-IDF search tool."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="search",
                description=(
                    "Semantic search over document corpus. "
                    "Returns ranked list of relevant documents with excerpts. "
                    "Use this to find documents by topic, concept, or natural language query."
                ),
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "query": {
                            "type": "string",
                            "description": (
                                "Natural language search query. "
                                "Examples: 'TMA configuration for shared memory', "
                                "'warp-specialized GEMM performance', 'CuTe layout broadcasting'"
                            ),
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return (default: 10)",
                        },
                    },
                ),
                required=["query"],
            ),
        )

    def _read_tool(self) -> Tool:
        """Read file content tool."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="read",
                description=(
                    "Read the full content of a file from the corpus. "
                    "Optionally specify line range to read specific sections. "
                    "Use this after finding relevant files to examine their content."
                ),
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "path": {
                            "type": "string",
                            "description": (
                                "Path to file (relative to corpus root). "
                                "Example: 'cutlass-docs/docs/cute/04-algorithms.md'"
                            ),
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Start line number (1-indexed, inclusive). Defaults to 1.",
                        },
                        "end_line": {
                            "type": "integer",
                            "description": "End line number (1-indexed, inclusive). Defaults to end of file.",
                        },
                    },
                ),
                required=["path"],
            ),
        )

    def _submit_tool(self) -> Tool:
        """Submit final answer tool."""
        return Tool(
            type="function",
            function=ToolFunction(
                name="submit",
                description=(
                    "Submit your final answer with source citations. "
                    "This ends the evaluation. Call this when you have found all relevant sources "
                    "and synthesized a complete answer to the query."
                ),
                parameters=ToolFunctionParameter(
                    type="object",
                    properties={
                        "sources": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "file": {
                                        "type": "string",
                                        "description": "File path relative to corpus root",
                                    },
                                    "start_line": {
                                        "type": "integer",
                                        "description": "Start line number (1-indexed)",
                                    },
                                    "end_line": {
                                        "type": "integer",
                                        "description": "End line number (1-indexed)",
                                    },
                                },
                                "required": ["file", "start_line", "end_line"],
                            },
                            "description": (
                                "Ordered list of source citations (file + line ranges) that support your answer. "
                                "Order by relevance (most relevant first)."
                            ),
                        },
                        "answer": {
                            "type": "string",
                            "description": (
                                "Natural language answer to the query, synthesized from the sources. "
                                "Should be clear, concise, and directly address the question."
                            ),
                        },
                    },
                ),
                required=["sources", "answer"],
            ),
        )

    async def handle_tool_call(self, tool_call: ToolCall) -> str:
        """Handle a tool call and return the result."""
        if self._submitted:
            return "Error: Already submitted answer. Cannot make more tool calls."

        # Log tool call for analysis
        self._tool_calls.append({
            "tool": tool_call.name,
            "args": tool_call.arguments,
        })

        # Dispatch to handler
        if tool_call.name == "grep":
            return await self._handle_grep(tool_call)
        elif tool_call.name == "glob":
            return await self._handle_glob(tool_call)
        elif tool_call.name == "search":
            return await self._handle_search(tool_call)
        elif tool_call.name == "read":
            return await self._handle_read(tool_call)
        elif tool_call.name == "submit":
            return await self._handle_submit(tool_call)
        else:
            return f"Error: Unknown tool '{tool_call.name}'"

    async def _handle_grep(self, tool_call: ToolCall) -> str:
        """Handle grep tool call."""
        args = tool_call.arguments
        pattern = args.get("pattern")
        search_path = args.get("path", ".")
        case_sensitive = args.get("case_sensitive", False)
        literal = args.get("literal", False)

        if not pattern:
            return "Error: 'pattern' is required"

        # Build regex
        if literal:
            regex_pattern = re.escape(pattern)
        else:
            regex_pattern = pattern

        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            regex = re.compile(regex_pattern, flags)
        except re.error as e:
            return f"Error: Invalid regex pattern: {e}"

        # Search files
        full_path = self.corpus_path / search_path
        if not full_path.exists():
            return f"Error: Path does not exist: {search_path}"

        matches = []
        files_to_search = []

        if full_path.is_file():
            files_to_search = [full_path]
        else:
            files_to_search = list(full_path.rglob("*"))
            files_to_search = [f for f in files_to_search if f.is_file()]

        for file_path in files_to_search[: self.max_results]:
            try:
                with open(file_path, encoding="utf-8", errors="ignore") as f:
                    for line_no, line in enumerate(f, 1):
                        if regex.search(line):
                            rel_path = file_path.relative_to(self.corpus_path)
                            matches.append({
                                "file": str(rel_path),
                                "line": line_no,
                                "content": line.rstrip()[:200],  # Truncate long lines
                            })
                            if len(matches) >= self.max_results:
                                break
            except Exception:
                continue  # Skip files that can't be read

            if len(matches) >= self.max_results:
                break

        if not matches:
            return f"No matches found for pattern: {pattern}"

        # Format results
        result_lines = [f"Found {len(matches)} matches:"]
        for match in matches:
            result_lines.append(f"  {match['file']}:{match['line']}: {match['content']}")

        return "\n".join(result_lines)

    async def _handle_glob(self, tool_call: ToolCall) -> str:
        """Handle glob tool call."""
        args = tool_call.arguments
        pattern = args.get("pattern")

        if not pattern:
            return "Error: 'pattern' is required"

        # Find matching files
        try:
            matches = list(self.corpus_path.glob(pattern))
            matches = [f for f in matches if f.is_file()]
            matches = matches[: self.max_results]
        except Exception as e:
            return f"Error: Invalid glob pattern: {e}"

        if not matches:
            return f"No files found matching pattern: {pattern}"

        # Format results
        rel_paths = [str(f.relative_to(self.corpus_path)) for f in matches]
        result_lines = [f"Found {len(rel_paths)} files:"]
        result_lines.extend(f"  {p}" for p in rel_paths)

        return "\n".join(result_lines)

    async def _handle_search(self, tool_call: ToolCall) -> str:
        """Handle search tool call."""
        args = tool_call.arguments
        query = args.get("query")
        top_k = args.get("top_k", 10)

        if not query:
            return "Error: 'query' is required"

        # TODO: Implement TF-IDF or semantic search
        # For now, return a placeholder
        return "Error: Search not yet implemented. Use grep or glob to find files, then read them."

    async def _handle_read(self, tool_call: ToolCall) -> str:
        """Handle read tool call."""
        args = tool_call.arguments
        path = args.get("path")
        start_line = args.get("start_line", 1)
        end_line = args.get("end_line")

        if not path:
            return "Error: 'path' is required"

        full_path = self.corpus_path / path
        if not full_path.exists():
            return f"Error: File does not exist: {path}"
        if not full_path.is_file():
            return f"Error: Path is not a file: {path}"

        # Read file
        try:
            with open(full_path, encoding="utf-8", errors="ignore") as f:
                lines = f.readlines()
        except Exception as e:
            return f"Error reading file: {e}"

        # Apply line range
        if end_line is None:
            end_line = len(lines)

        start_idx = max(0, start_line - 1)
        end_idx = min(len(lines), end_line)

        if start_idx >= len(lines):
            return f"Error: start_line {start_line} exceeds file length ({len(lines)} lines)"

        selected_lines = lines[start_idx:end_idx]

        if len(selected_lines) > self.max_file_lines:
            return (
                f"Error: Requested {len(selected_lines)} lines, "
                f"but max is {self.max_file_lines}. "
                f"Use smaller line ranges."
            )

        # Format with line numbers
        result_lines = [f"{path} (lines {start_line}-{end_idx}):"]
        for i, line in enumerate(selected_lines, start_line):
            result_lines.append(f"{i:4d} | {line.rstrip()}")

        return "\n".join(result_lines)

    async def _handle_submit(self, tool_call: ToolCall) -> str:
        """Handle submit tool call."""
        args = tool_call.arguments
        sources = args.get("sources", [])
        answer = args.get("answer")

        if not sources:
            return "Error: 'sources' is required and must be non-empty"
        if not answer:
            return "Error: 'answer' is required"

        # Validate sources format
        for i, source in enumerate(sources):
            if not isinstance(source, dict):
                return f"Error: source {i} must be an object"
            if "file" not in source:
                return f"Error: source {i} missing 'file'"
            if "start_line" not in source:
                return f"Error: source {i} missing 'start_line'"
            if "end_line" not in source:
                return f"Error: source {i} missing 'end_line'"

        # Mark as submitted
        self._submitted = True
        self._submission = {
            "sources": sources,
            "answer": answer,
        }

        return "Answer submitted successfully. Evaluation complete."

    @classmethod
    async def deserialize(cls, data: dict[str, Any]) -> SWEGrepEnvironment:
        """Deserialize environment from dict."""
        return cls(
            corpus_path=Path(data["corpus_path"]),
            use_semantic_search=data.get("use_semantic_search", True),
            max_results=data.get("max_results", 50),
            max_file_lines=data.get("max_file_lines", 5000),
        )
