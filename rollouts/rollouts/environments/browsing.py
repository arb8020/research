"""
Browsing environment with web search and fetch tools.

Tools: web_search (DuckDuckGo), web_fetch
For use with BrowseComp and other web research tasks.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlparse

import markdownify
import trio
from curl_cffi import requests as curl_requests

if TYPE_CHECKING:
    from ..frontends.tui.theme import Theme

from ..dtypes import (
    AgentState,
    Message,
    RunConfig,
    Tool,
    ToolCall,
    ToolFunction,
    ToolFunctionParameter,
    ToolResult,
)

# Web fetch constants (same as coding.py)
WEB_FETCH_MAX_SIZE = 10 * 1024 * 1024  # 10MB max download
WEB_FETCH_MAX_CONTENT = 100_000  # 100KB max content after conversion
WEB_FETCH_TIMEOUT = 30  # seconds

# Search constants
SEARCH_MAX_RESULTS = 10


async def _ddg_search(query: str, max_results: int = SEARCH_MAX_RESULTS) -> list[dict]:
    """Run DuckDuckGo search in thread pool (sync library)."""
    from ddgs import DDGS

    def _search() -> list[dict]:
        with DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    return await trio.to_thread.run_sync(_search)


def format_web_search(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Format web_search tool execution."""
    query = args.get("query", "")
    text = f"web_search(query={repr(query[:50] + '...' if len(query) > 50 else query)})"

    if result:
        from .coding import _get_text_output

        output = _get_text_output(result).strip()
        is_error = result.get("isError", False)

        if is_error:
            text += "\n⎿ Search failed"
            if output:
                text += f": {output[:100]}"
        else:
            # Count results
            lines = output.split("\n") if output else []
            result_count = sum(1 for line in lines if line.startswith("## "))
            text += f"\n⎿ Found {result_count} results"

    return text


def format_web_fetch(
    tool_name: str, args: dict, result: dict | None, expanded: bool, theme: Theme | None = None
) -> str:
    """Format web_fetch tool execution."""
    url = args.get("url", "")

    # Shorten URL for display
    try:
        parsed = urlparse(url)
        display_url = (
            f"{parsed.netloc}{parsed.path[:30]}..."
            if len(parsed.path) > 30
            else f"{parsed.netloc}{parsed.path}"
        )
    except Exception:
        display_url = url[:50] + "..." if len(url) > 50 else url

    text = f"web_fetch(url={repr(display_url)})"

    if result:
        from .coding import _get_text_output

        output = _get_text_output(result).strip()
        is_error = result.get("isError", False)

        if is_error:
            text += "\n⎿ Fetch failed"
            if output:
                text += f": {output[:100]}"
        else:
            lines = output.split("\n") if output else []
            text += f"\n⎿ Fetched {len(lines)} lines"

    return text


@dataclass
class BrowsingEnvironment:
    """Browsing environment with web search and fetch tools.

    Uses DuckDuckGo for search (no API key needed) and httpx for fetching.
    """

    max_search_results: int = field(default=SEARCH_MAX_RESULTS)

    def get_name(self) -> str:
        return "browsing"

    def get_status_info(self) -> dict[str, str] | None:
        return None

    async def serialize(self) -> dict:
        return {"env_kind": "browsing", "max_search_results": self.max_search_results}

    @staticmethod
    async def deserialize(data: dict) -> BrowsingEnvironment:
        return BrowsingEnvironment(
            max_search_results=data.get("max_search_results", SEARCH_MAX_RESULTS)
        )

    def requires_confirmation(self, tool_call: ToolCall) -> bool:
        return False

    def get_tool_formatter(
        self, tool_name: str
    ) -> Callable[[str, dict, dict | None, bool, Theme | None], str] | None:
        formatters = {
            "web_search": format_web_search,
            "web_fetch": format_web_fetch,
        }
        return formatters.get(tool_name)

    def get_tools(self) -> list[Tool]:
        return [
            # web_search tool
            Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web using DuckDuckGo. Returns titles, URLs, and snippets for top results.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": f"Maximum number of results (default: {SEARCH_MAX_RESULTS})",
                            },
                        },
                    ),
                    required=["query"],
                ),
            ),
            # web_fetch tool
            Tool(
                type="function",
                function=ToolFunction(
                    name="web_fetch",
                    description="Fetch content from a URL. Converts HTML to markdown. Use after web_search to read promising results.",
                    parameters=ToolFunctionParameter(
                        type="object",
                        properties={
                            "url": {
                                "type": "string",
                                "description": "The URL to fetch (must be valid http/https URL)",
                            },
                        },
                    ),
                    required=["url"],
                ),
            ),
        ]

    async def on_assistant_message(self, message: Message, state: AgentState) -> AgentState:
        return state

    async def exec_tool(
        self,
        tool_call: ToolCall,
        current_state: AgentState,
        run_config: RunConfig,
        cancel_scope: trio.CancelScope | None = None,
    ) -> ToolResult:
        try:
            if tool_call.name == "web_search":
                return await self._exec_web_search(tool_call)
            elif tool_call.name == "web_fetch":
                return await self._exec_web_fetch(tool_call)
            else:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Unknown tool: {tool_call.name}",
                )
        except trio.Cancelled:
            # Re-raise cancellation so agent loop can handle it
            raise
        except Exception as e:
            return ToolResult(tool_call_id=tool_call.id, is_error=True, content="", error=str(e))

    async def _exec_web_search(self, tool_call: ToolCall) -> ToolResult:
        """Execute DuckDuckGo search."""
        query = tool_call.args["query"]
        max_results = tool_call.args.get("max_results", self.max_search_results)

        try:
            results = await _ddg_search(query, max_results=max_results)
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Search failed: {e}",
            )

        if not results:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=False,
                content="No results found.",
            )

        # Format results as markdown
        output_lines = [f"Search results for: {query}\n"]
        for i, result in enumerate(results, 1):
            title = result.get("title", "No title")
            url = result.get("href", result.get("link", ""))
            snippet = result.get("body", result.get("snippet", ""))

            output_lines.append(f"## {i}. {title}")
            output_lines.append(f"URL: {url}")
            if snippet:
                output_lines.append(f"{snippet}")
            output_lines.append("")

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content="\n".join(output_lines),
        )

    async def _exec_web_fetch(self, tool_call: ToolCall) -> ToolResult:
        """Fetch content from URL, convert to markdown."""
        url = tool_call.args["url"]

        # Validate URL
        try:
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Invalid URL scheme: {parsed.scheme}. Must be http or https.",
                )
            if not parsed.netloc:
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error="Invalid URL: missing hostname",
                )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id, is_error=True, content="", error=f"Invalid URL: {e}"
            )

        # Upgrade http to https
        if parsed.scheme == "http":
            url = url.replace("http://", "https://", 1)

        # Fetch the URL using curl_cffi (impersonates browser TLS fingerprint)
        def _fetch() -> curl_requests.Response:
            return curl_requests.get(
                url,
                impersonate="chrome",
                timeout=WEB_FETCH_TIMEOUT,
                allow_redirects=True,
            )

        try:
            response = await trio.to_thread.run_sync(_fetch)
        except Exception as e:
            error_msg = str(e)
            if "timeout" in error_msg.lower():
                return ToolResult(
                    tool_call_id=tool_call.id,
                    is_error=True,
                    content="",
                    error=f"Request timed out after {WEB_FETCH_TIMEOUT} seconds",
                )
            return ToolResult(
                tool_call_id=tool_call.id, is_error=True, content="", error=f"Request failed: {e}"
            )

        if response.status_code >= 400:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"HTTP {response.status_code}",
            )

        # Check content size
        content_length = len(response.content)
        if content_length > WEB_FETCH_MAX_SIZE:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Content too large: {content_length} bytes (max {WEB_FETCH_MAX_SIZE})",
            )

        # Decode content
        try:
            text = response.text
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                is_error=True,
                content="",
                error=f"Failed to decode response: {e}",
            )

        # Convert HTML to markdown if needed
        content_type = response.headers.get("content-type", "")
        if "text/html" in content_type:
            try:
                text = markdownify.markdownify(text, heading_style="ATX", strip=["script", "style"])
            except Exception:
                pass

        # Truncate if too large
        if len(text) > WEB_FETCH_MAX_CONTENT:
            text = text[:WEB_FETCH_MAX_CONTENT] + "\n\n...[content truncated]"

        return ToolResult(
            tool_call_id=tool_call.id,
            is_error=False,
            content=f"URL: {url}\n\n---\n\n{text}",
        )
