from __future__ import annotations

import json
from dataclasses import asdict

from rollouts.drivers.claude import _ClaudeEventParser
from rollouts.drivers.codex import _CodexEventParser
from rollouts.drivers.runner import _EventAccumulator
from rollouts.drivers.session_adapter import claude_session_to_messages, codex_session_to_messages


def _messages_from_live_entries(parser, entries: list[dict]) -> list[dict]:
    accumulator = _EventAccumulator()
    for entry in entries:
        for event in parser.parse(entry):
            accumulator.handle(event)
    return [asdict(message) for message in accumulator.finalize()]


def _messages_from_session_file(tmp_path, name: str, entries: list[dict], loader) -> list[dict]:
    session_path = tmp_path / name
    session_path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    return [asdict(message) for message in loader(session_path)]


def test_claude_session_import_matches_live_parser(tmp_path) -> None:
    live_entries = [
        {"type": "system", "subtype": "init", "session_id": "sess-1", "model": "sonnet"},
        {
            "type": "assistant",
            "message": {
                "model": "sonnet",
                "content": [
                    {"type": "text", "text": "I will inspect the file."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "perf_takehome.py"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "tool_use_id": "toolu_1",
            "content": "file contents",
            "is_error": False,
        },
    ]
    session_entries = [
        {"type": "summary", "summary": "ignore"},
        {
            "type": "assistant",
            "timestamp": "2026-03-18T20:00:00Z",
            "message": {
                "model": "sonnet",
                "content": [
                    {"type": "text", "text": "I will inspect the file."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"path": "perf_takehome.py"}},
                ],
            },
        },
        {
            "type": "tool_result",
            "timestamp": "2026-03-18T20:00:01Z",
            "tool_use_id": "toolu_1",
            "content": "file contents",
            "is_error": False,
        },
    ]

    live_messages = _messages_from_live_entries(_ClaudeEventParser(), live_entries)
    session_messages = _messages_from_session_file(
        tmp_path,
        "claude.jsonl",
        session_entries,
        claude_session_to_messages,
    )

    for message in session_messages:
        message.pop("timestamp", None)

    assert session_messages == live_messages


def test_codex_session_import_matches_live_parser(tmp_path) -> None:
    entries = [
        {"type": "session_meta", "payload": {"id": "sess-2", "model": "gpt-5.1-codex-mini"}},
        {
            "type": "response_item",
            "payload": {
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": "Need to inspect the file first."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "I will inspect `perf_takehome.py`."}],
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read",
                "arguments": json.dumps({"path": "perf_takehome.py"}),
            },
        },
        {
            "type": "response_item",
            "payload": {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Chunk ID: abc\nWall time: 0.1 seconds\nProcess exited with code 0\nOriginal token count: 1\nOutput:\nfile contents\n",
            },
        },
    ]

    live_messages = _messages_from_live_entries(_CodexEventParser(), entries)
    session_messages = _messages_from_session_file(
        tmp_path,
        "codex.jsonl",
        entries,
        codex_session_to_messages,
    )

    assert session_messages == live_messages
