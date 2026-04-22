#!/usr/bin/env python3
"""agent-blame HTTP server — serves attribution data + stubbed chat.

Divergent fork of ~/research/dev/pr-bot/server.py. Both servers are
small HTTP scaffolding around a domain-specific JSON payload (pr-bot:
diff; agent-blame: line attribution). Shared pattern: threaded stdlib
HTTP server, CORS, per-route handlers. Expected to share base later.

Endpoints:

    GET  /api/repo          -> repo-level metadata (root, git scope, source kind)
    GET  /api/files         -> list of files we have attribution for, with
                                per-file coverage summary
    GET  /api/blame?path=X  -> per-line attribution for file X
    POST /api/chat          -> stubbed (returns echo; LLM wiring later)
    POST /api/session       -> stubbed (load session messages for a given
                                FileEdit reference; UI side panel consumer)

Usage:

    python server.py <repo> [--sha <ref>] [--port 7979]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, str(Path(__file__).parent))

from agent_blame.adapters import claude_code, codex
from agent_blame.fold import fold_edits
from agent_blame.git_blame import CommitInfo, blame_file
from agent_blame.reconcile import (
    FileAttribution,
    reconcile,
)
from agent_blame.sources import SourceReader, git_sha_reader, git_timestamp_seeder, working_tree_reader
from agent_blame.transcript import TranscriptNotFound, load_transcript

logger = logging.getLogger(__name__)


def _build_attributions(
    repo_root: Path,
    git_root: Path,
    scope_rel: Path,
    source: SourceReader,
) -> list[FileAttribution]:
    """Same pipeline as cli.main, producing FileAttributions for the UI."""
    def _in_scope(p: str) -> bool:
        try:
            Path(p).resolve().relative_to(repo_root)
            return True
        except ValueError:
            return False

    edits = []
    for s in claude_code.list_sessions_for_cwd(repo_root):
        for e in claude_code.iter_file_edits(s):
            if _in_scope(e.path):
                edits.append(e)
    for s in codex.list_sessions_for_cwd(repo_root):
        for e in codex.iter_file_edits(s):
            if _in_scope(e.path):
                edits.append(e)
    logger.info("parsed %d edits in scope of %s", len(edits), repo_root)
    virtual_states = fold_edits(
        edits,
        seed_reader=source,
        timestamped_seeder=git_timestamp_seeder(git_root),
    )
    tracked = _tracked_text_files(git_root, scope_rel)
    return reconcile(
        repo_root=git_root,
        virtual_states=virtual_states,
        source=source,
        files=tracked,
    )


def _tracked_text_files(git_root: Path, scope: Path) -> list[Path]:
    cmd = ["git", "-C", str(git_root), "ls-files"]
    if str(scope) != ".":
        cmd.append(str(scope))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return [git_root / line for line in result.stdout.splitlines() if line]


def _commit_info_to_json(c: CommitInfo | None) -> dict | None:
    if c is None:
        return None
    return {
        "sha": c.sha,
        "short_sha": c.short_sha,
        "author_name": c.author_name,
        "author_email": c.author_email,
        "timestamp": c.author_time_iso,
        "summary": c.summary,
        "agent_marker": c.agent_marker,
    }


def _attribution_to_json(
    fa: FileAttribution,
    commits: list[CommitInfo | None] | None = None,
) -> dict:
    """Shape the UI consumes. One record per line.

    Each line has:
      - `edit`: session-level attribution (None for unknown), from fold
      - `commit`: git-blame fallback (None if git blame failed), always
        provided so the UI can show *something* even for unknown edits

    `commits` is the parallel list from git blame, indexed 0-based by
    line. If None, the commit fields are omitted entirely (back-compat
    with callers not passing it).
    """
    lines = []
    for i, line in enumerate(fa.lines):
        commit = None
        if commits is not None and i < len(commits):
            commit = _commit_info_to_json(commits[i])
        if line.edit is None:
            entry = {
                "n": line.line_number,
                "text": line.text,
                "edit": None,
                "ambiguous": False,
                "match_kind": line.match_kind,  # 'unknown'
            }
        else:
            entry = {
                "n": line.line_number,
                "text": line.text,
                "edit": {
                    "source": line.edit.source,
                    "session_id": line.edit.session_id,
                    "tool_call_id": line.edit.tool_call_id,
                    "timestamp": line.edit.timestamp.isoformat(),
                },
                "ambiguous": line.ambiguous,
                "match_kind": line.match_kind,
            }
        if commit is not None:
            entry["commit"] = commit
        lines.append(entry)
    return {
        "path": str(fa.repo_path),
        "lines": lines,
        "attributed": fa.attributed_count,
        "unknown": fa.unknown_count,
    }


def _files_summary(attributions: list[FileAttribution]) -> list[dict]:
    """File-list payload: one row per file, no per-line content."""
    out = []
    for fa in attributions:
        # Top contributing session per file
        counter: dict[tuple[str, str], int] = defaultdict(int)
        for line in fa.lines:
            if line.edit is not None:
                counter[(line.edit.source, line.edit.session_id)] += 1
        top = max(counter.items(), key=lambda kv: kv[1], default=None)
        out.append({
            "path": str(fa.repo_path),
            "total_lines": len(fa.lines),
            "attributed": fa.attributed_count,
            "unknown": fa.unknown_count,
            "top_session": (
                {"source": top[0][0], "session_id": top[0][1], "lines": top[1]}
                if top else None
            ),
        })
    return out


def _sessions_summary(attributions: list[FileAttribution]) -> list[dict]:
    """Sessions view: one row per (source, session_id) with the files it touched.

    Each row: {source, session_id, total_lines, files: [{path, lines}...]}.
    Sorted by total attributed lines, desc. Files within a session sorted
    same way.
    """
    # (source, session_id) -> {path -> line count}
    agg: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    latest: dict[tuple[str, str], str] = {}
    for fa in attributions:
        for line in fa.lines:
            if line.edit is None:
                continue
            key = (line.edit.source, line.edit.session_id)
            agg[key][str(fa.repo_path)] += 1
            # Track latest edit timestamp per session for display/sort tiebreak
            ts = line.edit.timestamp.isoformat()
            if key not in latest or ts > latest[key]:
                latest[key] = ts
    out = []
    for (source, sid), files_map in agg.items():
        files = sorted(
            ({"path": p, "lines": n} for p, n in files_map.items()),
            key=lambda f: -f["lines"],
        )
        total = sum(f["lines"] for f in files)
        out.append({
            "source": source,
            "session_id": sid,
            "total_lines": total,
            "latest_edit": latest.get((source, sid)),
            "files": files,
        })
    out.sort(key=lambda s: -s["total_lines"])
    return out


def make_handler(
    *,
    repo_root: Path,
    git_root: Path,
    scope_rel: Path,
    attributions: list[FileAttribution],
    sha: str | None,
):
    by_path = {str(fa.repo_path): fa for fa in attributions}
    repo_info = {
        "repo_root": str(repo_root),
        "git_root": str(git_root),
        "scope": str(scope_rel),
        "source": "git_sha" if sha else "working_tree",
        "sha": sha,
        "file_count": len(attributions),
    }
    # Per-path cache of git-blame output. Populated lazily on /api/blame.
    # `git blame --line-porcelain` takes a few hundred ms on a 2000-line
    # file; caching keeps repeat hits instant.
    blame_cache: dict[str, list[CommitInfo | None]] = {}

    def _git_blame_for(rel_path: str) -> list[CommitInfo | None]:
        cached = blame_cache.get(rel_path)
        if cached is not None:
            return cached
        commits = blame_file(git_root, rel_path, ref=sha)
        blame_cache[rel_path] = commits
        return commits

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            logger.info(f"[{self.address_string()}] {fmt % args}")

        def _send_json(self, data, status=200):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)

        def _send_404(self, msg: str = "not found"):
            self._send_json({"error": msg}, status=404)

        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            parsed = urlparse(self.path)
            route = parsed.path
            params = parse_qs(parsed.query)

            if route == "/api/repo":
                self._send_json(repo_info)
                return
            if route == "/api/files":
                self._send_json({"files": _files_summary(attributions)})
                return
            if route == "/api/sessions":
                self._send_json({"sessions": _sessions_summary(attributions)})
                return
            if route == "/api/blame":
                path = (params.get("path") or [""])[0]
                fa = by_path.get(path)
                if fa is None:
                    self._send_404(f"no attribution for {path!r}")
                    return
                commits = _git_blame_for(path)
                self._send_json(_attribution_to_json(fa, commits))
                return
            if route == "/api/session":
                source = (params.get("source") or [""])[0]
                session_id = (params.get("session_id") or [""])[0]
                if not source or not session_id:
                    self._send_json({"error": "need ?source=X&session_id=Y"}, status=400)
                    return
                try:
                    messages = load_transcript(source, session_id)
                except TranscriptNotFound as e:
                    self._send_404(str(e))
                    return
                except ValueError as e:
                    self._send_json({"error": str(e)}, status=400)
                    return
                self._send_json({
                    "source": source,
                    "session_id": session_id,
                    "messages": messages,
                })
                return
            self._send_404()

        def do_POST(self):
            if self.path == "/api/chat":
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length))
                message = body.get("message", "")
                # TODO: wire to Claude API once we decide on context packing.
                self._send_json({
                    "response": f"[stub] received {message!r}. Chat not wired up yet.",
                })
                return
            self._send_404()

    return Handler


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", type=Path, help="repo root (or subdir of a git repo)")
    parser.add_argument("--sha", type=str, default=None,
                        help="serve attribution against this SHA (default: working tree)")
    parser.add_argument("--port", type=int, default=7979)
    args = parser.parse_args(argv)

    repo_root = args.repo.resolve()
    try:
        toplevel = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "--show-toplevel"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        print(f"error: {repo_root} is not inside a git repo", file=sys.stderr)
        return 2
    git_root = Path(toplevel)
    scope_rel = repo_root.relative_to(git_root) if repo_root != git_root else Path(".")

    if args.sha:
        try:
            source: SourceReader = git_sha_reader(args.sha, git_root)
        except ValueError as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
    else:
        source = working_tree_reader()

    logger.info("building attributions (this may take a few seconds)...")
    attributions = _build_attributions(repo_root, git_root, scope_rel, source)
    logger.info("ready: %d files", len(attributions))

    handler = make_handler(
        repo_root=repo_root, git_root=git_root, scope_rel=scope_rel,
        attributions=attributions, sha=args.sha,
    )
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"agent-blame api: http://127.0.0.1:{args.port}", flush=True)
    print(f"endpoints: /api/repo  /api/files  /api/blame?path=...  /api/chat  /api/session", flush=True)
    print(f"vite dev server: cd ui && npm install && npm run dev", flush=True)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
