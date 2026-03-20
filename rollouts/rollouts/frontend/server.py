#!/usr/bin/env python3
"""HTTP server for the rollouts run viewer.

Provides:
- Static UI serving
- Run listing and inspection APIs
- Live updates for active runs

Usage:
    python -m rollouts.frontend.server
    python -m rollouts.frontend.server --port 8080
    python -m rollouts.frontend.server --project ~/myproject
"""

import argparse
import json
import logging
import re
import webbrowser
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from ..export_html import run_to_html, sample_to_html
from .artifacts import (
    find_trace_dir,
    load_sample_payload,
    load_trace_payload,
)
from .live_runs import (
    active_run_ids,
    discover_watching_runs,
    get_run,
    kill_run,
    list_registered_runs,
)
from .live_streams import stream_registered_run, stream_watch_run
from .tags import build_live_run_tags, build_run_tags, update_user_tags
from .workspace_views import load_workspace_response

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Route:
    method: str
    pattern: re.Pattern[str]
    handler_name: str


GET_ROUTES = [
    Route("GET", re.compile(r"^/$"), "_serve_ui_index"),
    Route("GET", re.compile(r"^/index\.html$"), "_serve_ui_index"),
    Route("GET", re.compile(r"^/(?P<asset_path>assets/.+)$"), "_serve_ui_static"),
    Route("GET", re.compile(r"^/api/runs$"), "_list_runs"),
    Route("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/events$"), "_stream_run_events"),
    Route("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/export\.html$"), "_export_run_html"),
    Route(
        "GET",
        re.compile(r"^/api/runs/(?P<run_id>[^/]+)/samples/(?P<sample_id>[^/]+)/workspace$"),
        "_get_workspace",
    ),
    Route(
        "GET",
        re.compile(r"^/api/runs/(?P<run_id>[^/]+)/samples/(?P<sample_id>[^/]+)/export\.html$"),
        "_export_sample_html",
    ),
    Route(
        "GET",
        re.compile(r"^/api/runs/(?P<run_id>[^/]+)/samples/(?P<sample_id>[^/]+)$"),
        "_get_sample",
    ),
    Route("GET", re.compile(r"^/api/runs/(?P<run_id>[^/]+)$"), "_get_run"),
    Route("GET", re.compile(r"^/api/results-dirs$"), "_list_results_dirs"),
]

POST_ROUTES = [
    Route("POST", re.compile(r"^/api/set-results-dir$"), "_set_results_dir"),
    Route("POST", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/kill$"), "_kill_run"),
    Route("POST", re.compile(r"^/api/runs/(?P<run_id>[^/]+)/tags$"), "_set_run_tags"),
]


class RolloutsViewerServer(SimpleHTTPRequestHandler):
    """HTTP server for the rollouts run viewer.

    Serves the frontend UI and provides APIs for:
    - /api/runs - List visible runs
    - /api/runs/{id} - Load a saved run
    - /api/runs/{id}/events - Live SSE updates when available
    """

    # Class variables (set by main())
    results_dir: Path = Path.cwd() / "results"
    known_results_dirs: list[Path] = []

    def log_message(self, format: str, *args: object) -> None:
        """Override to use our logger instead of stderr."""
        logger.info(f"{self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        """Handle GET requests."""
        try:
            self._do_GET_inner()
        except Exception:
            logger.exception(f"Unhandled error in GET {self.path}")
            try:
                self.send_error(500, "Internal server error")
            except Exception:
                pass

    def _do_GET_inner(self) -> None:
        parsed = urlparse(self.path)
        self._dispatch_route("GET", parsed.path, GET_ROUTES)

    def do_POST(self) -> None:
        """Handle POST requests."""
        try:
            self._do_POST_inner()
        except Exception:
            logger.exception(f"Unhandled error in POST {self.path}")
            try:
                self.send_error(500, "Internal server error")
            except Exception:
                pass

    def _do_POST_inner(self) -> None:
        parsed = urlparse(self.path)
        self._dispatch_route("POST", parsed.path, POST_ROUTES)

    def _dispatch_route(self, method: str, path: str, routes: list[Route]) -> None:
        for route in routes:
            if route.method != method:
                continue
            match = route.pattern.match(path)
            if match is None:
                continue
            handler = getattr(self, route.handler_name)
            handler(**match.groupdict())
            return
        self.send_error(404, "Not found")

    def _serve_ui_index(self) -> None:
        """Serve the built React UI from ui/dist/index.html."""
        ui_dist = Path(__file__).parent / "ui" / "dist" / "index.html"
        if not ui_dist.exists():
            self.send_error(
                404, "UI build not found. Run `npm --prefix rollouts/frontend/ui run build`."
            )
            return

        content = ui_dist.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _serve_ui_static(self, asset_path: str) -> None:
        """Serve static assets from ui/dist/assets/."""
        import mimetypes

        asset_file = Path(__file__).parent / "ui" / "dist" / asset_path

        if not asset_file.exists() or not asset_file.is_file():
            self.send_error(404, f"Asset not found: {asset_path}")
            return

        content = asset_file.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(asset_file))
        self.send_response(200)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(content)

    def _resolve_run_dir(self, run_id: str) -> Path | None:
        return find_trace_dir(
            self.results_dir,
            list(self.__class__.known_results_dirs),
            run_id,
        )

    def _list_runs(self) -> None:
        """List saved runs and live/watchable runs in one normalized payload."""
        results_dir = self.results_dir

        runs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for search_dir in [results_dir, *list(self.__class__.known_results_dirs)]:
            if not search_dir.exists():
                continue
            for run_dir in sorted(search_dir.iterdir(), reverse=True):
                if not run_dir.is_dir():
                    continue
                report_path = run_dir / "report.json"
                if not report_path.exists():
                    continue
                report = json.loads(report_path.read_text())
                run_id = run_dir.name
                if run_id in seen_ids:
                    continue
                seen_ids.add(run_id)
                runs.append({
                    "id": run_id,
                    "name": run_id,
                    "timestamp": run_dir.stat().st_mtime,
                    "total_samples": report.get("total_samples", 0),
                    "mean_reward": report.get("summary_metrics", {}).get("mean_reward", 0),
                    "status": "completed",
                    "live": False,
                    "can_kill": False,
                    "tags": build_run_tags(run_dir, report),
                })

        live_runs = list_registered_runs()
        live_runs.extend(
            discover_watching_runs(
                self.__class__.results_dir,
                list(self.__class__.known_results_dirs),
            )
        )
        for run in live_runs:
            run_id = run["run_id"]
            if run_id in seen_ids:
                continue
            runs.append({
                "id": run_id,
                "name": run["config_name"],
                "timestamp": run["start_time"],
                "total_samples": None,
                "mean_reward": None,
                "status": run["status"],
                "live": True,
                "can_kill": run["status"] == "running" and get_run(run_id) is not None,
                "tags": build_live_run_tags(run),
            })

        runs.sort(key=lambda run: run["timestamp"], reverse=True)
        self._json_response({"runs": runs})

    def _get_run(self, run_id: str) -> None:
        """Load a saved run artifact."""
        trace_dir = self._resolve_run_dir(run_id)

        if trace_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return

        report_path = trace_dir / "report.json"
        if not report_path.exists():
            self.send_error(404, f"No report.json in run: {run_id}")
            return

        self._json_response(load_trace_payload(trace_dir, run_id))

    def _get_sample(self, run_id: str, sample_id: str) -> None:
        """Load a sample JSON and normalize it for the frontend."""
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return
        sample_path = run_dir / "samples" / f"{sample_id}.json"
        if not sample_path.exists():
            self.send_error(404, f"Sample not found: {run_id}/{sample_id}")
            return
        sample = load_sample_payload(run_dir.parent, run_dir.name, sample_id)

        content = json.dumps(sample).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _get_workspace(self, run_id: str, sample_id: str) -> None:
        """Return workspace snapshots and line history for a sample.

        Attempts to read live workspace_snapshot events from events.jsonl.
        Falls back to reconstructing from trajectory tool calls when none exist.
        """
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return

        if not (run_dir / "samples" / f"{sample_id}.json").exists():
            self.send_error(404, f"Sample not found: {sample_id}")
            return
        response = load_workspace_response(run_dir, sample_id)
        content = json.dumps(response).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(content)

    def _export_sample_html(self, run_id: str, sample_id: str) -> None:
        run_dir = self._resolve_run_dir(run_id)
        if run_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return

        sample_path = run_dir / "samples" / f"{sample_id}.json"
        if not sample_path.exists():
            self.send_error(404, f"Sample not found: {run_id}/{sample_id}")
            return

        sample = load_sample_payload(run_dir.parent, run_dir.name, sample_id)
        html_doc = sample_to_html(run_id, sample_id, sample)
        self._html_response(html_doc, filename=f"{run_id}_{sample_id}.html")

    def _export_run_html(self, run_id: str) -> None:
        trace_dir = self._resolve_run_dir(run_id)
        if trace_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return

        report_path = trace_dir / "report.json"
        if not report_path.exists():
            self.send_error(404, f"No report.json in run: {run_id}")
            return

        trace_payload = load_trace_payload(trace_dir, run_id)
        samples_dir = trace_dir / "samples"
        samples = [
            load_sample_payload(trace_dir.parent, run_id, sample_path.stem)
            for sample_path in sorted(samples_dir.glob("*.json"))
        ]
        html_doc = run_to_html(run_id, trace_payload["report"], samples)
        self._html_response(html_doc, filename=f"{run_id}.html")

    def _stream_run_events(self, run_id: str) -> None:
        """Stream live events for any streamable run."""
        logger.debug(f"📡 Stream connection opened for run_id: {run_id}")
        try:
            if get_run(run_id) is not None:
                stream_registered_run(self, run_id)
            else:
                stream_watch_run(
                    self,
                    run_id,
                    results_dir=self.__class__.results_dir,
                    known_results_dirs=list(self.__class__.known_results_dirs),
                )
        except KeyError:
            logger.debug(f"Available run_ids: {active_run_ids()}")
            self.send_error(404, f"Run not found: {run_id}")
        except FileNotFoundError:
            self.send_error(404, f"No live event source found for run: {run_id}")
        except Exception as e:
            logger.exception("Error streaming run %s: %s", run_id, e)
            raise

    def _list_results_dirs(self) -> None:
        """Return all known results directories and the currently active one."""
        dirs = []
        for d in self.__class__.known_results_dirs:
            label = f"{d.parent.name}/{d.name}" if d.name == "results" else d.name
            dirs.append({"path": str(d), "label": label, "exists": d.exists()})
        self._json_response({
            "current": str(self.__class__.results_dir),
            "dirs": dirs,
        })

    def _set_results_dir(self) -> None:
        """Hot-swap the results directory. Body: {"path": "/abs/path/to/results"}"""
        body = self._read_json_body()
        new_path = Path(body["path"]).expanduser().resolve()
        if not new_path.exists():
            self.send_error(400, f"Directory does not exist: {new_path}")
            return
        self.__class__.results_dir = new_path
        logger.info(f"Results dir switched to: {new_path}")
        self._json_response({"ok": True, "path": str(new_path)})

    def _set_run_tags(self, run_id: str) -> None:
        trace_dir = self._resolve_run_dir(run_id)
        if trace_dir is None:
            self.send_error(404, f"Run not found: {run_id}")
            return

        body = self._read_json_body()
        raw_updates = body.get("tags")
        if not isinstance(raw_updates, dict):
            self.send_error(400, "Body must contain object field 'tags'")
            return

        updates: dict[str, str | None] = {}
        for key, value in raw_updates.items():
            if value is not None and not isinstance(value, str):
                self.send_error(400, f"Tag {key!r} must be a string or null")
                return
            updates[str(key)] = value

        tags = update_user_tags(trace_dir, updates)
        self._json_response({"ok": True, "user": tags})

    def _kill_run(self, run_id: str) -> None:
        """Kill a running process."""
        logger.debug(f"🛑 Kill request received for run_id: {run_id}")
        success, message = kill_run(run_id)
        if not success and message.startswith("Run not found:"):
            logger.error("Kill failed: %s", message)
            logger.debug(f"Available run_ids: {active_run_ids()}")
            self.send_error(404, message)
            return
        if success:
            logger.info("successfully killed run %s", run_id)
        else:
            logger.warning(message)
        self._json_response({"success": success, "message": message})

    def _json_response(self, data: Any) -> None:
        """Send JSON response."""
        json_data = json.dumps(data, indent=2)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(json_data)))
        self.end_headers()
        self.wfile.write(json_data.encode("utf-8"))

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))
        if not isinstance(body, dict):
            raise ValueError("Request body must be a JSON object")
        return body

    def _html_response(self, document: str, *, filename: str) -> None:
        content = document.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    """Run the rollouts run viewer server."""
    parser = argparse.ArgumentParser(description="Rollouts run viewer")
    parser.add_argument(
        "--port", type=int, default=8080, help="Port to run server on (default: 8080)"
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path.cwd(),
        help="Project root; defaults results dir to PROJECT/results",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="Don't automatically open browser"
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        type=Path,
        default=[],
        help="Additional results directories to offer in the UI (space-separated paths)",
    )

    args = parser.parse_args()

    # Build known results dirs: project's own results/ first, then any extras
    primary = args.project.resolve() / "results"
    extras = [Path(p).expanduser().resolve() for p in (args.results_dirs or [])]
    all_dirs: list[Path] = []
    seen: set[Path] = set()
    for d in [primary] + extras:
        if d not in seen:
            all_dirs.append(d)
            seen.add(d)
    RolloutsViewerServer.results_dir = all_dirs[0] if all_dirs else primary
    RolloutsViewerServer.known_results_dirs = all_dirs

    # Create server
    server = ThreadingHTTPServer(("localhost", args.port), RolloutsViewerServer)

    url = f"http://localhost:{args.port}"
    print(f"\n{'=' * 60}")
    print("🚀 Rollouts Run Viewer")
    print(f"{'=' * 60}")
    print(f"URL: {url}")
    print(f"Results: {RolloutsViewerServer.results_dir}")
    print(f"{'=' * 60}\n")

    # Open browser
    if not args.no_browser:
        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped")


if __name__ == "__main__":
    main()
