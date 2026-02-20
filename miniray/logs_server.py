"""LogsServer: Serve training log files over TCP (wandb at home).

A lightweight file server using MiniRay's JSON-over-TCP protocol.
Training node runs this alongside the training job. Monitor connects
as a RemoteWorker and polls for file updates.

Protocol:
    → {"cmd": "list"}
    ← {"files": ["metrics.jsonl", "training.log", ...]}

    → {"cmd": "tail", "file": "metrics.jsonl", "offset": 0}
    ← {"lines": ["...", "..."], "offset": 1234}

    → {"cmd": "tail", "file": "metrics.jsonl", "offset": 1234}
    ← {"lines": ["new line"], "offset": 1290}

    → {"cmd": "ping"}
    ← {"status": "alive"}

    → {"cmd": "shutdown"}
    ← (connection closed)

Usage:
    # Terminal 1: start server
    python -m miniray.logs_server --port 9100 --dir /path/to/run/output

    # Terminal 2: connect
    from miniray import RemoteWorker
    w = RemoteWorker("localhost", 9100)
    w.connect()
    w.send({"cmd": "list"})
    print(w.recv())
"""

from __future__ import annotations

import json
import logging
import signal
import socket
import threading
from pathlib import Path
from types import FrameType

logger = logging.getLogger(__name__)


def serve_client(sock: socket.socket, watch_dir: Path) -> None:
    """Handle a single client connection.

    Reads commands from the socket, serves file contents.
    Runs in a thread per client.
    """
    r = sock.makefile("r")
    w = sock.makefile("w")

    def send(msg: dict) -> None:
        json.dump(msg, w)
        w.write("\n")
        w.flush()

    try:
        while True:
            line = r.readline()
            if not line:
                break  # Client disconnected

            msg = json.loads(line)
            cmd = msg.get("cmd")

            if cmd == "ping":
                send({"status": "alive"})

            elif cmd == "list":
                files = [
                    f.name
                    for f in watch_dir.iterdir()
                    if f.is_file() and not f.name.startswith(".")
                ]
                send({"files": sorted(files)})

            elif cmd == "tail":
                filename = msg.get("file")
                offset = msg.get("offset", 0)
                max_lines = msg.get("max_lines", 1000)  # Pagination limit
                max_bytes = msg.get("max_bytes", 512 * 1024)  # 512KB per response

                assert filename, "tail requires 'file'"
                assert isinstance(offset, int) and offset >= 0, f"bad offset: {offset}"

                # Prevent path traversal
                target = (watch_dir / filename).resolve()
                if target.parent != watch_dir.resolve():
                    send({"lines": [], "offset": 0, "error": "path_traversal"})
                    continue

                if not target.exists():
                    send({"lines": [], "offset": 0, "error": "not_found"})
                    continue

                with open(target) as f:
                    f.seek(offset)
                    # Read limited bytes, not entire file
                    content = f.read(max_bytes)

                    # Split into lines
                    lines = content.splitlines()

                    # If we hit max_bytes mid-line, drop the last partial line
                    # (unless it's the only line, meaning single line > max_bytes)
                    if len(content) == max_bytes and len(lines) > 1 and not content.endswith("\n"):
                        partial_line = lines.pop()
                        content = content[: -len(partial_line)]

                    # Apply line limit
                    if len(lines) > max_lines:
                        lines = lines[:max_lines]
                        # Recalculate content to match truncated lines
                        content = "\n".join(lines) + "\n"

                    new_offset = offset + len(content.encode("utf-8"))
                    has_more = len(content) == max_bytes or f.read(1) != ""

                send({"lines": lines, "offset": new_offset, "has_more": has_more})

            elif cmd == "read":
                # Read entire file (for config.json etc.)
                filename = msg.get("file")
                assert filename, "read requires 'file'"

                target = (watch_dir / filename).resolve()
                if target.parent != watch_dir.resolve():
                    send({"content": "", "error": "path_traversal"})
                    continue

                if not target.exists():
                    send({"content": "", "error": "not_found"})
                    continue

                content = target.read_text()
                send({"content": content})

            elif cmd == "shutdown":
                break

            else:
                send({"error": f"unknown command: {cmd}"})

    except (json.JSONDecodeError, AssertionError, BrokenPipeError, ConnectionResetError) as e:
        logger.warning("Client error: %s", e)
    finally:
        r.close()
        w.close()
        sock.close()


class LogsServer:
    """TCP server that serves log files from a directory.

    Accepts multiple concurrent client connections. Each client
    gets its own thread. Uses the same JSON-over-TCP protocol
    as MiniRay's WorkerServer/RemoteWorker.

    Example:
        >>> server = LogsServer(port=9100, watch_dir="/tmp/run_output")
        >>> server.serve_forever()  # Blocks, Ctrl-C to stop
    """

    def __init__(self, port: int, watch_dir: str | Path, host: str = "0.0.0.0") -> None:
        assert port > 0 and port <= 65535, f"bad port: {port}"

        self.host = host
        self.port = port
        self.watch_dir = Path(watch_dir).resolve()
        self._sock: socket.socket | None = None
        self._shutdown = False

        assert self.watch_dir.is_dir(), f"watch_dir must exist: {self.watch_dir}"

    def serve_forever(self) -> None:
        """Listen for connections and spawn a handler thread per client."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.settimeout(1.0)  # So we can check _shutdown flag
        self._sock.bind((self.host, self.port))
        self._sock.listen(8)

        # Signal handlers
        def on_signal(signum: int, _frame: FrameType | None) -> None:
            logger.info("Signal %s, shutting down...", signum)
            self._shutdown = True

        signal.signal(signal.SIGTERM, on_signal)
        signal.signal(signal.SIGINT, on_signal)

        logger.info("Serving %s", self.watch_dir)
        logger.info("Listening on %s:%s", self.host, self.port)

        while not self._shutdown:
            try:
                client_sock, client_addr = self._sock.accept()
                logger.info("Client connected: %s", client_addr)
                t = threading.Thread(
                    target=serve_client,
                    args=(client_sock, self.watch_dir),
                    daemon=True,
                )
                t.start()
            except TimeoutError:
                continue  # Check _shutdown flag

        self._sock.close()
        logger.info("Stopped")


# ============================================================================
# CLI
# ============================================================================


def main() -> None:
    """CLI entry point: python -m miniray.logs_server --port 9100 --dir /path/to/output"""
    import argparse

    parser = argparse.ArgumentParser(description="MiniRay LogsServer — serve log files over TCP")
    parser.add_argument("--port", type=int, default=9100, help="Port to listen on (default: 9100)")
    parser.add_argument("--dir", required=True, help="Directory to serve files from")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    server = LogsServer(port=args.port, watch_dir=args.dir, host=args.host)
    server.serve_forever()


if __name__ == "__main__":
    main()
