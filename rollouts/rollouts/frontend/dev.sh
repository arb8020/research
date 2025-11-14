#!/usr/bin/env bash
set -euo pipefail

# Simple dev runner: starts agent dev loop server, prefixes logs, stops on ENTER/Ctrl+C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROLLOUTS_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Default to current directory as project, or pass as argument
PROJECT_DIR="${1:-$(pwd)}"
PORT="${2:-8080}"

# Logs
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$LOG_DIR/devloop_${TS}.log"

cleanup() {
  echo "\n🛑 Stopping dev loop server..."
  [[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" >/dev/null 2>&1 || true
  wait >/dev/null 2>&1 || true
  echo "✅ Stopped."
}
trap cleanup INT TERM

echo "🔧 Checking prerequisites..."

if [[ ! -d "$PROJECT_DIR" ]]; then
  echo "❌ Project directory not found: $PROJECT_DIR" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "❌ Need python3 installed to run the server" >&2
  exit 1
fi

SERVER_CMD=(python3 "$SCRIPT_DIR/run.py" --project "$PROJECT_DIR" --port "$PORT")

echo "🚀 Starting Agent Dev Loop Server on :$PORT"
echo "   ↳ Project: $PROJECT_DIR"
echo "   ↳ Logging to: $SERVER_LOG"
"${SERVER_CMD[@]}" 2>&1 | tee -a "$SERVER_LOG" | sed -e 's/^/[SERVER] /' & SERVER_PID=$!

sleep 2
echo ""
echo "➡️  Server logs are prefixed with [SERVER]"
echo "🔗 Dev Loop: http://localhost:$PORT"
echo ""
echo "Press ENTER to stop, or Ctrl+C"
read -r _ || true

cleanup
