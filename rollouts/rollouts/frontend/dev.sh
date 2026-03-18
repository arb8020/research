#!/usr/bin/env bash
set -euo pipefail

# Rollouts UI - tmux dev launcher
#
# Two-pane layout:
#   [left]   python server.py  (API server on PORT)
#   [right]  bun run dev       (Vite HMR on VITE_PORT)
#
# Finds free ports in 8080-8099 (server) and 5173-5192 (vite) automatically.
#
# Usage:
#   ./dev.sh [project_dir]      # default: cwd
#   ./dev.sh ~/research/rollouts
#
# Attach later:  tmux attach -t rollouts-ui
# Kill:          tmux kill-session -t rollouts-ui
# Server logs:   tail -f <project>/logs/server_*.log

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "${1:-$(pwd)}" && pwd)"
SESSION="rollouts-ui"
UI_DIR="$ROOT_DIR/ui"

# Extra results dirs to surface in the UI (space-separated, or set EXTRA_RESULTS_DIRS env var)
# Defaults to charisma's results dir if it exists
_default_extras=""
[[ -d "$HOME/silares_stuff/charisma/results" ]] && _default_extras="$HOME/silares_stuff/charisma/results"
EXTRA_RESULTS_DIRS="${EXTRA_RESULTS_DIRS:-$_default_extras}"

# ── Port finder ────────────────────────────────────────────────────────────────
find_free_port() {
  local start=$1 end=$2
  for port in $(seq "$start" "$end"); do
    if ! lsof -ti ":$port" >/dev/null 2>&1; then
      echo "$port"
      return 0
    fi
  done
  echo "No free port in $start-$end" >&2
  exit 1
}

SERVER_PORT=$(find_free_port 8080 8099)
VITE_PORT=$(find_free_port 5173 5192)

# ── Prereqs ────────────────────────────────────────────────────────────────────
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found" >&2; exit 1
fi
if ! command -v bun >/dev/null 2>&1; then
  echo "bun not found" >&2; exit 1
fi
if [[ ! -d "$UI_DIR/node_modules" ]]; then
  echo "Installing UI dependencies..."
  bun install --cwd "$UI_DIR"
fi

# ── Logs ───────────────────────────────────────────────────────────────────────
LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
SERVER_LOG="$LOG_DIR/server_${TS}.log"

# ── tmux ───────────────────────────────────────────────────────────────────────
_extra_args=()
[[ -n "$EXTRA_RESULTS_DIRS" ]] && _extra_args=(--results-dirs $EXTRA_RESULTS_DIRS)

if ! command -v tmux >/dev/null 2>&1; then
  echo "tmux not found - running foreground (no split view)" >&2
  "$_PYTHON" "$ROOT_DIR/server.py" --project "$PROJECT_DIR" --port "$SERVER_PORT" "${_extra_args[@]}" 2>&1 | tee "$SERVER_LOG" &
  SERVER_PID=$!
  trap "kill $SERVER_PID 2>/dev/null; wait" INT TERM
  cd "$UI_DIR" && API_PORT=$SERVER_PORT bun run dev --port "$VITE_PORT"
  exit 0
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Session '$SESSION' already running."
  echo "  Attach: tmux attach -t $SESSION"
  echo "  Kill:   tmux kill-session -t $SESSION"
  exit 0
fi

# Resolve venv Python — prefer workspace venv, fall back to python3
_VENV_PYTHON=""
for _candidate in \
    "$PROJECT_DIR/../../.venv/bin/python3" \
    "$PROJECT_DIR/../.venv/bin/python3" \
    "$PROJECT_DIR/.venv/bin/python3"; do
  if [[ -x "$_candidate" ]]; then
    _VENV_PYTHON="$(realpath "$_candidate")"
    break
  fi
done
_PYTHON="${_VENV_PYTHON:-python3}"

# Server pane (left)
tmux new-session -d -s "$SESSION" -n ui -c "$ROOT_DIR" \
  "'$_PYTHON' '$ROOT_DIR/server.py' --project '$PROJECT_DIR' --port $SERVER_PORT ${_extra_args[*]:-} \
   2>&1 | tee '$SERVER_LOG'; echo '[server exited — press enter]'; read"

# Vite pane (right) — pass API_PORT so vite.config.ts proxies to the right server
tmux split-window -t "$SESSION:0" -h -c "$UI_DIR" \
  "API_PORT=$SERVER_PORT bun run dev --port $VITE_PORT; echo '[vite exited — press enter]'; read"

# Wider left pane (server logs are denser)
tmux select-layout -t "$SESSION:0" main-vertical
tmux resize-pane -t "$SESSION:0.0" -x "55%"
tmux select-pane -t "$SESSION:0.0"

echo ""
echo "Session '$SESSION' started"
echo "  API server : http://localhost:$SERVER_PORT  (log: $SERVER_LOG)"
echo "  Vite UI    : http://localhost:$VITE_PORT   (proxies /api -> :$SERVER_PORT)"
echo ""
echo "  Attach : tmux attach -t $SESSION"
echo "  Kill   : tmux kill-session -t $SESSION"
echo ""

if [[ "${TMUX:-}" == "" ]]; then
  tmux attach -t "$SESSION"
fi
