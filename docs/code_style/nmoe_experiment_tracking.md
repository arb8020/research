# nmoe Experiment Tracking & Logging Analysis

Analysis of https://github.com/Noumena-Network/nmoe logging/metrics infrastructure compared to our rollouts codebase.

## High-Level Summary

nmoe has a three-layer observability stack: (1) an SQLite experiment registry that tracks run metadata with git state, (2) per-rank DuckDB files for queryable timeseries metrics, and (3) a Next.js dashboard (NVIZ) that reads both for live visualization.

The experiment registry is the main thing we're missing. It stores `(run_id, experiment_id, config_json, git_hash, git_dirty, status, started_at, ended_at, results_json)` in SQLite, giving you a queryable history of what ran, with what config, from what commit. This is useful for comparing runs and catching dirty-commit experiments.

Their metrics storage uses DuckDB instead of JSONL. The schema is `(run, tag, step, ts_ms, value)` with a composite primary key. This lets you run SQL queries like `SELECT AVG(value) FROM metrics WHERE tag='loss' AND step > 1000` without loading everything into memory. Each rank writes its own `.duckdb` file to avoid write contention.

They also have CUDA event timing that records start/end events without synchronization, then flushes elapsed times lazily at log boundaries. This avoids mid-step GPU stalls from `cudaEventSynchronize`.

Our rollouts codebase already does some things better: protocol-based `MetricsLogger` with dependency injection, async-safe `QueueHandler` with `QueueListener` (nmoe doesn't do this), and multiple formatter options (JSON, color, rich). We also have composite logging to multiple backends.

The GPU telemetry (C++ NVML poller for utilization, temperature, power, ECC errors, throttle state) and web dashboard are probably overkill for our use case—W&B or similar would be simpler if we need dashboards.

## What to Consider Adding

1. **Experiment registry** - SQLite table tracking run metadata, git state, config snapshots, and completion status. Enables "what ran" queries and config comparison.

2. **Queryable timeseries** - DuckDB or similar instead of flat JSONL. Enables SQL aggregations over metrics without full file loads.

3. **Deferred CUDA timing** - Record events without sync, flush lazily. Useful if we add GPU training loops.

## Reference Files

### nmoe (cloned to /tmp/nmoe)

- `/tmp/nmoe/nmoe/experiments.py` - SQLite experiment/run registry with git tracking
- `/tmp/nmoe/nmoe/metrics.py` - DuckDB metrics writer, CUDA timing, GPU telemetry
- `/tmp/nmoe/nmoe/log.py` - glog-style colored formatter with level prefixes
- `/tmp/nmoe/nviz/README.md` - Dashboard overview and metrics schema docs
- `/tmp/nmoe/nviz/lib/` - Dashboard data fetching (reads DuckDB + SQLite)

### Our rollouts codebase

- `/Users/chiraagbalu/research/rollouts/rollouts/training/metrics.py` - JSONLLogger, WandbLogger, CompositeLogger, MetricsLogger protocol
- `/Users/chiraagbalu/research/rollouts/rollouts/_logging/logging_config.py` - QueueHandler setup, JSON/color/rich formatters
- `/Users/chiraagbalu/research/shared/shared/logging_config.py` - Shared logging config (duplicate of above)
- `/Users/chiraagbalu/research/rollouts/rollouts/_logging/color_formatter.py` - ANSI color formatter
- `/Users/chiraagbalu/research/rollouts/rollouts/_logging/json_formatter.py` - JSON formatter for structured logs
