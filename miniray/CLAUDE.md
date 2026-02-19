# miniray

Lightweight distributed computing over TCP + fork. Used in rollouts for streaming training logs from remote pods via LogsServer/RemoteWorker.

## LogsServer / RemoteWorker (main use case)

The rollouts monitor uses these to stream log files from a remote training job:

```python
# On remote (training node) — started automatically by rollouts
from miniray import LogsServer
server = LogsServer(watch_dir="results/rl/run_xyz/", port=9100)
server.start()

# On local (monitor) — used by rollouts monitor --attach
from miniray import RemoteWorker
worker = RemoteWorker("remote-host", 9100)
worker.connect()

worker.send({"cmd": "list"})
files = worker.recv()["files"]

worker.send({"cmd": "tail", "file": "training.log", "offset": 0})
result = worker.recv()
lines, new_offset = result["lines"], result["offset"]

worker.close()
```

## Local multi-process workers (fork-based)

```python
from miniray import Worker

def work(handle):
    data = handle.recv()
    handle.send(data * 2)

workers = [Worker(work) for _ in range(4)]
for w in workers:
    w.send(42)
results = [w.recv() for w in workers]
```

## Key gotchas

- `RemoteWorker.connect()` is explicit — must call before send/recv
- LogsServer protocol: `list`, `tail` (with offset), `ping`
- JSON serialization by default; byte offsets track position for incremental tailing
