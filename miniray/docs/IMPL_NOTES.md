# Implementation Notes: Adding Heinrich Features Safely

## Current State

miniray has:
- Worker (local fork + socketpair)
- RemoteWorker (TCP client)
- WorkerServer (TCP server, forks workers)
- wait_any() for select() multiplexing
- fileno() on Worker and RemoteWorker

Wire protocol: **newline-delimited JSON text**
```python
json.dump(msg, self.w)
self.w.write("\n")
self.w.flush()
```

## Features to Add

### 1. PDEATHSIG (Linux only) - SAFE

Child dies when parent dies. One function, one call site.

```python
# Add to worker.py

import ctypes
import signal
import sys

def _set_pdeathsig() -> None:
    """Set PDEATHSIG so child dies when parent dies (Linux only)."""
    if sys.platform != "linux":
        return
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM, 0, 0, 0)
    except OSError:
        pass  # Best effort

# In Worker.__init__, child process section:
if not pid:
    # === Child process ===
    _set_pdeathsig()  # <-- ADD THIS LINE
    sock0, sock = sock, sock0
    ...
```

No protocol change. No API change. Safe.

---

### 2. marshal option - SAFE (if done right)

Keep the **same newline-delimited text protocol**. Just change encoding.

**WRONG approach** (breaks RemoteWorker):
```python
# DON'T DO THIS - changes wire protocol to binary
length_bytes = len(data).to_bytes(8, "big")
self._sock.sendall(length_bytes + data)
```

**RIGHT approach** (keeps text protocol):
```python
import base64
import marshal

SerializationFormat = Literal["json", "marshal"]

class Worker:
    def __init__(self, work_fn, format: SerializationFormat = "json"):
        self._format = format
        ...

    def send(self, msg: Any) -> None:
        if self._format == "json":
            text = json.dumps(msg)
        else:
            # marshal -> bytes -> base64 -> text (keeps newline protocol!)
            text = base64.b64encode(marshal.dumps(msg)).decode("ascii")

        self.w.write(text)
        self.w.write("\n")
        self.w.flush()

    def recv(self, max_size: int) -> Any:
        line = self.r.readline(max_size).rstrip("\n")
        if self._format == "json":
            return json.loads(line)
        else:
            return marshal.loads(base64.b64decode(line))
```

base64 adds ~33% overhead but marshal is 5-10x faster than JSON, so net win for large messages.

**Also update**: RemoteWorker needs same format option to interoperate.

---

### 3. SharedMemory (memfd) - SAFE

New class, doesn't touch existing Worker/RemoteWorker code.

```python
import mmap
import os
import tempfile
import sys

@dataclass
class SharedMemory:
    """Shared memory region using memfd (Linux) or tempfile (macOS)."""
    fd: int
    size: int
    _mmap: mmap.mmap | None = None

    @classmethod
    def create(cls, name: str, size: int) -> "SharedMemory":
        """Create shared memory region."""
        if sys.platform == "linux":
            fd = os.memfd_create(name)
        else:
            # macOS fallback - use tempfile
            f = tempfile.NamedTemporaryFile(delete=False)
            fd = os.dup(f.fileno())  # Keep fd alive
            f.close()
            os.unlink(f.name)  # Unlink but fd keeps it alive

        os.ftruncate(fd, size)
        return cls(fd=fd, size=size)

    def map(self) -> mmap.mmap:
        """Memory-map the region."""
        if self._mmap is None:
            self._mmap = mmap.mmap(self.fd, self.size)
        return self._mmap

    def close(self) -> None:
        """Close the shared memory."""
        if self._mmap:
            self._mmap.close()
        os.close(self.fd)
```

Usage:
```python
# Create shared memory
shm = SharedMemory.create("weights", 1024 * 1024 * 100)  # 100MB
buf = shm.map()

# Write tensor data
buf[:len(data)] = data

# Pass fd to child (survives fork)
# Child can mmap the same fd
```

---

## Order of Implementation

1. **PDEATHSIG** - 10 lines, zero risk
2. **SharedMemory** - 40 lines, zero risk (new class)
3. **marshal** - 20 lines, low risk if you keep text protocol

---

## Testing Checklist

Before committing:
```python
# Test 1: Worker still works
worker = Worker(echo_fn)
worker.send({"test": 123})
assert worker.recv(1024) == {"test": 123}

# Test 2: RemoteWorker interop (if marshal added)
# Start WorkerServer, connect with RemoteWorker
remote = RemoteWorker("localhost", 10000)
remote.send({"test": 123})
assert remote.recv(1024) == {"test": 123}

# Test 3: wait_any still works
ready = wait_any([worker], timeout=1.0)
assert len(ready) == 1
```

---

## Key Principle

From the failed attempt:
> "Write usage code first"

Before changing Worker, ask:
1. Does this work with RemoteWorker?
2. Does this work with WorkerServer?
3. Does this change the wire protocol?

If protocol changes, ALL THREE must change together.
