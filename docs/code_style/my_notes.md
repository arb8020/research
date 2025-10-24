# My Code Style Notes

## Error Handling Boundaries

**Core principle:** Try/except looks gross and should be avoided whenever possible.

From Tiger Style: Use **assertions for programmer errors** (bugs in our code) and **error handling for expected operating errors** (external failures).

**In practice:**
- ❌ **No try/except in internal code** - use assertions, let it crash
- ✅ **Only at external boundaries** - CLI entry points, network I/O, file operations, optional imports

**Rationale:** Internal code should fail fast on bugs. Only the outermost boundary (main, CLI handlers) should catch exceptions and present them nicely to users.

```python
# ✅ Good: Internal function uses assertions
def process(data):
    assert data is not None
    assert len(data) > 0
    return transform(data)  # Let exceptions propagate

# ✅ Good: Boundary catches and handles
def main():
    try:
        result = process(load_data())
        print(result)
    except Exception as e:
        print(f"Error: {e}")
        return 1
```
