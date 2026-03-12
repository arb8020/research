## Test Layout

- `tests/unit`
  Small focused tests for stable cut points.

- `tests/integration`
  Default automated suite. Subsystem-boundary tests that should run locally and in CI.

- `tests/regression`
  Bug reproductions that should stay fixed.

- `tests/live`
  Pytest tests that consume real APIs, CLIs, GPUs, or other costly resources.
  Not part of default pytest discovery.

- `tests/manual`
  Human-run scripts and trusted repro/config flows.
  These are not part of pytest discovery.

- `tests/archive`
  Old or exploratory tests kept for reference only.
