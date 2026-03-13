## Service-scoped dependency ownership

Runtime dependencies belong to the service/process that owns them.

For RL this means at least:

- `trainer.deps`: training/distributed/backend stack
- `inference.deps`: inference/server/runtime stack

`hardware.deps` is still the fallback contract for older single-env runners, but
it should not remain the long-term semantic source of truth for multi-service
jobs.

Current state:

- the single-sandbox Modal path still uses one shared runtime env
- configs may now declare `trainer.deps` / `inference.deps`
- Argus rejects those service-scoped deps on Modal for now, because the launcher
  cannot realize separate service runtimes yet

Target state:

- each service gets its own image/runtime env
- heavy deps are image-owned
- workspace/bootstrap steps do not redefine the heavy runtime stack
- the contract between services is explicit:
  endpoint discovery, weight publication, and version visibility
