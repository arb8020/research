## Service-scoped dependency ownership

Runtime dependencies belong to the service/process that owns them.

For RL this means at least:

- `trainer.deps`: training/distributed/backend stack
- `inference.deps`: inference/server/runtime stack

Current state:

- the current Argus launcher realizes exactly one shared runtime env
- that shared env must be declared explicitly in `hardware.deps`
- `trainer.deps` / `inference.deps` are not merged back into `hardware.deps`
- current launchers reject service-scoped deps instead of pretending to realize
  them inside one shared env

Target state:

- each service gets its own image/runtime env
- heavy deps are image-owned
- workspace/bootstrap steps do not redefine the heavy runtime stack
- the contract between services is explicit:
  endpoint discovery, weight publication, and version visibility
