# Job Groups: Rollouts Use Cases

## 1) Remote distributed evaluation: proxy + workers + driver

This mirrors the “job group” pattern directly:
- **proxy**: centralizes rate limiting and/or caching
- **workers**: run evaluation loops / tool envs / model servers
- **driver (primary)**: submits work and exits when done

Without job groups, we tend to:
- start pieces manually in tmux
- forget cleanup (or kill sessions by hand)
- lose the “this is one experiment run” notion

With job groups, a single “run” owns the lifecycle.

## 2) Multi-node training scaffolding

Even if miniray handles runtime coordination, we still need reliable:
- bootstrapping (env + code deploy)
- starting per-node worker servers
- a driver that validates cluster membership and exits
- teardown after completion/failure

Job groups provide that outer shell.

