# Consumer project threading

## Problem

Argus/rollouts conflates three distinct directories into one `REPO_ROOT`
constant resolved off `__file__`:

1. **Package install dir** — where argus/rollouts/bifrost code lives. Used
   legitimately for `importlib.resources`-style bundled asset lookup.
2. **Primary workspace root** — the directory bifrost ships to the remote as
   a bundle. Today always assumed to be `REPO_ROOT.parent`, which is the
   research monorepo.
3. **Consumer project root** — the project whose config the user is running
   (the directory with a `pyproject.toml` that depends on rollouts).

In the research monorepo (1)+(2)+(3) are all `~/research/` so the conflation
is invisible. In courier, rollouts is vendored at
`~/silares_stuff/courier/third_party/research_deps/rollouts/`, so
`REPO_ROOT.parent` points at `third_party/research_deps/`, which is neither
the workspace to deploy nor the consumer project. Four courier configs fail
as a result.

An escape hatch exists (`argus.run._external_config_projects`) that
detects "config is outside the primary workspace" and returns a
`PythonProjectMaterialization`. It is computed but never threaded through
to the bifrost call site, supervisor cwd, or output_dir resolution.

## Fix scope

Small pass. Name one concept, thread it through once, remove the per-site
lies that would otherwise keep regrowing. Explicitly **not** doing:

- Breaking the argus ↔ rollouts import cycle.
- Replacing rollouts `REPO_ROOT` for bundled asset lookup with
  `importlib.resources`.
- Eliminating the remote sys.path hack.
- Collapsing primary workspace and consumer project into one concept.

These are real follow-ups but out of scope for unblocking courier.

## The concept

**Consumer project**: the Python project that depends on rollouts/argus and
whose config is being run. Resolved by walking up from the config file
looking for `pyproject.toml` or `.git`.

Type: reuse `bifrost.types.PythonProjectMaterialization`. Singular, not a
collection — there is exactly one consumer project per run.

## Signature changes

### New helper (one source of truth)

`rollouts.remote_runtime.resolve_consumer_project(config_path: Path) -> PythonProjectMaterialization`

Replaces duplicated `_find_config_project_root` in both
`rollouts/eval/run.py` and `argus/run.py`. Replaces
`argus.run._external_config_projects` (escape hatch becomes unneeded because
value is always known).

### Argus CLI boundary

`argus.run.run_main`:
```
consumer_project = resolve_consumer_project(config_path)
```

Resolved once, at the CLI entry point. Threaded down.

### Launch plan

`rollouts.launch_plan.build_local_workload_plan` and the per-kind builders
(`build_local_eval_launch_plan`, `build_local_serving_launch_plan`) accept
`consumer_project: PythonProjectMaterialization` and use it for:

- `run_dir = Path(consumer_project.local_root) / "results" / <kind> / <run>`
  (replaces `repo_root / "results" / ...`).
- No change to the subprocess command; we do *not* serialize
  `consumer_project` across the subprocess boundary. The supervisor
  recomputes it from its `--config` arg using the same helper. Rationale:
  the value is derived, not ambient — one source of truth (the helper),
  called at both boundaries.

### Supervisor subprocesses

`rollouts.eval.supervisor.main` and `rollouts.serving.supervisor.main`:

- Compute `consumer_project = resolve_consumer_project(config_path)` after
  arg parse.
- `_spawn_child(..., cwd=consumer_project.local_root)` — replaces
  `cwd=str(REPO_ROOT.parent)`.
- Pass `consumer_project` to `realize_worker_backed_endpoint`.

### Endpoint realization

`rollouts.eval.endpoint_realization.realize_worker_backed_endpoint` gains
`consumer_project: PythonProjectMaterialization | None` (None for the
monorepo-internal case where config is inside the primary workspace).

- `_realize_ssh_endpoint` passes it as
  `WorkspaceMaterializationSpec(extra_python_projects=(consumer_project,) if consumer_project else ())`.
- `_realize_modal_endpoint` — see bifrost decision below.

### Bifrost decision: unify `extra_source_roots`?

`WorkspaceMaterializationSpec.extra_python_projects: tuple[PythonProjectMaterialization, ...]` is
the rich, validated type. `ModalExecutionRequest.extra_source_roots: tuple[str, ...]` is
the impoverished stringly-typed version of the same concept. Two shapes, one
meaning — the smell we identified.

**Decision: unify.** Add `extra_python_projects: tuple[PythonProjectMaterialization, ...] = ()` to
`ModalExecutionRequest`. Deprecate (but keep for one release) `extra_source_roots`
by having it coexist. Lowering inside `modal_backend.py` converts to whatever
internal representation the Modal materialize path actually needs.

Rationale: our fix has to do the conversion somewhere. Doing it once inside
bifrost is better than doing it at every external call site and leaving the
next caller to rediscover the shape mismatch. Scope: ~30-50 lines of bifrost,
localized to `modal_backend.py` and `types.py`.

If this proves larger than expected during implementation, fall back to:
convert `PythonProjectMaterialization → str` at the single call site in
`_realize_modal_endpoint` and file a follow-up. Don't let bifrost cleanup
block courier.

## What gets deleted

- `argus.run._external_config_projects`
- `argus.run._find_config_project_root`
- `rollouts.eval.run._find_config_project_root`
- The `REPO_ROOT.parent` reads in `rollouts/eval/supervisor.py:77` and
  `rollouts/serving/supervisor.py:76` (as cwd).
- The `repo_root / "results"` reads in `rollouts/launch_plan.py:58, :92`.

The top-level `REPO_ROOT = Path(__file__)...` constants in
`argus/run.py:186` and `rollouts/eval/run.py:59` stay. They still serve as
package-dir anchors for things like the bundled
`examples/inference/evals/configs/` lookup. Not our problem today.

## What stays a lie (deferred)

- Argus still assumes the primary workspace to deploy is `REPO_ROOT.parent`.
  For courier, this is `third_party/research_deps/` — still wrong. We
  compensate by threading courier through as `extra_python_projects`
  alongside the "primary workspace." The honest fix (collapse primary +
  consumer into one concept, or thread primary workspace explicitly too) is
  the bigger refactor we're deferring.

- Rollouts still has `REPO_ROOT = Path(__file__).parent.parent.parent` in
  `eval/run.py`. Still used for bundled example-config lookup. Correct use
  of the constant; wrong mechanism (should be `importlib.resources`). Defer.

## Acceptance test

From courier, all four configs succeed:

```
cd ~/silares_stuff/courier
uv run python -m argus run --config courier/serving/tau2_retail_c4_smoke.py --force-deploy-committed
uv run python -m argus run --config courier/serving/harbor_tb2_modal_c4_smoke.py --force-deploy-committed
uv run python -m argus run --config courier/serving/kimi_verifier_deepseek_v32_mi355x_smoke.py --force-deploy-committed
uv run python -m argus run --config courier/bench/bench_deepseek_v3_2_amd_mi355x_sharegpt.py --force-deploy-committed
```

Observable outcomes:

- Results land in `~/silares_stuff/courier/results/`, not in
  `third_party/research_deps/rollouts/results/`.
- For MI355X configs: remote bundle contains courier's own code
  (verify via `ssh` to the node, inspect the deployed workspace).
- Existing research-monorepo invocations still work (regression check):
  `cd ~/research && .venv/bin/python -m argus run --config rollouts/examples/serving/*.py` for
  the same four configs.

## Implementation order

1. Add `resolve_consumer_project` helper in `rollouts.remote_runtime`.
2. Thread through argus → launch_plan → supervisor. Fix cwd and output_dir.
3. Run tau2 from courier (local, fastest iteration). Verify results land in
   courier.
4. Run tau2 from research (regression).
5. Thread through to `realize_worker_backed_endpoint` and SSH bifrost call.
6. Run kimi_verifier from courier. Verify remote bundle has courier.
7. Bifrost unification of `extra_source_roots`. Thread into Modal path.
8. Run harbor from courier.
9. Run sharegpt bench from courier.
10. Delete the now-dead helpers (`_external_config_projects` etc).
