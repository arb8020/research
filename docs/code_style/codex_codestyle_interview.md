# Codex Code Style Interview

This note records a set of code-style preferences surfaced through concrete
"which code do you want and why" exercises.

It is not a generic style guide. It is a compressed statement of what kinds of
shapes feel honest, locally understandable, and maintainable in this codebase.

## Core orientation

The recurring theme is:

- normalize external junk early
- keep the hot path typed and boring
- make ownership and behavior choice explicit
- use heavier abstraction only when it has clearly earned itself

Another way to say it:

- code should say what capabilities it needs
- boundary code should choose concrete behavior when that choice is real
- not every implementation detail deserves to become a protocol, class, or
  injected abstraction

## Boundaries and normalization

Prefer to normalize at the boundary and then operate on honest internal types.

Good:

- parse env/CLI/JSON into one internal shape if they denote the same fact
- reject malformed input early
- turn optional-arg sludge into a real product type or sum type

Bad:

- carrying input-format distinctions deeper than necessary
- leaving useless branches alive after parse time
- fallback defaults that silently change semantics

If several input sources all mean the same thing internally, erase the source
distinction and keep only the real domain distinction.

Example:

```python
UploadProvider = S3Args | SftpArgs | WebDavArgs


def parse_upload_provider(raw: EnvInput | CliInput | JsonInput) -> UploadProvider:
    ...
```

Not:

```python
def upload_from_input(raw: EnvInput | CliInput | JsonInput, path: Path) -> UploadResult:
    ...
```

unless the input-source distinction remains semantically relevant later.

## Types over arg soup

Prefer honest product types and sum types over giant signatures full of
optional parameters.

Bad:

```python
def upload_attachment(
    file_path: Path,
    *,
    provider: str,
    s3_bucket: str | None = None,
    sftp_host: str | None = None,
    webdav_url: str | None = None,
) -> UploadResult:
    ...
```

Better:

```python
UploadProvider = S3Args | SftpArgs | WebDavArgs


@dataclass(frozen=True)
class UploadPlan:
    provider: UploadProvider
    encrypt_before_upload: bool
    generate_preview: bool
```

Then the hot path can operate on `UploadPlan` rather than re-decoding a little
config language hidden in the function signature.

## Sum types before protocols

When the variants are part of the domain model, visible branching over an
honest sum type is often preferable to erasing the distinction behind a
protocol too early.

Good default:

```python
StoreConfig = LocalStore | S3Store | HFHubStore


def publish_checkpoint(store: StoreConfig, path: Path, key: str) -> PublishResult:
    if isinstance(store, LocalStore):
        ...
    if isinstance(store, S3Store):
        ...
    if isinstance(store, HFHubStore):
        ...
    raise AssertionError(store)
```

This keeps the real branch visible and preserves local reasoning.

A protocol/capability layer starts to earn itself when:

- the implementation owns real resources or cleanup
- repeated setup would otherwise happen per call
- many call sites only care about one capability
- the choice of implementation belongs to an outer boundary

## Dependency construction at boundaries

Construct dependencies at the earliest honest boundary.

Usually that means:

- process startup
- request boundary
- tenant/model/provider resolution boundary

Which boundary is correct depends on where the information to choose the
dependency actually lives.

Good:

```python
def main(config: AppConfig, logger: Logger) -> None:
    db = connect_db(config.db)
    store = build_store(config.storage)
    app = App(db=db, store=store, logger=logger)
    app.run()
```

Also good when the choice is request-scoped:

```python
def handle_upload(req: UploadRequest) -> UploadResult:
    company = authenticate(req)
    store = build_store_for_company(company)
    return upload_attachment(store, req.file_path)
```

Bad:

- ambient globals that hide where the dependency came from
- inner code reaching into config/env and constructing the world for itself
- framework/container magic that obscures ownership

## Composition roots and fail-fast setup

Prefer explicit composition roots.

Required resources should usually be built up front so failure happens early
and obviously.

Good default:

```python
def main(cfg: AppConfig) -> None:
    db = connect_db(cfg.db)
    cache = connect_cache(cfg.cache)
    app = App(db=db, cache=cache)
    app.run()
```

Lazy construction is justified when:

- the resource is expensive
- it may never be needed
- availability is inherently runtime-dependent
- request validity and resource availability are separate concerns

But lazy init should be the exception, not the way required setup gets hidden.

## Functions before noun-classes

Prefer functions over classes when the code is primarily a verb operating over
visible state.

Good:

```python
@dataclass
class RunState:
    jobs: dict[str, Job]
    next_id: int = 0


def submit_job(state: RunState, spec: JobSpec) -> JobId:
    ...
```

Less preferred:

```python
class JobManager:
    def submit_job(self, spec: JobSpec) -> JobId:
        ...
```

unless the class truly owns lifecycle, cleanup, long-lived state, or a real
resource boundary.

If it is basically a function, let it be a function.

## Classes should earn themselves

A class is justified when it owns something real:

- a network connection
- a process
- a cache with meaningful policy/state
- a lifecycle with setup and cleanup
- a stateful long-lived service

Do not create a class just to bundle related verbs or to satisfy OO symmetry.

Similarly, inheritance is disfavored. Shared helpers or composition are easier
to reason about locally than behavior hidden up a class hierarchy.

## Caches are often real dependencies

Caches are not automatically "just private optimizations."

If cache policy affects:

- semantics
- staleness
- observability
- performance in ways callers care about

then the cache is often a real dependency and should be visible.

Good:

```python
class Cache(Protocol):
    def get(self, key: str) -> bytes | None: ...
    def put(self, key: str, value: bytes) -> None: ...


def get_artifact(client: ArtifactClient, cache: Cache, key: str) -> bytes:
    ...
```

This makes policy and behavior explicit instead of hiding them behind a
"helpful" object with secret state.

## Logging, env, and cross-cutting concerns

Cross-cutting concerns should be explicit, but explicit does not mean threading
everything through every function signature.

Preferences here:

- logger setup should be explicit
- env/config ingress should be explicit
- avoid ambient `os.environ` reads deep inside the program
- avoid mysterious global observability state

Good:

```python
def main(env: Mapping[str, str], logger: Logger) -> None:
    db_url = env["DB_URL"]
    app = App(db_url=db_url, logger=logger)
    app.run()
```

Bad:

```python
class App:
    def run(self) -> None:
        db_url = os.environ["DB_URL"]
        ...
```

Metrics and tracing are trickier. The preference is still for explicit setup,
but not at the cost of turning every function signature into plumbing noise.

## Testing seams

Default stance: skepticism toward mocks and fakes.

Do not invent seams mainly to make mocking easy.

Prefer fakes only at coarse, honest cut points where you are replacing:

- an external system
- a nondeterministic boundary
- a resource-heavy dependency

If a seam only exists because a test wants to stub a library call, the better
question is often whether the cut point is wrong.

Corollary:

- a real storage backend seam is good
- a fake `ImageScaler` protocol created only to dodge a library in tests is
  suspicious

## Assertions, errors, and result types

Use the failure channel that matches the kind of failure.

### `raise`

Use `raise` at boundaries for:

- invalid input
- violated preconditions
- malformed external data

### `assert`

Use `assert` for:

- programmer errors
- broken invariants
- impossible internal states

This means the same check may be a `raise` or an `assert` depending on where it
appears in the pipeline.

### `None` vs explicit result variants

`None` is fine when optionality is trivial and universally understood.

Use an explicit result variant when absence has semantics or may grow richer.

Good enough:

```python
def maybe_get_parent(run_id: str) -> Run | None:
    ...
```

Better when the "none-ness" means something specific:

```python
@dataclass(frozen=True)
class NoCheckpoint:
    path: Path


ResumeLoadResult = Checkpoint | NoCheckpoint
```

The point is semantic honesty, not blindly following "Pythonic" folklore.

## No dishonest fallbacks

Do not silently repair bad input with fake defaults.

Bad:

```python
def get_storage_kind(row: dict[str, object]) -> str:
    kind = row.get("storage_kind", "s3")
    if kind not in {"s3", "sftp", "webdav"}:
        kind = "s3"
    return kind
```

Good:

```python
def parse_storage_kind(row: dict[str, object]) -> str:
    kind = row.get("storage_kind")
    if kind is None:
        raise ValueError("missing storage_kind")
    if kind not in {"s3", "sftp", "webdav"}:
        raise ValueError(f"invalid storage_kind: {kind!r}")
    return kind
```

If the variant matters, bad input should fail honestly.

## Helper extraction

Do not extract helpers just to create the feeling of organization.

Keep code inline when:

- the branch body is small
- extraction does not improve local readability
- the helper would only add indirection

Extract helpers when:

- the branch body becomes large enough to obscure parent control flow
- the extracted chunk is a real semantic unit
- the parent remains the owner of control flow and branching

In other words:

- parent owns the `if`
- helper owns the straight-line chunk, if that chunk is large enough to earn a
  name

## Continuous granularity

Expose staged lower-level APIs when the stages are real, but allow a compressed
entrypoint on top.

Good:

```python
RawProviderRow -> ParsedProviderConfig -> UploadTarget
```

with:

```python
parse_provider_row(...)
normalize_provider_config(...)
resolve_upload_target(...)
```

and also:

```python
parse_upload_target(raw_row) -> UploadTarget
```

The high-level API should be implemented in terms of the lower-level honest
stages, not replace them. Do not make a hole in the API.

## Planning vs execution

Do not split "plan" and "execute" by default.

Keep them together until the plan becomes a real thing in the problem.

A plan object earns itself when it supports:

- recovery
- retries
- auditing
- orchestration across steps
- staged side effects with meaningful boundaries

Otherwise, a forced plan/execute split can just make straightforward code less
readable.

## Practical summary

When choosing between two designs, bias toward the one that:

- keeps ownership visible
- models real distinctions with types
- normalizes external mess at the boundary
- keeps inner code on trusted shapes
- fails honestly instead of silently repairing semantics
- uses classes only for real lifecycle/state/resource ownership
- introduces abstraction only where variation or ownership is real

And push back on designs that:

- hide dependency choice in globals or framework magic
- spread env/config reads throughout the codebase
- encode domain variants as optional-arg sludge
- invent protocols mainly for tests
- add indirection without improving local reasoning
