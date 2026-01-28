Handoff: Broker Async Migration (trio)

DONE (committed fc68b3d on main):
  - broker/broker/providers/modal.py — full Modal provider, async via
    trio.to_thread.run_sync(). Smoke tested: T4 provision → nvidia-smi
    → PyTorch CUDA → terminate. All working.
  - broker/providers/__init__.py — "modal" case added
  - broker/api.py — "modal" in PROVIDER_MODULES
  - broker/types.py — modal_token_id + modal_token_secret in ProviderCredentials
  - broker/pyproject.toml — modal>=0.64.0, trio, httpx, anyio added
  - docs/code_style/drafts/outside_in.md — draft on outside-in programming

GOAL (this handoff): Migrate broker's existing providers from sync
  (requests) to async (trio + httpx). Modal provider is already async.
  The rest of broker is sync. Make it all async.

ASYNC LAYERING (decided):
  trio (core) <-> anyio (compat layer) <-> asyncio (Modal SDK bridge)
  - Broker code: trio directly (task groups, cancel scopes, trio.sleep)
  - Shared/reusable code: anyio (portable)
  - Modal SDK: trio.to_thread.run_sync() wrapping sync API
    (Modal's .aio() methods are asyncio coroutines, not trio-compatible.
    Confirmed: awaiting modal.Sandbox.create.aio() from trio fails with
    "unrecognized yield message <Future pending>")
  - httpx: native anyio support, no bridge needed

READ:
  ~/research/broker/broker/providers/runpod.py — template sync provider
  ~/research/broker/broker/api.py — PROVIDER_MODULES, search/create flows
  ~/research/broker/broker/types.py — ProviderModule protocol (currently sync)
  ~/research/shared/shared/retry.py — has async_retry() using trio.sleep()
  ~/research/docs/code_style/archive/domain/anyio_advice.md — async style guide

CHANGE 1: ProviderModule protocol → async
  ~/research/broker/broker/types.py lines 493-524
  All methods become async:
    async def provision_instance(...) → GPUInstance | None
    async def get_instance_details(...) → GPUInstance | None
    async def list_instances(...) → list[GPUInstance]
    async def terminate_instance(...) → bool
    async def search_gpu_offers(...) → list[GPUOffer]

CHANGE 2: Migrate each provider (6 total)
  For each of runpod, lambdalabs, vast, primeintellect, digitalocean,
  digitalocean_amd:
    - Replace `import requests` with `import httpx`
    - All public functions → async def
    - requests.get/post/put/delete → async with httpx.AsyncClient() as client:
        response = await client.get/post/put/delete(...)
    - Replace shared/retry.py's retry() with async_retry() where used
    - response.json() stays the same (httpx has .json() too)
    - response.status_code → response.status_code (same)
    - response.raise_for_status() → response.raise_for_status() (same)

  Start with runpod.py (most used, good template). Then do the rest.

CHANGE 3: api.py → async
  ~/research/broker/broker/api.py
  - search() → async def search()
  - create() → async def create()
  - terminate_instance() → async def terminate_instance()
  - get_instance() → async def get_instance()
  - list_instances() → async def list_instances()
  - _try_provision_from_offer() → async def
  - All provider_module.X() calls become await provider_module.X()

CHANGE 4: GPUInstance methods → async
  types.py GPUInstance:
  - exec() → keep sync (for backward compat) but add aexec() that
    works for both SSH and Modal
  - terminate() → consider async version
  - wait_until_ready() → async with trio.sleep instead of time.sleep
  - wait_until_ssh_ready() → async

CHANGE 5: Client/CLI layer
  - broker/client.py — GPUClient methods → async
  - broker/cli.py — typer commands may need trio.run() wrappers
    (typer is sync, so CLI entrypoints do trio.run(async_main))

VERIFY:
  # Modal still works (no regression)
  uv run python << 'EOF'
  import trio
  from broker.providers.modal import provision_instance, terminate_instance, exec_on_sandbox
  from broker.types import ProvisionRequest
  async def main():
      request = ProvisionRequest(gpu_type="T4", gpu_count=1, name="verify")
      instance = await provision_instance(request)
      assert instance is not None
      result = await exec_on_sandbox(instance.id, "nvidia-smi")
      assert result.success
      await terminate_instance(instance.id)
      print("Modal PASS")
  trio.run(main)
  EOF

  # RunPod search works async
  uv run python << 'EOF'
  import trio
  from broker.providers import runpod
  async def main():
      offers = await runpod.search_gpu_offers(gpu_count=1, api_key="...")
      print(f"{len(offers)} RunPod offers")
  trio.run(main)
  EOF

  # Full create flow works async
  uv run python << 'EOF'
  import trio
  from broker.api import search
  async def main():
      offers = await search(provider="modal")
      for o in offers:
          print(f"{o.gpu_type} ${o.price_per_hour}/hr")
  trio.run(main)
  EOF

KEY RISKS:
  1. shared/retry.py async_retry() uses trio.sleep() — good, but check
     all retry callsites in providers to make sure they switch.
  2. broker/cli.py uses typer (sync). Entrypoints need trio.run() wrappers.
     typer doesn't natively support async commands.
  3. Tests (pytest-asyncio) may need pytest-trio instead.
  4. GPUInstance.wait_until_ready() uses time.sleep(15) in a loop —
     must become await trio.sleep(15) in async version.

ORDER:
  1. ProviderModule protocol → async (types.py)
  2. runpod.py → async (template)
  3. Remaining providers one by one
  4. api.py → async
  5. client.py / cli.py → async wrappers
  6. Run all verify scripts

KEYWORDS: trio, httpx, async_retry, ProviderModule, PROVIDER_MODULES,
  httpx.AsyncClient, trio.run, trio.sleep
