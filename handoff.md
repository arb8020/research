Handoff: Modal Provider + Broker Async Migration

GOAL: broker supports Modal Sandboxes as a compute provider AND
  broker's provider interface migrates to async (trio).
  These are a single task because adding Modal forces touching
  every provider callsite anyway.

CONTEXT (design decisions already made):
  - Modal sandboxes don't have SSH. bifrost is SSH-only.
    Wrapping Modal in a fake SSH layer would be "dishonest about
    problem shape" (see keeping_llm_code_honest.md). So: Modal
    provider returns GPUInstances that exec() directly via
    sandbox.exec(), no bifrost in the loop.
  - broker is currently 100% sync (requests library, no trio/anyio).
    Per anyio_advice.md: use trio for personal projects where
    correctness is paramount. Bridge to asyncio (Modal SDK) via
    trio.to_thread.run_sync() — Modal's SDK is sync-looking but
    runs asyncio internally. If trio_asyncio import conflicts
    surface (as they did in wafer), fall back to subprocess
    isolation pattern from wafer.
  - "Don't abstract until you've done something twice." Modal is
    the second compute backend with a non-SSH exec model (first
    was... well, there wasn't one). So we DON'T create a unified
    transport abstraction yet. Modal gets its own codepath.
    When/if a third non-SSH provider appears, compress then.

READ (lowest-level primitives — Modal's action space):
  Modal Sandbox API:
    modal.Sandbox.create(app, image, gpu, timeout, volumes, ...) → Sandbox
    sandbox.exec("bash", "-c", cmd, timeout, workdir) → ContainerProcess
    process.stdout  — iterable, streams line-by-line in real-time
    process.stderr  — iterable
    process.wait()  — blocks until done
    process.returncode — int
    sandbox.terminate()
    modal.Sandbox.from_id(sandbox_id) → reconnect to existing
    modal.Sandbox.from_name(app_name, name) → lookup by name
    sandbox.object_id — string identifier

  File upload: NO direct upload API. Two options:
    (a) image.add_local_file() at build time (baked into image)
    (b) sandbox.exec("bash", "-c", "printf '%s' '{b64}' | base64 -d > /path")
    Wafer uses (b) for runtime file sync: shell_app.py:884-895

  GPU types (plain strings):
    "T4" (~$0.59/hr), "L4" (~$0.80/hr), "A10G" (~$1.10/hr),
    "L40S" (~$1.95/hr), "A100-40GB" (~$2.10/hr), "A100-80GB" (~$2.50/hr),
    "H100" (~$3.95/hr), "H200" (~$4.54/hr), "B200" (~$6.25/hr)
  Multi-GPU: "H100:4" or modal.gpu.A100(count=4). Up to 8 GPUs.
  Max sandbox lifetime: 24 hours. Also supports idle_timeout.
  No search/pricing API — hardcode known GPU types.

  Auth: reads ~/.modal.toml automatically. Already configured
    (arb8020 account, active=true). No env vars needed unless
    overriding. ProviderCredentials doesn't need a modal field
    for basic usage — but add modal_token_id + modal_token_secret
    fields for explicit credential passing.

READ (existing broker code to modify):
  ~/research/broker/broker/providers/__init__.py — provider registry
  ~/research/broker/broker/providers/runpod.py — template for new provider
  ~/research/broker/broker/types.py — GPUInstance, ProviderModule protocol
  ~/research/broker/broker/api.py — PROVIDER_MODULES dict, search/create flows
  ~/research/broker/broker/ssh_clients_compat.py — SSH exec wrappers

READ (reference implementations):
  ~/research/dev/jax_basic/run_integration_test_modal.py — clean Sandbox usage
  ~/wafer/services/wafer-api/src/modal/shell_app.py:606-730 — Named Sandbox
  ~/wafer/packages/wafer-core/wafer_core/utils/modal_execution/modal_execution.py
    — subprocess isolation pattern (trio_asyncio conflict workaround)
  ~/research/docs/code_style/archive/domain/anyio_advice.md — async style guide

READ (usage code — what callers want to write):
  ~/research/rollouts/examples/sft/base_config.py — SFT remote execution
  ~/research/rollouts/examples/rl/base_config.py — RL remote execution
  ~/research/examples/provision_and_serve.py — inference serving

CHANGE 1: Modal provider (new file)
  ~/research/broker/broker/providers/modal.py
  Implements ProviderModule protocol:
    async def provision_instance(request, ssh_startup_script, api_key) → GPUInstance | None
      - modal.Sandbox.create(app, image, gpu=request.gpu_type, timeout=86400)
      - Store sandbox.object_id as instance ID
      - Return GPUInstance with provider="modal", no public_ip/ssh_port
    async def get_instance_details(instance_id, api_key) → GPUInstance | None
      - modal.Sandbox.from_id(instance_id), check if still alive
    async def list_instances(api_key) → list[GPUInstance]
      - May not be possible (Modal has no list-my-sandboxes API)
      - Return empty list or use modal.Sandbox.list() if it exists
    async def terminate_instance(instance_id, api_key) → bool
      - modal.Sandbox.from_id(instance_id).terminate()
    async def search_gpu_offers(...) → list[GPUOffer]
      - Return hardcoded offers for known GPU types + pricing
      - No availability check (Modal handles this at create time)

  GPUInstance.exec() override for Modal:
    Option: ModalGPUInstance subclass that overrides exec()/aexec()
    to call sandbox.exec() instead of SSH. Store sandbox reference
    in raw_data or as a dedicated field.

  All Modal SDK calls wrapped in trio.to_thread.run_sync() since
  they're blocking. If trio_asyncio assertion errors appear, switch
  to wafer's subprocess isolation pattern.

CHANGE 2: Async migration of existing providers
  All provider functions become async def.
  Replace `requests` with `httpx` (async client).
  Add trio, httpx to broker/pyproject.toml dependencies.
  Update ProviderModule protocol to async.
  Update api.py dispatch to await provider calls.

  Migration per provider (runpod, lambdalabs, vast, primeintellect,
  digitalocean, digitalocean_amd):
    - import httpx instead of requests
    - async def for all public functions
    - httpx.AsyncClient() for HTTP calls
    - Keep retry decorator (update shared/retry.py if needed for async)

CHANGE 3: Registry + types updates
  ~/research/broker/broker/providers/__init__.py — add "modal" case
  ~/research/broker/broker/types.py:
    - Add modal_token_id, modal_token_secret to ProviderCredentials
    - Update ProviderModule protocol methods to async
    - Consider ModalGPUInstance subclass for exec() override
  ~/research/broker/broker/api.py:
    - Add "modal" to PROVIDER_MODULES
    - Make search/create/terminate async

CHANGE 4: Dependencies
  ~/research/broker/pyproject.toml:
    - Add modal>=0.64.0
    - Add trio
    - Add httpx
    - Add anyio (comes with httpx but be explicit)

VERIFY:
  # Basic import
  python -c "from broker.providers.modal import provision_instance; print('ok')"

  # Smoke test: create sandbox, exec nvidia-smi, terminate
  python -c "
  import trio
  from broker.providers import modal as modal_provider
  from broker.types import ProvisionRequest

  async def main():
      request = ProvisionRequest(gpu_type='T4', gpu_count=1, name='broker-test')
      instance = await modal_provider.provision_instance(request)
      assert instance is not None
      result = instance.exec('nvidia-smi')
      print(result.stdout)
      assert result.success
      await modal_provider.terminate_instance(instance.id)
      print('PASS')

  trio.run(main)
  "

  # Verify existing providers still work after async migration
  python -c "
  import trio
  from broker.providers import runpod
  async def main():
      offers = await runpod.search_gpu_offers(gpu_count=1, api_key='...')
      print(f'{len(offers)} offers')
  trio.run(main)
  "

KEY RISKS:
  1. trio_asyncio conflict: Modal SDK uses asyncio internally.
     If trio is the event loop and trio_asyncio gets imported,
     Modal crashes with AssertionError. Mitigation: don't depend
     on trio_asyncio. If conflict appears, use subprocess isolation.
  2. Async migration scope: 6 providers + api.py + types.py + tests.
     May want to do providers incrementally (Modal first as async,
     others one-by-one) rather than big-bang.
  3. shared/retry.py may need async variant for httpx calls.

KEYWORDS: modal.Sandbox, sandbox.exec, trio.to_thread.run_sync,
  ProviderModule, PROVIDER_MODULES, httpx.AsyncClient, provision_instance
