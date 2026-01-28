# Outside-In Programming (draft)

quotes from conversation (exact):

> "a lot of times the best way to write code is outside in. you start
> at the lowest level primitive you care to control (ex: ssh client) and
> the highest level usage code (usage code someone wants to write), and
> then the path becomes clear."

> "outside in programming would help us think about the lowest level
> primitive we control (what is the action space modal gives us), the
> exact behaviors we want (vllm endpoint, trainer backend, etc), in
> order to help us design the abstractions (changes that might need to
> happen in bifrost to support modal)"

---

## worked example: adding Modal to broker

### pin the bottom end (lowest primitive we control)

Modal Sandbox API — this is the action space:

    Sandbox.create(app, image, gpu, timeout, volumes) → Sandbox
    sandbox.exec("bash", "-c", cmd) → ContainerProcess (stdout/stderr/returncode)
    sandbox.terminate()
    Sandbox.from_id(id) → reconnect to existing sandbox

no SSH. no public IP. no SFTP. no port forwarding (just HTTP tunnels).
file upload = exec("base64 -d > /path") or bake into image.

### pin the top end (usage code someone wants to write)

three real use cases from rollouts/:

1. eval (exec + collect):
   instance = broker.create(provider="modal", gpu_type="A100")
   result = instance.exec("python eval.py")
   instance.terminate()

2. training smoke-test (push code + run + stream logs):
   instance = broker.create(provider="modal", gpu_type="A100")
   # somehow get code onto the sandbox...
   result = instance.exec("cd /workspace && python train.py")
   instance.terminate()

3. inference serving (provision + deploy server):
   instance = broker.create(provider="modal", gpu_type="H100")
   # server runs, you hit it via HTTP tunnel
   instance.terminate()

### what the middle reveals

with both ends pinned, the design questions answer themselves:

Q: should bifrost support Modal?
bifrost is paramiko all the way down. Modal has no SSH.
wrapping one to look like the other = "dishonest about problem shape"
(keeping_llm_code_honest.md). answer: no, separate codepath.

Q: should we build a unified transport abstraction?
"don't abstract until you've done something twice." Modal is the
first non-SSH backend. we don't have two examples to compress from.
answer: not yet. when a third non-SSH provider appears, we'll see
the real seam.

Q: how does GPUInstance.exec() work for Modal?
bottom end says sandbox.exec() returns stdout/stderr/returncode.
top end says instance.exec("cmd") returns SSHResult-shaped thing.
answer: ModalGPUInstance subclass overrides exec() to call
sandbox.exec() instead of paramiko. same interface, honest
implementation.

Q: how do we get code onto the sandbox?
bottom end says: image.add_local_file() (build time) or
sandbox.exec("base64 -d > /path") (runtime). no SFTP.
top end says: caller wants to run a script that exists locally.
answer: for now, bake code into image or base64-exec. bifrost's
git-sync doesn't apply. this is an honest difference — Modal
sandboxes are ephemeral containers, not persistent VMs.

---

## the principle

you have two things you can't change:
- the lowest level primitive you care to control
- what the caller wants to write

everything in between is your design space. pin both ends and
the middle becomes constrained enough to see clearly.

this is "write usage code first" (casey) applied bidirectionally.
casey says pin the top. this says pin the top AND the bottom.
the bottom matters because it tells you what's actually possible,
which prevents you from designing APIs that require impossible
implementations.

bad code happens when you design the middle without pinning
either end. you end up with abstractions that don't match the
primitives (leaky) or don't match the usage (awkward).
