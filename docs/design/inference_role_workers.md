## Inference Roles Over Named Workers

This is the intended denotation for shared eval/RL inference ownership.

We want three distinct layers:

1. `HardwareConfig`
   - what allocation are we provisioning?
   - example: `4xH100`

2. `WorkerTopologyConfig`
   - what named long-lived workers run on that allocation?
   - examples:
     - `actor` inference worker on GPU 0
     - `judge` inference worker on GPU 1
     - `trainer` worker on GPUs 2,3

3. workload-level role binding
   - what semantic role points at which worker?
   - examples:
     - eval actor -> worker `actor`
     - eval judge -> worker `judge`
     - RL teacher -> worker `teacher`

The endpoint URL is a derived property of an inference worker, not the primary
modeling unit.

That means the intended authoring shape is:

```python
hardware = HardwareConfig(gpu_type="H100", gpu_count=4, ...)

topology = WorkerTopologyConfig(
    hardware=hardware,
    inference_workers=(
        InferenceWorkerConfig(
            worker_id="actor",
            model="model-a",
            inference=InferenceConfig(cuda_device_ids=(0,), ...),
        ),
        InferenceWorkerConfig(
            worker_id="judge",
            model="model-b",
            inference=InferenceConfig(cuda_device_ids=(1,), ...),
        ),
    ),
    training_workers=(
        TrainingWorkerConfig(
            worker_id="trainer",
            trainer=TrainerConfig(cuda_device_ids=(2, 3), ...),
        ),
    ),
    role_bindings=(
        InferenceRoleBinding(role="actor", worker_id="actor"),
        InferenceRoleBinding(role="judge", worker_id="judge"),
    ),
)
```

Current migration status:

- RL configs still primarily consume `trainer` + `inference` directly.
- eval configs still primarily consume `endpoint + server + hardware`.
- `WorkerTopologyConfig` is additive for now and lowers back into today's eval
  runner surface via `endpoint_and_server_for_role(...)`.

TODO(boundary): once eval and RL both consume named workers natively, stop
threading endpoint/server tuples separately through new config surfaces.
