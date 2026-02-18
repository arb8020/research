# Metal graph execution - STUBBED for agent restoration exercise
#
# Original implementation had:
# - MetalGraph: Batches multiple kernel executions using indirect command buffers
#
# The agent must implement this to restore Metal graph support.

from tinygrad.engine.jit import GraphRunner

class MetalGraph(GraphRunner):
    def __init__(self, jit_cache, input_buffers, var_vals, orig_valid_positions=None):
        raise NotImplementedError("TODO: implement MetalGraph - batches kernel executions using indirect command buffers")

    def __call__(self, input_buffers, var_vals, wait=False):
        raise NotImplementedError("TODO: implement __call__ - executes the batched graph")
