"""K2VV-shape tool-call conformance verifier.

See runner.run_tool_call_verifier_workload for the public entrypoint.
"""

from .runner import run_tool_call_verifier_workload

__all__ = ["run_tool_call_verifier_workload"]
