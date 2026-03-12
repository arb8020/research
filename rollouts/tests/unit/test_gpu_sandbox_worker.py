from __future__ import annotations

from rollouts.gpu_sandbox.worker import _parse_scoring_output


def test_parse_scoring_output_extracts_runtime_provenance() -> None:
    stdout = """
PROVENANCE_RESULT:{"hostname":"gpu-worker-1","machine":"x86_64","torch":{"cuda_version":"12.4","device_capability":[8,0],"device_name":"NVIDIA A100","version":"2.8.0"}}
COMPILE_SUCCESS
CORRECTNESS_RESULT:3/3
SPEEDUP_RESULT:1.2345
"""

    result = _parse_scoring_output(stdout, "", 0)

    assert result["compiled"] == 1.0
    assert result["correct"] == 1.0
    assert result["speedup"] == 1.2345
    assert result["runtime_provenance"] == {
        "hostname": "gpu-worker-1",
        "machine": "x86_64",
        "torch": {
            "cuda_version": "12.4",
            "device_capability": [8, 0],
            "device_name": "NVIDIA A100",
            "version": "2.8.0",
        },
    }
