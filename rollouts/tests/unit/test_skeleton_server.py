from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import ModuleType


def _load_skeleton_server_module() -> ModuleType:
    fastapi_stub = types.ModuleType("fastapi")
    fastapi_stub.FastAPI = object
    fastapi_stub.HTTPException = Exception
    sys.modules.setdefault("fastapi", fastapi_stub)

    responses_stub = types.ModuleType("fastapi.responses")
    responses_stub.StreamingResponse = object
    sys.modules.setdefault("fastapi.responses", responses_stub)

    uvicorn_stub = types.ModuleType("uvicorn")
    uvicorn_stub.run = lambda *args, **kwargs: None
    sys.modules.setdefault("uvicorn", uvicorn_stub)

    module_path = (
        Path(__file__).resolve().parents[2] / "rollouts" / "inference" / "skeleton_server.py"
    )
    spec = importlib.util.spec_from_file_location("test_skeleton_server_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_skeleton_server_fixed_stub_result_has_logprobs() -> None:
    module = _load_skeleton_server_module()

    result = module._fixed_stub_result()

    assert result.reply_text == "<answer>stub</answer>"
    assert result.finish_reason == "stop"
    assert len(result.token_logprobs) == 3
    assert result.token_logprobs[0].token == "<answer>"
    assert result.token_logprobs[1].top_logprobs[0] == ("stub", -0.02)


def test_skeleton_server_formats_choice_logprobs() -> None:
    module = _load_skeleton_server_module()

    formatted = module._format_choice_logprobs(module._fixed_stub_result().token_logprobs)

    assert formatted is not None
    assert formatted["content"][0]["token"] == "<answer>"
    assert formatted["content"][1]["logprob"] == -0.02
    assert formatted["content"][2]["top_logprobs"][0]["token"] == "</answer>"
