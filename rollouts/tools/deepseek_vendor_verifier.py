"""DeepSeek V3.2 vendor verification against an OpenAI-compatible endpoint.

This adapts the Kimi Vendor Verifier workflow to a text-only, non-thinking model.
It checks:
1. Basic chat completion behavior and parameter acceptance
2. Streaming usage accounting (`stream_options.include_usage`)
3. Tool call formatting
4. A minimal multi-turn tool-use loop
5. Optional AIME 2025 smoke run through the local KVV repo
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import time
from dataclasses import dataclass
from typing import Any

import httpx
from openai import OpenAI

DEFAULT_BASE_URL = "http://localhost:30000/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_API_KEY = "dummy"


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


def make_client(base_url: str, api_key: str, timeout: float) -> OpenAI:
    return OpenAI(
        base_url=base_url,
        api_key=api_key,
        http_client=httpx.Client(timeout=timeout),
    )


def short_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


def require(condition: bool, detail: str) -> None:
    if not condition:
        raise AssertionError(detail)


def check_models_schema(client: OpenAI, model: str) -> CheckResult:
    try:
        response = client.models.list()
        models = list(response.data)
        require(models, "/v1/models returned no models")

        matched = next((item for item in models if getattr(item, "id", None) == model), None)
        require(matched is not None, f"model {model!r} missing from /v1/models")
        require(
            getattr(matched, "object", None) == "model", f"unexpected object={matched.object!r}"
        )
        require(bool(getattr(matched, "owned_by", None)), "owned_by missing")

        detail = {
            "id": matched.id,
            "owned_by": matched.owned_by,
            "root": getattr(matched, "root", None),
            "max_model_len": getattr(matched, "max_model_len", None),
        }
        return CheckResult("models_schema", True, short_json(detail))
    except Exception as exc:
        return CheckResult("models_schema", False, f"{type(exc).__name__}: {exc}")


def check_basic_params(client: OpenAI, model: str) -> list[CheckResult]:
    cases = [
        (
            "params_no_optional",
            {},
        ),
        (
            "params_kvv_nonthinking_defaults",
            {
                "temperature": 0.6,
                "top_p": 0.95,
                "presence_penalty": 0,
                "frequency_penalty": 0,
            },
        ),
        (
            "params_nondefault_values_accepted",
            {
                "temperature": 0.2,
                "top_p": 0.7,
                "presence_penalty": 0.1,
                "frequency_penalty": 0.1,
            },
        ),
    ]

    results: list[CheckResult] = []
    for name, extra in cases:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Reply with OK and nothing else."}],
                max_tokens=16,
                **extra,
            )
            content = response.choices[0].message.content or ""
            results.append(
                CheckResult(
                    name=name,
                    passed=True,
                    detail=f"accepted {short_json(extra)} -> {content[:40]!r}",
                )
            )
        except Exception as exc:
            results.append(
                CheckResult(
                    name=name,
                    passed=False,
                    detail=f"rejected {short_json(extra)} -> {type(exc).__name__}: {exc}",
                )
            )
    return results


def check_stream_usage(client: OpenAI, model: str) -> CheckResult:
    saw_usage = False
    saw_content = False
    final_usage: dict[str, int] | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": "Write exactly one short sentence about San Francisco."}
        ],
        max_tokens=48,
        temperature=0.6,
        top_p=0.95,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if chunk.choices:
            delta = chunk.choices[0].delta
            if delta.content:
                saw_content = True
        if chunk.usage:
            saw_usage = True
            final_usage = {
                "prompt_tokens": getattr(chunk.usage, "prompt_tokens", 0) or 0,
                "completion_tokens": getattr(chunk.usage, "completion_tokens", 0) or 0,
                "total_tokens": getattr(chunk.usage, "total_tokens", 0) or 0,
            }

    if not saw_content:
        return CheckResult("stream_usage", False, "stream produced no content deltas")
    if not saw_usage or not final_usage:
        return CheckResult("stream_usage", False, "stream never emitted usage")
    if final_usage["total_tokens"] <= 0:
        return CheckResult("stream_usage", False, f"invalid usage payload: {final_usage}")
    return CheckResult("stream_usage", True, f"usage={final_usage}")


def weather_tools() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "unit": {"type": "string", "enum": ["C", "F"]},
                    },
                    "required": ["city", "unit"],
                    "additionalProperties": False,
                },
            },
        }
    ]


def check_structured_output(client: OpenAI, model: str) -> CheckResult:
    schema = {
        "name": "city_state",
        "schema": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "state": {"type": "string"},
            },
            "required": ["city", "state"],
            "additionalProperties": False,
        },
    }
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Return San Francisco, California as JSON matching the schema.",
                }
            ],
            response_format={"type": "json_schema", "json_schema": schema},
            max_tokens=128,
            temperature=0.0,
        )
        content = response.choices[0].message.content or ""
        payload = json.loads(content)
        require(payload.get("city") == "San Francisco", f"unexpected payload: {payload}")
        require(
            payload.get("state") in {"CA", "California"},
            f"unexpected payload: {payload}",
        )
        return CheckResult("structured_output", True, short_json(payload))
    except Exception as exc:
        return CheckResult("structured_output", False, f"{type(exc).__name__}: {exc}")


def check_tool_call(client: OpenAI, model: str) -> CheckResult:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": "Call get_weather for San Francisco in Fahrenheit. Do not answer directly.",
            }
        ],
        tools=weather_tools(),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=256,
        temperature=0.0,
    )
    message = response.choices[0].message
    tool_calls = message.tool_calls or []
    if not tool_calls:
        detail = f"assistant returned no tool calls; content={message.content!r}"
        return CheckResult("tool_call", False, detail)

    tool_call = tool_calls[0]
    if tool_call.function.name != "get_weather":
        return CheckResult(
            "tool_call",
            False,
            f"wrong tool name: {tool_call.function.name!r}",
        )

    try:
        args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        return CheckResult(
            "tool_call",
            False,
            f"tool arguments were not valid JSON: {exc}: {tool_call.function.arguments!r}",
        )

    city = args.get("city")
    unit = args.get("unit")
    if city != "San Francisco" or unit != "F":
        return CheckResult(
            "tool_call",
            False,
            f"unexpected tool args: {args}",
        )

    return CheckResult(
        "tool_call",
        True,
        f"id={tool_call.id} args={args}",
    )


def check_agent_loop(client: OpenAI, model: str) -> CheckResult:
    user_prompt = "Use the tool then answer: what is the weather in San Francisco?"
    first = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        tools=weather_tools(),
        tool_choice={"type": "function", "function": {"name": "get_weather"}},
        max_tokens=256,
        temperature=0.0,
    )
    first_message = first.choices[0].message
    tool_calls = first_message.tool_calls or []
    if not tool_calls:
        detail = f"first turn produced no tool call; content={first_message.content!r}"
        return CheckResult("agent_loop", False, detail)

    tool_call = tool_calls[0]
    second = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            },
            {
                "role": "assistant",
                "content": first_message.content,
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps({
                    "city": "San Francisco",
                    "unit": "F",
                    "temperature": 61,
                    "condition": "foggy",
                }),
            },
        ],
        tools=weather_tools(),
        tool_choice="none",
        max_tokens=256,
        temperature=0.0,
    )
    final_text = (second.choices[0].message.content or "").lower()
    if "61" not in final_text or "fog" not in final_text:
        return CheckResult(
            "agent_loop",
            False,
            f"final answer did not use tool result: {second.choices[0].message.content!r}",
        )
    return CheckResult("agent_loop", True, second.choices[0].message.content or "")


def check_sustained_load(
    base_url: str,
    api_key: str,
    model: str,
    timeout: float,
    requests: int,
    concurrency: int,
) -> CheckResult:
    start = time.monotonic()
    failures: list[str] = []

    def run_one(index: int) -> int:
        client = make_client(base_url, api_key, timeout)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Reply exactly with OK and nothing else.",
                }
            ],
            max_tokens=8,
            temperature=0.0,
        )
        content = (response.choices[0].message.content or "").strip()
        require(content == "OK", f"unexpected content for request {index}: {content!r}")
        usage = response.usage
        require(
            usage is not None and (usage.total_tokens or 0) > 0,
            f"missing usage for request {index}",
        )
        return int(usage.total_tokens or 0)

    total_tokens = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        future_to_index = {executor.submit(run_one, index): index for index in range(requests)}
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                total_tokens += future.result()
            except Exception as exc:
                failures.append(f"req={index} {type(exc).__name__}: {exc}")

    elapsed = time.monotonic() - start
    success_count = requests - len(failures)
    if failures:
        return CheckResult(
            "sustained_load",
            False,
            f"{success_count}/{requests} succeeded in {elapsed:.2f}s; first_failure={failures[0]}",
        )

    rps = success_count / elapsed if elapsed > 0 else 0.0
    return CheckResult(
        "sustained_load",
        True,
        f"requests={requests} concurrency={concurrency} elapsed={elapsed:.2f}s rps={rps:.2f} total_tokens={total_tokens}",
    )


def run_aime_smoke(base_url: str, api_key: str, model: str, max_connections: int) -> CheckResult:
    cmd = [
        "uv",
        "run",
        "python",
        "eval.py",
        "aime2025",
        "--model",
        f"kimi/{model}",
        "--max-tokens",
        "16384",
        "--think-mode",
        "none",
        "--stream",
        "--epochs",
        "1",
        "--max-connections",
        str(max_connections),
    ]
    env = os.environ.copy()
    env["KIMI_BASE_URL"] = base_url
    env["KIMI_API_KEY"] = api_key
    proc = subprocess.run(
        cmd,
        cwd="/tmp/kvv",
        env=env,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = (proc.stdout + "\n" + proc.stderr).strip()[-1200:]
        return CheckResult("aime_smoke", False, detail)
    tail = (proc.stdout + "\n" + proc.stderr).strip()[-1200:]
    return CheckResult("aime_smoke", True, tail)


def print_result(result: CheckResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}: {result.detail}")


def main() -> int:
    parser = argparse.ArgumentParser(description="DeepSeek V3.2 vendor verification")
    parser.add_argument("--base-url", default=os.environ.get("KIMI_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("KIMI_API_KEY", DEFAULT_API_KEY))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--skip-aime", action="store_true")
    parser.add_argument("--skip-load", action="store_true")
    parser.add_argument("--aime-max-connections", type=int, default=8)
    parser.add_argument("--load-requests", type=int, default=32)
    parser.add_argument("--load-concurrency", type=int, default=8)
    args = parser.parse_args()

    client = make_client(args.base_url, args.api_key, args.timeout)

    results: list[CheckResult] = []
    results.append(check_models_schema(client, args.model))
    results.extend(check_basic_params(client, args.model))
    results.append(check_stream_usage(client, args.model))
    results.append(check_structured_output(client, args.model))
    results.append(check_tool_call(client, args.model))
    results.append(check_agent_loop(client, args.model))
    if not args.skip_load:
        results.append(
            check_sustained_load(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                timeout=args.timeout,
                requests=args.load_requests,
                concurrency=args.load_concurrency,
            )
        )

    if not args.skip_aime:
        results.append(
            run_aime_smoke(
                base_url=args.base_url,
                api_key=args.api_key,
                model=args.model,
                max_connections=args.aime_max_connections,
            )
        )

    print()
    for result in results:
        print_result(result)

    failed = [result for result in results if not result.passed]
    print()
    print(f"Summary: {len(results) - len(failed)}/{len(results)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
