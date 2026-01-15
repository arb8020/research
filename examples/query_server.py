#!/usr/bin/env python3
"""Demo: Query a deployed SGLang/vLLM server.

Usage:
    # Interactive mode
    python examples/query_server.py --url http://gpu.example.com:30000/v1

    # Single query
    python examples/query_server.py --url http://gpu.example.com:30000/v1 --prompt "What is 2+2?"

    # With custom model name (if server hosts multiple models)
    python examples/query_server.py --url http://gpu.example.com:30000/v1 --model "Qwen/Qwen2.5-7B-Instruct"

Works with any OpenAI-compatible endpoint (SGLang, vLLM, etc).
"""

from __future__ import annotations

import argparse
import asyncio
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query SGLang/vLLM server")

    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="API base URL (e.g., http://gpu:30000/v1)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: auto-detect from server)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Single prompt to run (default: interactive mode)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens to generate (default: 512)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming output",
    )

    return parser.parse_args()


async def get_model_name(api_base: str) -> str | None:
    """Auto-detect model name from server."""
    import httpx

    # Normalize URL
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        models_url = f"{base}/models"
    else:
        models_url = f"{base}/v1/models"

    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(models_url, timeout=10.0)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("data") and len(data["data"]) > 0:
                    return data["data"][0]["id"]
    except Exception as e:
        print(f"Warning: Could not auto-detect model: {e}")

    return None


async def query_streaming(
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Query with streaming output."""
    from openai import AsyncOpenAI

    # Normalize URL
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    client = AsyncOpenAI(
        base_url=base,
        api_key="EMPTY",  # SGLang/vLLM don't require API key
    )

    response_text = ""

    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )

    async for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end="", flush=True)
            response_text += content

    print()  # Newline after streaming
    return response_text


async def query_sync(
    api_base: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
) -> str:
    """Query without streaming."""
    from openai import AsyncOpenAI

    # Normalize URL
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"

    client = AsyncOpenAI(
        base_url=base,
        api_key="EMPTY",
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        stream=False,
    )

    content = response.choices[0].message.content or ""
    print(content)
    return content


async def interactive_mode(
    api_base: str,
    model: str,
    temperature: float,
    max_tokens: int,
    stream: bool,
) -> None:
    """Run interactive chat loop."""
    print(f"Connected to: {api_base}")
    print(f"Model: {model}")
    print(f"Temperature: {temperature}, Max tokens: {max_tokens}")
    print()
    print("Enter prompts (Ctrl+C to exit):")
    print("-" * 40)

    query_fn = query_streaming if stream else query_sync

    while True:
        try:
            prompt = input("\n> ").strip()
            if not prompt:
                continue

            print()
            await query_fn(api_base, model, prompt, temperature, max_tokens)

        except KeyboardInterrupt:
            print("\n\nExiting.")
            break
        except Exception as e:
            print(f"Error: {e}")


async def main() -> int:
    args = parse_args()

    # Check dependencies
    try:
        from openai import AsyncOpenAI  # noqa: F401
    except ImportError:
        print("Missing dependency: openai")
        print("Install with: pip install openai")
        return 1

    # Auto-detect model if not specified
    model = args.model
    if model is None:
        print(f"Auto-detecting model from {args.url}...")
        model = await get_model_name(args.url)
        if model is None:
            print("Could not auto-detect model. Please specify --model")
            return 1
        print(f"Detected model: {model}")

    # Single query or interactive mode
    if args.prompt:
        query_fn = query_streaming if args.stream else query_sync
        await query_fn(
            args.url,
            model,
            args.prompt,
            args.temperature,
            args.max_tokens,
        )
    else:
        await interactive_mode(
            args.url,
            model,
            args.temperature,
            args.max_tokens,
            args.stream,
        )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
