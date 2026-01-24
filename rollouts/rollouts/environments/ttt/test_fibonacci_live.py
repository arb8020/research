"""Live test: call a real model and grade the response."""

import trio
from anthropic import AsyncAnthropic

from .code_challenge import FibonacciEnv


async def main() -> None:
    env = FibonacciEnv(timeout=2.0)
    client = AsyncAnthropic()

    print("Prompt:")
    print(env.prompt)
    print("\n" + "=" * 60 + "\n")

    # Call the model
    print("Calling claude-sonnet-4-20250514...")
    response = await client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": env.prompt}],
    )

    response_text = response.content[0].text
    print("Response:")
    print(response_text)
    print("\n" + "=" * 60 + "\n")

    # Grade it
    score = await env.grade(response_text)

    print("Grading:")
    print(f"  Reward: {score.reward:.3f}")
    for m in score.metrics:
        meta = f" {m.metadata}" if m.metadata else ""
        print(f"  {m.name}: {m.value:.3f} (weight={m.weight}){meta}")


if __name__ == "__main__":
    trio.run(main)
