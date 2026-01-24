"""Quick smoke test for FibonacciEnv."""

import trio

from .code_challenge import FibonacciEnv

# Test responses of varying quality
NAIVE_RECURSIVE = """
Here's a simple recursive solution:

```python
def fib(n):
    if n <= 1:
        return n
    return fib(n-1) + fib(n-2)
```
"""

MEMOIZED = """
```python
def fib(n):
    memo = {0: 0, 1: 1}
    def helper(k):
        if k not in memo:
            memo[k] = helper(k-1) + helper(k-2)
        return memo[k]
    return helper(n)
```
"""

ITERATIVE = """
```python
def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```
"""

CLOSED_FORM = """
```python
def fib(n):
    phi = (1 + 5**0.5) / 2
    psi = (1 - 5**0.5) / 2
    return int((phi**n - psi**n) / 5**0.5 + 0.5)
```
"""

BROKEN = """
```python
def fib(n):
    return n  # wrong!
```
"""


async def main() -> None:
    env = FibonacciEnv(timeout=2.0)
    print(f"Prompt:\n{env.prompt[:200]}...\n")

    test_cases = [
        ("Naive recursive (will timeout)", NAIVE_RECURSIVE),
        ("Memoized", MEMOIZED),
        ("Iterative", ITERATIVE),
        ("Closed form", CLOSED_FORM),
        ("Broken", BROKEN),
    ]

    for name, response in test_cases:
        score = await env.grade(response)
        print(f"{name}:")
        print(f"  Reward: {score.reward:.3f}")
        for m in score.metrics:
            print(f"  {m.name}: {m.value:.3f} (weight={m.weight})")
        print()


if __name__ == "__main__":
    trio.run(main)
