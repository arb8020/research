"""Wordle - matches prime-rl nightly CI config (STUB - needs environment).

prime-rl config: examples/wordle/rl.toml
prime-rl test: tests/nightly/test_wordle.py

NOTE: This requires implementing a Wordle environment. prime-rl uses
`primeintellect/wordle` which is a hosted environment not available here.

TODO: Implement WordleEnvironment with:
- Tool: guess(word) -> returns feedback (green/yellow/gray for each letter)
- Max 6 guesses per game
- Reward: 1.0 for solving, 0.0 otherwise (or partial credit based on guesses)

prime-rl config reference:
    max_steps = 200
    seq_len = 8192
    [model] name = "PrimeIntellect/Qwen3-1.7B-Wordle-SFT"
    [orchestrator] batch_size = 1024, rollouts_per_example = 16
    [orchestrator.sampling] max_tokens = 1024
    inference_gpu_ids = [0,1,2,3,4,5]
    trainer_gpu_ids = [6,7]
    [inference.parallel] dp = 6

Run (once environment is implemented):
    python examples/rl/basic/wordle.py --modal
"""

raise NotImplementedError(
    "Wordle requires implementing WordleEnvironment. "
    "See prime-rl's primeintellect/wordle environment for reference."
)
