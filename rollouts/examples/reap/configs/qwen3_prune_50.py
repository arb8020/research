"""Qwen3 MoE 50% compression config.

Run locally:
    python examples/reap/run_reap.py --config examples/reap/configs/qwen3_prune_50.py

Run remotely:
    python examples/reap/configs/qwen3_prune_50.py --provision
    python examples/reap/configs/qwen3_prune_50.py --node-id sf:abc123
"""

from pathlib import Path

from examples.reap.base_config import run_reap  # noqa: F401
from examples.reap.config import PruneMethod, ReapConfig

config = ReapConfig(
    model_name="Qwen/Qwen3-30B-A3B",
    dataset_name="theblackcat102/evol-codealpaca-v1",
    compression_ratio=0.5,
    prune_method=PruneMethod.REAP,
    num_samples=1024,
    max_seq_len=2048,
    seed=42,
    output_dir=Path("results/reap"),
    preserve_super_experts=True,
    cache_observations=True,
)


# For remote execution compatibility
def train(config=config):
    return run_reap(config)


if __name__ == "__main__":
    import argparse
    import functools

    import trio

    parser = argparse.ArgumentParser()
    parser.add_argument("--provision", action="store_true")
    parser.add_argument("--keep-alive", action="store_true")
    parser.add_argument("--node-id", type=str, default=None)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--gpu-type", type=str, default="A100")
    args = parser.parse_args()

    if args.provision or args.node_id:
        from examples.rl.base_config import run_remote

        trio.run(
            functools.partial(
                run_remote,
                script_path=__file__,
                keep_alive=args.keep_alive,
                node_id=args.node_id,
                gpu_count=args.gpu_count,
                gpu_type=args.gpu_type,
            )
        )
    else:
        result = run_reap(config)
        print(f"Output: {result['output_path']}")
        print(f"Experts: {result['original_num_experts']} -> {result['pruned_num_experts']}")
