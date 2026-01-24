"""Fibonacci GRPO training experiment.

Run with:
    # Local (requires GPU + SGLang)
    python examples/rl/fibonacci/grpo_01_01.py

    # Remote (provisions GPU automatically)
    python examples/rl/fibonacci/grpo_01_01.py --provision
"""

from examples.rl.fibonacci.base_config import train
from rollouts.training.grpo import GRPOConfig

config = GRPOConfig(
    experiment_name="fibonacci_grpo_01",
    model_name="Qwen/Qwen2.5-0.5B-Instruct",
    num_steps=10,
    checkpoint_every=5,
    batch_size=4,
    n_samples_per_prompt=4,
    temperature=0.8,
    lr=1e-6,
    max_seq_len=1024,
    max_tokens=512,
    max_turns=1,  # Single turn
    inference_cuda_device_ids=(0,),
    trainer_cuda_device_ids=(0,),
)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fibonacci GRPO training")
    parser.add_argument("--provision", action="store_true", help="Provision new GPU instance")
    parser.add_argument("--keep-alive", action="store_true", help="Keep GPU after completion")
    parser.add_argument("--node-id", type=str, help="Reuse existing instance ID")
    args = parser.parse_args()

    if args.provision or args.node_id:
        from examples.rl.base_config import run_remote

        run_remote(
            __file__,
            keep_alive=args.keep_alive,
            node_id=args.node_id,
        )
    else:
        results = train(config=config, n_prompts=16)
        print(f"Training complete. {len(results.get('metrics_history', []))} steps")
