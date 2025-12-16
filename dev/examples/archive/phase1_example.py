"""
Example showing Phase 1 API changes - explicit credentials and SSH keys
"""

import os

from broker import GPUClient

# Phase 1: Explicit credentials and SSH key (no auto-discovery)
client = GPUClient(
    credentials={
        "runpod": os.environ["RUNPOD_API_KEY"],
        # "vast": os.environ.get("VAST_API_KEY", ""),  # Ready for multi-provider
    },
    ssh_key_path="~/.ssh/id_ed25519",  # Required, no auto-discovery
)

# Search with provider filter (multi-provider ready)
offers = client.search(
    client.provider.in_(["runpod"]) & client.gpu_type.contains("RTX") & client.price_per_hour < 0.50
)

print(f"Found {len(offers)} offers")

# Create instance
if offers:
    instance = client.create(offers[0])
    if instance:
        print(f"Instance created: {instance.id}")

        # Wait for SSH (uses logging instead of prints)
        if instance.wait_until_ssh_ready(timeout=600):
            print("SSH ready!")

            # Execute command
            result = instance.exec("nvidia-smi")
            print(result.stdout)

            # Terminate
            instance.terminate()
            print("Instance terminated")
