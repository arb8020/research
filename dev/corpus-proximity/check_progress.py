#!/usr/bin/env python3
"""Quick script to check clustering progress on remote instance."""

import sys
from broker import GPUClient
from bifrost import BifrostClient
from shared.config import get_runpod_key, get_prime_key
import os

def main():
    instance_id = sys.argv[1] if len(sys.argv) > 1 else None

    credentials = {}
    if key := get_runpod_key():
        credentials['runpod'] = key
    if key := get_prime_key():
        credentials['primeintellect'] = key

    client = GPUClient(credentials=credentials, ssh_key_path=os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))
    instances = client.list_instances()

    if instance_id:
        target = next((i for i in instances if i.id == instance_id or i.name == instance_id), None)
    else:
        # Find corpus-proximity instance
        target = next((i for i in instances if i.name and 'corpus' in i.name.lower()), None)

    if not target:
        print(f"No instance found matching: {instance_id or 'corpus*'}")
        print("\nAvailable instances:")
        for i in instances:
            print(f"  {i.id}: {i.name} ({i.status})")
        return 1

    print(f"Checking instance: {target.name} ({target.id})")
    print(f"Status: {target.status}")
    print(f"SSH: {target.ssh_connection_string()}")
    print()

    try:
        bf = BifrostClient(target.ssh_connection_string(), os.getenv('SSH_KEY_PATH', '~/.ssh/id_ed25519'))

        # Check for completion markers
        result = bf.exec('cd ~/.bifrost/workspace/examples/corpus-proximity && ls -la .clustering_* 2>/dev/null')
        if result.stdout.strip():
            print("Completion markers:")
            print(result.stdout)
            print()

        # Get recent log lines
        result = bf.exec('cd ~/.bifrost/workspace/examples/corpus-proximity && tail -50 pipeline.log 2>/dev/null || echo "No pipeline.log"')
        print("Recent pipeline log:")
        print("=" * 80)
        print(result.stdout)

    except Exception as e:
        print(f"Error connecting: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
