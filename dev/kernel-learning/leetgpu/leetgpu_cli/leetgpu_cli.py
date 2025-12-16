#!/usr/bin/env python3
"""
LeetGPU CLI - Command-line interface for LeetGPU challenges
"""

import argparse
from pathlib import Path

from leetgpu_client import LeetGPUClient

USER_ID = "0e7dac69-c082-4aee-b652-36824ffe2f62"


def list_challenges(args):
    """List all available challenges."""
    client = LeetGPUClient(user_id=USER_ID)
    challenges = client.get_all_challenges()

    # Sort by ID
    challenges.sort(key=lambda c: c["id"])

    # Filter by difficulty if specified
    if args.difficulty:
        challenges = [c for c in challenges if c["difficulty"].lower() == args.difficulty.lower()]

    # Print header
    print(f"\n{'ID':<4} {'Title':<40} {'Difficulty':<10}")
    print("=" * 55)

    for c in challenges:
        print(f"{c['id']:<4} {c['title']:<40} {c['difficulty']:<10}")

    print(f"\nTotal: {len(challenges)} challenges")
    print("\nUse 'leetgpu init <challenge_id>' to start working on a challenge\n")


def init_challenge(args):
    """Initialize a challenge directory with starter code."""
    client = LeetGPUClient(user_id=USER_ID)

    # Determine if input is ID or name
    challenge_id = None
    if args.challenge.isdigit():
        challenge_id = int(args.challenge)
    else:
        # Search by name
        challenges = client.get_all_challenges()
        matches = [c for c in challenges if args.challenge.lower() in c["title"].lower()]

        if not matches:
            print(f"Error: No challenge found matching '{args.challenge}'")
            return
        elif len(matches) > 1:
            print(f"Multiple challenges found matching '{args.challenge}':")
            for m in matches:
                print(f"  {m['id']}: {m['title']}")
            print("\nPlease specify the exact challenge ID")
            return
        else:
            challenge_id = matches[0]["id"]

    # Fetch challenge details
    challenges = client.get_all_challenges()
    challenge = next((c for c in challenges if c["id"] == challenge_id), None)

    if not challenge:
        print(f"Error: Challenge {challenge_id} not found")
        return

    # Get starter code
    starter = client.get_starter_code(challenge_id)

    # Determine language
    language = args.language or "cute"
    if language not in starter["languages"]:
        print(f"Error: Language '{language}' not available for this challenge")
        print(f"Available languages: {', '.join(starter['languages'])}")
        return

    # Create directory in current working directory
    problem_name = challenge["title"].lower().replace(" ", "-")
    challenge_dir = Path.cwd() / problem_name

    if challenge_dir.exists() and not args.force:
        print(f"Error: Directory '{challenge_dir}' already exists")
        print("Use --force to overwrite")
        return

    challenge_dir.mkdir(exist_ok=True)

    # Determine file extension
    extensions = {
        "cute": "py",
        "cuda": "cu",
        "triton": "py",
        "mojo": "mojo",
        "pytorch": "py",
        "tinygrad": "py",
    }
    ext = extensions.get(language, "txt")

    # Write starter code
    code_file = challenge_dir / f"starter_kernel.{ext}"
    with open(code_file, "w") as f:
        f.write(starter["starter_code"][language])

    # Write problem spec
    spec_file = challenge_dir / "problem.md"
    with open(spec_file, "w") as f:
        f.write(f"# {challenge['title']}\n\n")
        f.write(f"**Difficulty:** {challenge['difficulty']}\n\n")
        f.write(f"**Challenge ID:** {challenge_id}\n\n")
        f.write(f"**Language:** {language}\n\n")
        f.write("## Problem Description\n\n")
        # Strip HTML tags for markdown (simple approach)
        import re

        spec_text = re.sub("<[^<]+?>", "", challenge["spec"])
        f.write(spec_text)

    # Write metadata
    metadata_file = challenge_dir / "metadata.json"
    import json

    with open(metadata_file, "w") as f:
        json.dump(
            {
                "challenge_id": challenge_id,
                "title": challenge["title"],
                "difficulty": challenge["difficulty"],
                "language": language,
            },
            f,
            indent=2,
        )

    # Create solution file with problem description as docstring + starter code
    solution_file = challenge_dir / f"solution_kernel.{ext}"
    with open(solution_file, "w") as f:
        import re

        spec_text = re.sub("<[^<]+?>", "", challenge["spec"])

        # For Python-like languages, use docstring
        if ext in ["py", "mojo"]:
            f.write('"""\n')
            f.write(f"{challenge['title']}\n\n")
            f.write(f"Difficulty: {challenge['difficulty']}\n")
            f.write(f"Challenge ID: {challenge_id}\n")
            f.write(f"Language: {language}\n\n")
            f.write(spec_text.strip())
            f.write('\n"""\n\n')
        else:
            # For C-like languages, use block comments
            f.write("/*\n")
            f.write(f" * {challenge['title']}\n")
            f.write(" *\n")
            f.write(f" * Difficulty: {challenge['difficulty']}\n")
            f.write(f" * Challenge ID: {challenge_id}\n")
            f.write(f" * Language: {language}\n")
            f.write(" *\n")
            for line in spec_text.strip().split("\n"):
                f.write(f" * {line}\n")
            f.write(" */\n\n")

        # Write starter code
        f.write(starter["starter_code"][language])

    print(f"✓ Created challenge directory: {challenge_dir}")
    print(f"  - {code_file.name} (starter code)")
    print(f"  - {solution_file.name} (problem + starter code for working)")
    print(f"  - {spec_file.name} (problem description)")
    print(f"  - {metadata_file.name} (metadata)")
    print("\nNext steps:")
    print(f"  1. cd {challenge_dir}")
    print(f"  2. Edit {solution_file.name}")
    print(f"  3. Run: leetgpu submit {solution_file.name}")


def submit_solution(args):
    """Submit a solution to LeetGPU."""
    import json

    solution_file = Path(args.file)
    if not solution_file.exists():
        print(f"Error: File '{solution_file}' not found")
        return

    # Try to find metadata.json
    metadata_file = solution_file.parent / "metadata.json"
    if not metadata_file.exists():
        print("Error: metadata.json not found in the challenge directory")
        print("Use 'leetgpu init' to create a proper challenge directory")
        return

    with open(metadata_file) as f:
        metadata = json.load(f)

    # Read solution code
    with open(solution_file) as f:
        code = f.read()

    # Submit
    client = LeetGPUClient(user_id=USER_ID)
    gpu = args.gpu or "NVIDIA TESLA T4"

    print(f"Submitting solution for challenge {metadata['challenge_id']} ({metadata['title']})...")
    print(f"  Language: {metadata['language']}")
    print(f"  GPU: {gpu}")

    try:
        result = client.submit_code(
            challenge_id=metadata["challenge_id"],
            code=code,
            language=metadata["language"],
            gpu=gpu,
            is_public=not args.private,
        )
        print("\n✓ Submission successful!")
        print(f"  Submission ID: {result.submission_id}")
        print(f"  Status: {result.status}")
        if result.runtime > 0:
            print(f"  Runtime: {result.runtime}ms")
        if result.percentile > 0:
            print(f"  Percentile: {result.percentile}%")
    except Exception as e:
        print(f"\n✗ Submission failed: {e}")


def show_info(args):
    """Show information about a challenge."""
    client = LeetGPUClient(user_id=USER_ID)

    # Determine if input is ID or name
    challenge_id = None
    if args.challenge.isdigit():
        challenge_id = int(args.challenge)
    else:
        # Search by name
        challenges = client.get_all_challenges()
        matches = [c for c in challenges if args.challenge.lower() in c["title"].lower()]

        if not matches:
            print(f"Error: No challenge found matching '{args.challenge}'")
            return
        elif len(matches) > 1:
            print("Multiple challenges found:")
            for m in matches:
                print(f"  {m['id']}: {m['title']}")
            return
        else:
            challenge_id = matches[0]["id"]

    # Fetch challenge
    challenges = client.get_all_challenges()
    challenge = next((c for c in challenges if c["id"] == challenge_id), None)

    if not challenge:
        print(f"Error: Challenge {challenge_id} not found")
        return

    # Get starter code to show available languages
    starter = client.get_starter_code(challenge_id)

    print(f"\n{'=' * 60}")
    print(f"{challenge['title']}")
    print(f"{'=' * 60}")
    print(f"ID:         {challenge['id']}")
    print(f"Difficulty: {challenge['difficulty']}")
    print(f"Languages:  {', '.join(starter['languages'])}")
    print("\nDescription:")
    print(challenge["spec"])
    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="LeetGPU CLI - Work with GPU programming challenges",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_parser = subparsers.add_parser("list", help="List all challenges")
    list_parser.add_argument(
        "-d", "--difficulty", choices=["easy", "medium", "hard"], help="Filter by difficulty"
    )
    list_parser.set_defaults(func=list_challenges)

    # Init command
    init_parser = subparsers.add_parser("init", help="Initialize a challenge directory")
    init_parser.add_argument("challenge", help="Challenge ID or name")
    init_parser.add_argument(
        "-l", "--language", default="cute", help="Programming language (default: cute)"
    )
    init_parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing directory"
    )
    init_parser.set_defaults(func=init_challenge)

    # Submit command
    submit_parser = subparsers.add_parser("submit", help="Submit a solution")
    submit_parser.add_argument("file", help="Solution file to submit")
    submit_parser.add_argument("-g", "--gpu", help="GPU to run on (default: NVIDIA TESLA T4)")
    submit_parser.add_argument(
        "-p", "--private", action="store_true", help="Make submission private"
    )
    submit_parser.set_defaults(func=submit_solution)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show challenge information")
    info_parser.add_argument("challenge", help="Challenge ID or name")
    info_parser.set_defaults(func=show_info)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
