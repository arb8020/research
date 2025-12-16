"""
Python client for submitting code to LeetGPU programmatically.
"""

from dataclasses import dataclass
from typing import Literal

import requests


@dataclass
class SubmissionResult:
    """Result of a code submission."""

    submission_id: int
    status: str
    gpu: str
    language: str
    runtime: float
    percentile: float
    created_at: str
    is_public: bool


class LeetGPUClient:
    """Client for interacting with the LeetGPU API."""

    BASE_URL = "https://api.leetgpu.com/api/v1"

    def __init__(self, user_id: str):
        """
        Initialize the LeetGPU client.

        Args:
            user_id: Your LeetGPU user ID (found in browser requests)
        """
        self.user_id = user_id
        self.session = requests.Session()
        self.session.headers.update({
            "accept": "*/*",
            "origin": "https://leetgpu.com",
            "referer": "https://leetgpu.com/",
            "x-user-id": user_id,
        })

    def submit_code(
        self,
        challenge_id: int,
        code: str,
        language: Literal["cute", "cuda", "triton", "mojo", "pytorch", "tinygrad"],
        gpu: str = "NVIDIA TESLA T4",
        is_public: bool = True,
    ) -> SubmissionResult:
        """
        Submit code to a LeetGPU challenge.

        Args:
            challenge_id: The challenge ID (e.g., 1)
            code: Your source code as a string
            language: Programming language/framework
            gpu: GPU model to run on
            is_public: Whether to make submission public

        Returns:
            SubmissionResult with submission details

        Example:
            >>> client = LeetGPUClient(user_id="your-user-id")
            >>> code = '''
            ... # Your CuteDSL code here
            ... '''
            >>> result = client.submit_code(
            ...     challenge_id=1,
            ...     code=code,
            ...     language="cute",
            ...     gpu="NVIDIA TESLA T4"
            ... )
            >>> print(f"Submission {result.submission_id}: {result.status}")
        """
        # Based on the API endpoint structure, we'll POST to submissions
        url = f"{self.BASE_URL}/challenges/submissions"

        # Construct payload (structure to be confirmed by capture script)
        payload = {
            "challenge_id": challenge_id,
            "language": language,
            "gpu": gpu,
            "code": code,
            "is_public": is_public,
        }

        response = self.session.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return SubmissionResult(**data)

    def get_all_challenges(self) -> list[dict]:
        """
        Get all available challenges.

        Returns:
            List of challenge dictionaries with id, title, difficulty, spec, etc.

        Example:
            >>> client = LeetGPUClient(user_id="your-user-id")
            >>> challenges = client.get_all_challenges()
            >>> for c in challenges:
            ...     print(f"{c['id']}: {c['title']} ({c['difficulty']})")
        """
        url = f"{self.BASE_URL}/challenges/fetch-all"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_starter_code(self, challenge_id: int) -> dict:
        """
        Get starter code for a challenge in all available languages.

        Args:
            challenge_id: The challenge ID

        Returns:
            Dictionary with 'languages' list and 'starter_code' dict keyed by language

        Example:
            >>> client = LeetGPUClient(user_id="your-user-id")
            >>> starter = client.get_starter_code(challenge_id=1)
            >>> print(starter['languages'])  # ['triton', 'mojo', 'pytorch', 'cuda', 'cute']
            >>> print(starter['starter_code']['cute'])  # CuteDSL starter code
        """
        url = f"{self.BASE_URL}/challenges/{challenge_id}/starter-code"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_submissions(
        self,
        challenge_id: int,
        language: str | None = None,
        gpu: str | None = None,
    ) -> list[dict]:
        """
        Get submissions for a challenge.

        Args:
            challenge_id: The challenge ID
            language: Filter by language (optional)
            gpu: Filter by GPU (optional)

        Returns:
            List of submission dictionaries
        """
        url = f"{self.BASE_URL}/challenges/submissions"
        params = {"challenge_id": challenge_id}

        if language:
            params["language"] = language
        if gpu:
            params["gpu"] = gpu

        response = self.session.get(url, params=params)
        response.raise_for_status()

        return response.json()

    def submit_file(
        self,
        challenge_id: int,
        file_path: str,
        language: str,
        gpu: str = "NVIDIA TESLA T4",
        is_public: bool = True,
    ) -> SubmissionResult:
        """
        Submit code from a file.

        Args:
            challenge_id: The challenge ID
            file_path: Path to source file
            language: Programming language/framework
            gpu: GPU model to run on
            is_public: Whether to make submission public

        Returns:
            SubmissionResult with submission details
        """
        with open(file_path) as f:
            code = f.read()

        return self.submit_code(
            challenge_id=challenge_id,
            code=code,
            language=language,
            gpu=gpu,
            is_public=is_public,
        )


def main():
    """Example usage."""
    # Replace with your user ID from browser requests
    USER_ID = "0e7dac69-c082-4aee-b652-36824ffe2f62"

    client = LeetGPUClient(user_id=USER_ID)

    # Example 1: List all challenges
    print("=" * 60)
    print("Available Challenges")
    print("=" * 60)
    challenges = client.get_all_challenges()
    for c in challenges[:5]:
        print(f"{c['id']:2d}. {c['title']:30s} [{c['difficulty']}]")
    print(f"\n... and {len(challenges) - 5} more challenges\n")

    # Example 2: Get starter code for challenge 1
    print("=" * 60)
    print("Starter Code for Challenge 1 (Vector Addition)")
    print("=" * 60)
    starter = client.get_starter_code(challenge_id=1)
    print(f"Available languages: {', '.join(starter['languages'])}\n")
    print("CuteDSL starter code:")
    print("-" * 60)
    print(starter["starter_code"]["cute"])
    print("-" * 60)

    # Example 3: Get previous submissions
    print("\n" + "=" * 60)
    print("Previous Submissions for Challenge 1")
    print("=" * 60)
    try:
        submissions = client.get_submissions(challenge_id=1, language="cute")
        if submissions:
            print(f"Found {len(submissions)} submissions")
            for sub in submissions[:3]:
                print(f"  - {sub['id']}: {sub['status']} on {sub['gpu']}")
        else:
            print("No previous submissions found")
    except Exception as e:
        print(f"Could not fetch submissions: {e}")

    # Example 4: Submit code (commented out)
    print("\n" + "=" * 60)
    print("To Submit Code")
    print("=" * 60)
    print("Uncomment the following to submit:")
    print('''
    code = """
    import cutlass
    import cutlass.cute as cute

    @cute.jit
    def solve(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor, N: cute.Uint32):
        # Your solution here
        pass
    """

    result = client.submit_code(
        challenge_id=1,
        code=code,
        language="cute",
        gpu="NVIDIA TESLA T4"
    )
    print(f"Submission {result.submission_id}: {result.status}")
    ''')


if __name__ == "__main__":
    main()
