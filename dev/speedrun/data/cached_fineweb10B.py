import os
import sys
import time

from huggingface_hub import hf_hub_download

# Set increased timeout for HuggingFace downloads via environment variable
# HF uses httpx internally and respects this timeout (in seconds)
# Default timeout is too short for large files on slow connections
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")  # 10 minutes


# Download the GPT-2 tokens of Fineweb10B from huggingface. This
# saves about an hour of startup time compared to regenerating them.
def get(fname, max_retries=5):
    """Download a file with retry logic.

    Args:
        fname: Filename to download
        max_retries: Maximum number of retry attempts (default: 5)
    """
    local_dir = os.path.join(os.path.dirname(__file__), "fineweb10B")
    if os.path.exists(os.path.join(local_dir, fname)):
        return  # File already exists

    for attempt in range(max_retries):
        try:
            print(f"Downloading {fname} (attempt {attempt + 1}/{max_retries})...")

            hf_hub_download(
                repo_id="kjj0/fineweb10B-gpt2",
                filename=fname,
                repo_type="dataset",
                local_dir=local_dir,
            )

            print(f"✓ Successfully downloaded {fname}")
            return  # Success!

        except Exception as e:
            error_str = str(e)
            is_timeout = "timeout" in error_str.lower() or "timed out" in error_str.lower()

            if is_timeout:
                print(f"✗ Timeout on attempt {attempt + 1}/{max_retries}")
            else:
                print(f"✗ Error on attempt {attempt + 1}/{max_retries}: {type(e).__name__}")

            if attempt < max_retries - 1:
                # Exponential backoff: wait longer between retries
                wait_time = 2**attempt  # 1s, 2s, 4s, 8s, 16s
                print(f"  Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"✗ Failed to download {fname} after {max_retries} attempts")
                print(f"  Final error: {e}")
                raise


get("fineweb_val_%06d.bin" % 0)
num_chunks = 103  # full fineweb10B. Each chunk is 100M tokens
if len(sys.argv) >= 2:  # we can pass an argument to download less
    num_chunks = int(sys.argv[1])
for i in range(1, num_chunks + 1):
    get("fineweb_train_%06d.bin" % i)
