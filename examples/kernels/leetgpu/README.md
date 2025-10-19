# LeetGPU CLI

Command-line interface for working with LeetGPU challenges programmatically, with support for CuteDSL and other GPU programming frameworks.

## Installation

### Using uv (recommended)

From the research workspace root:

```bash
uv sync --extra example-leetgpu
```

### Using pip

```bash
cd examples/kernels
pip install -e .
```

### Running the CLI

If the `leetgpu` command doesn't work after installation, run the CLI directly from the `examples/kernels` directory:

```bash
python leetgpu/leetgpu_cli/leetgpu_cli.py list
python leetgpu/leetgpu_cli/leetgpu_cli.py init 2 --language cute
python leetgpu/leetgpu_cli/leetgpu_cli.py submit starter_kernel.py
```

## Usage

### List all challenges

```bash
leetgpu list
```

Filter by difficulty:
```bash
leetgpu list --difficulty easy
leetgpu list --difficulty medium
leetgpu list --difficulty hard
```

### Initialize a challenge

By challenge ID:
```bash
leetgpu init 1
```

By challenge name (partial match):
```bash
leetgpu init "vector addition"
leetgpu init "matrix mult"
```

With a specific language:
```bash
leetgpu init 1 --language cute     # CuteDSL (default)
leetgpu init 1 --language cuda     # CUDA
leetgpu init 1 --language triton   # Triton
leetgpu init 1 --language mojo     # Mojo
leetgpu init 1 --language pytorch  # PyTorch
```

This creates a directory in your current working directory:
```
vector-addition/
  ├── starter_kernel.py   # Starter code template
  ├── solution_kernel.py  # Problem description (as comments) + starter code for working
  ├── problem.md          # Problem description
  └── metadata.json       # Challenge metadata
```

### Show challenge information

```bash
leetgpu info 1
leetgpu info "prefix sum"
```

### Submit a solution

```bash
cd vector-addition
# Edit solution_kernel.py with your solution (contains problem description as comments)
leetgpu submit solution_kernel.py
```

With options:
```bash
leetgpu submit starter_kernel.py --gpu "NVIDIA RTX 3070"
leetgpu submit starter_kernel.py --private  # Make submission private
```

## Supported Languages

- **cute** - CuteDSL (Python-based CUDA DSL)
- **cuda** - NVIDIA CUDA C++
- **triton** - OpenAI Triton
- **mojo** - Mojo
- **pytorch** - PyTorch
- **tinygrad** - TinyGrad

## Available GPUs

- NVIDIA GTX TITAN X
- NVIDIA GV100
- NVIDIA QV100
- NVIDIA TITAN V
- NVIDIA RTX 2060 Super
- NVIDIA RTX 3070
- NVIDIA TESLA T4 (default)

## Example Workflow

```bash
# 1. List challenges to find one you want to solve
leetgpu list

# 2. Initialize the challenge (e.g., Vector Addition)
leetgpu init 1

# 3. Navigate to the challenge directory
cd vector-addition

# 4. Read the problem
cat problem.md

# 5. Edit the solution file (has problem description as comments + starter code)
vim solution_kernel.py

# 6. Submit your solution
leetgpu submit solution_kernel.py

# 7. Check results
# The submission will show status, runtime, and percentile ranking
```

## Files

- `leetgpu_cli.py` - Main CLI script
- `leetgpu_client.py` - Python API client
- `setup.py` - Package setup for pip installation
- `capture_leetgpu_api.py` - Playwright script for API exploration (optional)

## Notes

- Your user ID is configured in `leetgpu_cli.py`
- Submissions are public by default (use `--private` flag to make them private)
- The CLI automatically handles language-specific file extensions (.py, .cu, .mojo, etc.)
- Challenge directories are created in your current working directory
