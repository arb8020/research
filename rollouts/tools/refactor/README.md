# refactor

Multi-file AI refactoring tool. Write a task as a trailing comment, run the tool, get your files edited.

Inspired by [Victor Taelin's refactor.ts](https://github.com/VictorTaelin/AI-scripts).

## Usage

```bash
cd ~/research/rollouts
source .venv/bin/activate

# Basic usage
python -m tools.refactor.refactor <file>

# With options
python -m tools.refactor.refactor <file> --model openai/gpt-5.1
python -m tools.refactor.refactor <file> --thinking high
python -m tools.refactor.refactor <file> --dry-run
```

## Workflow

1. Open a file at the top of your dependency tree
2. Write your task as a trailing comment:

```python
from utils import helper_function

def main():
    result = helper_function(42)
    print(result)

# move helper_function to a new file called helpers.py
# update the import
```

3. Run the tool:

```bash
python -m tools.refactor.refactor main.py
```

4. Done! Files are edited automatically.

## How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  1. Extract task from trailing comments                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Collect context (follow imports recursively)            │
│     - Python: from .foo import bar                          │
│     - JS/TS: import x from './foo'                          │
│     - Lua: require('foo')                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Split files into blocks with IDs                        │
│                                                             │
│     !0                                                      │
│     from utils import helper                                │
│                                                             │
│     !1                                                      │
│     def main():                                             │
│         ...                                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Compaction (if >32k tokens)                             │
│     Fast model identifies irrelevant blocks to omit         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Main model generates commands                           │
│     <write file="helpers.py">...</write>                    │
│     <patch id=0>new import</patch>                          │
│     <delete file="old.py"/>                                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Apply commands to filesystem                            │
└─────────────────────────────────────────────────────────────┘
```

## Options

| Flag | Description |
|------|-------------|
| `--model`, `-m` | Model in `provider/model` format (default: `anthropic/claude-sonnet-4-5-20250929`) |
| `--thinking`, `-t` | Thinking level: `none`, `low`, `medium`, `high` (default: `medium`) |
| `--dry-run`, `-n` | Print prompt without calling API |
| `--no-compact` | Skip compaction pass |

## Neovim Keybindings

Add to your `init.lua` (already configured):

| Key | Model |
|-----|-------|
| `<leader>rs` | Claude Sonnet |
| `<leader>rS` | Claude Sonnet (high thinking) |
| `<leader>rg` | GPT-5.1 |
| `<leader>rG` | GPT-5.1 (high) |
| `<leader>ri` | Gemini 3 |
| `<leader>rI` | Gemini 3 (high) |
| `<leader>ro` | Claude Opus |
| `<leader>rO` | Claude Opus (high) |
| `<leader>rd` | Dry run (preview) |

## Example

```bash
$ python -m tools.refactor.refactor main.py

model: anthropic/claude-sonnet-4-5-20250929
thinking: medium
task: move helper_function to helpers.py
Collecting context...
context: 3 files
blocks: 9
tokens: ~1250
compaction: skipped (under 32000 tokens)

** Calling AI... **
Applying 2 commands...
  ✓ Wrote helpers.py
  ✓ Patched block !2 in main.py

Done!
```

## Logs

All prompts and responses are saved to `~/.ai/`:

```
~/.ai/
├── refactor-prompt.txt      # Latest prompt
├── refactor-response.txt    # Latest response
└── refactor-history/        # Timestamped history
    ├── 20241211-143022-main-prompt.txt
    └── 20241211-143022-main-response.txt
```

## API Keys

Set via environment variable:

```bash
export ANTHROPIC_API_KEY="sk-..."
export OPENAI_API_KEY="sk-..."
export GOOGLE_API_KEY="..."
```

Or create token files:

```bash
echo "sk-..." > ~/.config/anthropic.token
echo "sk-..." > ~/.config/openai.token
```
