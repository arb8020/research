# csearch

Code search CLI for symbol definitions and references. Built for use with LLM agents.

## Install

```bash
uv pip install -e .
```

## Usage

```bash
# Find symbol definitions (functions, classes)
csearch defs MyClass -p /path/to/repo
csearch defs my_function -p . --snippet

# Find references (call sites)
csearch refs MyClass -p /path/to/repo

# With limits
csearch defs MyClass -p . --limit 10
csearch refs MyClass -p . --no-limit

# Build trigram index for faster searches
csearch index /path/to/repo
csearch index /path/to/repo --status
```

## Output Format

```
path/to/file.py:10-25 function my_func
path/to/file.py:30-45 class MyClass
```

With `--snippet`:
```
path/to/file.py:10-25
def my_func():
    ...
```

## How It Works

1. **Without index**: Scans all files, parses with tree-sitter
2. **With index**: Uses trigram index to find candidate files, then parses only those

The trigram index maps 3-character sequences to files containing them. Query "NewPod" extracts trigrams `["New", "ewP", "wPo", "Pod"]`, intersects posting lists to get ~50 candidate files, then parses only those with tree-sitter.

## Backends

- **tree-sitter** (default): AST-aware, precise symbol extraction
- **ctags** (fallback): Broader language support, less precise

### Tree-sitter languages

Python, JavaScript, TypeScript, Go, Rust

### Ctags (optional)

For additional language support (C, C++, Java, Ruby, etc.), install universal-ctags:

```bash
# macOS
brew install universal-ctags

# Ubuntu/Debian
sudo apt install universal-ctags
```

Note: macOS ships with BSD ctags which is not compatible. You need universal-ctags.

## Performance

On kubernetes (16k files):

| Operation | Without index | With index |
|-----------|---------------|------------|
| `defs NewPod` | 16s | 1.3s |
| `refs NewPod` | 16s | 1.4s |
| `csearch index` | - | 4 min |

## Files

- `.csearch.db` - SQLite trigram index (created by `csearch index`)
- `.csearch.tags` - ctags index (created by `csearch index` if universal-ctags available)
