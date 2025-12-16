# pyvimdiff

Vim-native git diff viewer with syntax highlighting. Like vimdiff but with a file picker and prettier colors.

## Install

```bash
# From source
uv pip install -e .

# Or run directly
uv run --from . pyvimdiff
```

## Usage

```bash
pyvimdiff                    # Show unstaged changes
pyvimdiff --staged           # Show staged changes
pyvimdiff HEAD~1             # Compare to previous commit
pyvimdiff main..feature      # Compare branches
pyvimdiff -C /path/to/repo   # Run in different directory
```

## Git Difftool Integration

Set up as your default difftool:

```bash
git config --global diff.tool pyvimdiff
git config --global difftool.pyvimdiff.cmd 'pyvimdiff --difftool "$LOCAL" "$REMOTE"'
git config --global difftool.prompt false
```

Then use:

```bash
git difftool                 # View changes file-by-file
git difftool --staged        # Staged changes
git difftool HEAD~1          # Compare commits
```

## Keybindings

### Navigation

| Key | Action |
|-----|--------|
| `j` / `k` / `↑` / `↓` | Scroll up/down |
| `Ctrl+d` / `Ctrl+u` | Half-page down/up |
| `gg` | Jump to start |
| `G` | Jump to end |

### File Navigation

| Key | Action |
|-----|--------|
| `h` / `l` / `←` / `→` | Previous/next file |
| `H` | First file |
| `L` | Last file |

### Hunk Navigation

| Key | Action |
|-----|--------|
| `]c` | Next hunk |
| `[c` | Previous hunk |

### Search

| Key | Action |
|-----|--------|
| `/` | Start search |
| `n` | Next match |
| `N` | Previous match |

### Other

| Key | Action |
|-----|--------|
| `q` / `Escape` | Quit |

## Features

- **Alternate screen buffer** - exits cleanly like vim, doesn't pollute scrollback
- **File tabs** - see all changed files, navigate with `h`/`l`
- **Colored diffs** - green for additions, red for removals, blue hunk headers
- **Vim keybindings** - `hjkl`, `gg`/`G`, `Ctrl+d`/`Ctrl+u`, `]c`/`[c`, `/search`
- **Git difftool support** - use as your `git difftool`
