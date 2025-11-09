# Bifrost SSH Key Investigation - Complete Documentation Index

This directory contains a comprehensive investigation into the bifrost CLI SSH key configuration issue.

## Quick Summary

**Problem:** Error message suggests using `bifrost exec --ssh-key` but the flag only works as `bifrost --ssh-key` (before the command).

**Root Cause:** Typer framework requires global options to be placed before subcommand names.

**Solution:** Fix error message immediately, move flag to command level in next release.

## Documents

### 1. SSH_KEY_ISSUE_QUICK_REF.txt
**What:** Quick reference guide for users and developers  
**Best For:** Quick lookups, copy-paste examples, testing  
**Length:** 3.8 KB  

Contents:
- Problem statement
- Configuration hierarchy
- Correct usage examples
- What doesn't work
- Code flow diagram
- Affected commands
- Key implementation files

### 2. SSH_KEY_ISSUE_ANALYSIS.md
**What:** Comprehensive technical analysis  
**Best For:** Deep understanding of the issue  
**Length:** 7.5 KB  

Contents:
- Problem summary
- Root cause analysis
- How SSH key resolution works
- Code flow explanation
- The two configuration methods
- Affected commands list
- Implementation details with code
- Example usage scenarios
- Key takeaways

### 3. SSH_KEY_CODE_SNIPPETS.md
**What:** Exact code from all relevant files with annotations  
**Best For:** Code review, understanding implementation, testing  
**Length:** 8.5 KB  

Contents:
- Global callback definition
- SSH key resolution function
- Exec command implementation
- Helper functions in shared config
- Init command
- All 8 commands using resolve_ssh_key()
- Client integration
- Execution flow diagram
- Testing examples

### 4. SSH_KEY_FIX_RECOMMENDATIONS.md
**What:** 3 proposed solutions with implementation details  
**Best For:** Planning fixes, understanding trade-offs  
**Length:** 10+ KB  

Contents:
- Executive summary
- Problem statement
- 3 solutions with pros/cons/effort
- Hybrid approach recommendation
- Step-by-step implementation for Solution 2
- Error message improvements
- Backward compatibility considerations
- Testing checklist
- Priority assessment

## Key Files in Bifrost Repository

### CLI Implementation
**File:** `bifrost/cli.py`
- **Lines 106-129:** `@app.callback()` defines `--ssh-key` flag
- **Lines 58-85:** `resolve_ssh_key()` function (the core logic)
- **Lines 214-265:** `exec` command implementation
- **Lines 131-145:** `init` command (creates .env template)
- **Lines 148-212, 268-334, 336-405, 408-471, 474-505, 508-537, 540-569:** Other commands

### SSH Configuration Helpers
**File:** `shared/shared/config.py`
- **Lines 85-87:** `get_ssh_key_path()` - reads SSH_KEY_PATH from environment
- **Lines 90-97:** `discover_ssh_keys()` - finds keys in ~/.ssh/
- **Lines 100-126:** `create_env_template()` - generates .env file

### Client Integration
**File:** `bifrost/client.py`
- BifrostClient constructor accepts `ssh_key_path` parameter

### Testing
**File:** `tests/smoke_exec_stream.py`
- Demonstrates correct BifrostClient usage

## Understanding the Issue

### SSH Key Resolution Precedence

The bifrost CLI tries to find SSH keys in this order:

1. **CLI flag:** `bifrost --ssh-key PATH ...` (highest priority)
2. **Environment variable:** `SSH_KEY_PATH` (from .env or shell)
3. **Auto-discovery:** `~/.ssh/id_{ed25519,rsa,ecdsa}`
4. **Error with suggestions:** If nothing found

Code location: `bifrost/cli.py` lines 58-85

### The Bug

**File:** `bifrost/cli.py` line 78

Current error message:
```python
logger.info(f"Or use: --ssh-key {found_keys[0]}")
```

This is misleading because users expect to put the flag after the command, but Typer requires global options before the command.

## Recommended Usage

### For Users (Current)

**Best practice:**
```bash
bifrost init                    # Creates .env
# Edit .env and set: SSH_KEY_PATH=~/.ssh/id_ed25519
bifrost exec user@host "cmd"    # Works without flags
```

**Alternative (if needed):**
```bash
bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "cmd"
```

**NOT working:**
```bash
bifrost exec --ssh-key ~/.ssh/id_ed25519 user@host "cmd"  # FAILS!
```

### For Maintainers

See `SSH_KEY_FIX_RECOMMENDATIONS.md` for:
- 3 proposed solutions (effort: 5 min to 1 hour)
- Recommended hybrid approach
- Step-by-step implementation details
- Testing checklist
- Backward compatibility plan

## Affected Commands

All 8 SSH-based bifrost commands use `resolve_ssh_key()`:

1. `bifrost push` - Deploy code
2. `bifrost exec` - Execute command [PRIMARY ISSUE]
3. `bifrost deploy` - Deploy + execute
4. `bifrost run` - Background job (tmux)
5. `bifrost jobs` - List jobs
6. `bifrost logs` - View job logs
7. `bifrost download` - Download files
8. `bifrost upload` - Upload files

## Investigation Methodology

This analysis was conducted by:

1. Reading the main CLI implementation (`bifrost/cli.py`)
2. Tracing SSH key resolution logic
3. Checking shared configuration module
4. Examining the init command and .env template generation
5. Looking at test files for usage patterns
6. Analyzing the Typer framework's callback behavior
7. Comparing error message suggestions with actual CLI behavior

All findings are documented with exact file paths, line numbers, and code snippets.

## Next Steps

### Immediate (Patch Release)
1. Fix error message (line 78 of `bifrost/cli.py`)
2. Test with all commands
3. Update README with correct syntax

### Next Release (Minor)
1. Move `--ssh-key` to command level
2. Support both syntaxes for backward compatibility
3. Update documentation
4. Add command-level tests

### Long-term
1. Consider other global options behavior
2. Improve CLI UX for configuration discovery
3. Add interactive setup mode

## Questions?

- See `SSH_KEY_QUICK_REF.txt` for common questions
- See `SSH_KEY_CODE_SNIPPETS.md` for code details
- See `SSH_KEY_FIX_RECOMMENDATIONS.md` for implementation guidance
