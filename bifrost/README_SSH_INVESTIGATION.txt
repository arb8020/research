BIFROST SSH KEY CONFIGURATION INVESTIGATION - QUICK START
=========================================================

THE PROBLEM
===========
$ bifrost exec user@host "ls"
✗ No SSH key specified
Or use: --ssh-key /Users/user/.ssh/id_ed25519

$ bifrost exec --ssh-key ~/.ssh/id_ed25519 user@host "ls"
Error: No such option: --ssh-key

WHY? The --ssh-key flag is defined in @app.callback() (global scope), not 
in the exec command. Typer requires global options BEFORE the subcommand.

CORRECT SYNTAX
==============
$ bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "ls"
              ^^^^^ Must come BEFORE exec


BETTER SOLUTION (Recommended)
=============================
$ bifrost init
$ # Edit .env and set: SSH_KEY_PATH=~/.ssh/id_ed25519
$ bifrost exec user@host "ls"  # Works without flags


DOCUMENTATION FILES
===================

Start Here:
  SSH_KEY_INVESTIGATION_INDEX.md
  - Overview of all documents
  - Quick navigation guide

For Quick Answers:
  SSH_KEY_ISSUE_QUICK_REF.txt
  - Problem summary
  - Working examples
  - Code flow diagram
  - What doesn't work

For Technical Details:
  SSH_KEY_ISSUE_ANALYSIS.md
  - Complete explanation
  - Code flow details
  - All scenarios
  - Example usage

For Implementation:
  SSH_KEY_CODE_SNIPPETS.md
  - Exact code locations
  - Line numbers
  - Implementation details

For Fixes:
  SSH_KEY_FIX_RECOMMENDATIONS.md
  - 3 solution options
  - Effort estimates
  - Step-by-step guidance
  - Testing checklist


KEY FILES IN BIFROST
====================

bifrost/cli.py
  Lines 106-129: @app.callback() defines --ssh-key
  Lines 58-85:   resolve_ssh_key() function
  Lines 214-265: exec command
  Line 78:       THE BUG (misleading error message)

shared/shared/config.py
  Lines 85-87:   get_ssh_key_path()
  Lines 90-97:   discover_ssh_keys()


SSH KEY RESOLUTION ORDER
========================

1. CLI flag:        bifrost --ssh-key PATH ...
2. Env variable:    SSH_KEY_PATH (from .env)
3. Auto-discovery:  ~/.ssh/id_ed25519, ~/.ssh/id_rsa, ~/.ssh/id_ecdsa
4. Error:           If nothing found


ALL AFFECTED COMMANDS
====================

bifrost push      - Deploy code
bifrost exec      - Execute command (PRIMARY ISSUE)
bifrost deploy    - Deploy + execute
bifrost run       - Background job
bifrost jobs      - List jobs
bifrost logs      - View logs
bifrost download  - Download files
bifrost upload    - Upload files


THE FIX
=======

Immediate (Patch Release - 5 minutes):
  File: bifrost/cli.py, Line 78
  From: logger.info(f"Or use: --ssh-key {found_keys[0]}")
  To:   logger.info(f"Or use: bifrost --ssh-key {found_keys[0]} <command> ...")

Next Release (Minor - 30-45 minutes):
  Move --ssh-key from @app.callback() to individual commands
  Results in: bifrost exec --ssh-key PATH ... (intuitive!)


TESTING
=======

Current (wrong):
  $ bifrost exec --ssh-key ~/.ssh/id_ed25519 user@host "ls" 
  ✗ Error: No such option: --ssh-key

Currently works:
  $ bifrost --ssh-key ~/.ssh/id_ed25519 exec user@host "ls"
  ✓ Works (but unintuitive)

Recommended:
  $ bifrost init && bifrost exec user@host "ls"
  ✓ Works (best UX)

Will work after fix:
  $ bifrost exec --ssh-key ~/.ssh/id_ed25519 user@host "ls"
  ✓ Works (after moving flag to command level)


INVESTIGATION STATISTICS
========================

Files Analyzed:              4
  - bifrost/cli.py
  - shared/shared/config.py
  - bifrost/client.py
  - tests/smoke_exec_stream.py

Documentation Created:       5
  - 1,188 total lines
  - 34.1 KB combined

Commands Documented:         8
Key Functions Identified:    4
Lines Referenced:          ~100+


QUICK REFERENCE TABLE
====================

Scenario              | Command                            | Works?
---------------------|------------------------------------|---------
.env set             | bifrost exec user@host "ls"        | YES
--ssh-key flag       | bifrost --ssh-key ~/.ssh/id ...    | YES
--ssh-key flag       | bifrost exec --ssh-key ~/.ssh/id..| NO
Export SSH_KEY_PATH  | bifrost exec user@host "ls"        | YES
Auto-discover keys   | bifrost exec user@host "ls"        | YES


NEXT STEPS
==========

1. Read SSH_KEY_INVESTIGATION_INDEX.md for complete overview
2. Choose your immediate action based on your role:
   - User: See Quick Ref for workarounds
   - Developer: See Code Snippets for implementation
   - Maintainer: See Fix Recommendations for solutions


CONTACT / MORE INFO
===================

All investigation documents are in this directory:
/Users/chiraagbalu/research/bifrost/

Files starting with "SSH_KEY_" contain the complete analysis
