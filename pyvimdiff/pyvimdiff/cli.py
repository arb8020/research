"""
CLI entry point - spawns vim/nvim with a file picker and diff viewer.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile


def get_changed_files(ref: str | None = None, staged: bool = False) -> list[str]:
    """Get list of changed files."""
    args = ["git", "diff", "--name-only"]
    if staged:
        args.append("--cached")
    elif ref:
        args.append(ref)

    result = subprocess.run(args, capture_output=True, text=True)
    if result.returncode != 0:
        return []

    return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Vim-native git diff viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  pyvimdiff                    # Show unstaged changes
  pyvimdiff --staged           # Show staged changes
  pyvimdiff HEAD~1             # Compare to previous commit
""",
    )

    parser.add_argument(
        "ref",
        nargs="?",
        help="Git ref to diff against (e.g., HEAD~1, main, commit hash)",
    )
    parser.add_argument(
        "remote",
        nargs="?",
        help="Remote file (used with --difftool)",
    )
    parser.add_argument(
        "--staged",
        "-s",
        action="store_true",
        help="Show staged changes",
    )
    parser.add_argument(
        "--difftool",
        action="store_true",
        help="Run in difftool mode (LOCAL REMOTE)",
    )

    args = parser.parse_args()

    # Difftool mode: show unified diff between two files in our TUI
    if args.difftool and args.ref and args.remote:
        from .app import DiffViewer, diff_files

        diff_text = diff_files(args.ref, args.remote)
        if not diff_text:
            print("Files are identical.")
            return 0

        filename = os.path.basename(args.remote)
        viewer = DiffViewer(diff_text, filename)
        viewer.run()
        return 0

    # Get changed files
    files = get_changed_files(ref=args.ref, staged=args.staged)

    if not files:
        print("No changes.")
        return 0

    # Create vim script for file picker + diff viewing
    vim_script = """
" pyvimdiff - git diff file picker
" Navigation: j/k move, Enter view, h/l prev/next file, q quit

let g:pvd_files = FILES_LIST
let g:pvd_ref = 'REF_VALUE'
let g:pvd_staged = STAGED_VALUE
let g:pvd_idx = 0

function! PvdShowList()
    enew
    setlocal buftype=nofile bufhidden=wipe noswapfile nomodifiable
    setlocal cursorline nonumber norelativenumber
    file [Changed\\ Files]

    let lines = ['Changed Files (' . len(g:pvd_files) . ')', '']
    for i in range(len(g:pvd_files))
        call add(lines, '  ' . g:pvd_files[i])
    endfor
    call add(lines, '')
    call add(lines, 'j/k:move  Enter:view  q:quit')

    setlocal modifiable
    call setline(1, lines)
    setlocal nomodifiable
    normal! 3G

    nnoremap <buffer> q :qa!<CR>
    nnoremap <buffer> <CR> :call PvdOpenSelected()<CR>
endfunction

function! PvdOpenSelected()
    let idx = line('.') - 3
    if idx >= 0 && idx < len(g:pvd_files)
        let g:pvd_idx = idx
        call PvdOpenDiff(idx)
    endif
endfunction

function! PvdOpenDiff(idx)
    let file = g:pvd_files[a:idx]
    let cmd = 'git diff'
    if g:pvd_staged
        let cmd .= ' --cached'
    elseif g:pvd_ref != ''
        let cmd .= ' ' . g:pvd_ref
    endif
    let cmd .= ' -- ' . shellescape(file)

    enew
    setlocal buftype=nofile bufhidden=wipe noswapfile
    execute 'file [' . (a:idx + 1) . '/' . len(g:pvd_files) . ']\\ ' . file
    execute 'silent r !' . cmd
    normal! ggdd
    setlocal nomodifiable filetype=diff

    nnoremap <buffer> q :call PvdShowList()<CR>
    nnoremap <buffer> <Esc> :call PvdShowList()<CR>
    nnoremap <buffer> h :call PvdPrev()<CR>
    nnoremap <buffer> l :call PvdNext()<CR>
    nnoremap <buffer> <Left> :call PvdPrev()<CR>
    nnoremap <buffer> <Right> :call PvdNext()<CR>
    nnoremap <buffer> ]c /^@@<CR>zt
    nnoremap <buffer> [c ?^@@<CR>zt
endfunction

function! PvdPrev()
    if g:pvd_idx > 0
        let g:pvd_idx -= 1
        call PvdOpenDiff(g:pvd_idx)
    else
        echo "First file"
    endif
endfunction

function! PvdNext()
    if g:pvd_idx < len(g:pvd_files) - 1
        let g:pvd_idx += 1
        call PvdOpenDiff(g:pvd_idx)
    else
        echo "Last file"
    endif
endfunction

call PvdShowList()
"""

    # Substitute values
    files_list = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    vim_script = vim_script.replace("FILES_LIST", files_list)
    vim_script = vim_script.replace("REF_VALUE", args.ref or "")
    vim_script = vim_script.replace("STAGED_VALUE", "1" if args.staged else "0")

    # Write script and run vim
    with tempfile.NamedTemporaryFile(mode="w", suffix=".vim", delete=False) as f:
        f.write(vim_script)
        script_path = f.name

    try:
        editor = (
            "nvim"
            if subprocess.run(["which", "nvim"], capture_output=True).returncode == 0
            else "vim"
        )
        subprocess.run([editor, "-S", script_path])
        return 0
    finally:
        os.unlink(script_path)


if __name__ == "__main__":
    sys.exit(main())
