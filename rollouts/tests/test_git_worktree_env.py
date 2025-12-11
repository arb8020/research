"""Test GitWorktreeEnvironment in isolation."""

import shutil
import tempfile
from pathlib import Path

import trio

from rollouts.environments.git_worktree import GitWorktreeEnvironment
from rollouts.dtypes import ToolCall


async def test_git_worktree_environment():
    """Test the GitWorktreeEnvironment lifecycle and auto-commit behavior."""

    # Create a temp directory to work in
    temp_dir = Path(tempfile.mkdtemp(prefix="git_worktree_test_"))
    print(f"Test directory: {temp_dir}")

    try:
        # Create some initial files
        (temp_dir / "README.md").write_text("# Test Project\n")
        (temp_dir / "src").mkdir()
        (temp_dir / "src" / "main.py").write_text("print('hello')\n")

        # Create environment
        env = GitWorktreeEnvironment(working_dir=temp_dir)
        print(f"✓ Created environment")

        # Setup with a session ID
        session_id = "test-session-123"
        await env.on_session_start(session_id)
        print(f"✓ Setup complete")
        print(f"  Branch: {env._current_branch}")
        print(f"  Worktree: {env._worktree_path}")

        # Verify .rollouts structure was created
        assert (temp_dir / ".rollouts").exists(), ".rollouts dir should exist"
        assert (temp_dir / ".rollouts" / "repo.git").exists(), "bare repo should exist"
        assert env._worktree_path.exists(), "worktree should exist"
        print(f"✓ Directory structure verified")

        # Verify initial files were copied to worktree
        assert (env._worktree_path / "README.md").exists(), "README should be in worktree"
        assert (env._worktree_path / "src" / "main.py").exists(), "main.py should be in worktree"
        print(f"✓ Initial files copied to worktree")

        # Test read tool
        read_result = await env.exec_tool(
            ToolCall(id="1", name="read", args={"path": "README.md"}),
            current_state=None,
            run_config=None,
        )
        assert not read_result.is_error, f"read failed: {read_result.error}"
        assert "Test Project" in read_result.content
        print(f"✓ Read tool works")

        # Test write tool (should auto-commit)
        write_result = await env.exec_tool(
            ToolCall(id="2", name="write", args={
                "path": "new_file.txt",
                "content": "Hello from test!"
            }),
            current_state=None,
            run_config=None,
        )
        assert not write_result.is_error, f"write failed: {write_result.error}"
        assert env._commit_count == 1, f"Should have 1 commit, got {env._commit_count}"
        print(f"✓ Write tool works + auto-committed")

        # Verify file was created
        assert (env._worktree_path / "new_file.txt").exists()
        assert (env._worktree_path / "new_file.txt").read_text() == "Hello from test!"
        print(f"✓ File verified in worktree")

        # Test edit tool (should auto-commit)
        edit_result = await env.exec_tool(
            ToolCall(id="3", name="edit", args={
                "path": "README.md",
                "old_text": "# Test Project",
                "new_text": "# Test Project\n\nEdited!"
            }),
            current_state=None,
            run_config=None,
        )
        assert not edit_result.is_error, f"edit failed: {edit_result.error}"
        assert env._commit_count == 2, f"Should have 2 commits, got {env._commit_count}"
        print(f"✓ Edit tool works + auto-committed")

        # Verify edit
        content = (env._worktree_path / "README.md").read_text()
        assert "Edited!" in content
        print(f"✓ Edit verified in worktree")

        # Test bash tool (should auto-commit)
        bash_result = await env.exec_tool(
            ToolCall(id="4", name="bash", args={"command": "echo 'test' > bash_file.txt"}),
            current_state=None,
            run_config=None,
        )
        assert not bash_result.is_error, f"bash failed: {bash_result.error}"
        assert env._commit_count == 3, f"Should have 3 commits, got {env._commit_count}"
        print(f"✓ Bash tool works + auto-committed")

        # Verify bash created file
        assert (env._worktree_path / "bash_file.txt").exists()
        print(f"✓ Bash file verified in worktree")

        # Test status info
        status = env.get_status_info()
        assert status is not None
        assert "branch" in status
        assert "commits" in status
        assert status["commits"] == "3"
        print(f"✓ Status info: {status}")

        # Test serialize/deserialize
        state = await env.serialize()
        assert state["session_id"] == session_id
        assert state["commit_count"] == 3
        assert state["head_commit"] is not None
        print(f"✓ Serialize works: {state['head_commit'][:8]}...")

        # Verify git log has our commits
        import subprocess
        result = subprocess.run(
            ["git", "log", "--oneline"],
            cwd=str(env._worktree_path),
            capture_output=True,
            text=True,
        )
        log_lines = result.stdout.strip().split("\n")
        print(f"✓ Git log ({len(log_lines)} commits):")
        for line in log_lines[:5]:
            print(f"    {line}")

        # Verify user's original directory is untouched
        assert not (temp_dir / ".git").exists(), "Should NOT have created .git in user's dir"
        original_readme = (temp_dir / "README.md").read_text()
        assert "Edited!" not in original_readme, "User's original file should be untouched"
        print(f"✓ User's original directory is untouched")

        print(f"\n✅ All tests passed!")

    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up {temp_dir}")


if __name__ == "__main__":
    trio.run(test_git_worktree_environment)
