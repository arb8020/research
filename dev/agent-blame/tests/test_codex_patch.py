"""Test the apply_patch parser on real-world-ish inputs.

Codex's apply_patch format is content-anchored (no @@ -N,M +N,M @@ headers),
so the fold's content match via old_content/new_content is load-bearing.
"""

from __future__ import annotations

from agent_blame.adapters.codex import _parse_apply_patch


def test_single_update_hunk():
    patch = """*** Begin Patch
*** Update File: /a/b.py
@@
-    x = 1
+    x = 2
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert result == [("/a/b.py", "edit", "    x = 1", "    x = 2")]


def test_update_with_context():
    patch = """*** Begin Patch
*** Update File: /a/b.py
@@
 def foo():
-    return 1
+    return 2
     # done
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert len(result) == 1
    path, op, old, new = result[0]
    assert path == "/a/b.py"
    assert op == "edit"
    assert old == "def foo():\n    return 1\n    # done"
    assert new == "def foo():\n    return 2\n    # done"


def test_multiple_hunks_one_file_emits_multiple_edits():
    patch = """*** Begin Patch
*** Update File: /a/b.py
@@
-old1
+new1
@@
-old2
+new2
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert len(result) == 2
    assert result[0] == ("/a/b.py", "edit", "old1", "new1")
    assert result[1] == ("/a/b.py", "edit", "old2", "new2")


def test_multiple_files():
    patch = """*** Begin Patch
*** Update File: /a/b.py
@@
-a
+A
*** Update File: /a/c.py
@@
-b
+B
*** End Patch"""
    result = _parse_apply_patch(patch)
    paths = [r[0] for r in result]
    assert paths == ["/a/b.py", "/a/c.py"]


def test_add_file_emits_write():
    patch = """*** Begin Patch
*** Add File: /a/new.py
+line one
+line two
+
+line four
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert len(result) == 1
    path, op, old, new = result[0]
    assert path == "/a/new.py"
    assert op == "write"
    assert old is None
    assert new == "line one\nline two\n\nline four"


def test_delete_file_emits_nothing():
    patch = """*** Begin Patch
*** Delete File: /a/gone.py
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert result == []


def test_real_codex_style_patch():
    """Based on an actual apply_patch seen in ~/.codex/sessions."""
    patch = """*** Begin Patch
*** Update File: /Users/x/configs.py
@@
-    provider: Literal["modal", "runpod"] = "runpod"
+    provider: Literal["modal", "runpod", "ssh"] = "runpod"
@@
     container_disk_gb: int = 100
     persistent_volume_mount_path: str = "/workspace"
     persistent_volume_location: str | None = None
+    ssh: str | None = None
+    ssh_key_path: str | None = None
*** End Patch"""
    result = _parse_apply_patch(patch)
    assert len(result) == 2
    # First hunk: type change
    assert result[0][0] == "/Users/x/configs.py"
    assert "Literal" in result[0][2]
    assert "ssh" in result[0][3]
    # Second hunk: two new fields with three context lines
    _, op, old, new = result[1]
    assert op == "edit"
    # old should have the three context lines
    old_lines = old.split("\n")
    assert old_lines[0].startswith("    container_disk_gb")
    assert "ssh" not in old
    # new should have context + two additions
    new_lines = new.split("\n")
    assert any("ssh: str | None" in l for l in new_lines)
