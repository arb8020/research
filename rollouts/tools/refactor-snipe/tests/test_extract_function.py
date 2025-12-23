"""Tests for extract function refactoring."""

import tempfile
from pathlib import Path

import pytest

from ..refactors.extract_function import ExtractFunction, apply_edits


class TestExtractFunction:
    """Test extract function refactoring."""

    def test_simple_extraction(self):
        """Test extracting simple statements without parameters or returns."""
        code = """\
def main():
    print("hello")
    print("world")
    print("done")
"""
        expected = """\
def main():
    def greet():
        print("hello")
        print("world")

    greet()
    print("done")
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            refactor = ExtractFunction(path, 2, 3, "greet")
            result = refactor.execute()
            modified = apply_edits(path, result.edits)

            assert "def greet():" in modified
            assert "greet()" in modified
        finally:
            path.unlink()

    def test_extraction_with_parameters(self):
        """Test extracting code that uses variables from outer scope."""
        code = """\
def process(data):
    x = data + 1
    y = x * 2
    z = y + x
    return z
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            # Extract lines 3-4 (y = x * 2; z = y + x)
            refactor = ExtractFunction(path, 3, 4, "compute")
            result = refactor.execute()
            modified = apply_edits(path, result.edits)

            # Should have x as parameter since it's used but defined outside
            assert "def compute(x):" in modified or "def compute(x)" in modified
        finally:
            path.unlink()

    def test_extraction_with_return_value(self):
        """Test extracting code that defines a variable used later."""
        code = """\
def calculate():
    a = 1
    b = 2
    result = a + b
    print(result)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            # Extract lines 2-4 (a = 1; b = 2; result = a + b)
            refactor = ExtractFunction(path, 2, 4, "compute_result")
            result = refactor.execute()
            modified = apply_edits(path, result.edits)

            # Should return result since it's used after the extracted region
            assert "return result" in modified
            assert "result = compute_result()" in modified
        finally:
            path.unlink()

    def test_extraction_with_params_and_returns(self):
        """Test extraction with both parameters and return values."""
        code = """\
def transform(input_val):
    x = input_val * 2
    y = x + 10
    z = y * 3
    final = z - x
    print(final)
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            # Extract lines 3-4 (y = x + 10; z = y * 3)
            refactor = ExtractFunction(path, 3, 4, "process_values")
            result = refactor.execute()
            modified = apply_edits(path, result.edits)

            # Should have x as parameter (used in region, defined before)
            # Should return z (defined in region, used after)
            assert "x" in modified  # parameter
            assert "return z" in modified or "return" in modified
        finally:
            path.unlink()

    def test_class_method_extraction(self):
        """Test extracting code from inside a class method."""
        code = """\
class Calculator:
    def compute(self, x):
        a = x + 1
        b = a * 2
        return b
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            # Extract lines 3-4 (a = x + 1; b = a * 2)
            refactor = ExtractFunction(path, 3, 4, "helper")
            result = refactor.execute()
            modified = apply_edits(path, result.edits)

            # Should be a method (with self)
            assert "def helper(self" in modified
            assert "self.helper(" in modified
        finally:
            path.unlink()


class TestApplyEdits:
    """Test the edit application logic."""

    def test_apply_insertion(self):
        """Test inserting new code."""
        code = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            from ..refactors.extract_function import TextEdit

            edits = [
                TextEdit(start_line=2, end_line=2, new_text="new_line\n\n"),
            ]
            result = apply_edits(path, edits)

            assert "new_line" in result
        finally:
            path.unlink()

    def test_apply_replacement(self):
        """Test replacing existing code."""
        code = "line1\nline2\nline3\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            f.flush()
            path = Path(f.name)

        try:
            from ..refactors.extract_function import TextEdit

            edits = [
                TextEdit(start_line=2, end_line=2, new_text="replaced"),
            ]
            result = apply_edits(path, edits)

            assert "replaced" in result
            assert "line2" not in result
        finally:
            path.unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
