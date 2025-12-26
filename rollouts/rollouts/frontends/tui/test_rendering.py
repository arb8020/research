#!/usr/bin/env python3
"""
Test TUI rendering logic (no terminal required).
"""

from .components import Markdown, Spacer, Text
from .tui import Container
from .utils import visible_width, wrap_text_with_ansi


def bold(text: str) -> str:
    return f"\x1b[1m{text}\x1b[0m"


def cyan(text: str) -> str:
    return f"\x1b[36m{text}\x1b[0m"


def test_visible_width() -> None:
    """Test ANSI-aware width calculation."""
    assert visible_width("hello") == 5
    assert visible_width(bold("hello")) == 5
    assert visible_width(cyan("hello")) == 5
    assert visible_width(bold(cyan("hello"))) == 5
    assert visible_width("") == 0
    assert visible_width("\t") == 3  # Tab = 3 spaces
    print("test_visible_width: PASSED")


def test_wrap_text() -> None:
    """Test word wrapping with ANSI codes."""
    # Simple wrap
    lines = wrap_text_with_ansi("hello world", 5)
    assert len(lines) == 2
    assert lines[0] == "hello"
    assert lines[1] == "world"

    # With ANSI codes
    lines = wrap_text_with_ansi(bold("hello") + " " + cyan("world"), 6)
    assert len(lines) == 2
    # Width should be correct even with ANSI
    assert visible_width(lines[0]) <= 6
    assert visible_width(lines[1]) <= 6

    # With newlines
    lines = wrap_text_with_ansi("line1\nline2", 80)
    assert len(lines) == 2
    assert lines[0] == "line1"
    assert lines[1] == "line2"

    print("test_wrap_text: PASSED")


def test_text_component() -> None:
    """Test Text component rendering."""
    text = Text("Hello World", padding_x=1, padding_y=0)
    lines = text.render(80)

    assert len(lines) == 1
    # Should have padding on left and right
    assert lines[0].startswith(" ")
    assert "Hello World" in lines[0]
    # Should be padded to full width
    assert visible_width(lines[0]) == 80

    print("test_text_component: PASSED")


def test_container() -> None:
    """Test Container composing components."""
    container = Container()
    container.add_child(Text("First", padding_x=0, padding_y=0))
    container.add_child(Spacer(1))
    container.add_child(Text("Second", padding_x=0, padding_y=0))

    lines = container.render(80)
    assert len(lines) == 3  # First + Spacer + Second
    assert "First" in lines[0]
    assert lines[1].strip() == ""  # Spacer
    assert "Second" in lines[2]

    print("test_container: PASSED")


def test_ansi_preservation() -> None:
    """Test that ANSI codes are preserved through rendering."""
    styled_text = bold("Hello") + " " + cyan("World")
    text = Text(styled_text, padding_x=1, padding_y=0)
    lines = text.render(80)

    # ANSI codes should still be in the output
    assert "\x1b[1m" in lines[0]  # bold
    assert "\x1b[36m" in lines[0]  # cyan

    print("test_ansi_preservation: PASSED")


def test_markdown() -> None:
    """Test markdown rendering."""
    md = Markdown(
        """# Heading

This is **bold** and *italic* text.

- List item 1
- List item 2

```python
print("hello")
```
""",
        padding_x=0,
        padding_y=0,
    )

    lines = md.render(80)

    # Should have multiple lines
    assert len(lines) > 5

    # Check that heading is styled (bold)
    assert "\x1b[1m" in lines[0]

    # Check that list bullets are present
    list_lines = [l for l in lines if "- " in l or "List item" in l]
    assert len(list_lines) >= 2

    print("test_markdown: PASSED")


def main() -> None:
    """Run all tests."""
    test_visible_width()
    test_wrap_text()
    test_text_component()
    test_container()
    test_ansi_preservation()
    test_markdown()
    print("\nAll tests passed!")


if __name__ == "__main__":
    main()
