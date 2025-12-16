"""Basic tests for csearch."""

from pathlib import Path

from csearch.types import Location, SearchResult


def test_location_str():
    loc = Location(path=Path("foo/bar.py"), line_start=10, line_end=20)
    assert str(loc) == "foo/bar.py:10-20"

    loc_single = Location(path=Path("foo/bar.py"), line_start=10, line_end=10)
    assert str(loc_single) == "foo/bar.py:10"


def test_search_result_format():
    loc = Location(path=Path("foo/bar.py"), line_start=10, line_end=20)
    result = SearchResult(
        location=loc,
        name="my_function",
        kind="function",
        snippet="def my_function():\n    pass",
    )

    compact = result.format_compact()
    assert "foo/bar.py:10-20" in compact
    assert "function" in compact
    assert "my_function" in compact

    with_snippet = result.format_with_snippet()
    assert "foo/bar.py:10-20" in with_snippet
    assert "def my_function" in with_snippet
