#!/usr/bin/env python3
"""Compatibility wrapper for the Argus-owned run implementation."""

from __future__ import annotations

import sys

from argus import run as _argus_run


def __getattr__(name: str):
    return getattr(_argus_run, name)


def main(argv: list[str] | None = None) -> int:
    return _argus_run.main(argv)


if __name__ == "__main__":
    sys.exit(main())
