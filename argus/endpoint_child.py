"""Detached child process for `argus endpoint up`. Not invoked directly."""

from __future__ import annotations

import sys

from argus.endpoint import _run_detached_main

if __name__ == "__main__":
    sys.exit(_run_detached_main(sys.argv[1:]))
