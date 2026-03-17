"""Process-wide SGLang instrumentation shim.

Loaded automatically by Python when present on sys.path. This lets us propagate
the same runtime tracing hooks into SGLang child interpreters spawned after the
main launcher process.
"""

from __future__ import annotations

import json
import os
import sys
import traceback


def _emit_fallback(payload: dict[str, object]) -> None:
    try:
        sys.stderr.write(f"__ARGUS_DIAG__{json.dumps(payload, sort_keys=True)}\n")
        sys.stderr.flush()
    except Exception:
        pass


if os.environ.get("ROLLOUTS_SGLANG_SITE_TRACE") == "1":
    try:
        from rollouts.training.sglang_launcher import (
            _emit_argus_diag,
            _instrument_sglang_runtime_methods,
            _process_context,
        )

        _emit_argus_diag(
            "sglang_sitecustomize_active",
            process=_process_context(),
        )
        _instrument_sglang_runtime_methods()
    except Exception as exc:
        _emit_fallback({
            "event": "sglang_sitecustomize_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback_tail": traceback.format_exc().splitlines()[-8:],
        })
