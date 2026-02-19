"""SGLang launcher with transformers compatibility patches.

Patches transformers.utils.hub.list_repo_templates to handle 404s gracefully.
See: https://github.com/huggingface/transformers/issues/41813

Usage:
    python -m rollouts.training.sglang_launcher --model-path ... --port ...
"""

import sys


def _patch_transformers() -> None:
    """Patch list_repo_templates to catch RemoteEntryNotFoundError."""
    try:
        import transformers.utils.hub as hub
    except ImportError:
        return  # transformers not installed

    _original = hub.list_repo_templates

    def _patched_list_repo_templates(*args, **kwargs):
        try:
            yield from _original(*args, **kwargs)
        except Exception:
            # additional_chat_templates directory doesn't exist - that's fine
            return

    hub.list_repo_templates = _patched_list_repo_templates


# Apply patch before importing sglang
_patch_transformers()

# Forward to sglang.launch_server
if __name__ == "__main__":
    import sys

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args

    server_args = prepare_server_args(sys.argv[1:])
    run_server(server_args)
