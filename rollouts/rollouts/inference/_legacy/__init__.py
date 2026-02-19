"""Legacy inference code.

This package contains legacy tokenization and generation utilities.
The main exports are in the backends submodule:
    from rollouts.inference._legacy.backends import tokenize_chat, compute_suffix_ids
"""

# Don't import anything at package level to avoid circular imports
# Use direct imports from submodules instead:
#   from rollouts.inference._legacy.backends.tokenize import tokenize_chat
#   from rollouts.inference._legacy.backends.generate import generate_sglang
