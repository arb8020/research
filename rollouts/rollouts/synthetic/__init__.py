"""Synthetic data generators for pretraining and evaluation.

Provides on-the-fly data generation for:
- iGSM: Grade-school math problems (Physics of Language Models Part 2)
- iGSM-Retry: Math problems with error-correction tokens

Usage:
    from rollouts.synthetic import build_igsm_loader, build_igsm_retry_loader

    # Clean math problems
    loader = build_igsm_loader(difficulty="med", seq_len=768, batch_size=4)
    input_ids, labels = loader.next()

    # Math with retry/correction tokens
    retry_loader = build_igsm_retry_loader(
        difficulty="med",
        retry_rate=0.1,
        seq_len=768,
        batch_size=4,
    )

Requirements:
    iGSM must be cloned to /tmp/iGSM:
    git clone https://github.com/facebookresearch/iGSM.git /tmp/iGSM
"""

from .igsm import (
    build_igsm_loader,
    ensure_igsm_available,
    iGSMConfig,
    iGSMDataLoader,
)
from .igsm_retry import (
    RETRY_TOKEN,
    build_igsm_retry_loader,
    iGSMRetryConfig,
    iGSMRetryDataLoader,
)

__all__ = [
    # Clean iGSM
    "iGSMConfig",
    "iGSMDataLoader",
    "build_igsm_loader",
    "ensure_igsm_available",
    # Retry iGSM
    "iGSMRetryConfig",
    "iGSMRetryDataLoader",
    "build_igsm_retry_loader",
    "RETRY_TOKEN",
]
