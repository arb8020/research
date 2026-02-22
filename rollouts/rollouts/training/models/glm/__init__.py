"""GLM model support for torchtitan.

This module provides GLM-4.7-Flash and GLM-5 model implementations
that integrate with torchtitan's training infrastructure.

Usage:
    # Register GLM with torchtitan at import time
    from rollouts.training.models import glm

    # Or explicitly register
    from rollouts.training.models.glm import register_glm_with_torchtitan
    register_glm_with_torchtitan()

    # Then use torchtitan normally
    from torchtitan.protocols.train_spec import get_train_spec
    spec = get_train_spec("glm")
"""

from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.optimizer import build_optimizers
from torchtitan.components.tokenizer import build_hf_tokenizer
from torchtitan.components.validate import build_validator
from torchtitan.hf_datasets.text_datasets import build_text_dataloader
from torchtitan.protocols.train_spec import TrainSpec, register_train_spec

from .args import GLM_4_7_FLASH, GLM_5, GLM_DEBUG, GLMModelArgs
from .model import GLMModel
from .parallelize import parallelize_glm
from .state_dict_adapter import GLMStateDictAdapter

__all__ = [
    "GLMModelArgs",
    "GLMModel",
    "GLMStateDictAdapter",
    "parallelize_glm",
    "GLM_4_7_FLASH",
    "GLM_5",
    "GLM_DEBUG",
    "get_train_spec",
    "register_glm_with_torchtitan",
]


# Model configs
glm_model_args = {
    "debugmodel": GLM_DEBUG,
    "4.7-flash": GLM_4_7_FLASH,
    "5": GLM_5,
}


def get_train_spec() -> TrainSpec:
    """Get GLM TrainSpec for torchtitan."""
    return TrainSpec(
        model_cls=GLMModel,
        model_args=glm_model_args,
        parallelize_fn=parallelize_glm,
        pipelining_fn=None,  # TODO: Add pipeline parallelism support
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_text_dataloader,
        build_tokenizer_fn=build_hf_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
        build_validator_fn=build_validator,
        state_dict_adapter=GLMStateDictAdapter,
    )


def register_glm_with_torchtitan() -> None:
    """Register GLM model with torchtitan's train spec registry.

    Call this once at startup to make GLM available via:
        get_train_spec("glm")
    """
    register_train_spec("glm", get_train_spec())


# Auto-register on import
register_glm_with_torchtitan()
