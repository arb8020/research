"""Tests for pretrain config loading."""

from rollouts.pretrain.config import ModelConfig, TrainConfig


def test_model_config_defaults() -> None:
    """Test ModelConfig fills in defaults correctly."""
    config = ModelConfig(dim=256, n_layers=4, n_heads=4)

    assert config.n_kv_heads == 4  # defaults to n_heads
    assert config.head_dim == 64  # dim // n_heads
    assert config.mlp_dim == 1024  # 4 * dim


def test_model_config_explicit() -> None:
    """Test ModelConfig with explicit values."""
    config = ModelConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        n_kv_heads=2,  # GQA
        head_dim=32,
        mlp_dim=512,
    )

    assert config.n_kv_heads == 2
    assert config.head_dim == 32
    assert config.mlp_dim == 512


def test_train_config_fingerprint_stable() -> None:
    """Test fingerprint is stable for same config."""
    config1 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=100,
    )
    config2 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=100,
    )

    assert config1.fingerprint() == config2.fingerprint()


def test_train_config_fingerprint_changes() -> None:
    """Test fingerprint changes when config changes."""
    config1 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=100,
    )
    config2 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        steps=200,  # different
    )

    assert config1.fingerprint() != config2.fingerprint()


def test_train_config_fingerprint_ignores_runtime() -> None:
    """Test fingerprint ignores output_dir and run_id."""
    config1 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        output_dir="/path/a",
        run_id="run1",
    )
    config2 = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
        output_dir="/path/b",
        run_id="run2",
    )

    assert config1.fingerprint() == config2.fingerprint()


def test_python_config_import() -> None:
    """Test Python config files import correctly."""
    from rollouts.pretrain.configs.small import config as small_config
    from rollouts.pretrain.configs.tiny import config as tiny_config

    assert tiny_config.model.dim == 256
    assert tiny_config.steps == 100

    assert small_config.model.dim == 512
    assert small_config.steps == 1000


def test_grad_accum_steps_default() -> None:
    """Test grad_accum_steps defaults to 1."""
    config = TrainConfig(
        model=ModelConfig(dim=256, n_layers=4, n_heads=4),
    )

    assert config.grad_accum_steps == 1
