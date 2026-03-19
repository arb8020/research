"""Configuration for reasoning-theater experiments."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class ModelConfig:
    """Model and serving configuration for the main rollout model."""

    model_name: str
    provider: str
    tokenizer_name: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 32000
    use_tito: bool = False


@dataclass
class DatasetConfig:
    """Task source configuration."""

    name: str
    split: str
    input_path: Path
    limit: int | None = None
    seed: int | None = None


@dataclass
class PrefixConfig:
    """How to choose reasoning prefixes."""

    fractions: list[float] = field(default_factory=lambda: [0.1, 0.25, 0.5, 0.75, 1.0])
    prefer_token_prefixes: bool = True
    min_chars: int = 32


@dataclass
class OutputConfig:
    """Filesystem outputs for this experiment."""

    run_name: str
    output_dir: Path
    activation_manifest_dir: Path | None = None
    probe_artifact_dir: Path | None = None


@dataclass
class ExperimentConfig:
    """Top-level experiment config."""

    model: ModelConfig
    dataset: DatasetConfig
    prefixes: PrefixConfig
    output: OutputConfig

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, default=str))

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        data = json.loads(path.read_text())
        return cls(
            model=ModelConfig(**data["model"]),
            dataset=DatasetConfig(
                **{
                    **data["dataset"],
                    "input_path": Path(data["dataset"]["input_path"]),
                }
            ),
            prefixes=PrefixConfig(**data["prefixes"]),
            output=OutputConfig(
                **{
                    **data["output"],
                    "output_dir": Path(data["output"]["output_dir"]),
                    "activation_manifest_dir": (
                        Path(data["output"]["activation_manifest_dir"])
                        if data["output"]["activation_manifest_dir"] is not None
                        else None
                    ),
                    "probe_artifact_dir": (
                        Path(data["output"]["probe_artifact_dir"])
                        if data["output"]["probe_artifact_dir"] is not None
                        else None
                    ),
                }
            ),
        )
