from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class DataConfig:
    raw_dir: str = "data/raw/flowers"
    processed_dir: str = "data/processed"
    split_manifest: str = "data/processed/split-manifest.json"
    image_size: int = 128
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    train_batch_size: int = 32
    eval_batch_size: int = 64
    num_workers: int = 0
    augmentation: str = "basic"
    mean: list[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std: list[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


@dataclass(slots=True)
class ModelConfig:
    name: str = "custom_cnn"
    num_classes: int = 5
    pretrained: bool = False
    dropout: float = 0.0


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    learning_rate: float = 0.0003
    weight_decay: float = 0.0001
    optimizer: str = "adamw"
    early_stopping_patience: int = 5
    checkpoint_metric: str = "val_accuracy"
    label_smoothing: float = 0.0
    warmup_epochs: int = 0
    use_amp: bool = True
    resume_from: str | None = None


@dataclass(slots=True)
class ArtifactsConfig:
    root_dir: str = "artifacts/runs"
    benchmark_dir: str = "reports/benchmark"


@dataclass(slots=True)
class AppConfig:
    experiment_name: str = "flower-classifier"
    seed: int = 7
    device: str = "auto"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    artifacts: ArtifactsConfig = field(default_factory=ArtifactsConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return project_root() / path

    @property
    def raw_dir(self) -> Path:
        return self.resolve_path(self.data.raw_dir)

    @property
    def processed_dir(self) -> Path:
        return self.resolve_path(self.data.processed_dir)

    @property
    def split_manifest_path(self) -> Path:
        return self.resolve_path(self.data.split_manifest)

    @property
    def runs_root(self) -> Path:
        return self.resolve_path(self.artifacts.root_dir)

    @property
    def benchmark_dir(self) -> Path:
        return self.resolve_path(self.artifacts.benchmark_dir)


def _load_section(data: dict[str, Any], key: str, cls: type[Any]) -> Any:
    values = data.get(key, {})
    return cls(**values)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig(
        experiment_name=raw.get("experiment_name", "flower-classifier"),
        seed=raw.get("seed", 7),
        device=raw.get("device", "auto"),
        data=_load_section(raw, "data", DataConfig),
        model=_load_section(raw, "model", ModelConfig),
        training=_load_section(raw, "training", TrainingConfig),
        artifacts=_load_section(raw, "artifacts", ArtifactsConfig),
    )
