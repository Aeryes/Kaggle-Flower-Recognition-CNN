from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from flower_classifier.config import AppConfig

CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    Image.fromarray(pixels, mode="RGB").save(path)


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "raw" / "flowers"
    for label_index, class_name in enumerate(CLASS_NAMES):
        class_dir = root / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        for image_index in range(6):
            _write_image(
                class_dir / f"{class_name}-{image_index}.png", seed=label_index * 100 + image_index
            )
    return root


@pytest.fixture()
def tiny_config(tmp_path: Path, tiny_dataset: Path) -> AppConfig:
    config = AppConfig()
    config.seed = 11
    config.data.raw_dir = str(tiny_dataset)
    config.data.processed_dir = str(tmp_path / "processed")
    config.data.split_manifest = str(tmp_path / "processed" / "split-manifest.json")
    config.data.image_size = 32
    config.data.train_batch_size = 4
    config.data.eval_batch_size = 4
    config.data.num_workers = 0
    config.model.num_classes = len(CLASS_NAMES)
    config.model.name = "custom_cnn"
    config.training.epochs = 1
    config.training.early_stopping_patience = 1
    config.artifacts.root_dir = str(tmp_path / "artifacts" / "runs")
    config.artifacts.benchmark_dir = str(tmp_path / "reports" / "benchmark")
    return config


@pytest.fixture()
def config_file(tmp_path: Path, tiny_config: AppConfig) -> Path:
    config_path = tmp_path / "tiny-config.yaml"
    with config_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(tiny_config.to_dict(), handle, sort_keys=False)
    return config_path
