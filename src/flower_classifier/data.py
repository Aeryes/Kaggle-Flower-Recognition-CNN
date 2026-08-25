from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flower_classifier.config import AppConfig
from flower_classifier.utils import (
    build_generator,
    ensure_directory,
    is_image_file,
    load_json,
    relative_to,
    utc_now,
    worker_init_fn,
    write_json,
)


class ImageManifestDataset(Dataset[tuple[torch.Tensor, int]]):
    def __init__(
        self, root_dir: Path, records: list[dict[str, Any]], transform: transforms.Compose
    ):
        self.root_dir = root_dir
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        image_path = self.root_dir / record["relative_path"]
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            return self.transform(image), int(record["label"])


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_records(raw_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw dataset directory does not exist: {raw_dir}")
    class_dirs = sorted(path for path in raw_dir.iterdir() if path.is_dir())
    if not class_dirs:
        raise FileNotFoundError(f"No class directories found under {raw_dir}")

    class_names = [path.name for path in class_dirs]
    records: list[dict[str, Any]] = []
    for label, class_dir in enumerate(class_dirs):
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and is_image_file(image_path):
                records.append(
                    {
                        "relative_path": relative_to(image_path, raw_dir),
                        "label": label,
                        "label_name": class_dir.name,
                        "sha256": _hash_file(image_path),
                    }
                )
    if not records:
        raise FileNotFoundError(f"No supported image files found under {raw_dir}")
    return class_names, records


def fingerprint_records(records: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(record["relative_path"].encode("utf-8"))
        digest.update(str(record["label"]).encode("utf-8"))
        digest.update(record["sha256"].encode("utf-8"))
    return digest.hexdigest()


def _fallback_split_indices(
    records: list[dict[str, Any]], train_ratio: float, val_ratio: float
) -> dict[str, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        grouped[int(record["label"])].append(index)

    split_indices = {"train": [], "val": [], "test": []}
    for indices in grouped.values():
        size = len(indices)
        train_cutoff = max(1, round(size * train_ratio))
        val_cutoff = max(train_cutoff + 1, round(size * (train_ratio + val_ratio)))
        split_indices["train"].extend(indices[:train_cutoff])
        split_indices["val"].extend(indices[train_cutoff:val_cutoff])
        split_indices["test"].extend(indices[val_cutoff:])
    return split_indices


def stratified_split(
    records: list[dict[str, Any]],
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[int]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError("Train, validation, and test ratios must sum to 1.0")

    indices = list(range(len(records)))
    labels = [int(record["label"]) for record in records]
    try:
        train_indices, temp_indices = train_test_split(
            indices,
            train_size=train_ratio,
            stratify=labels,
            random_state=seed,
        )
        temp_labels = [labels[index] for index in temp_indices]
        val_share = val_ratio / (val_ratio + test_ratio)
        val_indices, test_indices = train_test_split(
            temp_indices,
            train_size=val_share,
            stratify=temp_labels,
            random_state=seed,
        )
        return {
            "train": sorted(train_indices),
            "val": sorted(val_indices),
            "test": sorted(test_indices),
        }
    except ValueError:
        fallback = _fallback_split_indices(records, train_ratio, val_ratio)
        return {name: sorted(values) for name, values in fallback.items()}


def build_manifest(config: AppConfig) -> dict[str, Any]:
    class_names, records = discover_records(config.raw_dir)
    split_indices = stratified_split(
        records,
        seed=config.seed,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
    )
    manifest = {
        "version": 1,
        "created_at": utc_now(),
        "seed": config.seed,
        "dataset_fingerprint": fingerprint_records(records),
        "class_names": class_names,
        "ratios": {
            "train": config.data.train_ratio,
            "val": config.data.val_ratio,
            "test": config.data.test_ratio,
        },
        "splits": {
            split_name: [records[index] for index in split_indices[split_name]]
            for split_name in ("train", "val", "test")
        },
    }
    manifest["counts"] = {name: len(items) for name, items in manifest["splits"].items()}
    return manifest


def prepare_data(config: AppConfig, force: bool = False) -> dict[str, Any]:
    ensure_directory(config.processed_dir)
    if config.split_manifest_path.exists() and not force:
        manifest = load_json(config.split_manifest_path)
        current_classes, current_records = discover_records(config.raw_dir)
        current_fingerprint = fingerprint_records(current_records)
        if (
            manifest.get("dataset_fingerprint") == current_fingerprint
            and manifest.get("class_names") == current_classes
            and manifest.get("seed") == config.seed
        ):
            return manifest
    manifest = build_manifest(config)
    write_json(config.split_manifest_path, manifest)
    return manifest


def build_transforms(config: AppConfig) -> dict[str, transforms.Compose]:
    normalize = transforms.Normalize(mean=config.data.mean, std=config.data.std)
    image_size = config.data.image_size
    if config.data.augmentation == "strong":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    image_size,
                    scale=(0.7, 1.0),
                    ratio=(0.8, 1.25),
                ),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.05,
                ),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
            ]
        )
    elif config.data.augmentation == "basic":
        train_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        raise ValueError(f"Unsupported augmentation profile: {config.data.augmentation}")

    return {
        "train": train_transform,
        "eval": transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    }


def build_dataloaders(
    config: AppConfig, manifest: dict[str, Any]
) -> tuple[dict[str, DataLoader], dict[str, ImageManifestDataset]]:
    transforms_by_split = build_transforms(config)
    datasets = {
        "train": ImageManifestDataset(
            config.raw_dir, manifest["splits"]["train"], transforms_by_split["train"]
        ),
        "val": ImageManifestDataset(
            config.raw_dir, manifest["splits"]["val"], transforms_by_split["eval"]
        ),
        "test": ImageManifestDataset(
            config.raw_dir, manifest["splits"]["test"], transforms_by_split["eval"]
        ),
    }
    dataloaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=config.data.train_batch_size,
            shuffle=True,
            num_workers=config.data.num_workers,
            worker_init_fn=worker_init_fn,
            generator=build_generator(config.seed),
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=config.data.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=config.data.eval_batch_size,
            shuffle=False,
            num_workers=config.data.num_workers,
        ),
    }
    return dataloaders, datasets
