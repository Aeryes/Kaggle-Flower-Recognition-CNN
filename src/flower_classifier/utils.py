from __future__ import annotations

import json
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_yaml(path: Path, payload: Any) -> None:
    ensure_directory(path.parent)
    if is_dataclass(payload):
        payload = asdict(payload)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def build_generator(seed: int) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def environment_report() -> dict[str, Any]:
    report = {
        "timestamp_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "torch": getattr(torch, "__version__", "unavailable"),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "numpy": getattr(np, "__version__", "unavailable"),
    }
    report.update(git_report())
    return report


def git_report() -> dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        branch = subprocess.check_output(
            ["git", "branch", "--show-current"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        commit = "unknown"
        dirty = True
        branch = "unknown"
    return {"git_commit": commit, "git_branch": branch, "git_dirty": dirty}


def copy_file(source: Path, target: Path) -> None:
    ensure_directory(target.parent)
    target.write_bytes(source.read_bytes())


def relative_to(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def summarize_topk(
    probabilities: list[float], class_names: list[str], top_k: int
) -> list[dict[str, Any]]:
    pairs = list(zip(class_names, probabilities, strict=True))
    pairs.sort(key=lambda item: item[1], reverse=True)
    return [
        {"class_name": class_name, "probability": float(probability)}
        for class_name, probability in pairs[:top_k]
    ]


def set_torch_num_threads(max_threads: int = 1) -> None:
    current = torch.get_num_threads()
    if current > max_threads:
        torch.set_num_threads(max_threads)
    interop = torch.get_num_interop_threads()
    if interop > max_threads:
        torch.set_num_interop_threads(max_threads)
