from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from flower_classifier.config import AppConfig, load_config
from flower_classifier.data import build_dataloaders, prepare_data
from flower_classifier.inference import predict_image
from flower_classifier.training import benchmark_runs, evaluate_checkpoint, train_model


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reproducible flower classification CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in ("prepare-data", "train", "evaluate", "predict", "preview"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", default="configs/custom-cnn.yaml")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument("--run-dir", action="append", required=True)
    benchmark_parser.add_argument("--output-dir", default="reports/benchmark")

    train_parser = subparsers.choices["train"]
    train_parser.add_argument("--resume-checkpoint")

    evaluate_parser = subparsers.choices["evaluate"]
    evaluate_parser.add_argument("--checkpoint", required=True)
    evaluate_parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    evaluate_parser.add_argument("--output-dir")

    predict_parser = subparsers.choices["predict"]
    predict_parser.add_argument("--checkpoint", required=True)
    predict_parser.add_argument("--image", required=True)
    predict_parser.add_argument("--top-k", type=int, default=3)

    preview_parser = subparsers.choices["preview"]
    preview_parser.add_argument("--split", default="train", choices=["train", "val", "test"])
    preview_parser.add_argument("--count", type=int, default=4)
    return parser


def _load(args: argparse.Namespace) -> AppConfig:
    return load_config(args.config)


def _entry_args(argv: Sequence[str] | None) -> list[str]:
    return list(argv) if argv is not None else sys.argv[1:]


def prepare_data_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["prepare-data", *_entry_args(argv)])
    config = _load(args)
    manifest = prepare_data(config)
    print(
        json.dumps(
            {"manifest": str(config.split_manifest_path), "counts": manifest["counts"]}, indent=2
        )
    )


def train_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["train", *_entry_args(argv)])
    config = _load(args)
    run_dir = train_model(config, Path(args.resume_checkpoint) if args.resume_checkpoint else None)
    print(run_dir)


def evaluate_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["evaluate", *_entry_args(argv)])
    config = _load(args)
    output_dir = Path(args.output_dir) if args.output_dir else None
    metrics = evaluate_checkpoint(
        config, Path(args.checkpoint), split=args.split, output_dir=output_dir
    )
    print(json.dumps(metrics, indent=2))


def predict_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["predict", *_entry_args(argv)])
    config = _load(args)
    prediction = predict_image(config, Path(args.checkpoint), Path(args.image), top_k=args.top_k)
    print(json.dumps(prediction, indent=2))


def benchmark_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["benchmark", *_entry_args(argv)])
    output_dir = benchmark_runs([Path(path) for path in args.run_dir], Path(args.output_dir))
    print(output_dir)


def preview_command(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(["preview", *_entry_args(argv)])
    config = _load(args)
    manifest = prepare_data(config)
    dataloaders, _ = build_dataloaders(config, manifest)
    images, labels = next(iter(dataloaders[args.split]))

    count = min(args.count, len(images))
    figure, axes = plt.subplots(1, count, figsize=(count * 3, 3))
    if count == 1:
        axes = [axes]

    mean = np.array(config.data.mean).reshape(3, 1, 1)
    std = np.array(config.data.std).reshape(3, 1, 1)
    for axis, image, label in zip(axes, images[:count], labels[:count], strict=True):
        restored = (image.numpy() * std) + mean
        restored = np.clip(np.transpose(restored, (1, 2, 0)), 0.0, 1.0)
        axis.imshow(restored)
        axis.set_title(manifest["class_names"][int(label.item())])
        axis.axis("off")
    figure.tight_layout()
    plt.show()


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "prepare-data":
        prepare_data_command(["--config", args.config])
        return
    if args.command == "train":
        command_args = ["--config", args.config]
        if args.resume_checkpoint:
            command_args.extend(["--resume-checkpoint", args.resume_checkpoint])
        train_command(command_args)
        return
    if args.command == "evaluate":
        command_args = [
            "--config",
            args.config,
            "--checkpoint",
            args.checkpoint,
            "--split",
            args.split,
        ]
        if args.output_dir:
            command_args.extend(["--output-dir", args.output_dir])
        evaluate_command(command_args)
        return
    if args.command == "predict":
        predict_command(
            [
                "--config",
                args.config,
                "--checkpoint",
                args.checkpoint,
                "--image",
                args.image,
                "--top-k",
                str(args.top_k),
            ]
        )
        return
    if args.command == "benchmark":
        benchmark_args: list[str] = []
        for run_dir in args.run_dir:
            benchmark_args.extend(["--run-dir", run_dir])
        benchmark_args.extend(["--output-dir", args.output_dir])
        benchmark_command(benchmark_args)
        return
    if args.command == "preview":
        preview_command(
            ["--config", args.config, "--split", args.split, "--count", str(args.count)]
        )
        return


if __name__ == "__main__":
    main()
