from __future__ import annotations

import csv
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch import nn
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    SequentialLR,
)
from torch.utils.tensorboard import SummaryWriter

from flower_classifier.config import AppConfig
from flower_classifier.data import build_dataloaders, prepare_data
from flower_classifier.metrics import (
    classification_summary,
    predictions_frame,
    save_confusion_matrix,
)
from flower_classifier.models import build_model
from flower_classifier.utils import (
    copy_file,
    ensure_directory,
    environment_report,
    seed_everything,
    select_device,
    summarize_topk,
    utc_now,
    write_json,
    write_yaml,
)


@dataclass(slots=True)
class RunPaths:
    run_id: str
    root: Path
    tensorboard_dir: Path
    checkpoints_dir: Path
    evaluation_dir: Path
    metrics_csv: Path
    metrics_json: Path
    config_path: Path
    environment_path: Path
    status_path: Path
    split_manifest_path: Path


def _create_run_paths(config: AppConfig, run_id: str) -> RunPaths:
    root = ensure_directory(config.runs_root / run_id)
    return RunPaths(
        run_id=run_id,
        root=root,
        tensorboard_dir=ensure_directory(root / "tensorboard"),
        checkpoints_dir=ensure_directory(root / "checkpoints"),
        evaluation_dir=ensure_directory(root / "evaluation"),
        metrics_csv=root / "metrics.csv",
        metrics_json=root / "metrics.json",
        config_path=root / "resolved-config.yaml",
        environment_path=root / "environment.json",
        status_path=root / "status.json",
        split_manifest_path=root / "split-manifest.json",
    )


def _new_run_id(config: AppConfig) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{config.model.name.replace('_', '-')}"


def _initialize_run(config: AppConfig, run_id: str | None = None) -> RunPaths:
    run_id = run_id or _new_run_id(config)
    paths = _create_run_paths(config, run_id)
    write_yaml(paths.config_path, config.to_dict())
    write_json(paths.environment_path, environment_report())
    write_json(paths.status_path, {"state": "running", "updated_at": utc_now(), "run_id": run_id})
    return paths


def _resume_paths_from_checkpoint(checkpoint_path: Path) -> RunPaths:
    run_root = checkpoint_path.resolve().parents[1]
    run_id = run_root.name
    return RunPaths(
        run_id=run_id,
        root=run_root,
        tensorboard_dir=run_root / "tensorboard",
        checkpoints_dir=run_root / "checkpoints",
        evaluation_dir=run_root / "evaluation",
        metrics_csv=run_root / "metrics.csv",
        metrics_json=run_root / "metrics.json",
        config_path=run_root / "resolved-config.yaml",
        environment_path=run_root / "environment.json",
        status_path=run_root / "status.json",
        split_manifest_path=run_root / "split-manifest.json",
    )


def _create_optimizer(config: AppConfig, model: nn.Module) -> Optimizer:
    optimizer_name = config.training.optimizer.lower()
    kwargs = {
        "lr": config.training.learning_rate,
        "weight_decay": config.training.weight_decay,
    }
    if optimizer_name == "adam":
        return Adam(model.parameters(), **kwargs)
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), **kwargs)
    raise ValueError(f"Unsupported optimizer: {config.training.optimizer}")


def _create_scheduler(config: AppConfig, optimizer: Optimizer) -> LRScheduler:
    warmup_epochs = min(config.training.warmup_epochs, max(config.training.epochs - 1, 0))
    if warmup_epochs == 0:
        return CosineAnnealingLR(optimizer, T_max=max(config.training.epochs, 1))
    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(config.training.epochs - warmup_epochs, 1),
    )
    return SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_epochs],
    )


def _append_metric_row(path: Path, row: dict[str, Any]) -> None:
    ensure_directory(path.parent)
    file_exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    epoch: int,
    best_metric: float,
    config: AppConfig,
    class_names: list[str],
    manifest: dict[str, Any],
    history: list[dict[str, Any]],
    scaler: torch.amp.GradScaler,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_metric": best_metric,
            "config": config.to_dict(),
            "class_names": class_names,
            "manifest_fingerprint": manifest["dataset_fingerprint"],
            "preprocessing": {
                "image_size": config.data.image_size,
                "mean": config.data.mean,
                "std": config.data.std,
            },
            "history": history,
        },
        path,
    )


def load_checkpoint(path: Path, model: nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


@torch.no_grad()
def _run_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> tuple[float, float, list[int], list[int]]:
    model.eval()
    running_loss = 0.0
    total_examples = 0
    labels: list[int] = []
    predictions: list[int] = []
    for images, batch_labels in dataloader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)
        logits = model(images)
        loss = criterion(logits, batch_labels)
        running_loss += loss.item() * images.size(0)
        total_examples += images.size(0)
        batch_predictions = torch.argmax(logits, dim=1)
        labels.extend(batch_labels.cpu().tolist())
        predictions.extend(batch_predictions.cpu().tolist())
    accuracy = sum(
        int(pred == label) for pred, label in zip(predictions, labels, strict=True)
    ) / max(total_examples, 1)
    average_loss = running_loss / max(total_examples, 1)
    return average_loss, accuracy, labels, predictions


def _train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: Optimizer,
    scaler: torch.amp.GradScaler,
    use_amp: bool,
) -> tuple[float, float]:
    model.train()
    running_loss = 0.0
    total_examples = 0
    correct_predictions = 0
    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        total_examples += images.size(0)
        correct_predictions += int((torch.argmax(logits, dim=1) == labels).sum().item())
    return (
        running_loss / max(total_examples, 1),
        correct_predictions / max(total_examples, 1),
    )


@torch.no_grad()
def evaluate_checkpoint(
    config: AppConfig,
    checkpoint_path: Path,
    split: str = "test",
    output_dir: Path | None = None,
) -> dict[str, Any]:
    seed_everything(config.seed)
    manifest = prepare_data(config)
    dataloaders, datasets = build_dataloaders(config, manifest)
    device = select_device(config.device)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device)

    class_names = checkpoint.get("class_names", manifest["class_names"])
    criterion = nn.CrossEntropyLoss()
    loss, accuracy, labels, predictions = _run_epoch(model, dataloaders[split], device, criterion)

    probabilities_output: list[dict[str, Any]] = []
    model.eval()
    start = time.perf_counter()
    for index, (images, batch_labels) in enumerate(dataloaders[split]):
        images = images.to(device)
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1).cpu().tolist()
        predicted_indices = torch.argmax(logits, dim=1).cpu().tolist()
        for batch_index, probability_row in enumerate(probabilities):
            record = datasets[split].records[index * dataloaders[split].batch_size + batch_index]
            probabilities_output.append(
                {
                    "relative_path": record["relative_path"],
                    "label": int(batch_labels[batch_index].item()),
                    "label_name": class_names[int(batch_labels[batch_index].item())],
                    "prediction": int(predicted_indices[batch_index]),
                    "prediction_name": class_names[int(predicted_indices[batch_index])],
                    "top_predictions": summarize_topk(probability_row, class_names, top_k=3),
                }
            )
    latency_ms = ((time.perf_counter() - start) / max(len(probabilities_output), 1)) * 1000.0

    metrics = classification_summary(labels, predictions, class_names)
    metrics.update(
        {
            "split": split,
            "loss": float(loss),
            "accuracy": float(accuracy),
            "inference_latency_ms_per_image": latency_ms,
            "generated_at": utc_now(),
        }
    )
    sample_errors = [item for item in probabilities_output if item["label"] != item["prediction"]][
        :10
    ]

    if output_dir is None:
        output_dir = checkpoint_path.resolve().parents[1] / "evaluation" / split
    ensure_directory(output_dir)
    predictions_frame(probabilities_output).to_csv(output_dir / "predictions.csv", index=False)
    write_json(output_dir / "summary.json", metrics)
    write_json(output_dir / "sample_errors.json", sample_errors)
    save_confusion_matrix(labels, predictions, class_names, output_dir / "confusion_matrix.png")
    return metrics


def train_model(config: AppConfig, resume_checkpoint: Path | None = None) -> Path:
    seed_everything(config.seed)
    manifest = prepare_data(config)
    dataloaders, datasets = build_dataloaders(config, manifest)
    device = select_device(config.device)
    class_names = manifest["class_names"]

    if resume_checkpoint is not None:
        run_paths = _resume_paths_from_checkpoint(resume_checkpoint)
    else:
        run_paths = _initialize_run(config)
    write_json(run_paths.split_manifest_path, manifest)

    model = build_model(config).to(device)
    optimizer = _create_optimizer(config, model)
    scheduler = _create_scheduler(config, optimizer)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    amp_enabled = config.training.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    best_metric = float("-inf")
    history: list[dict[str, Any]] = []
    if resume_checkpoint is not None:
        checkpoint = load_checkpoint(resume_checkpoint, model, device)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = float(checkpoint["best_metric"])
        history = list(checkpoint.get("history", []))

    writer = SummaryWriter(log_dir=str(run_paths.tensorboard_dir))
    try:
        dummy_input = torch.randn(
            1,
            3,
            config.data.image_size,
            config.data.image_size,
            device=device,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            writer.add_graph(model, dummy_input)
    except Exception:
        pass

    epochs_without_improvement = 0
    latest_checkpoint_path = run_paths.checkpoints_dir / "latest.pt"
    best_checkpoint_path = run_paths.checkpoints_dir / "best.pt"

    try:
        for epoch in range(start_epoch, config.training.epochs):
            epoch_started = time.perf_counter()
            train_loss, train_accuracy = _train_epoch(
                model,
                dataloaders["train"],
                device,
                criterion,
                optimizer,
                scaler,
                amp_enabled,
            )
            val_loss, val_accuracy, val_labels, val_predictions = _run_epoch(
                model, dataloaders["val"], device, criterion
            )
            scheduler.step()

            epoch_metrics = classification_summary(val_labels, val_predictions, class_names)
            row = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            row.update(
                {
                    "val_macro_f1": epoch_metrics["macro_f1"],
                    "val_weighted_f1": epoch_metrics["weighted_f1"],
                }
            )
            history.append(row)
            _append_metric_row(run_paths.metrics_csv, row)
            write_json(run_paths.metrics_json, history)

            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            writer.add_scalar("accuracy/train", train_accuracy, epoch)
            writer.add_scalar("accuracy/val", val_accuracy, epoch)
            writer.add_scalar("f1/val_macro", epoch_metrics["macro_f1"], epoch)
            writer.add_scalar("f1/val_weighted", epoch_metrics["weighted_f1"], epoch)
            writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

            current_metric = float(row[config.training.checkpoint_metric])
            _save_checkpoint(
                latest_checkpoint_path,
                model,
                optimizer,
                scheduler,
                epoch,
                best_metric,
                config,
                class_names,
                manifest,
                history,
                scaler,
            )

            if current_metric > best_metric:
                best_metric = current_metric
                epochs_without_improvement = 0
                _save_checkpoint(
                    best_checkpoint_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_metric,
                    config,
                    class_names,
                    manifest,
                    history,
                    scaler,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

        evaluate_checkpoint(config, best_checkpoint_path, split="test")
        write_json(
            run_paths.status_path,
            {
                "state": "completed",
                "updated_at": utc_now(),
                "run_id": run_paths.run_id,
                "best_checkpoint": str(best_checkpoint_path),
            },
        )
    except KeyboardInterrupt:
        write_json(
            run_paths.status_path,
            {
                "state": "interrupted",
                "updated_at": utc_now(),
                "run_id": run_paths.run_id,
                "resume_checkpoint": str(latest_checkpoint_path),
            },
        )
        raise
    except Exception as exc:
        write_json(
            run_paths.status_path,
            {
                "state": "failed",
                "updated_at": utc_now(),
                "run_id": run_paths.run_id,
                "error": repr(exc),
            },
        )
        raise
    finally:
        writer.close()

    return run_paths.root


def benchmark_runs(run_dirs: list[Path], output_dir: Path) -> Path:
    ensure_directory(output_dir)
    summaries: list[dict[str, Any]] = []
    lines = [
        "# Benchmark methodology",
        "",
        "- Compare runs generated from the same split manifest and evaluation flow.",
        "- Report the best checkpoint from each run.",
        "- Treat metrics as measured results, not targets.",
        "",
    ]
    for run_dir in run_dirs:
        summary_path = run_dir / "evaluation" / "test" / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing evaluation summary: {summary_path}")
        summary = pd.read_json(summary_path, typ="series").to_dict()
        summary["run_dir"] = str(run_dir)
        summaries.append(summary)
        for artifact_name in (
            "confusion_matrix.png",
            "predictions.csv",
            "sample_errors.json",
            "summary.json",
        ):
            source = run_dir / "evaluation" / "test" / artifact_name
            if source.exists():
                copy_file(source, output_dir / run_dir.name / artifact_name)
        if (run_dir / "metrics.csv").exists():
            copy_file(run_dir / "metrics.csv", output_dir / run_dir.name / "training_curves.csv")
    comparison = pd.DataFrame(summaries).sort_values("accuracy", ascending=False)
    comparison.to_json(output_dir / "comparison.json", orient="records", indent=2)
    comparison.to_csv(output_dir / "comparison.csv", index=False)
    lines.extend(
        [
            f"- `{row['run_dir']}` accuracy: {row['accuracy']:.4f}"
            for _, row in comparison.iterrows()
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_dir
