from __future__ import annotations

from pathlib import Path

from flower_classifier.training import benchmark_runs, evaluate_checkpoint, train_model


def test_end_to_end_pipeline_generates_artifacts(tiny_config: object, tmp_path: Path):
    run_dir = train_model(tiny_config)  # type: ignore[arg-type]
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    summary = evaluate_checkpoint(tiny_config, checkpoint_path, split="test")  # type: ignore[arg-type]
    benchmark_dir = benchmark_runs([run_dir], tmp_path / "benchmark")

    assert (run_dir / "metrics.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "evaluation" / "test" / "summary.json").exists()
    assert (run_dir / "evaluation" / "test" / "predictions.csv").exists()
    assert (benchmark_dir / "comparison.json").exists()
    assert summary["split"] == "test"
