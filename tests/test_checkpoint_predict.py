from __future__ import annotations

from pathlib import Path

from flower_classifier.inference import predict_image
from flower_classifier.training import train_model


def test_checkpoint_roundtrip_supports_prediction(tiny_config, tiny_dataset: Path):
    run_dir = train_model(tiny_config)
    checkpoint_path = run_dir / "checkpoints" / "best.pt"
    image_path = next((tiny_dataset / "daisy").glob("*.png"))

    prediction = predict_image(tiny_config, checkpoint_path, image_path)

    assert checkpoint_path.exists()
    assert prediction["predicted_class"] in {"daisy", "dandelion", "rose", "sunflower", "tulip"}
    assert len(prediction["top_predictions"]) >= 1
