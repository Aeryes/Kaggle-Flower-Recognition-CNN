from __future__ import annotations

import os
from pathlib import Path

import gradio as gr

from flower_classifier.config import load_config
from flower_classifier.inference import predict_image


def _default_checkpoint() -> Path | None:
    configured = os.getenv("FLOWER_CHECKPOINT")
    if configured:
        return Path(configured)
    candidates = sorted(Path("artifacts/runs").glob("*/checkpoints/best.pt"))
    return candidates[-1] if candidates else None


CONFIG_PATH = Path(os.getenv("FLOWER_CONFIG", "configs/custom-cnn.yaml"))
CONFIG = load_config(CONFIG_PATH)
CHECKPOINT = _default_checkpoint()


def classify(image):
    if image is None:
        raise gr.Error("Upload an image to classify.")
    if CHECKPOINT is None or not CHECKPOINT.exists():
        raise gr.Error(
            "No checkpoint was found. Train a model first or set "
            "FLOWER_CHECKPOINT to a best.pt file."
        )
    result = predict_image(CONFIG, CHECKPOINT, Path(image), top_k=5)
    scores = {item["class_name"]: item["probability"] for item in result["top_predictions"]}
    metadata = (
        f"Predicted class: {result['predicted_class']}\n"
        f"Checkpoint: {CHECKPOINT}\n"
        f"Config: {CONFIG_PATH}"
    )
    return scores, metadata


def build_demo() -> gr.Blocks:
    with gr.Blocks() as demo:
        gr.Markdown(
            "# Flower Classifier Demo\n"
            "Upload an image to run deterministic inference with the configured checkpoint."
        )
        with gr.Row():
            image = gr.Image(type="filepath", label="Flower image")
            label = gr.Label(num_top_classes=5, label="Predictions")
        metadata = gr.Textbox(label="Run metadata", lines=3)
        image.change(fn=classify, inputs=image, outputs=[label, metadata])
    return demo


demo = build_demo()


if __name__ == "__main__":
    demo.launch()
