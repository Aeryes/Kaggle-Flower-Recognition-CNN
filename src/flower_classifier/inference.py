from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from flower_classifier.config import AppConfig
from flower_classifier.data import build_transforms
from flower_classifier.models import build_model
from flower_classifier.training import load_checkpoint
from flower_classifier.utils import select_device, summarize_topk


@torch.no_grad()
def predict_image(
    config: AppConfig, checkpoint_path: Path, image_path: Path, top_k: int = 3
) -> dict[str, Any]:
    device = select_device(config.device)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    class_names = checkpoint["class_names"]
    transform = build_transforms(config)["eval"]

    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    model.eval()
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
    top_predictions = summarize_topk(probabilities, class_names, top_k=top_k)
    predicted_index = int(torch.argmax(logits, dim=1).item())
    return {
        "image_path": str(image_path),
        "predicted_index": predicted_index,
        "predicted_class": class_names[predicted_index],
        "top_predictions": top_predictions,
    }
