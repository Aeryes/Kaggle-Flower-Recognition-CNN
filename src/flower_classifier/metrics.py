from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def classification_summary(
    labels: list[int], predictions: list[int], class_names: list[str]
) -> dict[str, Any]:
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )
    per_class = []
    for index, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_name": class_name,
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
        )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(macro[0]),
        "macro_recall": float(macro[1]),
        "macro_f1": float(macro[2]),
        "weighted_precision": float(weighted[0]),
        "weighted_recall": float(weighted[1]),
        "weighted_f1": float(weighted[2]),
        "per_class": per_class,
    }


def save_confusion_matrix(
    labels: list[int], predictions: list[int], class_names: list[str], output_path: Path
) -> None:
    matrix = confusion_matrix(labels, predictions, labels=list(range(len(class_names))))
    figure, axis = plt.subplots(figsize=(7, 5))
    image = axis.imshow(matrix)
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    for row_index, row in enumerate(matrix):
        for column_index, value in enumerate(row):
            axis.text(column_index, row_index, str(value), ha="center", va="center")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def predictions_frame(predictions: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(predictions)
