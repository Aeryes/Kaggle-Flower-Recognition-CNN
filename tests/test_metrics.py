from __future__ import annotations

from flower_classifier.metrics import classification_summary


def test_classification_summary_contains_expected_keys():
    summary = classification_summary(
        labels=[0, 0, 1, 1],
        predictions=[0, 1, 1, 1],
        class_names=["a", "b"],
    )

    assert summary["accuracy"] == 0.75
    assert "macro_f1" in summary
    assert len(summary["per_class"]) == 2
