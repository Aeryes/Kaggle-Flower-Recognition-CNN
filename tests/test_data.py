from __future__ import annotations

from flower_classifier.data import build_dataloaders, prepare_data


def test_prepare_data_is_deterministic(tiny_config):
    manifest_a = prepare_data(tiny_config, force=True)
    manifest_b = prepare_data(tiny_config, force=False)

    assert manifest_a["dataset_fingerprint"] == manifest_b["dataset_fingerprint"]
    assert manifest_a["counts"] == manifest_b["counts"]
    assert manifest_a["splits"] == manifest_b["splits"]


def test_dataloaders_emit_expected_shape(tiny_config):
    manifest = prepare_data(tiny_config, force=True)
    dataloaders, _ = build_dataloaders(tiny_config, manifest)
    images, labels = next(iter(dataloaders["train"]))

    assert images.shape[1:] == (3, tiny_config.data.image_size, tiny_config.data.image_size)
    assert labels.ndim == 1
