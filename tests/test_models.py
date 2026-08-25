from __future__ import annotations

import torch

from flower_classifier.models import build_model


def test_custom_cnn_output_shape(tiny_config):
    model = build_model(tiny_config)
    outputs = model(torch.randn(2, 3, tiny_config.data.image_size, tiny_config.data.image_size))
    assert outputs.shape == (2, tiny_config.model.num_classes)


def test_custom_cnn_v2_output_shape(tiny_config):
    tiny_config.model.name = "custom_cnn_v2"
    tiny_config.model.dropout = 0.25
    model = build_model(tiny_config)
    outputs = model(torch.randn(2, 3, tiny_config.data.image_size, tiny_config.data.image_size))
    assert outputs.shape == (2, tiny_config.model.num_classes)
