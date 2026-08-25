from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from flower_classifier.config import AppConfig


class Unit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(inputs)))


class CustomCNN(nn.Module):
    """Modernized version of the original 14-unit CNN baseline."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            Unit(3, 32),
            Unit(32, 32),
            Unit(32, 32),
            nn.MaxPool2d(kernel_size=2),
            Unit(32, 64),
            Unit(64, 64),
            Unit(64, 64),
            Unit(64, 64),
            nn.MaxPool2d(kernel_size=2),
            Unit(64, 128),
            Unit(128, 128),
            Unit(128, 128),
            Unit(128, 128),
            nn.MaxPool2d(kernel_size=2),
            Unit(128, 128),
            Unit(128, 128),
            Unit(128, 128),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(inputs)
        outputs = torch.flatten(outputs, start_dim=1)
        return self.classifier(outputs)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        reduced_channels = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.scale = nn.Sequential(
            nn.Conv2d(channels, reduced_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs * self.scale(self.pool(inputs))


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.se = SqueezeExcitation(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.skip(inputs)
        outputs = self.activation(self.bn1(self.conv1(inputs)))
        outputs = self.se(self.bn2(self.conv2(outputs)))
        return self.activation(outputs + residual)


class CustomCNNV2(nn.Module):
    """Residual custom CNN with channel attention and regularized classification."""

    def __init__(self, num_classes: int, dropout: float = 0.25):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.features = nn.Sequential(
            ResidualSEBlock(32, 32),
            ResidualSEBlock(32, 32),
            ResidualSEBlock(32, 64, stride=2),
            ResidualSEBlock(64, 64),
            ResidualSEBlock(64, 128, stride=2),
            ResidualSEBlock(128, 128),
            ResidualSEBlock(128, 256, stride=2),
            ResidualSEBlock(256, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.features(self.stem(inputs))
        return self.classifier(self.pool(outputs))


def build_model(config: AppConfig) -> nn.Module:
    model_name = config.model.name.lower()
    if model_name == "custom_cnn":
        return CustomCNN(config.model.num_classes)
    if model_name == "custom_cnn_v2":
        return CustomCNNV2(config.model.num_classes, dropout=config.model.dropout)
    if model_name == "resnet18":
        weights = ResNet18_Weights.DEFAULT if config.model.pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, config.model.num_classes)
        return model
    raise ValueError(f"Unsupported model name: {config.model.name}")
