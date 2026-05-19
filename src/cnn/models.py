"""Dynamic CNN assembly utilities and model configuration structures."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


ACTIVATIONS: dict[str, nn.Module] = {
    "lrelu": nn.LeakyReLU(),
    "relu": nn.ReLU(),
    "sigmoid": nn.Sigmoid(),
    "tanh": nn.Tanh(),
    "none": nn.Identity(),
}


def _activation(name: str) -> nn.Module:
    """Resolve an activation name to a concrete PyTorch module."""

    key = name.lower()
    if key not in ACTIVATIONS:
        raise ValueError(f"Unsupported activation: {name}")
    return ACTIVATIONS[key]


@dataclass
class ArchitectureConfig:
    """Full architecture specification required to build a dynamic CNN."""

    input_channels: int
    input_size: int
    output_size: int
    conv_out_channels: list[int]
    conv_kernel_sizes: list[int]
    conv_strides: list[int]
    conv_paddings: list[int]
    max_pool_enabled: list[bool]
    max_pool_kernel_sizes: list[int]
    max_pool_strides: list[int]
    max_pool_paddings: list[int]
    batch_norm_2d: list[bool]
    dropout2d_enabled: list[bool]
    dropout2d_probabilities: list[float]
    conv_activations: list[str]
    fc_hidden_features: list[int]
    linear_batch_norm: list[bool]
    linear_dropout_enabled: list[bool]
    linear_dropout_probabilities: list[float]
    fc_activations: list[str]
    input_dropout_enabled: bool = False
    input_dropout_probability: float = 0.0


class DynamicAutocorrCNN(nn.Module):
    """Configurable CNN for regression from interferometric trace images."""

    def __init__(self, config: ArchitectureConfig) -> None:
        super().__init__()
        self.config = config
        self.features = self._build_feature_extractor()
        self.flattened_features = self._infer_flattened_features()
        self.classifier = self._build_classifier()

    def _build_feature_extractor(self) -> nn.Sequential:
        """Build the convolutional feature extractor from the configuration."""

        layers: list[nn.Module] = []
        in_channels = self.config.input_channels
        for index, out_channels in enumerate(self.config.conv_out_channels):
            layers.append(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    self.config.conv_kernel_sizes[index],
                    self.config.conv_strides[index],
                    self.config.conv_paddings[index],
                )
            )
            if self.config.max_pool_enabled[index]:
                layers.append(
                    nn.MaxPool2d(
                        self.config.max_pool_kernel_sizes[index],
                        self.config.max_pool_strides[index],
                        self.config.max_pool_paddings[index],
                    )
                )
            if self.config.batch_norm_2d[index]:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(_activation(self.config.conv_activations[index]))
            if self.config.dropout2d_enabled[index]:
                layers.append(nn.Dropout2d(self.config.dropout2d_probabilities[index]))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _infer_flattened_features(self) -> int:
        """Infer flattened feature count by forwarding a dummy input tensor."""

        with torch.no_grad():
            dummy = torch.zeros(
                1,
                self.config.input_channels,
                self.config.input_size,
                self.config.input_size,
            )
            output = self.features(dummy)
        return int(output.reshape(1, -1).shape[1])

    def _build_classifier(self) -> nn.Sequential:
        """Build the linear classifier/regressor head."""

        layers: list[nn.Module] = []
        in_features = self.flattened_features

        if self.config.input_dropout_enabled:
            layers.append(nn.Dropout(self.config.input_dropout_probability))

        linear_out_features = [*self.config.fc_hidden_features, self.config.output_size]
        for index, out_features in enumerate(linear_out_features):
            layers.append(nn.Linear(in_features, out_features))
            if self.config.linear_batch_norm[index]:
                layers.append(nn.BatchNorm1d(out_features))
            layers.append(_activation(self.config.fc_activations[index]))
            if self.config.linear_dropout_enabled[index]:
                layers.append(nn.Dropout(self.config.linear_dropout_probabilities[index]))
            in_features = out_features
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a full forward pass from image tensor to regression output."""

        x = self.features(x)
        x = x.reshape(x.shape[0], -1)
        return self.classifier(x)
