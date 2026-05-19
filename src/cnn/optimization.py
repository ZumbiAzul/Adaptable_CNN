"""Optuna search-space assembly, pruning logic, and study objective."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import optuna
import torch

from .data import DatasetBundle
from .models import ArchitectureConfig, DynamicAutocorrCNN
from .training import ArtifactManager, TrainingConfig, TrainingResult, train_model


@dataclass
class OptimizationContext:
    """Everything required to sample, train, and evaluate one Optuna trial."""

    repo_root: Path
    bundle: DatasetBundle
    database_name: str
    study_name: str
    input_channels: int
    input_size: int
    output_size: int
    num_conv_layers: int
    num_fc_hidden_layers: int
    fixed_parameters: dict[str, list[Any]]
    parameter_ranges: dict[str, tuple[Any, Any]]
    fixed_hyperparameters: dict[str, Any]
    hyperparameter_ranges: dict[str, tuple[Any, Any]]
    conv_activations: list[str]
    fc_activations: list[str]
    optimization_run_id: str


def _fixed_or_suggest_int(
    trial: optuna.Trial,
    name: str,
    values: list[Any],
    index: int,
    low: int,
    high: int,
    *,
    step: int = 1,
) -> int:
    """Return a fixed integer or ask Optuna to sample one."""

    if index < len(values) and values[index] is not None:
        return int(values[index])
    return int(trial.suggest_int(name, low, high, step=step))


def _fixed_or_suggest_float(
    trial: optuna.Trial,
    name: str,
    values: list[Any],
    index: int,
    low: float,
    high: float,
    *,
    step: float | None = None,
    log: bool = False,
) -> float:
    """Return a fixed float or ask Optuna to sample one."""

    if index < len(values) and values[index] is not None:
        return float(values[index])
    kwargs: dict[str, Any] = {"log": log}
    if step is not None:
        kwargs["step"] = step
    return float(trial.suggest_float(name, low, high, **kwargs))


def _conv_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Compute the 1D spatial output size of a convolution layer."""

    return int((input_size + 2 * padding - kernel_size) / stride + 1)


def _pool_output_size(input_size: int, kernel_size: int, stride: int, padding: int) -> int:
    """Compute the 1D spatial output size of a pooling layer."""

    return int((input_size + 2 * padding - kernel_size) / stride + 1)


def build_architecture_config(trial: optuna.Trial, context: OptimizationContext) -> ArchitectureConfig:
    """Sample or fix one CNN architecture and prune invalid size collapses."""

    fixed = context.fixed_parameters
    ranges = context.parameter_ranges
    current_size = context.input_size

    conv_out_channels: list[int] = []
    conv_kernel_sizes: list[int] = []
    conv_strides: list[int] = []
    conv_paddings: list[int] = []
    max_pool_enabled: list[bool] = []
    max_pool_kernel_sizes: list[int] = []
    max_pool_strides: list[int] = []
    max_pool_paddings: list[int] = []
    batch_norm_2d: list[bool] = []
    dropout2d_enabled: list[bool] = []
    dropout2d_probabilities: list[float] = []

    for index in range(context.num_conv_layers):
        conv_out_channels.append(
            _fixed_or_suggest_int(
                trial,
                f"conv_out_channels_{index}",
                fixed["conv_layer_out_channels"],
                index,
                *ranges["conv_layer_out_channels"],
            )
        )
        conv_kernel_sizes.append(
            _fixed_or_suggest_int(
                trial,
                f"conv_kernel_size_{index}",
                fixed["conv_layer_kernel_sizes"],
                index,
                *ranges["conv_layer_kernel_sizes"],
                step=2,
            )
        )
        conv_strides.append(
            _fixed_or_suggest_int(
                trial,
                f"conv_stride_{index}",
                fixed["conv_layer_strides"],
                index,
                *ranges["conv_layer_strides"],
            )
        )
        conv_paddings.append(
            _fixed_or_suggest_int(
                trial,
                f"conv_padding_{index}",
                fixed["conv_layer_paddings"],
                index,
                ranges["conv_layer_paddings"][0],
                min(ranges["conv_layer_paddings"][1], (conv_kernel_sizes[index] - 1) // 2),
            )
        )
        current_size = _conv_output_size(
            current_size,
            conv_kernel_sizes[index],
            conv_strides[index],
            conv_paddings[index],
        )
        if current_size < 1:
            raise optuna.TrialPruned(
                f"Conv layer {index} reduces spatial size below 1."
            )

        pool_enabled = bool(
            _fixed_or_suggest_int(
                trial,
                f"max_pool_enabled_{index}",
                fixed["max_pool_layer_numbers"],
                index,
                *ranges["max_pool_layer_numbers"],
            )
        )
        max_pool_enabled.append(pool_enabled)
        if pool_enabled:
            max_pool_kernel_sizes.append(
                _fixed_or_suggest_int(
                    trial,
                    f"max_pool_kernel_size_{index}",
                    fixed["max_pool_kernel_sizes"],
                    index,
                    *ranges["max_pool_kernel_sizes"],
                )
            )
            max_pool_strides.append(
                _fixed_or_suggest_int(
                    trial,
                    f"max_pool_stride_{index}",
                    fixed["max_pool_strides"],
                    index,
                    ranges["max_pool_strides"][0],
                    min(ranges["max_pool_strides"][1], max_pool_kernel_sizes[index]),
                )
            )
            max_pool_paddings.append(
                _fixed_or_suggest_int(
                    trial,
                    f"max_pool_padding_{index}",
                    fixed["max_pool_paddings"],
                    index,
                    ranges["max_pool_paddings"][0],
                    min(ranges["max_pool_paddings"][1], max_pool_kernel_sizes[index] // 2),
                )
            )
            current_size = _pool_output_size(
                current_size,
                max_pool_kernel_sizes[index],
                max_pool_strides[index],
                max_pool_paddings[index],
            )
            if current_size <= 1:
                raise optuna.TrialPruned(
                    f"Pooling layer {index} reduces spatial size to 1 or below."
                )
        else:
            max_pool_kernel_sizes.append(0)
            max_pool_strides.append(0)
            max_pool_paddings.append(0)

        batch_norm_2d.append(
            bool(
                _fixed_or_suggest_int(
                    trial,
                    f"batch_norm_2d_{index}",
                    fixed["batch_norm_2d_layer_numbers"],
                    index,
                    *ranges["batch_norm_2d_layer_numbers"],
                )
            )
        )
        dropout_enabled = bool(
            _fixed_or_suggest_int(
                trial,
                f"dropout2d_enabled_{index}",
                fixed["dropout2d_layer_numbers"],
                index,
                *ranges["dropout2d_layer_numbers"],
            )
        )
        dropout2d_enabled.append(dropout_enabled)
        if dropout_enabled:
            dropout2d_probabilities.append(
                _fixed_or_suggest_float(
                    trial,
                    f"dropout2d_probability_{index}",
                    fixed["dropout2d_probabilities"],
                    index,
                    *ranges["dropout2d_probabilities"],
                    step=0.02,
                )
            )
        else:
            dropout2d_probabilities.append(0.0)

    fc_hidden_features: list[int] = []
    for index in range(context.num_fc_hidden_layers):
        fc_hidden_features.append(
            _fixed_or_suggest_int(
                trial,
                f"fc_hidden_feature_{index}",
                fixed["fc_hidden_features"],
                index,
                *ranges["linear_layer_out_features"],
            )
        )

    linear_batch_norm = [
        bool(fixed["linear_batch_norm"][index])
        if index < len(fixed["linear_batch_norm"])
        else False
        for index in range(context.num_fc_hidden_layers)
    ]
    linear_batch_norm.append(False)

    linear_dropout_enabled: list[bool] = []
    linear_dropout_probabilities: list[float] = []
    for index in range(context.num_fc_hidden_layers):
        enabled = bool(
            _fixed_or_suggest_int(
                trial,
                f"linear_dropout_hidden_{index}",
                fixed["linear_dropout_enabled"],
                index,
                *ranges["dropout1d_layer_numbers"],
            )
        )
        linear_dropout_enabled.append(enabled)
        if enabled:
            linear_dropout_probabilities.append(
                _fixed_or_suggest_float(
                    trial,
                    f"linear_dropout_probability_{index}",
                    fixed["linear_dropout_probabilities"],
                    index,
                    *ranges["dropout1d_probabilities"],
                    step=0.02,
                )
            )
        else:
            linear_dropout_probabilities.append(0.0)
    linear_dropout_enabled.append(False)
    linear_dropout_probabilities.append(0.0)

    input_dropout_enabled = bool(
        _fixed_or_suggest_int(
            trial,
            "input_dropout_enabled",
            fixed["input_dropout_enabled"],
            0,
            *ranges["dropout1d_layer_numbers"],
        )
    )
    input_dropout_probability = (
        _fixed_or_suggest_float(
            trial,
            "input_dropout_probability",
            fixed["input_dropout_probability"],
            0,
            *ranges["dropout1d_probabilities"],
            step=0.02,
        )
        if input_dropout_enabled
        else 0.0
    )

    flattened_features = current_size * current_size * conv_out_channels[-1]
    if flattened_features == 0:
        raise optuna.TrialPruned("Flattened feature size is zero.")

    return ArchitectureConfig(
        input_channels=context.input_channels,
        input_size=context.input_size,
        output_size=context.output_size,
        conv_out_channels=conv_out_channels,
        conv_kernel_sizes=conv_kernel_sizes,
        conv_strides=conv_strides,
        conv_paddings=conv_paddings,
        max_pool_enabled=max_pool_enabled,
        max_pool_kernel_sizes=max_pool_kernel_sizes,
        max_pool_strides=max_pool_strides,
        max_pool_paddings=max_pool_paddings,
        batch_norm_2d=batch_norm_2d,
        dropout2d_enabled=dropout2d_enabled,
        dropout2d_probabilities=dropout2d_probabilities,
        conv_activations=context.conv_activations,
        fc_hidden_features=fc_hidden_features,
        linear_batch_norm=linear_batch_norm,
        linear_dropout_enabled=linear_dropout_enabled,
        linear_dropout_probabilities=linear_dropout_probabilities,
        fc_activations=context.fc_activations,
        input_dropout_enabled=input_dropout_enabled,
        input_dropout_probability=input_dropout_probability,
    )


def build_training_config(trial: optuna.Trial, context: OptimizationContext) -> TrainingConfig:
    """Sample or fix the training hyperparameters for one trial."""

    fixed = context.fixed_hyperparameters
    ranges = context.hyperparameter_ranges

    learning_rate_decay_name = fixed["learning_rate_decay"]
    if learning_rate_decay_name is None:
        decay_mode = trial.suggest_int("learning_rate_decay_mode", *ranges["learning_rate_decay"])
        learning_rate_decay_name = {0: "cnst", 1: "exp", 2: "sqrt"}[decay_mode]

    batch_size = (
        fixed["batch_size"]
        if fixed["batch_size"] is not None
        else trial.suggest_int("batch_size", *ranges["batch_size"])
    )
    learning_rate = (
        fixed["learning_rate"]
        if fixed["learning_rate"] is not None
        else trial.suggest_float("learning_rate", *ranges["learning_rate"], log=True)
    )
    learning_rate_decay_const = (
        fixed["learning_rate_decay_const"]
        if fixed["learning_rate_decay_const"] is not None
        else trial.suggest_int("learning_rate_decay_const", *ranges["learning_rate_decay_const"])
    )
    beta1 = fixed["beta1"] if fixed["beta1"] is not None else trial.suggest_float("beta1", *ranges["beta1"])
    beta2 = fixed["beta2"] if fixed["beta2"] is not None else trial.suggest_float("beta2", *ranges["beta2"])
    epochs = fixed["epochs"] if fixed["epochs"] is not None else trial.suggest_int("epochs", *ranges["epochs"])

    return TrainingConfig(
        batch_size=int(batch_size),
        learning_rate=float(learning_rate),
        learning_rate_decay=str(learning_rate_decay_name),
        learning_rate_decay_const=float(learning_rate_decay_const),
        beta1=float(beta1),
        beta2=float(beta2),
        epochs=int(epochs),
        num_workers=int(fixed["num_workers"]),
        device=str(fixed["device"]),
        shuffle=bool(fixed["shuffle"]),
        trainset=str(fixed["trainset"]),
        initialization_trials=int(fixed.get("initialization_trials", 0)),
        initialization_epochs=int(fixed.get("initialization_epochs", 1)),
    )


class OptunaAutocorrObjective:
    """Callable Optuna objective that trains a sampled CNN and returns dev loss."""

    def __init__(self, context: OptimizationContext) -> None:
        self.context = context
        self.artifacts = ArtifactManager(
            repo_root=self.context.repo_root,
            database_name=self.context.database_name,
            study_name=self.context.study_name,
            run_id=self.context.optimization_run_id,
        )

    def __call__(self, trial: optuna.Trial) -> float:
        """Train one sampled configuration and return the final dev loss."""

        architecture = build_architecture_config(trial, self.context)
        training = build_training_config(trial, self.context)

        run_name = f"{self.artifacts.run_id}_trial_{trial.number:04d}"
        try:
            result: TrainingResult = train_model(
                bundle=self.context.bundle,
                architecture=architecture,
                training=training,
                artifacts=self.artifacts,
                run_name=run_name,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "kernel size can't be greater than actual input size" in message:
                raise optuna.TrialPruned(str(exc)) from exc
            raise

        trial.set_user_attr("run_id", result.run_id)
        trial.set_user_attr("checkpoint_path", str(result.checkpoint_path))
        trial.set_user_attr("best_dev_loss", result.best_dev_loss)
        trial.set_user_attr("selected_initialization_trial", result.selected_initialization_trial)
        if result.initialization_screen_loss is not None:
            trial.set_user_attr("initialization_screen_loss", result.initialization_screen_loss)
        return result.final_dev_loss
