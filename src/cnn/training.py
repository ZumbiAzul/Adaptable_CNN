"""Training loop, initialization screening, evaluation, and artifact writing."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from .data import DatasetBundle
from .models import ArchitectureConfig, DynamicAutocorrCNN


plt.ioff()


@dataclass
class TrainingConfig:
    """Concrete training hyperparameters for one run."""

    batch_size: int
    learning_rate: float
    learning_rate_decay: str
    learning_rate_decay_const: float
    beta1: float
    beta2: float
    epochs: int
    num_workers: int
    device: str
    shuffle: bool
    trainset: str = "norm"
    initialization_trials: int = 0
    initialization_epochs: int = 1


@dataclass
class TrainingResult:
    """Outputs and metadata produced by one completed training run."""

    history: list[dict[str, float]]
    final_dev_loss: float
    best_dev_loss: float
    run_id: str
    checkpoint_path: Path
    selected_initialization_trial: int = 0
    initialization_screen_loss: float | None = None


class ArtifactManager:
    """Create and manage output folders and files for one optimization run."""

    def __init__(
        self,
        repo_root: Path,
        database_name: str,
        study_name: str,
        run_id: str | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.database_name = database_name
        self.study_name = study_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d%H%M%S")

        self.results_dir = repo_root / "results" / self.run_id
        self.params_dir = repo_root / "saved_model_params" / self.run_id
        self.plots_dir = repo_root / "plots" / database_name / study_name / self.run_id
        self.logs_dir = repo_root / "logs" / database_name / study_name

        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.params_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def save_history(self, run_name: str, history: list[dict[str, float]]) -> None:
        """Save epoch-by-epoch training history as CSV and JSON."""

        dataframe = pd.DataFrame(history)
        dataframe.to_csv(self.results_dir / f"{run_name}.csv", index=False)
        with (self.results_dir / f"{run_name}.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

    def save_plot(self, run_name: str, history: list[dict[str, float]]) -> None:
        """Save a non-interactive train-versus-dev loss plot."""

        epochs = [row["epoch"] for row in history]
        train_losses = [row["train_loss"] for row in history]
        dev_losses = [row["dev_loss"] for row in history]

        figure, axis = plt.subplots(figsize=(8, 4.5))
        axis.plot(epochs, train_losses, label="train")
        axis.plot(epochs, dev_losses, label="dev")
        axis.set_xlabel("Epoch")
        axis.set_ylabel("MSE loss")
        axis.legend()
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(self.plots_dir / f"{run_name}_loss.png")
        plt.close(figure)

    def save_checkpoint(
        self,
        run_name: str,
        model: DynamicAutocorrCNN,
        architecture: ArchitectureConfig,
        training: TrainingConfig,
        normalization: dict[str, float],
        history: list[dict[str, float]],
    ) -> Path:
        """Save a PyTorch checkpoint bundle and return its path."""

        checkpoint_path = self.params_dir / f"{run_name}.pth"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "architecture": asdict(architecture),
                "training": asdict(training),
                "normalization": normalization,
                "history": history,
            },
            checkpoint_path,
        )
        return checkpoint_path

    def write_study_summary(self, payload: dict[str, object]) -> None:
        """Write a timestamped study-summary JSON file."""

        with (self.logs_dir / f"study_summary_{self.run_id}.json").open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)


def epoch_learning_rate(config: TrainingConfig, epoch_index: int) -> float:
    """Compute the learning rate for a specific epoch."""

    if config.learning_rate_decay == "cnst":
        return config.learning_rate
    if config.learning_rate_decay == "exp":
        return (config.learning_rate_decay_const / 10.0) ** epoch_index * config.learning_rate
    if config.learning_rate_decay == "sqrt":
        return (config.learning_rate_decay_const / 10.0) / math.sqrt(epoch_index + 1.0) * config.learning_rate
    raise ValueError(f"Unsupported learning rate decay: {config.learning_rate_decay}")


def evaluate_model(
    model: DynamicAutocorrCNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> float:
    """Evaluate the average loss over a dataloader."""

    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device).float()
            labels = labels.to(device).float()
            predictions = model(images)
            loss = criterion(predictions, labels)
            total_loss += float(loss.item()) * len(images)
            total_samples += len(images)
    return total_loss / max(total_samples, 1)


def _build_loaders(bundle: DatasetBundle, training: TrainingConfig) -> tuple[DataLoader, DataLoader]:
    """Create training and development dataloaders from a dataset bundle."""

    train_loader = DataLoader(
        bundle.trainsets[training.trainset],
        batch_size=training.batch_size,
        shuffle=training.shuffle,
        num_workers=training.num_workers,
    )
    dev_loader = DataLoader(
        bundle.dev_normalized,
        batch_size=min(600, len(bundle.dev_normalized)),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, dev_loader


def _run_training_epochs(
    model: DynamicAutocorrCNN,
    bundle: DatasetBundle,
    training: TrainingConfig,
    *,
    epochs: int,
) -> tuple[list[dict[str, float]], float]:
    """Train a model for a specified number of epochs and track dev loss."""

    device = torch.device(training.device)
    model = model.to(device).float()
    train_loader, dev_loader = _build_loaders(bundle, training)

    criterion = nn.MSELoss()
    history: list[dict[str, float]] = []
    best_dev_loss = float("inf")

    for epoch in range(epochs):
        current_lr = epoch_learning_rate(training, epoch)
        optimizer = optim.Adam(
            model.parameters(),
            lr=current_lr,
            betas=(training.beta1, training.beta2),
            eps=1e-8,
            amsgrad=False,
        )

        model.train()
        total_train_loss = 0.0
        total_samples = 0
        for images, labels in train_loader:
            images = images.to(device).float()
            labels = labels.to(device).float()

            predictions = model(images)
            loss = criterion(predictions, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += float(loss.item()) * len(images)
            total_samples += len(images)

        train_loss = total_train_loss / max(total_samples, 1)
        dev_loss = evaluate_model(model, dev_loader, device, criterion)
        best_dev_loss = min(best_dev_loss, dev_loss)
        history.append(
            {
                "epoch": float(epoch + 1),
                "learning_rate": float(current_lr),
                "train_loss": float(train_loss),
                "dev_loss": float(dev_loss),
            }
        )

    return history, best_dev_loss


def _select_best_initialization(
    architecture: ArchitectureConfig,
    bundle: DatasetBundle,
    training: TrainingConfig,
) -> tuple[dict[str, torch.Tensor] | None, int, float | None]:
    """Screen several random initializations and keep the best starting weights."""

    if training.initialization_trials <= 0:
        return None, 0, None

    best_state_dict: dict[str, torch.Tensor] | None = None
    best_trial_index = 0
    best_screen_loss = float("inf")

    for trial_index in range(training.initialization_trials):
        model = DynamicAutocorrCNN(architecture)
        initial_state = copy.deepcopy(model.state_dict())
        history, _ = _run_training_epochs(
            model,
            bundle,
            training,
            epochs=training.initialization_epochs,
        )
        screen_loss = float(history[-1]["dev_loss"])
        if screen_loss < best_screen_loss:
            best_screen_loss = screen_loss
            best_state_dict = initial_state
            best_trial_index = trial_index

    return best_state_dict, best_trial_index, best_screen_loss


def train_model(
    bundle: DatasetBundle,
    architecture: ArchitectureConfig,
    training: TrainingConfig,
    artifacts: ArtifactManager,
    run_name: str,
) -> TrainingResult:
    """Run optional initialization screening, full training, and artifact saving."""

    initial_state, selected_trial, screen_loss = _select_best_initialization(
        architecture,
        bundle,
        training,
    )
    model = DynamicAutocorrCNN(architecture)
    if initial_state is not None:
        model.load_state_dict(initial_state)

    history, best_dev_loss = _run_training_epochs(
        model,
        bundle,
        training,
        epochs=training.epochs,
    )

    artifacts.save_history(run_name, history)
    artifacts.save_plot(run_name, history)
    checkpoint_path = artifacts.save_checkpoint(
        run_name,
        model,
        architecture,
        training,
        normalization=asdict(bundle.normalization),
        history=history,
    )
    return TrainingResult(
        history=history,
        final_dev_loss=float(history[-1]["dev_loss"]),
        best_dev_loss=float(best_dev_loss),
        run_id=artifacts.run_id,
        checkpoint_path=checkpoint_path,
        selected_initialization_trial=int(selected_trial),
        initialization_screen_loss=screen_loss,
    )
