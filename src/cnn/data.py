"""Dataset loading, splitting, normalization, and PyTorch dataset wrappers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .constants import (
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_IMAGE_SIZE,
    DEFAULT_LABEL_SIZE,
    FREQ_DOMAIN_FILENAME,
    NOISY_INPUT_FILENAME,
    TIME_DOMAIN_FILENAME,
    TIME_DOMAIN_SIO2_FILENAME,
)


NOISY_COLUMN_OFFSET = {
    1: 4,
    2: 8,
    5: 12,
    10: 16,
    20: 20,
}


@dataclass
class NormalizationStats:
    """Mean and standard deviation used to normalize input images."""

    mean: float
    std: float


@dataclass
class SplitConfig:
    """Train, development, and test split sizes."""

    n_train: int = 19_000
    n_dev: int = 600
    n_test: int = 400


@dataclass
class DatasetBundle:
    """Grouped normalized and raw dataset splits used by the training pipeline."""

    trainsets: dict[str, "AutocorrDataset"]
    dev_normalized: "AutocorrDataset"
    test_normalized: "AutocorrDataset"
    all_normalized: "AutocorrDataset"
    normalization: NormalizationStats


class AutocorrDataset(Dataset):
    """PyTorch dataset for interferometric-trace images and regression labels."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        normalization: NormalizationStats | None = None,
    ) -> None:
        self.images = np.asarray(images, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.normalization = normalization

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""

        return len(self.images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image tensor and one label tensor."""

        image = self.images[index].copy()
        if self.normalization is not None:
            image = (image - self.normalization.mean) / self.normalization.std
        label = self.labels[index]
        return torch.from_numpy(image), torch.from_numpy(label)


def _sorted_sample_dirs(root: Path) -> list[Path]:
    """Return dataset sample directories sorted numerically when possible."""

    candidates = [path for path in root.iterdir() if path.is_dir()]
    return sorted(candidates, key=lambda path: int(path.name) if path.name.isdigit() else path.name)


def load_autocorr_arrays(
    root: Path,
    input_channels: int = DEFAULT_INPUT_CHANNELS,
    image_size: int = DEFAULT_INPUT_IMAGE_SIZE,
    label_size: int = DEFAULT_LABEL_SIZE,
    noisy_snr: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load active input channels and `current_Et` targets from a dataset."""

    sample_dirs = _sorted_sample_dirs(root)
    images = np.empty((len(sample_dirs), input_channels, image_size, image_size), dtype=np.float32)
    labels = np.empty((len(sample_dirs), label_size), dtype=np.float32)

    for index, sample_dir in enumerate(sample_dirs):
        if noisy_snr is None:
            freq_data = np.genfromtxt(sample_dir / FREQ_DOMAIN_FILENAME)
            time_data = np.genfromtxt(sample_dir / TIME_DOMAIN_FILENAME)
            time_sio2_data = np.genfromtxt(sample_dir / TIME_DOMAIN_SIO2_FILENAME)

            spectrum_1 = freq_data[1250 - 800 : 1250 + 800, 1].reshape(1, image_size, image_size)
            spectrum_2 = freq_data[1250 - 800 : 1250 + 800, 3].reshape(1, image_size, image_size)
            crosscorr_2 = time_data[:, 11].reshape(1, image_size, image_size)
            crosscorr_2_sio2_2 = time_sio2_data[:, 12].reshape(1, image_size, image_size)
        else:
            noisy_inputs = np.genfromtxt(sample_dir / NOISY_INPUT_FILENAME)
            offset = NOISY_COLUMN_OFFSET[noisy_snr]
            spectrum_1 = noisy_inputs[:, offset + 0].reshape(1, image_size, image_size)
            spectrum_2 = noisy_inputs[:, offset + 1].reshape(1, image_size, image_size)
            crosscorr_2 = noisy_inputs[:, offset + 2].reshape(1, image_size, image_size)
            crosscorr_2_sio2_2 = noisy_inputs[:, offset + 3].reshape(1, image_size, image_size)

            time_data = np.genfromtxt(sample_dir / TIME_DOMAIN_FILENAME)

        current_et = time_data[800 - 500 : 800 + 500, 5]
        images[index] = np.concatenate((spectrum_1, spectrum_2, crosscorr_2, crosscorr_2_sio2_2))
        labels[index] = current_et

    return images, labels


def compute_normalization(images: np.ndarray) -> NormalizationStats:
    """Compute global mean and standard deviation for an image array."""

    mean = float(np.mean(images))
    std = float(np.std(images))
    if std == 0.0:
        std = 1.0
    return NormalizationStats(mean=mean, std=std)


def split_arrays(
    images: np.ndarray,
    labels: np.ndarray,
    split: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train, development, and test partitions."""

    train_images = images[: split.n_train]
    dev_images = images[split.n_train : split.n_train + split.n_dev]
    test_images = images[split.n_train + split.n_dev : split.n_train + split.n_dev + split.n_test]

    train_labels = labels[: split.n_train]
    dev_labels = labels[split.n_train : split.n_train + split.n_dev]
    test_labels = labels[split.n_train + split.n_dev : split.n_train + split.n_dev + split.n_test]

    return train_images, dev_images, test_images, train_labels, dev_labels, test_labels


def prepare_dataset_bundle(
    root: Path,
    split: SplitConfig,
    noisy_snr: int | None = None,
) -> DatasetBundle:
    """Load a dataset, normalize it, split it, and package it for training."""

    images, labels = load_autocorr_arrays(root=root, noisy_snr=noisy_snr)
    normalization = compute_normalization(images)
    (
        train_images,
        dev_images,
        test_images,
        train_labels,
        dev_labels,
        test_labels,
    ) = split_arrays(images, labels, split)

    train_raw = AutocorrDataset(train_images, train_labels)
    train_normalized = AutocorrDataset(train_images, train_labels, normalization)
    dev_normalized = AutocorrDataset(dev_images, dev_labels, normalization)
    test_normalized = AutocorrDataset(test_images, test_labels, normalization)
    all_normalized = AutocorrDataset(images, labels, normalization)

    return DatasetBundle(
        trainsets={"not_norm": train_raw, "norm": train_normalized},
        dev_normalized=dev_normalized,
        test_normalized=test_normalized,
        all_normalized=all_normalized,
        normalization=normalization,
    )
