# API Reference

This document describes the active functions and classes in `src/cnn/` and the
two supported entry scripts in `scripts/`.

## `src/cnn/constants.py`

### `REPO_ROOT`

- Absolute path to the repository root.

### `DEFAULT_DATASET_NAME`

- Default dataset folder name used by generation and optimization scripts.

### `DEFAULT_DATA_ROOT`

- Default root directory for generated datasets.

### `DEFAULT_DATASET_PATH`

- Convenience path combining `DEFAULT_DATA_ROOT` and `DEFAULT_DATASET_NAME`.

### `DEFAULT_INPUT_CHANNELS`

- Number of input channels expected by the current model and loader.

### `DEFAULT_INPUT_IMAGE_SIZE`

- Lateral size of each input channel image.

### `DEFAULT_LABEL_SIZE`

- Length of the output regression vector.

### `DEFAULT_DATABASE_NAME`

- Default SQLite database name used for Optuna studies.

### `DEFAULT_STUDY_NAME`

- Default Optuna study name.

## `src/cnn/generation.py`

### `time_to_frequency(time_axis, num_points) -> np.ndarray`

- Purpose: construct a frequency axis corresponding to a time axis.
- Inputs:
  - `time_axis`: evenly spaced time samples
  - `num_points`: number of output points
- Output:
  - frequency-axis array with `num_points` elements

### `fourier_transform(data, num_points) -> np.ndarray`

- Purpose: centered FFT helper used by the pulse-generation pipeline.
- Inputs:
  - `data`: array to transform
  - `num_points`: FFT length
- Output:
  - complex transformed array

### `inverse_fourier_transform(data, num_points) -> np.ndarray`

- Purpose: centered inverse FFT helper.
- Inputs:
  - `data`: array to transform back
  - `num_points`: inverse FFT length
- Output:
  - complex inverse-transformed array

### `gaussian_spectrum(freq_axis, center_freq, fwhm_freq) -> np.ndarray`

- Purpose: evaluate a Gaussian spectral envelope.
- Inputs:
  - `freq_axis`: frequency samples
  - `center_freq`: Gaussian center
  - `fwhm_freq`: full width at half maximum in frequency
- Output:
  - Gaussian envelope sampled on `freq_axis`

### `remove_linear_phase(phase_, time_axis, freq_axis, carrier_freq) -> np.ndarray`

- Purpose: subtract the dominant linear phase term from a phase trace.
- Inputs:
  - `phase_`: unwrapped phase array
  - `time_axis`: time samples
  - `freq_axis`: frequency samples
  - `carrier_freq`: carrier angular frequency
- Output:
  - phase array centered so that the midpoint phase is zero

### `calc_fwhm(x_axis, y_axis) -> float`

- Purpose: estimate full width at half maximum.
- Inputs:
  - `x_axis`: sample positions
  - `y_axis`: signal values
- Output:
  - estimated FWHM in x-axis units

### `interferometric_autocorrelation(pulse_1, pulse_2, time_axis, order) -> np.ndarray`

- Purpose: compute a normalized interferometric correlation trace.
- Inputs:
  - `pulse_1`: first complex field
  - `pulse_2`: second complex field
  - `time_axis`: time samples
  - `order`: nonlinear order, usually `1`, `2`, or `3`
- Output:
  - normalized correlation trace

### `random_phase_2(freq_axis, sigma_smooth) -> np.ndarray`

- Purpose: generate a smoothed random spectral phase.
- Inputs:
  - `freq_axis`: frequency samples
  - `sigma_smooth`: Gaussian smoothing width
- Output:
  - random phase array in radians

### `refractive_index(wavelength_microns, medium="air") -> np.ndarray`

- Purpose: compute refractive index from a Sellmeier-like model.
- Inputs:
  - `wavelength_microns`: wavelengths in microns
  - `medium`: one of `air`, `SiO2`, `bk7`, or `sf10`
- Output:
  - refractive-index array

### `propagation(time_axis, freq_axis, carrier_freq, spectral_field, length_m, num_z_points, medium="air") -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]`

- Purpose: propagate a spectral field through a medium.
- Inputs:
  - `time_axis`: time samples
  - `freq_axis`: frequency samples
  - `carrier_freq`: carrier angular frequency
  - `spectral_field`: complex spectral field
  - `length_m`: propagation length in meters
  - `num_z_points`: number of sampled propagation positions
  - `medium`: material name
- Outputs:
  - `fwhm`: FWHM at each propagation position
  - `z_axis`: sampled propagation positions
  - `temporal_evolution`: propagated time-domain fields
  - `phase_evolution`: propagated phases

### `PulseData`

- Purpose: container for one synthetic pulse and its derived metadata.
- Main fields:
  - time-domain field
  - frequency-domain field
  - time and frequency axes
  - temporal and spectral phases
  - central wavelength/frequency information
  - Gaussian-peak decomposition parameters
  - polynomial phase coefficients

### `generate_pulse(...) -> PulseData`

- Purpose: synthesize one pulse with either random or polynomial spectral phase.
- Inputs:
  - pulse-shape and width parameters
  - optional polynomial phase coefficients
  - `sigma_smooth`
  - `phase_mode`
- Output:
  - populated `PulseData` object

### `_find_nearest(array, value) -> int`

- Purpose: return the index of the closest value in an array.
- Inputs:
  - `array`: numeric array
  - `value`: target value
- Output:
  - nearest index

### `_window_background(signal, center_half_width=50, smooth_width=10) -> np.ndarray`

- Purpose: estimate a smoothed background envelope for noisy-correlation scaling.
- Inputs:
  - `signal`: 1D trace
  - `center_half_width`: central window half-width
  - `smooth_width`: edge smoothness parameter
- Output:
  - filtered background-like signal

### `_build_noisy_inputs(spectrum_1, spectrum_2, ixc1, ixc2, label, snr_levels) -> tuple[np.ndarray, np.ndarray]`

- Purpose: generate noisy input variants at several SNR levels.
- Inputs:
  - clean spectra and interferometric traces
  - `label`: target vector
  - `snr_levels`: iterable of SNR values
- Outputs:
  - matrix of clean and noisy input channels
  - label column vector

### `GenerationConfig`

- Purpose: configuration for dataset generation.
- Key fields:
  - dataset name and output root
  - number of samples
  - raw signal dimensions
  - phase modes
  - propagation-medium parameters
  - noisy-variant settings
  - random seed

### `GenerationConfig.dataset_path -> Path`

- Purpose: resolve the final dataset directory from the configuration.
- Output:
  - full dataset path

### `AutocorrDataGenerator`

- Purpose: class wrapper around dataset generation.

### `AutocorrDataGenerator.generate() -> Path`

- Purpose: generate a full dataset on disk.
- Inputs:
  - none directly; reads from `self.config`
- Output:
  - dataset root path that was written

## `src/cnn/data.py`

### `NOISY_COLUMN_OFFSET`

- Purpose: map SNR value to the starting column of that noisy variant in
  `noisy_input_data.txt`.

### `NormalizationStats`

- Purpose: store dataset normalization mean and standard deviation.

### `SplitConfig`

- Purpose: define train/dev/test split sizes.

### `DatasetBundle`

- Purpose: package the normalized and non-normalized dataset splits together.

### `AutocorrDataset`

- Purpose: PyTorch dataset wrapper around image and label arrays.

### `AutocorrDataset.__len__() -> int`

- Output:
  - number of samples

### `AutocorrDataset.__getitem__(index) -> tuple[torch.Tensor, torch.Tensor]`

- Inputs:
  - `index`: sample index
- Outputs:
  - image tensor
  - label tensor

### `_sorted_sample_dirs(root) -> list[Path]`

- Purpose: enumerate sample directories in numeric order when possible.
- Input:
  - dataset root directory
- Output:
  - sorted list of sample subdirectories

### `load_autocorr_arrays(root, input_channels=4, image_size=40, label_size=1000, noisy_snr=None) -> tuple[np.ndarray, np.ndarray]`

- Purpose: load the active four-channel input representation and `current_Et`
  targets from a generated dataset.
- Inputs:
  - dataset root
  - expected image and label dimensions
  - optional noisy SNR level
- Outputs:
  - image array shaped `(n_samples, channels, image_size, image_size)`
  - label array shaped `(n_samples, label_size)`

### `compute_normalization(images) -> NormalizationStats`

- Purpose: compute dataset-wide mean and standard deviation.
- Input:
  - image array
- Output:
  - `NormalizationStats`

### `split_arrays(images, labels, split) -> tuple[...]`

- Purpose: split arrays into train/dev/test partitions.
- Inputs:
  - image array
  - label array
  - split configuration
- Outputs:
  - train images
  - dev images
  - test images
  - train labels
  - dev labels
  - test labels

### `prepare_dataset_bundle(root, split, noisy_snr=None) -> DatasetBundle`

- Purpose: load arrays, normalize them, split them, and package them as
  datasets.
- Inputs:
  - dataset root
  - split configuration
  - optional noisy SNR
- Output:
  - `DatasetBundle`

## `src/cnn/models.py`

### `ACTIVATIONS`

- Purpose: map activation names to instantiated PyTorch modules.

### `_activation(name) -> nn.Module`

- Purpose: resolve a string activation name to a module.
- Input:
  - activation name
- Output:
  - `nn.Module`

### `ArchitectureConfig`

- Purpose: hold the full model architecture specification needed to build a
  dynamic CNN.

### `DynamicAutocorrCNN`

- Purpose: dynamically constructed CNN for regression on interferometric traces.

### `DynamicAutocorrCNN._build_feature_extractor() -> nn.Sequential`

- Purpose: build the convolutional front end.
- Output:
  - sequential module of conv, optional pool, normalization, activation, and
    optional dropout layers

### `DynamicAutocorrCNN._infer_flattened_features() -> int`

- Purpose: infer the flattened feature size by forwarding a dummy tensor.
- Output:
  - flattened feature count entering the classifier

### `DynamicAutocorrCNN._build_classifier() -> nn.Sequential`

- Purpose: build the linear back end.
- Output:
  - sequential module of linear, optional normalization, activation, and
    optional dropout layers

### `DynamicAutocorrCNN.forward(x) -> torch.Tensor`

- Input:
  - input tensor shaped `(batch, channels, height, width)`
- Output:
  - predicted regression vector

## `src/cnn/optimization.py`

### `OptimizationContext`

- Purpose: package everything the Optuna objective needs:
  search space, fixed parameters, dataset, naming, and model dimensions.

### `_fixed_or_suggest_int(...) -> int`

- Purpose: return a fixed integer value when provided, otherwise ask Optuna to
  sample one.

### `_fixed_or_suggest_float(...) -> float`

- Purpose: return a fixed float value when provided, otherwise ask Optuna to
  sample one.

### `_conv_output_size(input_size, kernel_size, stride, padding) -> int`

- Purpose: compute the output size of a convolution layer.

### `_pool_output_size(input_size, kernel_size, stride, padding) -> int`

- Purpose: compute the output size of a pooling layer.

### `build_architecture_config(trial, context) -> ArchitectureConfig`

- Purpose: construct the CNN architecture for one Optuna trial.
- Inputs:
  - `trial`: current Optuna trial
  - `context`: optimization context
- Output:
  - `ArchitectureConfig`
- Side behavior:
  - prunes architectures that collapse spatial size

### `build_training_config(trial, context) -> TrainingConfig`

- Purpose: construct the training configuration for one Optuna trial.
- Inputs:
  - `trial`: current Optuna trial
  - `context`: optimization context
- Output:
  - `TrainingConfig`

### `OptunaAutocorrObjective`

- Purpose: callable Optuna objective that samples, trains, saves artifacts, and
  returns final development loss.

### `OptunaAutocorrObjective.__call__(trial) -> float`

- Inputs:
  - `trial`: current Optuna trial
- Output:
  - final development loss for Optuna to minimize

## `src/cnn/training.py`

### `TrainingConfig`

- Purpose: hold the concrete training setup for one run.
- Main fields:
  - batch size
  - learning rate and decay settings
  - Adam betas
  - epochs
  - worker count
  - device
  - trainset choice
  - initialization-screening settings

### `TrainingResult`

- Purpose: store the outputs of one completed training run.
- Main fields:
  - training history
  - final and best development losses
  - run id
  - checkpoint path
  - selected initialization index
  - initialization-screen loss

### `ArtifactManager`

- Purpose: manage all file outputs of a training or optimization run.

### `ArtifactManager.save_history(run_name, history) -> None`

- Purpose: save per-epoch history as CSV and JSON.

### `ArtifactManager.save_plot(run_name, history) -> None`

- Purpose: save a train-vs-dev loss plot.

### `ArtifactManager.save_checkpoint(run_name, model, architecture, training, normalization, history) -> Path`

- Purpose: save a PyTorch checkpoint bundle.
- Output:
  - checkpoint path

### `ArtifactManager.write_study_summary(payload) -> None`

- Purpose: write a study-summary JSON file.

### `epoch_learning_rate(config, epoch_index) -> float`

- Purpose: compute epoch-specific learning rate for `cnst`, `exp`, or `sqrt`
  decay modes.

### `evaluate_model(model, loader, device, criterion) -> float`

- Purpose: evaluate average loss on a dataset loader.
- Output:
  - scalar average loss

### `_build_loaders(bundle, training) -> tuple[DataLoader, DataLoader]`

- Purpose: create train and development loaders from a dataset bundle.

### `_run_training_epochs(model, bundle, training, *, epochs) -> tuple[list[dict[str, float]], float]`

- Purpose: run the main epoch loop for a specified number of epochs.
- Outputs:
  - per-epoch history
  - best development loss seen during those epochs

### `_select_best_initialization(architecture, bundle, training) -> tuple[dict[str, torch.Tensor] | None, int, float | None]`

- Purpose: run short initialization-screening trials and keep the best starting
  weights.
- Outputs:
  - best initial state dict, or `None`
  - selected initialization index
  - best screening loss

### `train_model(bundle, architecture, training, artifacts, run_name) -> TrainingResult`

- Purpose: perform optional initialization screening, full training, and artifact
  saving.
- Output:
  - `TrainingResult`

## `scripts/generate_data.py`

### `build_parser() -> argparse.ArgumentParser`

- Purpose: define the command-line interface for dataset generation.

### `main() -> None`

- Purpose: parse arguments, build `GenerationConfig`, run the generator, and
  print the output path.

## `scripts/optimize_cnn.py`

### `build_parser() -> argparse.ArgumentParser`

- Purpose: define the command-line interface for optimization.

### `main() -> None`

- Purpose: parse arguments, prepare data, configure the search space, run
  Optuna, and write the study summary.
