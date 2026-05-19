# Framework Details

This document describes how the active framework works internally and what is
currently configurable.

## Optimization Engine

The optimization workflow is driven by Optuna:

- docs: [https://optuna.readthedocs.io/en/stable/](https://optuna.readthedocs.io/en/stable/)
- paper: Akiba, Sano, Yanase, Ohta, and Koyama. 2019. Optuna:
  A Next-generation Hyperparameter Optimization Framework. In KDD.

The active objective is implemented in
[src/cnn/optimization.py](../src/cnn/optimization.py)
and receives its configuration from
[scripts/optimize_cnn.py](../scripts/optimize_cnn.py).

## How `scripts/optimize_cnn.py` Works

When you run `python3 scripts/optimize_cnn.py`, the following happens:

1. A timestamp `optimization_run_id` is created in `YYYYMMDDHHMMSS` format.
2. The selected dataset is loaded from `data/<dataset-name>/`.
3. The dataset is split into train, development, and test subsets.
4. A normalization mean and standard deviation are computed over the loaded
   images.
5. An `OptimizationContext` object is created. It contains:
   - dataset bundle
   - architecture search-space configuration
   - training hyperparameters
   - naming information for logs and artifacts
6. Optuna creates or reuses a study stored in `databases/<database-name>.db`.
7. For each trial:
   - architecture parameters are sampled or read from fixed values
   - training parameters are sampled or read from fixed values
   - spatial-size pruning is applied before model construction
   - if enabled, several random initializations are screened briefly
   - the chosen model is trained
   - final development loss is returned to Optuna
8. A study summary JSON file is written to `logs/<database>/<study>/`.

## Loss Function

At present, the active loss is:

- `torch.nn.MSELoss()` in
  [src/cnn/training.py](../src/cnn/training.py)

If you change the criterion in:

- `evaluate_model(...)`
- `_run_training_epochs(...)`

then Optuna will optimize toward that replacement loss automatically, because
`OptunaAutocorrObjective.__call__` returns `TrainingResult.final_dev_loss`.

So the current workflow is MSE-based, but the framework is structurally
general with respect to the loss definition.

## Current Active CNN Topology Search

The current default search is intentionally smaller and safer than the legacy
monolith:

- `3` convolutional layers
- `2` total linear layers
  - `1` hidden fully connected layer
  - `1` output layer

The dynamic model is assembled in
[src/cnn/models.py](../src/cnn/models.py).

## Optimizable Architecture Parameters

These are the architecture hyperpara-parameters that Optuna can optimize in the current
default setup:

Examples:

- `conv_out_channels_0..2`
  - range: `16..192`
- `conv_kernel_size_0..2`
  - odd values in `3..9`
- `conv_stride_0..2`
  - range: `1..2`
- `conv_padding_0..2`
  - range: `1..min(4, (kernel_size - 1) // 2)`
- `fc_hidden_feature_0`
  - range: `128..768`

The code can also optimize the following if you unfix them in
`scripts/optimize_cnn.py`:

- `max_pool_enabled_0..2`
- `max_pool_kernel_size_0..2`
- `max_pool_stride_0..2`
- `max_pool_padding_0..2`
- `batch_norm_2d_0..2`
- `dropout2d_enabled_0..2`
- `dropout2d_probability_0..2`
- `linear_dropout_hidden_0`
- `linear_dropout_probability_0`
- `input_dropout_enabled`
- `input_dropout_probability`

## Current Fixed Architecture Parameters

In the default configuration:

- max-pooling is fixed off for all three convolutional layers
- 2D batch normalization is fixed on for all three convolutional layers
- dropout is fixed off unless `--enable-dropout` is passed
- hidden linear batch normalization is fixed on
- output-layer batch normalization is fixed off

These choices are defined in
[scripts/optimize_cnn.py](../scripts/optimize_cnn.py).

## Optimizable Training Hyperparameters

The code supports Optuna optimization of these training hyperparameters:

- `batch_size`
- `learning_rate`
- `learning_rate_decay_mode`
- `learning_rate_decay_const`
- `beta1`
- `beta2`
- `epochs`

In the current shipped script, all of them are fixed except:

- `device`, which is set from the CLI
- `initialization_trials`, which is set from the CLI
- `initialization_epochs`, which is set from the CLI

That means the current default study is mainly an architecture search with fixed
training hyperparameters.

## How To Fix Some Parameters and Optimize Others

The optimization code follows one simple convention:

- if a value in `fixed_parameters` or `fixed_hyperparameters` is `None`, Optuna
  samples it
- if it is a concrete value, that parameter is fixed

Examples:

- `conv_layer_out_channels = [None, None, None]`
  means all three convolutional output-channel counts are optimizable
- `max_pool_layer_numbers = [0, 0, 0]`
  means max-pooling is fixed off for all three layers
- `learning_rate = 0.008149612926988513`
  means the learning rate is fixed, not optimized

This behavior is implemented by helper functions in
[src/cnn/optimization.py](../src/cnn/optimization.py):

- `_fixed_or_suggest_int(...)`
- `_fixed_or_suggest_float(...)`

## Pruning Logic

The framework prunes invalid architectures before full training when:

- a convolution would shrink spatial size below `1`
- a pooling operation would shrink spatial size to `1` or below
- flattened feature size would become zero

This pruning is implemented in
`build_architecture_config(...)` and exists specifically to avoid wasting time
on architectures that cannot be trained.

## Initialization Screening

The framework supports a legacy-style screening stage before full training:

- `--init-trials N`
  means try `N` short random initializations for the same architecture
- `--init-epochs M`
  means each screening run trains for `M` epochs

The best screened initialization is then used as the starting point for the full
training run.

This logic is implemented in
[src/cnn/training.py](../src/cnn/training.py):

- `_select_best_initialization(...)`
- `train_model(...)`

## Data Loading and Target Definition

The current active input uses four channels:

- `S1` (first pulse spectrum)
- `S2` (second pulse spectrum)
- `IC2_12` (interferometric correlation between pulses)
- `IC2_sio2_2` (interferometric correlation between pulses in the case when one pulse passed through a slab of fused silica glass)

The current target is:

- `current_Et` (pulse electric field)

The active extraction is defined in
[src/cnn/data.py](../src/cnn/data.py)
inside `load_autocorr_arrays(...)`.

## Artifact Layout

Each optimization run uses one shared timestamp in `YYYYMMDDHHMMSS` format.

Artifacts are written to:

- `results/<timestamp>/`
- `saved_model_params/<timestamp>/`
- `plots/<database>/<study>/<timestamp>/`
- `logs/<database>/<study>/study_summary_<timestamp>.json`

Each trial file inside the shared run folders starts with the same timestamp,
for example:

- `20260518153000_trial_0000.csv`
- `20260518153000_trial_0000.json`
- `20260518153000_trial_0000.pth`
- `20260518153000_trial_0000_loss.png`

## Non-Interactive Plotting

The active code forces Matplotlib into a non-interactive backend:

- backend: `Agg`
- interactive mode: off

So optimization plots are saved only to disk and should not open GUI windows.
