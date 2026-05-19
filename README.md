# Adaptable Convolutional Neural Network

This repository contains a modular framework for generating data, and training and optimizing a convolutional neural network. 

A specific example is given of generating synthetic
one-dimensional interferometric autocorrelation and cross-correlation traces,
then training and optimizing convolutional neural networks to predict the pulse
electric field in the time domain.

The example is connected to the work:

Pavel V. Kolesnichenko and Donatas Zigmantas, "Neural-network-powered pulse
reconstruction from one-dimensional interferometric correlation traces,"
Opt. Express 31, 11806-11819 (2023)  
[https://doi.org/10.1364/OE.479638](https://doi.org/10.1364/OE.479638)

## What This Framework Does

The active codebase implements the following workflow:

1. Generate synthetic datasets of interferometric traces into `data/`
2. Load those datasets into a image-like CNN-ready representation
3. Build CNN architectures dynamically from configuration and train them
4. Assess those architectures using a development loss function
5. Let Optuna search the architecture and hyperparameter space to minimize that loss
6. Save study summaries, plots, checkpoints, and per-trial histories

The framework is adaptable in two important ways:

- The CNN architecture is assembled dynamically from parameter lists rather than
  being hardcoded as one fixed network.
- The optimization target is determined by the loss function (e.g., mean-squared error) used in the training
  pipeline. The optimization layer is built on Optuna’s define-by-run framework.


- Optuna documentation and reference:
  [https://optuna.readthedocs.io/en/stable/](https://optuna.readthedocs.io/en/stable/)
  Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, and Masanori
  Koyama. 2019. Optuna: A Next-generation Hyperparameter Optimization Framework.
  In KDD.

## Repository Layout

`src/cnn/`

- reusable generation, dataset, model, training, and optimization code

`scripts/`

- user-facing entrypoints for dataset generation and optimization

`data/`

- generated or imported training datasets

`databases/`

- Optuna SQLite study files

`docs/`

- detailed framework and API documentation

`legacy/`

- archived code kept only for reference

## Documentation Guide

Start here:

- [docs/FRAMEWORK.md](docs/FRAMEWORK.md)
  explains the optimization workflow, search space, loss handling, artifact
  layout, and what is currently fixed versus optimizable.
- [docs/API_REFERENCE.md](docs/API_REFERENCE.md)
  documents the active functions, classes, inputs, and outputs in the codebase.

## Setup

Install dependencies first:

```bash
python3 -m pip install -r requirements.txt
```

## Main Commands

Generate the default dataset:

```bash
python3 scripts/generate_data.py
```

Run optimization:

```bash
python3 scripts/optimize_cnn.py
```

## Useful Options

For `scripts/generate_data.py`:

- `--dataset-name NAME`
  Select the dataset folder name under `data/`.
- `--num-samples 200`
  Set the number of generated samples.
- `--phase-mode-1 random|poly`
  Choose the phase model for pulse 1.
- `--phase-mode-2 random|poly`
  Choose the phase model for pulse 2.
- `--write-noisy-variants`
  Also write noisy input variants and corresponding labels.
- `--seed 42`
  Make dataset generation reproducible.

Examples:

```bash
python3 scripts/generate_data.py --num-samples 200 --seed 42
python3 scripts/generate_data.py --dataset-name trial_dataset
python3 scripts/generate_data.py --write-noisy-variants
python3 scripts/generate_data.py --phase-mode-1 random --phase-mode-2 poly
```

For `scripts/optimize_cnn.py`:

- `--dataset-name NAME`
  Train against a specific dataset under `data/`.
- `--database-name NAME`
  Store Optuna trials in `databases/NAME.db`.
- `--study-name NAME`
  Create or resume a named Optuna study.
- `--num-trials 100`
  Set the number of Optuna architecture trials.
- `--noise-snr 1|2|5|10|20`
  Use a noisy input variant instead of the clean default channels.
- `--device cpu`
  Force CPU execution.
- `--n-train 160`
  Set the training split size.
- `--n-dev 20`
  Set the development split size.
- `--n-test 20`
  Set the test split size.
- `--init-trials N`
  Screen several random weight initializations per architecture before full
  training.
- `--init-epochs N`
  Set how many short epochs each initialization screen receives.
- `--enable-dropout`
  Re-enable dropout sampling. Dropout is off by default in the current workflow.

Examples:

```bash
python3 scripts/optimize_cnn.py --num-trials 100
python3 scripts/optimize_cnn.py --device cpu --num-trials 20
python3 scripts/optimize_cnn.py --noise-snr 10 --num-trials 50
python3 scripts/optimize_cnn.py --init-trials 3 --init-epochs 1
python3 scripts/optimize_cnn.py --study-name exploratory_run --database-name exploratory_db
```

## Structure of Exemplary Data

Each generated sample lives under `data/<dataset-name>/<index>/` and contains:

- `time_domain_data_IAs.txt`
- `time_domain_data_IAs_sio2.txt`
- `freq_domain_data_IAs.txt`
- `other_params_IAs.txt`
- optionally `noisy_input_data.txt`
- optionally `label_for_noisy_data.txt`

The active optimizer uses four input channels:

- `S1` (spectrum of the first pulse)
- `S2` (spectrum of the second pulse)
- `IC2_12` (interferometric cross-corelation between the pulses)
- `IC2_sio2_2` (interferometric cross-correlation between the pulses with one pulse having propagated through an extra layer of fused silica glass)

The prediction target is `current_Et`(pulse electric field), read from `time_domain_data_IAs.txt`.

## What Happens During Optimization

At a high level, `scripts/optimize_cnn.py` does the following:

1. Parses command-line options
2. Loads the selected dataset and prepares normalized train/dev/test splits
3. Builds an `OptimizationContext` describing the architecture and training
   search space
4. Creates or reopens an Optuna study in SQLite storage
5. For each trial, samples a CNN architecture and training configuration
6. Prunes invalid architectures before training if they would collapse spatial
   dimensions
7. Optionally screens multiple random initializations
8. Fully trains the selected architecture
9. Returns the final development loss to Optuna
10. Saves artifacts to `results/`, `saved_model_params/`, `plots/`, and `logs/`

Detailed behavior is documented in
[docs/FRAMEWORK.md](docs/FRAMEWORK.md).

## Outputs

Optimization and training artifacts are written to:

- `results/` (csv and json formats)
- `saved_model_params/` (pth format)
- `plots/` (png format)
- `logs/`

Each optimization run uses a shared timestamp in `YYYYMMDDHHMMSS` format. That
timestamp is reused across folders and filenames created during that run.

## Notes

- `scripts/` is the supported interface.
- `legacy/` is preserved for provenance and comparison only.
- Generated datasets and most runtime artifacts are intentionally ignored by git.
