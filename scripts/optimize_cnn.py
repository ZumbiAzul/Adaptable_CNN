from __future__ import annotations

import argparse
from datetime import datetime
import json
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnn.constants import (
    DEFAULT_DATABASE_NAME,
    DEFAULT_DATASET_NAME,
    DEFAULT_INPUT_CHANNELS,
    DEFAULT_INPUT_IMAGE_SIZE,
    DEFAULT_LABEL_SIZE,
    DEFAULT_STUDY_NAME,
)
from cnn.data import SplitConfig, prepare_dataset_bundle


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for Optuna-based CNN optimization."""

    parser = argparse.ArgumentParser(
        description="Optimize the autocorr CNN on current_Et prediction using Optuna.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--database-name", default=DEFAULT_DATABASE_NAME)
    parser.add_argument("--study-name", default=DEFAULT_STUDY_NAME)
    parser.add_argument("--num-trials", type=int, default=100)
    parser.add_argument("--noise-snr", type=int, choices=(1, 2, 5, 10, 20), default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-train", type=int, default=160)
    parser.add_argument("--n-dev", type=int, default=20)
    parser.add_argument("--n-test", type=int, default=20)
    parser.add_argument("--init-trials", type=int, default=0)
    parser.add_argument("--init-epochs", type=int, default=1)
    parser.add_argument("--enable-dropout", action="store_true")
    return parser


def main() -> None:
    """Run the end-to-end optimization workflow for the active dataset."""

    args = build_parser().parse_args()
    optimization_run_id = datetime.now().strftime("%Y%m%d%H%M%S")
    try:
        import optuna
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "optuna is required for scripts/optimize_cnn.py. Install dependencies from requirements.txt."
        ) from exc
    from cnn.optimization import OptimizationContext, OptunaAutocorrObjective

    dataset_root = REPO_ROOT / "data" / args.dataset_name
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_root}. Run scripts/generate_data.py first."
        )

    bundle = prepare_dataset_bundle(
        root=dataset_root,
        split=SplitConfig(n_train=args.n_train, n_dev=args.n_dev, n_test=args.n_test),
        noisy_snr=args.noise_snr,
    )

    if args.enable_dropout:
        dropout2d_layer_numbers = []
        dropout2d_probabilities = []
        linear_dropout_enabled = []
        linear_dropout_probabilities = []
        input_dropout_enabled = [None]
        input_dropout_probability = [None]
    else:
        dropout2d_layer_numbers = [0, 0, 0]
        dropout2d_probabilities = [0.0, 0.0, 0.0]
        linear_dropout_enabled = [0]
        linear_dropout_probabilities = [0.0]
        input_dropout_enabled = [0]
        input_dropout_probability = [0.0]

    fixed_parameters = {
        "conv_layer_out_channels": [None, None, None],
        "conv_layer_kernel_sizes": [None, None, None],
        "conv_layer_strides": [None, None, None],
        "conv_layer_paddings": [None, None, None],
        "max_pool_layer_numbers": [0, 0, 0],
        "max_pool_kernel_sizes": [0, 0, 0],
        "max_pool_strides": [0, 0, 0],
        "max_pool_paddings": [0, 0, 0],
        "batch_norm_2d_layer_numbers": [1, 1, 1],
        "dropout2d_layer_numbers": dropout2d_layer_numbers,
        "dropout2d_probabilities": dropout2d_probabilities,
        "fc_hidden_features": [None],
        "linear_batch_norm": [1],
        "linear_dropout_enabled": linear_dropout_enabled,
        "linear_dropout_probabilities": linear_dropout_probabilities,
        "input_dropout_enabled": input_dropout_enabled,
        "input_dropout_probability": input_dropout_probability,
    }

    parameter_ranges = {
        "conv_layer_out_channels": (16, 192),
        "conv_layer_kernel_sizes": (3, 9),
        "conv_layer_strides": (1, 2),
        "conv_layer_paddings": (1, 4),
        "max_pool_layer_numbers": (0, 1),
        "max_pool_kernel_sizes": (2, 4),
        "max_pool_strides": (1, 2),
        "max_pool_paddings": (0, 2),
        "batch_norm_2d_layer_numbers": (0, 1),
        "dropout2d_layer_numbers": (0, 1),
        "dropout2d_probabilities": (0.0, 1.0),
        "linear_layer_out_features": (128, 768),
        "dropout1d_layer_numbers": (0, 1),
        "dropout1d_probabilities": (0.0, 1.0),
    }

    fixed_hyperparameters = {
        "epochs": 70,
        "batch_size": 81,
        "learning_rate": 0.008149612926988513,
        "learning_rate_decay": "sqrt",
        "learning_rate_decay_const": 2,
        "beta1": 0.883345587855772,
        "beta2": 0.9996328747283718,
        "num_workers": 0,
        "device": args.device,
        "shuffle": True,
        "trainset": "norm",
        "initialization_trials": args.init_trials,
        "initialization_epochs": args.init_epochs,
    }

    hyperparameter_ranges = {
        "batch_size": (10, 400),
        "learning_rate": (2e-3, 1e0),
        "learning_rate_decay": (0, 2),
        "learning_rate_decay_const": (1, 10),
        "beta1": (0.81, 0.99),
        "beta2": (0.9981, 0.9999),
        "epochs": (1, 2000),
    }

    context = OptimizationContext(
        repo_root=REPO_ROOT,
        bundle=bundle,
        database_name=args.database_name,
        study_name=args.study_name,
        input_channels=DEFAULT_INPUT_CHANNELS,
        input_size=DEFAULT_INPUT_IMAGE_SIZE,
        output_size=DEFAULT_LABEL_SIZE,
        num_conv_layers=3,
        num_fc_hidden_layers=1,
        fixed_parameters=fixed_parameters,
        parameter_ranges=parameter_ranges,
        fixed_hyperparameters=fixed_hyperparameters,
        hyperparameter_ranges=hyperparameter_ranges,
        conv_activations=["lrelu"] * 3,
        fc_activations=["lrelu", "none"],
        optimization_run_id=optimization_run_id,
    )

    storage = f"sqlite:///{REPO_ROOT / 'databases' / f'{args.database_name}.db'}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
    )
    objective = OptunaAutocorrObjective(context)
    study.optimize(objective, n_trials=args.num_trials)

    complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.FAIL]
    summary = {
        "study_name": args.study_name,
        "database_name": args.database_name,
        "num_trials": len(study.trials),
        "num_complete_trials": len(complete_trials),
        "num_pruned_trials": len(pruned_trials),
        "num_failed_trials": len(failed_trials),
    }
    if complete_trials:
        summary.update(
            {
                "best_value": study.best_value,
                "best_params": study.best_params,
                "best_user_attrs": study.best_trial.user_attrs,
            }
        )
    else:
        summary.update(
            {
                "best_value": None,
                "best_params": None,
                "best_user_attrs": None,
            }
        )
    summary_path = (
        REPO_ROOT
        / "logs"
        / args.database_name
        / args.study_name
        / f"study_summary_{optimization_run_id}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
