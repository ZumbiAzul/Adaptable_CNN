from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from cnn.constants import DEFAULT_DATASET_NAME
from cnn.generation import AutocorrDataGenerator, GenerationConfig


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for dataset generation."""

    parser = argparse.ArgumentParser(
        description="Generate 1D interferometric autocorrelation/cross-correlation training data.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--phase-mode-1", choices=("random", "poly"), default="random")
    parser.add_argument("--phase-mode-2", choices=("random", "poly"), default="random")
    parser.add_argument("--write-noisy-variants", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    """Parse arguments, generate a dataset, and print the output path."""

    args = build_parser().parse_args()
    config = GenerationConfig(
        dataset_name=args.dataset_name,
        output_root=REPO_ROOT / "data",
        num_samples=args.num_samples,
        phase_mode_1=args.phase_mode_1,
        phase_mode_2=args.phase_mode_2,
        write_noisy_variants=args.write_noisy_variants,
        seed=args.seed,
    )
    generator = AutocorrDataGenerator(config=config)
    dataset_path = generator.generate()
    print(f"[generate] wrote dataset to {dataset_path}")


if __name__ == "__main__":
    main()
