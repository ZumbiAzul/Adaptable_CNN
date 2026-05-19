"""Shared constants for dataset naming, paths, and default dimensions."""

from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_DATASET_NAME = (
    "interf_autocorr_crosscorr_123_0_phase_at_centr_freq_plus_sio2_2mm_"
    "unconstrained_random_phi_200samples-2"
)

TIME_DOMAIN_FILENAME = "time_domain_data_IAs.txt"
TIME_DOMAIN_SIO2_FILENAME = "time_domain_data_IAs_sio2.txt"
FREQ_DOMAIN_FILENAME = "freq_domain_data_IAs.txt"
OTHER_PARAMS_FILENAME = "other_params_IAs.txt"
NOISY_INPUT_FILENAME = "noisy_input_data.txt"
NOISY_LABEL_FILENAME = "label_for_noisy_data.txt"

DEFAULT_DATA_ROOT = REPO_ROOT / "data"
DEFAULT_DATASET_PATH = DEFAULT_DATA_ROOT / DEFAULT_DATASET_NAME

DEFAULT_INPUT_CHANNELS = 4
DEFAULT_INPUT_IMAGE_SIZE = 40
DEFAULT_LABEL_SIZE = 1000

DEFAULT_DATABASE_NAME = (
    "i-xcorr-0-phi-at-centr-freq-sio2-et-pred-"
    "unrestricted-phase-without-2nd-order-IA-200-2"
)
DEFAULT_STUDY_NAME = "architecture-search-17-random-phase-optimisation-without-dropout-3conv-2fc"
