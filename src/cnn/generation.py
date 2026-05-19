"""Synthetic pulse generation and interferometric-trace dataset creation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import scipy.interpolate as interp
import scipy.ndimage as ndi
from numpy.fft import fft, fftshift, ifft

from .constants import (
    DEFAULT_DATASET_NAME,
    DEFAULT_DATASET_PATH,
    FREQ_DOMAIN_FILENAME,
    NOISY_INPUT_FILENAME,
    NOISY_LABEL_FILENAME,
    OTHER_PARAMS_FILENAME,
    TIME_DOMAIN_FILENAME,
    TIME_DOMAIN_SIO2_FILENAME,
)


def time_to_frequency(time_axis: np.ndarray, num_points: int) -> np.ndarray:
    """Construct a frequency axis matching an evenly spaced time axis."""

    delta_t = time_axis[1] - time_axis[0]
    total_t = num_points * delta_t
    delta_w = 2.0 * np.pi / total_t
    total_w = delta_w * num_points
    return np.linspace(-total_w / 2.0, total_w / 2.0, num_points)


def fourier_transform(data: np.ndarray, num_points: int) -> np.ndarray:
    """Apply the centered Fourier transform used by the legacy workflow."""

    return fftshift(fft(fftshift(data), num_points))


def inverse_fourier_transform(data: np.ndarray, num_points: int) -> np.ndarray:
    """Apply the centered inverse Fourier transform used by propagation."""

    return fftshift(ifft(fftshift(data), num_points))


def gaussian_spectrum(freq_axis: np.ndarray, center_freq: float, fwhm_freq: float) -> np.ndarray:
    """Evaluate a Gaussian spectral envelope on a frequency axis."""

    return np.exp(-4.0 * np.log(2.0) * (freq_axis - center_freq) ** 2 / fwhm_freq**2)


def remove_linear_phase(
    phase_: np.ndarray,
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    carrier_freq: float,
) -> np.ndarray:
    """Remove the dominant linear phase term and center the phase trace."""

    index_min = int(len(time_axis) / 2 - len(time_axis) / 10)
    index_max = int(len(time_axis) / 2 + len(time_axis) / 10)
    linear_phase = -carrier_freq * (
        time_axis * len(freq_axis) / len(time_axis)
        + (float(np.max(time_axis)) - float(np.min(time_axis))) / 2.0 * len(time_axis) / len(freq_axis)
    )
    correction = np.polyval(
        np.polyfit(time_axis[index_min:index_max], linear_phase[index_min:index_max], 1),
        time_axis,
    )
    phase = phase_ - correction
    return phase - phase[len(time_axis) // 2]


def calc_fwhm(x_axis: np.ndarray, y_axis: np.ndarray) -> float:
    """Estimate full width at half maximum from sampled data."""

    index_max = int(np.argmax(y_axis))
    index_low = int(np.argmin(np.abs(y_axis[:index_max] - y_axis[index_max] / 2.0)))
    index_high = int(
        np.argmin(np.abs(y_axis[index_max:] - y_axis[index_max] / 2.0)) + index_max
    )
    return float(x_axis[index_high] - x_axis[index_low])


def interferometric_autocorrelation(
    pulse_1: np.ndarray,
    pulse_2: np.ndarray,
    time_axis: np.ndarray,
    order: int,
) -> np.ndarray:
    """Compute a normalized interferometric correlation trace of a given order."""

    del time_axis  # the legacy implementation does not use the explicit step size
    num_points = len(pulse_1)
    trace = np.zeros(num_points)
    for index in range(num_points):
        shifted = np.roll(pulse_2, index + 1)
        trace[index] = np.sum(np.abs((pulse_1 + shifted) ** order) ** 2)
    return fftshift(trace / trace[num_points // 2])


def random_phase_2(freq_axis: np.ndarray, sigma_smooth: float) -> np.ndarray:
    """Generate a smoothed random spectral phase profile."""

    noise = ndi.gaussian_filter1d(2.0 * np.pi * np.random.rand(len(freq_axis)) - np.pi, sigma_smooth)
    noise /= np.max(np.abs(noise))
    return noise * np.pi


def refractive_index(wavelength_microns: np.ndarray, medium: str = "air") -> np.ndarray:
    """Return refractive index samples for one supported material."""

    def sellmeier(
        wavelength: np.ndarray,
        b1: float,
        c1: float,
        b2: float,
        c2: float,
        b3: float,
        c3: float,
    ) -> np.ndarray:
        return np.sqrt(
            1.0
            + b1 * wavelength**2.0 / (wavelength**2 - c1)
            + b2 * wavelength**2.0 / (wavelength**2 - c2)
            + b3 * wavelength**2.0 / (wavelength**2 - c3)
        )

    if medium == "air":
        b1, c1 = 0.05792105, 238.0185
        b2, c2 = 0.00167917, 57.362
        return 1.0 + b1 / (c1 - wavelength_microns**-2) + b2 / (c2 - wavelength_microns**-2)
    if medium == "SiO2":
        return sellmeier(
            wavelength_microns,
            0.6961663,
            0.0684043**2,
            0.4079426,
            0.1162414**2,
            0.8974794,
            9.896161**2,
        )
    if medium == "bk7":
        return sellmeier(
            wavelength_microns,
            1.03961212,
            0.00600069867,
            0.231792344,
            0.0200179144,
            1.01046945,
            103.560653,
        )
    if medium == "sf10":
        return sellmeier(
            wavelength_microns,
            1.62153902,
            0.0122241457,
            0.256287842,
            0.0595736775,
            1.64447552,
            147.468793,
        )
    raise ValueError(f"Unsupported medium: {medium}")


def propagation(
    time_axis: np.ndarray,
    freq_axis: np.ndarray,
    carrier_freq: float,
    spectral_field: np.ndarray,
    length_m: float,
    num_z_points: int,
    medium: str = "air",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Propagate a spectral field through a medium and sample the evolution."""

    speed_of_light = 3e8
    num_time_points = len(time_axis)

    wavelength_microns = 2.0 * np.pi * speed_of_light / freq_axis * 1e6
    if medium != "air":
        wavelength_microns = wavelength_microns.copy()
        wavelength_microns[wavelength_microns > 6] = 0
        wavelength_microns[wavelength_microns < 0.25] = 0

    refr_index = refractive_index(wavelength_microns, medium)
    wavevector = freq_axis * refr_index / speed_of_light
    z_axis = np.linspace(0.0, length_m, num_z_points)

    spectral_evolution = np.zeros((len(z_axis), len(freq_axis)), dtype=complex)
    temporal_evolution = np.zeros((len(z_axis), len(time_axis)), dtype=complex)
    phase_evolution = np.zeros((len(z_axis), len(time_axis)))
    fwhm = np.zeros(len(z_axis))

    for index, position in enumerate(z_axis):
        phase = wavevector * position
        spectral_evolution[index, :] = spectral_field * np.exp(1j * phase)
        temporal_evolution[index, :] = inverse_fourier_transform(
            spectral_evolution[index, :],
            num_time_points,
        )
        temporal_evolution[index, :] = np.roll(
            temporal_evolution[index, :],
            round(num_time_points / 2) - int(np.argmax(np.abs(temporal_evolution[index, :]))),
        )
        temporal_evolution[index, :] = temporal_evolution[index, :] / np.max(
            np.abs(temporal_evolution[index, :])
        )
        phase_ = np.unwrap(np.angle(temporal_evolution[index, :]))
        phase_evolution[index, :] = remove_linear_phase(
            phase_,
            time_axis,
            freq_axis,
            carrier_freq,
        )
        fwhm[index] = calc_fwhm(time_axis, np.abs(temporal_evolution[index, :]) ** 2)

    return fwhm, z_axis, np.transpose(temporal_evolution), phase_evolution


@dataclass
class PulseData:
    """Container for one generated pulse and its derived metadata."""

    time_domain_field: np.ndarray
    frequency_domain_field: np.ndarray
    time_axis: np.ndarray
    frequency_axis: np.ndarray
    phase: np.ndarray
    spectral_phase: np.ndarray
    lambda0: float
    fwhm_lambda: float
    w0: float
    fwhm_w: float
    n_peaks: int
    central_lambdas: np.ndarray
    central_w0s: np.ndarray
    fwhm_lambdas: np.ndarray
    fwhm_ws: np.ndarray
    amplitudes: np.ndarray
    phi2: float
    phi3: float
    phi4: float


def generate_pulse(
    num_points: int = 4096,
    lambda0: float | None = None,
    fwhm_lambda: float | None = None,
    n_peaks: int | None = None,
    c1: float | None = None,
    c2: float | None = None,
    c3: float | None = None,
    sigma_smooth: float = 5.0,
    phase_mode: str = "random",
) -> PulseData:
    """Synthesize one pulse with random or polynomial spectral phase."""

    num_freq_points = num_points
    time_axis = np.linspace(-500e-15, 500e-15, num_points)
    frequency_axis = time_to_frequency(time_axis, num_freq_points)

    speed_of_light = 3e8
    if lambda0 is None:
        lambda0 = (550.0 + np.random.rand() * 300.0) * 1e-9
    if fwhm_lambda is None:
        fwhm_lambda = (20.0 + np.random.rand() * 200.0) * 1e-9

    w0 = 2.0 * np.pi * speed_of_light / lambda0
    fwhm_w = 2.0 * np.pi * speed_of_light * fwhm_lambda / lambda0**2
    intensity_freq = np.zeros_like(frequency_axis)

    if n_peaks is None:
        n_peaks = int(11 * np.random.rand()) + 1

    central_w0s = np.empty(n_peaks)
    fwhm_ws = np.empty(n_peaks)
    amplitudes = np.empty(n_peaks)
    central_lambdas = np.empty(n_peaks)
    fwhm_lambdas = np.empty(n_peaks)

    for index in range(n_peaks):
        delta_lambda = -fwhm_lambda / 4.0 + np.random.rand() * fwhm_lambda / 2.0
        central_lambdas[index] = lambda0 + delta_lambda
        central_w0s[index] = 2.0 * np.pi * speed_of_light / central_lambdas[index]
        fwhm_lambdas[index] = (20.0 + np.random.rand() * 50.0) * 1e-9
        fwhm_ws[index] = (
            2.0
            * np.pi
            * speed_of_light
            * fwhm_lambdas[index]
            / (lambda0 + delta_lambda) ** 2
        )
        amplitudes[index] = np.random.rand()
        intensity_freq += amplitudes[index] * gaussian_spectrum(
            frequency_axis,
            central_w0s[index],
            fwhm_ws[index],
        )

    intensity_freq /= np.max(intensity_freq)

    sign = np.random.choice([-1.0, 1.0])
    if c1 is None:
        c1 = sign * (15.0 + np.random.rand() * 10.0)
    if c2 is None:
        c2 = np.random.rand() * 400.0
    if c3 is None:
        c3 = -3000.0 + np.random.rand() * 6000.0

    phi2 = c1 / 2.0 * (1e-15) ** 2.0
    phi3 = c2 / 6.0 * (1e-15) ** 3.0
    phi4 = c3 / 24.0 * (1e-15) ** 4.0
    poly_phase = (
        phi2 * (frequency_axis - w0) ** 2
        + phi3 * (frequency_axis - w0) ** 3
        + phi4 * (frequency_axis - w0) ** 4
    )

    if phase_mode == "poly":
        spectral_phase_0 = poly_phase
    elif phase_mode == "random":
        spectral_phase_0 = random_phase_2(frequency_axis, sigma_smooth)
    else:
        raise ValueError(f"Unsupported phase mode: {phase_mode}")

    frequency_domain_field = np.sqrt(intensity_freq) * np.exp(1j * spectral_phase_0)
    spectral_phase = np.unwrap(np.angle(frequency_domain_field))

    time_domain_field = fourier_transform(frequency_domain_field, num_points)
    time_domain_field = time_domain_field / np.max(np.abs(time_domain_field))
    phase = np.unwrap(np.angle(time_domain_field))
    phase = remove_linear_phase(phase, time_axis, frequency_axis, w0)

    return PulseData(
        time_domain_field=time_domain_field,
        frequency_domain_field=frequency_domain_field,
        time_axis=time_axis,
        frequency_axis=frequency_axis,
        phase=phase,
        spectral_phase=spectral_phase,
        lambda0=lambda0,
        fwhm_lambda=fwhm_lambda,
        w0=w0,
        fwhm_w=fwhm_w,
        n_peaks=n_peaks,
        central_lambdas=central_lambdas,
        central_w0s=central_w0s,
        fwhm_lambdas=fwhm_lambdas,
        fwhm_ws=fwhm_ws,
        amplitudes=amplitudes,
        phi2=phi2,
        phi3=phi3,
        phi4=phi4,
    )


def _find_nearest(array: np.ndarray, value: float) -> int:
    """Return the index of the array element closest to the target value."""

    return int(np.abs(np.asarray(array) - value).argmin())


def _window_background(signal: np.ndarray, center_half_width: int = 50, smooth_width: int = 10) -> np.ndarray:
    """Estimate a smooth background envelope used for noisy-input scaling."""

    x_axis = np.linspace(0, len(signal), len(signal))
    window = -0.5 * (
        np.tanh((x_axis - len(signal) // 2 - center_half_width) / smooth_width)
        - np.tanh((x_axis - len(signal) // 2 + center_half_width) / smooth_width)
    )
    return np.real(ifft(fftshift(window * fftshift(fft(signal)))))


def _build_noisy_inputs(
    spectrum_1: np.ndarray,
    spectrum_2: np.ndarray,
    ixc1: np.ndarray,
    ixc2: np.ndarray,
    label: np.ndarray,
    snr_levels: Iterable[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Create clean-plus-noisy input matrices and matching labels."""

    base = [spectrum_1, spectrum_2, ixc1, ixc2]
    columns: list[np.ndarray] = [*base]

    background_1 = _window_background(ixc1)
    background_2 = _window_background(ixc2)
    ixc1_scale = max(np.max(background_1 - 1.0), 1e-8)
    ixc2_scale = max(np.max(background_2 - 1.0), 1e-8)
    s1_scale = max(np.max(spectrum_1), 1e-8)
    s2_scale = max(np.max(spectrum_2), 1e-8)

    for snr in snr_levels:
        columns.append(spectrum_1 + np.random.normal(0.0, (s1_scale / snr) ** 2, len(spectrum_1)))
        columns.append(spectrum_2 + np.random.normal(0.0, (s2_scale / snr) ** 2, len(spectrum_2)))
        columns.append(ixc1 + np.random.normal(0.0, (ixc1_scale / snr) ** 2, len(ixc1)))
        columns.append(ixc2 + np.random.normal(0.0, (ixc2_scale / snr) ** 2, len(ixc2)))

    return np.vstack(columns).T, np.asarray(label).reshape(-1, 1)


@dataclass
class GenerationConfig:
    """Configuration for synthetic dataset generation."""

    dataset_name: str = DEFAULT_DATASET_NAME
    output_root: Path = DEFAULT_DATASET_PATH.parent
    num_samples: int = 200
    num_points: int = 4096
    time_half_window: int = 800
    freq_half_window: int = 1250
    freq_offset: int = 400
    phase_mode_1: str = "random"
    phase_mode_2: str = "random"
    sigma_smooth: float = 5.0
    medium: str = "SiO2"
    medium_length_m: float = 2e-3
    num_medium_points: int = 2
    write_noisy_variants: bool = False
    noisy_snr_levels: tuple[int, ...] = (1, 2, 5, 10, 20)
    seed: int | None = None

    @property
    def dataset_path(self) -> Path:
        """Return the final dataset path resolved from root and dataset name."""

        return self.output_root / self.dataset_name


@dataclass
class AutocorrDataGenerator:
    """Generate interferometric-trace datasets on disk."""

    config: GenerationConfig = field(default_factory=GenerationConfig)

    def generate(self) -> Path:
        """Generate all requested samples and return the dataset root path."""

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        dataset_path = self.config.dataset_path
        dataset_path.mkdir(parents=True, exist_ok=True)

        num_points = self.config.num_points
        center_index = num_points // 2
        time_start = center_index - self.config.time_half_window
        time_stop = center_index + self.config.time_half_window
        freq_start = center_index - self.config.freq_half_window + self.config.freq_offset
        freq_stop = center_index + self.config.freq_half_window + self.config.freq_offset

        for sample_index in range(self.config.num_samples):
            if sample_index % 100 == 0 or sample_index == self.config.num_samples - 1:
                print(f"[generate] sample {sample_index + 1}/{self.config.num_samples}")

            pulse_1 = generate_pulse(
                num_points=num_points,
                sigma_smooth=self.config.sigma_smooth,
                phase_mode=self.config.phase_mode_1,
            )
            pulse_2 = generate_pulse(
                num_points=num_points,
                sigma_smooth=self.config.sigma_smooth,
                phase_mode=self.config.phase_mode_2,
            )

            autocorr_1 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_1.time_domain_field,
                pulse_1.time_axis,
                1,
            )
            crosscorr_1 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                1,
            )
            autocorr_2 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_1.time_domain_field,
                pulse_1.time_axis,
                2,
            )
            autocorr_3 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_1.time_domain_field,
                pulse_1.time_axis,
                3,
            )
            crosscorr_2 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                2,
            )
            crosscorr_3 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                3,
            )

            _, _, pulse_1_time_sio2, pulse_1_phase_sio2 = propagation(
                pulse_1.time_axis,
                pulse_1.frequency_axis,
                pulse_1.w0,
                pulse_1.frequency_domain_field,
                self.config.medium_length_m,
                self.config.num_medium_points,
                medium=self.config.medium,
            )
            _, _, pulse_2_time_sio2, pulse_2_phase_sio2 = propagation(
                pulse_2.time_axis,
                pulse_2.frequency_axis,
                pulse_2.w0,
                pulse_2.frequency_domain_field,
                self.config.medium_length_m,
                self.config.num_medium_points,
                medium=self.config.medium,
            )

            pulse_1_field_sio2 = pulse_1_time_sio2[:, -1]
            pulse_2_field_sio2 = pulse_2_time_sio2[:, -1]
            pulse_1_phase_t_sio2 = pulse_1_phase_sio2[:, -1]
            pulse_2_phase_t_sio2 = pulse_2_phase_sio2[:, -1]

            autocorr_1_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_1_field_sio2,
                pulse_1.time_axis,
                1,
            )
            autocorr_1_sio2_2 = interferometric_autocorrelation(
                pulse_2_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                1,
            )
            crosscorr_1_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                1,
            )
            crosscorr_1_sio2_2 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                1,
            )
            crosscorr_1_sio2_12 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                1,
            )

            autocorr_2_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_1_field_sio2,
                pulse_1.time_axis,
                2,
            )
            autocorr_2_sio2_2 = interferometric_autocorrelation(
                pulse_2_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                2,
            )
            crosscorr_2_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                2,
            )
            crosscorr_2_sio2_2 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                2,
            )
            crosscorr_2_sio2_12 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                2,
            )

            autocorr_3_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_1_field_sio2,
                pulse_1.time_axis,
                3,
            )
            autocorr_3_sio2_2 = interferometric_autocorrelation(
                pulse_2_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                3,
            )
            crosscorr_3_sio2_1 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2.time_domain_field,
                pulse_1.time_axis,
                3,
            )
            crosscorr_3_sio2_2 = interferometric_autocorrelation(
                pulse_1.time_domain_field,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                3,
            )
            crosscorr_3_sio2_12 = interferometric_autocorrelation(
                pulse_1_field_sio2,
                pulse_2_field_sio2,
                pulse_1.time_axis,
                3,
            )

            sample_path = dataset_path / str(sample_index)
            sample_path.mkdir(parents=True, exist_ok=True)

            freqs = pulse_1.frequency_axis[freq_start:freq_stop]
            times = pulse_1.time_axis[time_start:time_stop]

            spectra_1 = np.abs(pulse_1.frequency_domain_field[freq_start:freq_stop]) ** 2
            spectral_phase_1 = pulse_1.spectral_phase[freq_start:freq_stop]
            peak_1 = _find_nearest(freqs, pulse_1.w0)
            spectral_phase_1 = spectral_phase_1 - spectral_phase_1[peak_1]

            spectra_2 = np.abs(pulse_2.frequency_domain_field[freq_start:freq_stop]) ** 2
            spectral_phase_2 = pulse_2.spectral_phase[freq_start:freq_stop]
            peak_2 = _find_nearest(freqs, pulse_2.w0)
            spectral_phase_2 = spectral_phase_2 - spectral_phase_2[peak_2]

            intensities_1 = np.abs(pulse_1.time_domain_field[time_start:time_stop]) ** 2
            amplitudes_1 = np.real(pulse_1.time_domain_field[time_start:time_stop])
            phase_1 = pulse_1.phase[time_start:time_stop]

            intensities_2 = np.abs(pulse_2.time_domain_field[time_start:time_stop]) ** 2
            amplitudes_2 = np.real(pulse_2.time_domain_field[time_start:time_stop])
            phase_2 = pulse_2.phase[time_start:time_stop]

            time_domain_data = np.vstack(
                (
                    times,
                    intensities_1,
                    amplitudes_1,
                    phase_1,
                    intensities_2,
                    amplitudes_2,
                    phase_2,
                    autocorr_1[time_start:time_stop],
                    autocorr_2[time_start:time_stop],
                    autocorr_3[time_start:time_stop],
                    crosscorr_1[time_start:time_stop],
                    crosscorr_2[time_start:time_stop],
                    crosscorr_3[time_start:time_stop],
                )
            ).T

            time_domain_sio2_data = np.vstack(
                (
                    np.abs(pulse_1_field_sio2[time_start:time_stop]) ** 2,
                    np.real(pulse_1_field_sio2[time_start:time_stop]),
                    np.abs(pulse_2_field_sio2[time_start:time_stop]) ** 2,
                    np.real(pulse_2_field_sio2[time_start:time_stop]),
                    autocorr_1_sio2_1[time_start:time_stop],
                    autocorr_1_sio2_2[time_start:time_stop],
                    crosscorr_1_sio2_1[time_start:time_stop],
                    crosscorr_1_sio2_2[time_start:time_stop],
                    crosscorr_1_sio2_12[time_start:time_stop],
                    autocorr_2_sio2_1[time_start:time_stop],
                    autocorr_2_sio2_2[time_start:time_stop],
                    crosscorr_2_sio2_1[time_start:time_stop],
                    crosscorr_2_sio2_2[time_start:time_stop],
                    crosscorr_2_sio2_12[time_start:time_stop],
                    autocorr_3_sio2_1[time_start:time_stop],
                    autocorr_3_sio2_2[time_start:time_stop],
                    crosscorr_3_sio2_1[time_start:time_stop],
                    crosscorr_3_sio2_2[time_start:time_stop],
                    crosscorr_3_sio2_12[time_start:time_stop],
                )
            ).T

            frequency_domain_data = np.vstack(
                (
                    freqs,
                    spectra_1,
                    spectral_phase_1,
                    spectra_2,
                    spectral_phase_2,
                )
            ).T

            other_parameters = np.array(
                [
                    pulse_1.lambda0,
                    pulse_1.fwhm_lambda,
                    pulse_1.w0,
                    pulse_1.fwhm_w,
                    pulse_1.n_peaks,
                    *pulse_1.central_lambdas,
                    *pulse_1.central_w0s,
                    *pulse_1.fwhm_lambdas,
                    *pulse_1.fwhm_ws,
                    *pulse_1.amplitudes,
                    pulse_1.phi2,
                    pulse_1.phi3,
                    pulse_1.phi4,
                    pulse_2.lambda0,
                    pulse_2.fwhm_lambda,
                    pulse_2.w0,
                    pulse_2.fwhm_w,
                    pulse_2.n_peaks,
                    *pulse_2.central_lambdas,
                    *pulse_2.central_w0s,
                    *pulse_2.fwhm_lambdas,
                    *pulse_2.fwhm_ws,
                    *pulse_2.amplitudes,
                    pulse_2.phi2,
                    pulse_2.phi3,
                    pulse_2.phi4,
                ]
            )

            np.savetxt(
                sample_path / TIME_DOMAIN_FILENAME,
                time_domain_data.astype(float),
                fmt="%.8e",
                header=(
                    "times_, intensities_1_, amplitudes_1_, phases_1_, "
                    "intensities_2_, amplitudes_2_, phases_2_, "
                    "interf_autocorrelations_1st_order, interf_autocorrelations_2nd_order, "
                    "interf_autocorrelations_3rd_order, interf_crosscorrelations_1st_order, "
                    "interf_crosscorrelations_2nd_order, interf_crosscorrelations_3rd_order"
                ),
            )
            np.savetxt(
                sample_path / TIME_DOMAIN_SIO2_FILENAME,
                time_domain_sio2_data.astype(float),
                fmt="%.8e",
                header=(
                    "intensities_1_sio2_, amplitudes_1_sio2_, intensities_2_sio2_, "
                    "amplitudes_2_sio2_, interf_autocorrelations_1_sio2_1_, "
                    "interf_autocorrelations_1_sio2_2_, interf_crosscorrelations_1_sio2_1_, "
                    "interf_crosscorrelations_1_sio2_2_, interf_crosscorrelations_1_sio2_12_, "
                    "interf_autocorrelations_2_sio2_1_, interf_autocorrelations_2_sio2_2_, "
                    "interf_crosscorrelations_2_sio2_1_, interf_crosscorrelations_2_sio2_2_, "
                    "interf_crosscorrelations_2_sio2_12_, interf_autocorrelations_3_sio2_1_, "
                    "interf_autocorrelations_3_sio2_2_, interf_crosscorrelations_3_sio2_1_, "
                    "interf_crosscorrelations_3_sio2_2_, interf_crosscorrelations_3_sio2_12_"
                ),
            )
            np.savetxt(
                sample_path / FREQ_DOMAIN_FILENAME,
                frequency_domain_data.astype(float),
                fmt="%.8e",
                header="freqs_, spectra_1_, spectral_phases_1_, spectra_2_, spectral_phases_2_",
            )
            np.savetxt(
                sample_path / OTHER_PARAMS_FILENAME,
                other_parameters.astype(float),
                fmt="%.8e",
                header=(
                    "[lambda0_1,fwhm_lambda_1,w0_1,fwhm_w_1,n_peaks_1,*central_lambdas_1,*central_w0s_1,"
                    "*fwhm_lambdas_1,*fwhm_ws_1,*amplitudes_1,phi2_1,phi3_1,phi4_1,lambda0_2,"
                    "fwhm_lambda_2,w0_2,fwhm_w_2,n_peaks_2,*central_lambdas_2,*central_w0s_2,"
                    "*fwhm_lambdas_2,*fwhm_ws_2,*amplitudes_2,phi2_2,phi3_2,phi4_2]"
                ),
            )

            if self.config.write_noisy_variants:
                noisy_inputs, noisy_labels = _build_noisy_inputs(
                    spectra_1[1250 - 800 : 1250 + 800],
                    spectra_2[1250 - 800 : 1250 + 800],
                    crosscorr_2[time_start:time_stop],
                    crosscorr_2_sio2_2[time_start:time_stop],
                    time_domain_data[800 - 500 : 800 + 500, 5],
                    self.config.noisy_snr_levels,
                )
                np.savetxt(
                    sample_path / NOISY_INPUT_FILENAME,
                    noisy_inputs.astype(float),
                    fmt="%.8e",
                    header=(
                        "S1,S2,IXC1,IXC2,"
                        "S1_noisy_1,S2_noisy_1,IXC1_noisy_1,IXC2_noisy_1,"
                        "S1_noisy_2,S2_noisy_2,IXC1_noisy_2,IXC2_noisy_2,"
                        "S1_noisy_5,S2_noisy_5,IXC1_noisy_5,IXC2_noisy_5,"
                        "S1_noisy_10,S2_noisy_10,IXC1_noisy_10,IXC2_noisy_10,"
                        "S1_noisy_20,S2_noisy_20,IXC1_noisy_20,IXC2_noisy_20"
                    ),
                )
                np.savetxt(
                    sample_path / NOISY_LABEL_FILENAME,
                    noisy_labels.astype(float),
                    fmt="%.8e",
                    header="label",
                )

        return dataset_path
