#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


PS_TO_AU = 41341.37333518211
AU_OMEGA_TO_CM = 219474.63136320


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct dressed x-dipole trajectories from fix cboamd output "
            "and compute direct FFT spectra."
        )
    )
    parser.add_argument("workdir", type=Path, help="Task directory containing traj_XX/cboamd_output.dat.")
    parser.add_argument("--lambda-value", type=float, required=True, help="Cavity coupling lambda.")
    parser.add_argument("--chi", default="unknown", help="Label stored in output headers.")
    parser.add_argument("--dt-ps", type=float, default=0.0005)
    parser.add_argument("--ntraj", type=int, default=40)
    return parser.parse_args()


def direct_fft(signal: np.ndarray, dt_ps: float) -> tuple[np.ndarray, np.ndarray]:
    dt_au = dt_ps * PS_TO_AU
    n = signal.size
    ws = 2.0 * np.pi / n
    wnorm = np.arange(-np.pi, np.pi, ws)[:n]
    omega_au = wnorm / dt_au
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal))) * dt_au
    return omega_au * AU_OMEGA_TO_CM, np.abs(spectrum)


def hamming_smooth(values: np.ndarray, window_len: int = 11) -> np.ndarray:
    padded = np.r_[values[window_len - 1 : 0 : -1], values, values[-2 : -window_len - 1 : -1]]
    weights = np.hamming(window_len)
    smoothed = np.convolve(weights / weights.sum(), padded, mode="valid")
    return smoothed[window_len // 2 - 1 : -window_len // 2]


def read_cboamd_table(path: Path) -> tuple[list[str], np.ndarray]:
    header = None
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                fields = line[1:].split()
                if "Dipole_x" in fields:
                    header = fields
                    break
    if header is None:
        raise ValueError(f"Could not find cboamd column header in {path}")

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return header, data


def column(header: list[str], name: str) -> int:
    return header.index(name)


def peak(freq: np.ndarray, intensity: np.ndarray, lo: float, hi: float) -> float:
    mask = (freq >= lo) & (freq <= hi)
    if not np.any(mask):
        return float("nan")
    return float(freq[mask][np.argmax(intensity[mask])])


def main() -> None:
    args = parse_args()
    workdir = args.workdir.resolve()
    spectra = []
    freq_ref = None

    for itraj in range(args.ntraj):
        traj = workdir / f"traj_{itraj:02d}"
        header, data = read_cboamd_table(traj / "cboamd_output.dat")
        step = data[:, column(header, "Step")]
        time = data[:, column(header, "Time")]
        bare_x = data[:, column(header, "Dipole_x")]
        pol_xx = data[:, column(header, "Pol_xx")] if "Pol_xx" in header else np.zeros_like(bare_x)
        ea = data[:, column(header, "ea_0")] if "ea_0" in header else np.zeros_like(bare_x)

        correction = args.lambda_value * pol_xx * ea
        dressed_x = bare_x + correction
        np.savetxt(
            traj / "dressed_dipole_x.dat",
            np.column_stack([step, time, dressed_x, bare_x, correction, pol_xx, ea]),
            header=(
                "Step Time dressed_Dipole_x bare_Dipole_x lambda_Pol_xx_ea Pol_xx ea_0 "
                f"lambda={args.lambda_value} chi={args.chi} units=au"
            ),
        )

        freq, spec = direct_fft(dressed_x, args.dt_ps)
        np.savetxt(
            traj / "spectrum_direct_dressed_x_corrected.dat",
            np.column_stack([freq, spec]),
            header=(
                "freq_cm-1 intensity components=x method=direct window=none "
                f"dipole=dressed lambda={args.lambda_value} chi={args.chi}"
            ),
        )

        if freq_ref is None:
            freq_ref = freq
        elif len(freq) != len(freq_ref) or not np.allclose(freq, freq_ref):
            raise ValueError(f"Frequency grid mismatch in {traj}")
        spectra.append(spec)

    stack = np.vstack(spectra)
    mean = stack.mean(axis=0)
    std = stack.std(axis=0, ddof=1)
    sem = std / np.sqrt(len(spectra))
    mean_smooth = hamming_smooth(mean, 11)
    std_smooth = hamming_smooth(std, 11)
    sem_smooth = hamming_smooth(sem, 11)

    raw_low = peak(freq_ref, mean, 300, 1200)
    raw_stretch = peak(freq_ref, mean, 1800, 3200)
    smooth_low = peak(freq_ref, mean_smooth, 300, 1200)
    smooth_stretch = peak(freq_ref, mean_smooth, 1800, 3200)

    np.savetxt(
        workdir / "spectrum_direct_dressed_x_avg40.dat",
        np.column_stack([freq_ref, mean, std, sem]),
        header=(
            "freq_cm-1 mean_intensity std_intensity sem_intensity "
            "components=x method=direct window=none "
            f"n_spectra={args.ntraj} dipole=dressed lambda={args.lambda_value} chi={args.chi} "
            f"low_peak_cm-1={raw_low:.12f} stretch_peak_cm-1={raw_stretch:.12f}"
        ),
    )
    np.savetxt(
        workdir / "spectrum_direct_dressed_x_avg40_hamming11.dat",
        np.column_stack([freq_ref, mean, std, sem, mean_smooth, std_smooth, sem_smooth]),
        header=(
            "freq_cm-1 mean_intensity std_intensity sem_intensity "
            "mean_hamming11 std_hamming11 sem_hamming11 "
            "components=x method=direct window=none "
            f"n_spectra={args.ntraj} dipole=dressed lambda={args.lambda_value} chi={args.chi} "
            f"low_peak_cm-1={smooth_low:.12f} stretch_peak_cm-1={smooth_stretch:.12f}"
        ),
    )
    print(
        f"{workdir.name}: low={smooth_low:.1f} cm^-1, "
        f"stretch={smooth_stretch:.1f} cm^-1; wrote traj_XX/dressed_dipole_x.dat"
    )


if __name__ == "__main__":
    main()
