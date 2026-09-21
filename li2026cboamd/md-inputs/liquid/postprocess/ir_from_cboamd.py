#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np


CM_PER_THz = 33.35640951981521
PS_TO_AU = 41341.37333518211
AU_OMEGA_TO_CM = 219474.63136320


def parse_args():
    parser = argparse.ArgumentParser(description="Compute an IR-like spectrum from fix cboamd total dipoles.")
    parser.add_argument("--input", default="cboamd_output.dat", help="fix cboamd output table.")
    parser.add_argument("--output", default="spectrum_xy.dat", help="Output spectrum file.")
    parser.add_argument("--dt-ps", type=float, default=0.0005, help="Sampling interval in ps.")
    parser.add_argument(
        "--method",
        choices=("direct", "acf"),
        default="direct",
        help="direct matches cboamd.infrared: abs(FFT(mu(t))). acf uses an omega^2-weighted dipole autocorrelation spectrum.",
    )
    parser.add_argument(
        "--components",
        choices=("x", "y", "z", "xy", "xyz"),
        default="xy",
        help="Dipole components included in the spectrum.",
    )
    parser.add_argument(
        "--window",
        choices=("hann", "none"),
        default="none",
        help="Window applied before the FFT. Use none to match cboamd.infrared.",
    )
    return parser.parse_args()


def autocorr_fft(x):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = x.size
    padded = np.zeros(2 * n, dtype=float)
    padded[:n] = x
    f = np.fft.rfft(padded)
    acf = np.fft.irfft(f * np.conjugate(f), n=2 * n)[:n]
    norm = np.arange(n, 0, -1, dtype=float)
    return acf / norm


def direct_fft(x, dt_ps):
    x = np.asarray(x, dtype=float)
    dt_au = dt_ps * PS_TO_AU
    n = x.size
    ws = 2.0 * np.pi / n
    wnorm = np.arange(-np.pi, np.pi, ws)[:n]
    omega_au = wnorm / dt_au
    spectrum = np.fft.fftshift(np.fft.fft(np.fft.fftshift(x))) * dt_au
    return omega_au * AU_OMEGA_TO_CM, np.abs(spectrum)


def read_cboamd_table(path):
    header_columns = None
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                fields = line[1:].split()
                if {"Dipole_x", "Dipole_y", "Dipole_z"}.issubset(fields):
                    header_columns = fields
                    break
            elif line.strip():
                break

    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if header_columns is not None and len(header_columns) == data.shape[1] + 1 and "Energy" in header_columns:
        header_columns = [name for name in header_columns if name != "Energy"]
    if header_columns is not None:
        indices = [header_columns.index(name) for name in ("Dipole_x", "Dipole_y", "Dipole_z")]
    elif data.shape[1] >= 6:
        indices = [2, 3, 4]
    elif data.shape[1] >= 5:
        indices = [2, 3, 4]
    else:
        raise ValueError("Could not find Dipole_x, Dipole_y, Dipole_z columns in cboamd output.")
    return data, data[:, indices]


def main():
    args = parse_args()
    data, dip = read_cboamd_table(args.input)
    index = {"x": [0], "y": [1], "z": [2], "xy": [0, 1], "xyz": [0, 1, 2]}[args.components]

    if args.method == "direct":
        spectra = []
        freq_cm = None
        for i in index:
            signal = dip[:, i].copy()
            if args.window == "hann":
                signal = signal * np.hanning(signal.size)
            freq_cm, spec = direct_fft(signal, args.dt_ps)
            spectra.append(spec)
        intensity = sum(spectra)
    else:
        acf = sum(autocorr_fft(dip[:, i]) for i in index)
        if args.window == "hann":
            acf = acf * np.hanning(acf.size)
        freq_thz = np.fft.rfftfreq(acf.size, d=args.dt_ps)
        freq_cm = freq_thz * CM_PER_THz
        omega_au = freq_cm / AU_OMEGA_TO_CM
        power = np.real(np.fft.rfft(acf))
        intensity = np.clip(power, 0.0, None) * omega_au**2

    out = np.column_stack([freq_cm, intensity])
    header = f"freq_cm-1 intensity components={args.components} dt_ps={args.dt_ps} method={args.method} window={args.window}"
    np.savetxt(args.output, out, header=header)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
