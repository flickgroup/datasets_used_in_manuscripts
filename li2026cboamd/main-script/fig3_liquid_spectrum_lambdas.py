import sys
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np

# --- repository layout ------------------------------------------------------
# This script runs from inside its own folder in the li2026cboamd repository and
# reads only files shipped in that repository. tools/co2io.py hides the .npz
# encoding: its loaders return exactly the arrays the original ASCII files
# held.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import co2io  # noqa: E402

DATA = REPO / "data"



plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent
SPEC = DATA / "liquid" / "spectra" / "offresonant"
SPEC_RES = DATA / "liquid" / "spectra" / "resonant"
OUTDIR = ROOT

# resonant runs (omega_c retuned per lambda to compensate the refractive-index
# renormalization) exist for the chi-included column only
RES_FILES = {
    lam: SPEC_RES / d / "spectrum_direct_dressed_x_avg40_hamming11.npz"
    for lam, d in [
        (0.10, "12_cavity_ir_40traj_newmodels_polar_lam010"),
        (0.03, "19_cavity_ir_40traj_newmodels_polar_lam003"),
        (0.02, "18_cavity_ir_40traj_newmodels_polar_lam002"),
        (0.01, "14_cavity_ir_40traj_newmodels_polar_lam001"),
        (0.04, "26_cavity_ir_40traj_newmodels_polar_lam004"),
        (0.05, "27_cavity_ir_40traj_newmodels_polar_lam005"),
        (0.06, "28_cavity_ir_40traj_newmodels_polar_lam006"),
        (0.07, "29_cavity_ir_40traj_newmodels_polar_lam007"),
        (0.08, "30_cavity_ir_40traj_newmodels_polar_lam008"),
        (0.09, "31_cavity_ir_40traj_newmodels_polar_lam009"),
    ]
}
RES_FILES[0.00] = SPEC_RES / "10_out_of_cavity_ir_40traj_newmodels" / "spectrum_direct_x_avg40_hamming11.npz"

# bare cavity frequency omega_c (a.u.) used for each resonant (omega_c/n_eff) run
RES_OMEGA_AU = {0.01: 0.0116178, 0.02: 0.0133763, 0.03: 0.0158802, 0.04: 0.0188345,
                0.05: 0.0220589, 0.06: 0.0254511, 0.07: 0.0289521, 0.08: 0.0325267,
                0.09: 0.0361532, 0.10: 0.0398174}
AU_TO_CM = 219474.63

CASES = {
    "included": [
        (0.10, SPEC / "12_cavity_ir_40traj_newmodels_polar_lam010" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.03, SPEC / "19_cavity_ir_40traj_newmodels_polar_lam003" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.02, SPEC / "18_cavity_ir_40traj_newmodels_polar_lam002" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.01, SPEC / "14_cavity_ir_40traj_newmodels_polar_lam001" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.00, SPEC / "10_out_of_cavity_ir_40traj_newmodels" / "spectrum_direct_x_avg40_hamming11.npz"),
    ],
    "neglected": [
        (0.10, SPEC / "11_cavity_ir_40traj_newmodels_nopolar_lam010" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.03, SPEC / "17_cavity_ir_40traj_newmodels_nopolar_lam003" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.02, SPEC / "16_cavity_ir_40traj_newmodels_nopolar_lam002" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.01, SPEC / "13_cavity_ir_40traj_newmodels_nopolar_lam001" / "spectrum_direct_dressed_x_avg40_hamming11.npz"),
        (0.00, SPEC / "10_out_of_cavity_ir_40traj_newmodels" / "spectrum_direct_x_avg40_hamming11.npz"),
    ],
}

PEAK_MARKS = {
    ("included", 0.01): [2222.3, 2472.5],
    ("neglected", 0.01): [2299.0, 2529.2],
    ("neglected", 0.02): [2199.0, 2650.9],
}

OUT_OF_CAVITY_PEAKS = [674.6, 2407.4]


def parse_header(path):
    header = path.read_text().splitlines()[0]
    return {key: float(val) for key, val in re.findall(r"([A-Za-z0-9_\\-]+)=([0-9.]+)", header)}


def top_local_peaks(freq, intensity, lo, hi, min_sep=120, n=4):
    idx = np.flatnonzero((freq >= lo) & (freq <= hi))
    local = []
    for i in idx[1:-1]:
        if intensity[i] >= intensity[i - 1] and intensity[i] >= intensity[i + 1]:
            local.append(i)
    local = sorted(local, key=lambda i: intensity[i], reverse=True)
    selected = []
    for i in local:
        if all(abs(freq[i] - freq[j]) >= min_sep for j in selected):
            selected.append(i)
        if len(selected) == n:
            break
    return [float(freq[i]) for i in selected]


def load_spectrum(path, lo, hi):
    data = co2io.load_spectrum_usecols(path, (0, 4))
    freq = data[:, 0]
    intensity = data[:, 1]
    mask = (freq >= lo) & (freq <= hi)
    intensity = intensity / np.nanmax(intensity[mask])
    return freq, intensity


n_rows = len(CASES["included"])
plot_lo, plot_hi = 300, 4400
fig, axes = plt.subplots(
    n_rows,
    2,
    figsize=(6.5, 5.322),
    sharex=True,
    sharey=True,
    gridspec_kw={"hspace": 0.05, "wspace": 0.10},
)

colors = {
    "included": "#cc163a",
    "neglected": "#1177b0",
    "resonant": "#1b7837",
    "missing": "0.68",
    "peak": "#3b4a8a",
}

for col, group in enumerate(["included", "neglected"]):
    axes[0, col].set_title(r"$\chi$ included" if group == "included" else r"$\chi$ neglected", pad=4)
    for row, (lam, path) in enumerate(CASES[group]):
        ax = axes[row, col]
        ax.set_xlim(plot_lo, plot_hi)
        ax.set_ylim(-0.03, 1.28)
        ax.tick_params(direction="in", top=True, right=True, pad=2)
        ax.set_yticks([0, 1])
        for ref_peak in OUT_OF_CAVITY_PEAKS:
            ax.axvline(ref_peak, color="#3b4a8a", ls="--", lw=0.8, alpha=0.65)
        ax.text(
            0.96,
            0.82,
            rf"$\lambda={lam:g}$",
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=10,
        )
        if path is None or not path.exists():
            ax.text(
                0.50,
                0.50,
                "pending",
                transform=ax.transAxes,
                ha="center",
                va="center",
                color=colors["missing"],
                fontsize=10,
            )
            continue

        freq, intensity = load_spectrum(path, plot_lo, plot_hi)
        has_res = group == "included" and lam in RES_FILES and RES_FILES[lam].exists()
        ax.plot(freq, intensity, color=colors[group], lw=1.0,
                label=r"fixed $\omega_c$" if (row, col) == (0, 0) else None)
        if has_res:
            rf, ri = load_spectrum(RES_FILES[lam], plot_lo, plot_hi)
            # at lambda=0 the two runs coincide: dashed green on top of red
            ax.plot(rf, ri, color=colors["resonant"], lw=1.1,
                    ls="--" if lam == 0.00 else "-",
                    label=r"$\omega_c/n_{\mathrm{eff}}$" if row == 0 else None)
            if lam in RES_OMEGA_AU:
                ax.text(0.98, 0.55,
                        rf"$\omega_c={RES_OMEGA_AU[lam] * AU_TO_CM:.0f}\ \mathrm{{cm}}^{{-1}}$",
                        transform=ax.transAxes, ha="right", va="center",
                        fontsize=9, color=colors["resonant"])

        marks = PEAK_MARKS.get((group, lam))
        if marks is None and group == "neglected" and lam == 0.03:
            marks = top_local_peaks(freq, intensity, 1800, 3200, min_sep=160, n=2)
            marks = sorted(marks)
            PEAK_MARKS[(group, lam)] = marks

axes[0, 0].legend(frameon=False, fontsize=10, loc="upper left",
                  bbox_to_anchor=(0.14, 1.02), handlelength=1.0,
                  handletextpad=0.4, labelspacing=0.3, borderaxespad=0.2)
# panel letters as in fig1; shifted up-left of fig1's (0.03, 0.93) so they
# clear the bend peak at ~675 cm-1 that sits close to the left axis here
axes[0, 0].text(0.02, 0.97, "a", transform=axes[0, 0].transAxes, fontsize=14,
                fontfamily="cmb10", ha="left", va="top")
axes[0, 1].text(0.02, 0.97, "b", transform=axes[0, 1].transAxes, fontsize=14,
                fontfamily="cmb10", ha="left", va="top")

for ax in axes[-1, :]:
    ax.set_xlabel(r"Frequency [cm$^{-1}$]")
    ax.set_xticks([1000, 2000, 3000, 4000])

fig.text(0.070, 0.50, "IR Intensity [arb.u.]", rotation=90, va="center", ha="center")
fig.savefig(OUTDIR / "fig3_liquid_spectrum_lambdas.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig3_liquid_spectrum_lambdas.pdf", bbox_inches="tight")

print("peak_marks")
for key in sorted(PEAK_MARKS):
    group, lam = key
    marks = sorted(PEAK_MARKS[key])
    if len(marks) >= 2:
        print(group, lam, marks, "splitting", marks[-1] - marks[0])
    else:
        print(group, lam, marks)
