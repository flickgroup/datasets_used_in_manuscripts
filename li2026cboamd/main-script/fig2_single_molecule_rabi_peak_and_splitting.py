"""Manuscript Fig. 2: single-CO2 Rabi peak positions and splitting vs lambda.

Style/setup matches fig3-fig5. Peak positions are extracted here from the
MLIP(NEP)-driven MD dipole trajectories (|FFT| spectra computed on the fly,
same procedure as fig1_spectrum_lambdas.py; NO new simulations). The explicit
CBOA QEDFT reference of Bonini et al. [J. Chem. Phys. 161, 154104 (2024),
Fig. 2] is embedded numerically below, taken from octopus_CO2_CBOA_FD.txt in
the johannesflick/bonini2024_data repository (Octopus finite-difference
explicit-CBOA polariton frequencies, cm-1).

Data: PBE0/aug-cc-pVDZ MLIP runs in data/single_molecule/nep_md (retrained NEP
models on uniform conv=1e-8 labels, 100 ps trajectories). Previous LDA
root kept in the git history.
"""
import sys
from pathlib import Path

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
OUTDIR = ROOT
DATA_NEP = DATA / "single_molecule" / "nep_md"

AU_TO_CM = 219474.63
OMEGA_C = 2442.0
LAMBDAS = [0, 0.008, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

colors = {"inc": "#cc163a", "neg": "#1177b0", "ref": "#1b7837", "grid": "0.45"}

# Bonini et al. 2024, explicit-CBOA QEDFT (Octopus, finite differences):
# lambda, lower polariton, upper polariton [cm-1]
BONINI = np.array([
    [0.0000, 2429.59, 2438.20],
    [0.0080, 2416.49, 2449.33],
    [0.0200, 2387.76, 2467.86],
    [0.0500, 2294.20, 2500.51],
    [0.0707, 2213.03, 2512.44],
    [0.1000, 2084.05, 2527.05],
    [0.1300, 1941.70, 2532.38],
    [0.1581, 1808.79, 2536.82],
    [0.1700, 1754.07, 2538.22],
    [0.2000, 1622.50, 2543.14],
    [0.2250, 1520.61, 2542.46],
    [0.2500, 1427.02, 2543.63],
    [0.2750, 1341.35, 2544.54],
    [0.3000, 1263.47, 2547.54],
])


def spectrum(dipole_file):
    d = co2io.load_trace_columns(dipole_file)
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    spec = np.abs(np.fft.rfft(sig)) * dt
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return freq, spec


def strongest_peak(freq, spec, lo, hi, floor):
    idx = np.flatnonzero((freq >= lo) & (freq <= hi))
    cand = [i for i in idx[1:-1]
            if spec[i] >= spec[i - 1] and spec[i] >= spec[i + 1]
            and spec[i] >= floor]
    return float(freq[max(cand, key=lambda i: spec[i])]) if cand else np.nan


def collect(polar):
    rows = []
    tag = "polar" if polar else "nopolar"
    for lam in LAMBDAS:
        name = "lambda=0" if lam == 0 else f"lambda={lam:g}_{tag}"
        f = DATA_NEP / name / "dipole.npz"
        if not f.exists():
            continue
        freq, spec = spectrum(f)
        floor = 0.01 * spec[(freq >= 500) & (freq <= 3800)].max()
        if lam == 0:
            pk = strongest_peak(freq, spec, 2000, 3000, floor)
            rows.append((lam, pk, pk))
            continue
        lower = strongest_peak(freq, spec, 800, OMEGA_C - 1, floor)
        upper = strongest_peak(freq, spec, OMEGA_C + 1, 3800, floor)
        rows.append((lam, lower, upper))
    return rows


inc = collect(True)
neg = collect(False)

fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.609), gridspec_kw={"wspace": 0.34})

ax = axes[0]
ax.axhline(OMEGA_C, color=colors["grid"], lw=0.9, ls="--", zorder=0)
for rows, key in [(inc, "inc"), (neg, "neg")]:
    lam = [r[0] for r in rows]
    ax.plot(lam, [r[2] for r in rows], color=colors[key], marker="o", ms=4.2,
            lw=1.2, ls="--")
    ax.plot(lam, [r[1] for r in rows], color=colors[key], marker="o", ms=4.2,
            lw=1.2, ls="--")
ax.plot(BONINI[:, 0], BONINI[:, 2], color=colors["ref"], marker="s", ms=3.6,
        lw=1.0, ls="--", mfc="none")
ax.plot(BONINI[:, 0], BONINI[:, 1], color=colors["ref"], marker="s", ms=3.6,
        lw=1.0, ls="--", mfc="none")
ax.plot([], [], color=colors["ref"], marker="s", ms=3.6, lw=1.0, ls="--",
        mfc="none", label="CBOA QEDFT, Bonini et al.")
ax.plot([], [], color=colors["inc"], marker="o", ms=4.2, lw=1.2, ls="--",
        label=r"MLIP, $\chi$ included")
ax.plot([], [], color=colors["neg"], marker="o", ms=4.2, lw=1.2, ls="--",
        label=r"MLIP, $\chi$ neglected")
ax.set_xlim(-0.012, 0.312)
ax.set_ylim(1150, 3600)   # headroom so the legend (10 pt) clears the chi-neglected UP points
ax.set_xlabel(r"Coupling strength $\lambda$")
ax.set_ylabel(r"Frequency [cm$^{-1}$]")
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.tick_params(direction="in", top=True, right=True)
ax.legend(frameon=False, loc="upper left", handlelength=1.3,
          handletextpad=0.5, labelspacing=0.25, borderaxespad=0.3)
ax.text(-0.26, 0.973, "a", transform=ax.transAxes, fontsize=14,
        fontfamily="cmb10", va="center")

ax = axes[1]
for rows, key, lab in [(inc, "inc", r"MLIP, $\chi$ included"),
                       (neg, "neg", r"MLIP, $\chi$ neglected")]:
    lam = [r[0] for r in rows]
    ax.plot(lam, [r[2] - r[1] for r in rows], "o--", color=colors[key],
            ms=4.2, lw=1.1, label=lab)
ax.plot(BONINI[:, 0], BONINI[:, 2] - BONINI[:, 1], color=colors["ref"],
        marker="s", ms=3.6, lw=1.0, ls="--", mfc="none",
        label=None)
ax.set_xlim(-0.012, 0.312)
ax.set_ylim(-40, 1500)
ax.set_xlabel(r"Coupling strength $\lambda$")
ax.set_ylabel(r"Rabi Splitting [cm$^{-1}$]")
ax.set_xticks([0.0, 0.1, 0.2, 0.3])
ax.tick_params(direction="in", top=True, right=True)
ax.text(-0.26, 1.0, "b", transform=ax.transAxes, fontsize=14,
        fontfamily="cmb10", va="center")

fig.savefig(OUTDIR / "fig2_single_molecule_rabi_peak_and_splitting.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig2_single_molecule_rabi_peak_and_splitting.pdf", bbox_inches="tight")

print("chi      lambda   lower    upper   splitting")
for rows, tag in [(inc, "included"), (neg, "neglected")]:
    for lam, lower, upper in rows:
        print(f"{tag:9s} {lam:5.3f} {lower:8.1f} {upper:8.1f} {upper-lower:9.1f}")
