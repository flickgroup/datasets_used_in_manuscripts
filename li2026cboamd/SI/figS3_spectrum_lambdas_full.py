"""SI Fig. S3: IR spectra of a single CO2 molecule for ALL coupling strengths.

PBE0 regeneration of the earlier LDA-era full-spectrum figure, restyled to
match the main-text fig1 (Computer Modern fonts, AIMD red solid / MLIP blue
dashed, cavity line, grey nu1/nu2 markers, Rabi arrows in the top row).
Spectra are |FFT| of the stored dressed x-dipole trajectories, the same
procedure as fig1_spectrum_lambdas.py; NO new simulations.

Data: PBE0/aug-cc-pVDZ. AIMD 10 ps, MLIP (NEP) 100 ps.
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
DATA_AIMD = DATA / "single_molecule"
DATA_NEP = DATA_AIMD / "nep_md"

AU_TO_CM = 219474.63
OMEGA_C = 2442.0
NU1 = 1378.9
NU2 = 680.4
LAMBDAS = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.008, 0]   # rows, top to bottom

colors = {"aimd": "#cc163a", "nep": "#1177b0"}
styles = {"aimd": ("-", 1.3), "nep": ("--", 1.0)}


def run_dir(base, lam, polar):
    tag = "polar" if polar else "nopolar"
    if base == DATA_NEP:
        return base / ("lambda=0" if lam == 0 else f"lambda={lam:g}_{tag}")
    return base / ("lambda=0_qmode" if lam == 0
                   else f"lambda={lam:g}_emode_{tag}")


def spectrum(dipole_file):
    d = co2io.load_trace_columns(dipole_file)
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    spec = np.abs(np.fft.rfft(sig)) * dt
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return freq, spec


plot_lo, plot_hi = 500, 3500
fig, axes = plt.subplots(len(LAMBDAS), 2, figsize=(6.5, 7.5), sharex=True,
                         sharey=True, gridspec_kw={"hspace": 0.08, "wspace": 0.08})

for col, polar in enumerate([True, False]):
    axes[0, col].set_title(r"$\chi$ included" if polar else r"$\chi$ neglected", pad=4)
    for row, lam in enumerate(LAMBDAS):
        ax = axes[row, col]
        ax.set_xlim(plot_lo, plot_hi)
        ax.set_ylim(-0.03, 1.34)
        ax.set_yticks([0, 1])
        ax.tick_params(direction="in", top=True, right=True, pad=2)
        ax.axvline(OMEGA_C, color="#3b4a8a", ls="--", lw=0.7, alpha=0.65)
        for nu in (NU1, NU2):
            ax.axvline(nu, color="0.7", ls="--", lw=0.6, zorder=0)
        ax.text(0.975, 0.84, rf"$\lambda={lam:g}$", transform=ax.transAxes,
                ha="right", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        peaks = {}
        for key, base in [("aimd", DATA_AIMD), ("nep", DATA_NEP)]:
            f = run_dir(base, lam, polar) / "dipole.npz"
            if not f.exists():
                print(f"missing: {f}")
                continue
            freq, spec = spectrum(f)
            m = (freq >= plot_lo) & (freq <= plot_hi)
            ls, lw = styles[key]
            ax.plot(freq[m], spec[m] / spec[m].max(), color=colors[key],
                    lw=lw, ls=ls,
                    label={"aimd": "AIMD", "nep": "MLIP"}[key]
                    if (row, col) == (0, 0) else None)
            lo_m = (freq >= 800) & (freq <= OMEGA_C - 1)
            hi_m = (freq >= OMEGA_C + 1) & (freq <= 3800)
            peaks[key] = (freq[lo_m][np.argmax(spec[lo_m])],
                          freq[hi_m][np.argmax(spec[hi_m])])
        # Rabi-splitting arrow between the polariton peaks in the top row
        if row == 0 and "nep" in peaks:
            lp, up = peaks["nep"]
            gap = 100.0
            ax.annotate("", xy=(up - gap, 0.38), xytext=(lp + gap, 0.38),
                        arrowprops=dict(arrowstyle="<->", color="0.2", lw=0.9,
                                        shrinkA=0, shrinkB=0))
            ax.text(0.5 * (lp + up), 0.47, "Rabi splitting", ha="center",
                    va="bottom", fontsize=9)

for ax in axes[-1, :]:
    ax.set_xlabel(r"Frequency [cm$^{-1}$]")
    ax.set_xticks([1000, 2000, 3000])

fig.text(0.055, 0.50, "IR Intensity [arb.u.]", rotation=90, va="center",
         ha="center")
axes[0, 0].legend(frameon=False, loc="upper left", ncol=2, fontsize=9,
                  handlelength=1.6, borderaxespad=0.2, columnspacing=1.0)

fig.savefig(OUTDIR / "figS3_spectrum_lambdas_full.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figS3_spectrum_lambdas_full.pdf", bbox_inches="tight")
print("figS3_spectrum_lambdas_full written")
