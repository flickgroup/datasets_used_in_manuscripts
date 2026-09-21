"""Manuscript Fig. 1: IR spectra of a single CO2 molecule, AIMD vs MLIP.

Style/setup matches fig3-fig5 (fonts, sizes, layout, output naming). Two
columns (chi included / chi neglected), one row per coupling strength.
Spectra are computed here as |FFT| of the stored dressed x-dipole
trajectories (dipole.dat, columns: step, time[a.u.], dressed mu_xyz,
bare mu_xyz), the same procedure as scripts/infrared.py (mode=ase) in
cboamd. NO new simulations: only existing trajectory files
are read.

Data: PBE0/aug-cc-pVDZ. AIMD from the e-mode runs of the single-molecule campaign (
10 ps); MLIP from nep_md (retrained NEP models on uniform
conv=1e-8 labels, 100 ps). Previous LDA roots kept in the git history.
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
# Field-free molecular frequencies (PBE0/aug-cc-pVDZ). NU1: symmetric stretch
# from the lambda=0 analysis. NU2: bend, measured as the peak of |FFT(mu_y)|
# (identical peak in mu_z) of the field-free AIMD run lambda=0_qmode, window
# 400-900 cm-1, contrast ~170 over the background; the NEP lambda=0 run is
# exactly one-dimensional (mu_y = mu_z = 0), so it carries no bend signal.
NU1 = 1378.9
NU2 = 680.4
LAMBDAS = [0.3, 0.2, 0.1, 0]          # rows, top to bottom

colors = {"aimd": "#cc163a", "nep": "#1177b0"}
styles = {"aimd": ("-", 1.5), "nep": ("--", 1.1)}


def run_dir(base, lam, polar):
    """PBE0 campaign naming. AIMD: lambda=X_emode_{polar,nopolar} with the
    out-of-cavity run in lambda=0_qmode (photons off there, so it serves
    both columns). MLIP: nep_md/lambda=X_{polar,nopolar} and nep_md/lambda=0."""
    tag = "polar" if polar else "nopolar"
    if base == DATA_NEP:
        return base / ("lambda=0" if lam == 0 else f"lambda={lam:g}_{tag}")
    return base / ("lambda=0_qmode" if lam == 0
                   else f"lambda={lam:g}_emode_{tag}")


def spectrum(dipole_file):
    """|FFT| of the dressed x-dipole, frequency axis in cm-1."""
    d = co2io.load_trace_columns(dipole_file)
    t, mux = d[:, 1], d[:, 2]
    dt = t[1] - t[0]
    sig = mux - mux.mean()
    spec = np.abs(np.fft.rfft(sig)) * dt
    freq = 2 * np.pi * np.fft.rfftfreq(len(sig), d=dt) * AU_TO_CM
    return freq, spec


plot_lo, plot_hi = 500, 3500
fig, axes = plt.subplots(len(LAMBDAS), 2, figsize=(6.5, 4.333), sharex=True,
                         sharey=True, gridspec_kw={"hspace": 0.05, "wspace": 0.10})

for col, polar in enumerate([True, False]):
    axes[0, col].set_title(r"$\chi$ included" if polar else r"$\chi$ neglected", pad=4)
    for row, lam in enumerate(LAMBDAS):
        ax = axes[row, col]
        ax.set_xlim(plot_lo, plot_hi)
        ax.set_ylim(-0.03, 1.28)
        ax.set_yticks([0, 1])
        ax.tick_params(direction="in", top=True, right=True, pad=2)
        ax.axvline(OMEGA_C, color="#3b4a8a", ls="--", lw=0.8, alpha=0.65)
        # field-free nu1 (symmetric stretch) and nu2 (bend) markers; the
        # asymmetric stretch nu3 coincides with the cavity line above
        for nu in (NU1, NU2):
            ax.axvline(nu, color="0.7", ls="--", lw=0.7, zorder=0)
        ax.text(0.96, 0.90, rf"$\lambda={lam:g}$", transform=ax.transAxes,
                ha="right", va="center", fontsize=10)
        peaks = {}
        for key, base in [("aimd", DATA_AIMD), ("nep", DATA_NEP)]:
            f = run_dir(base, lam, polar) / "dipole.npz"
            if not f.exists():
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
        # Rabi-splitting arrow between the polariton peaks in the lambda=0.3 row
        if row == 0 and "nep" in peaks:
            lp, up = peaks["nep"]
            print(f"top-row arrow, col={'ab'[col]}: LP={lp:.1f}  UP={up:.1f}")
            # tips stop a small margin short of the polariton lines so the
            # arrow does not overlap the spectral curves
            gap = 100.0
            ax.annotate("", xy=(up - gap, 0.35), xytext=(lp + gap, 0.35),
                        arrowprops=dict(arrowstyle="<->", color="0.2", lw=1.0,
                                        shrinkA=0, shrinkB=0))
            ax.text(0.5 * (lp + up), 0.44, "Rabi splitting", ha="center",
                    va="bottom", fontsize=10)

for ax in axes[-1, :]:
    ax.set_xlabel(r"Frequency [cm$^{-1}$]")
    ax.set_xticks([1000, 2000, 3000])

fig.text(0.070, 0.50, "IR Intensity [arb.u.]", rotation=90, va="center", ha="center")
axes[0, 0].legend(frameon=False, loc="center right",
                  bbox_to_anchor=(1.00, 0.50), fontsize=10, handlelength=1.8,
                  borderaxespad=0.0)
axes[0, 0].text(0.03, 0.93, "a", transform=axes[0, 0].transAxes, fontsize=14,
                fontfamily="cmb10", ha="left", va="top")
axes[0, 1].text(0.03, 0.93, "b", transform=axes[0, 1].transAxes, fontsize=14,
                fontfamily="cmb10", ha="left", va="top")

fig.savefig(OUTDIR / "fig1_spectrum_lambdas.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig1_spectrum_lambdas.pdf", bbox_inches="tight")
print("fig1_spectrum_lambdas written")
