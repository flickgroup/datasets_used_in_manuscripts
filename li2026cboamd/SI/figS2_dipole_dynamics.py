"""SI Fig. S2: time evolution of the CO2 x-dipole, AIMD vs MLIP, all lambdas.

PBE0 regeneration of the earlier LDA-era dipole-dynamics figure, restyled to match
the main-text figures (Computer Modern fonts, AIMD red solid / MLIP blue
dashed as in fig1_spectrum_lambdas.py). Nine coupling strengths (rows), the
two chi channels (columns), first 0.6 ps of each trajectory. Data are the
dressed x-dipole columns of the stored dipole.dat files; NO new simulations.

Data: PBE0/aug-cc-pVDZ. AIMD from the e-mode runs of the single-molecule campaign (
10 ps); MLIP from nep_md (retrained NEP models, 100 ps).
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

AU_T_PS = 2.4188843265857e-5   # atomic time unit in ps
T_MAX = 0.6                    # ps shown
LAMBDAS = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.008, 0]   # rows, top to bottom

colors = {"aimd": "#cc163a", "nep": "#1177b0"}
styles = {"aimd": ("-", 1.2), "nep": ("--", 0.9)}


def run_dir(base, lam, polar):
    tag = "polar" if polar else "nopolar"
    if base == DATA_NEP:
        return base / ("lambda=0" if lam == 0 else f"lambda={lam:g}_{tag}")
    return base / ("lambda=0_qmode" if lam == 0
                   else f"lambda={lam:g}_emode_{tag}")


def dipole_trace(dipole_file):
    """Time [ps] and dressed x-dipole [a.u.], first T_MAX ps."""
    d = co2io.load_trace_columns(dipole_file)
    t = d[:, 1] * AU_T_PS
    m = t <= T_MAX
    return t[m], d[m, 2]


fig, axes = plt.subplots(len(LAMBDAS), 2, figsize=(6.5, 7.5), sharex=True,
                         sharey=True, gridspec_kw={"hspace": 0.08, "wspace": 0.08})

for col, polar in enumerate([True, False]):
    axes[0, col].set_title(r"$\chi$ included" if polar else r"$\chi$ neglected", pad=4)
    for row, lam in enumerate(LAMBDAS):
        ax = axes[row, col]
        ax.set_xlim(0, T_MAX)
        ax.set_ylim(-0.062, 0.062)
        ax.set_yticks([-0.05, 0, 0.05])
        ax.tick_params(direction="in", top=True, right=True, pad=2)
        ax.text(0.975, 0.86, rf"$\lambda={lam:g}$", transform=ax.transAxes,
                ha="right", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))
        for key, base in [("aimd", DATA_AIMD), ("nep", DATA_NEP)]:
            f = run_dir(base, lam, polar) / "dipole.npz"
            if not f.exists():
                print(f"missing: {f}")
                continue
            t, mu = dipole_trace(f)
            ls, lw = styles[key]
            ax.plot(t, mu, color=colors[key], lw=lw, ls=ls,
                    label={"aimd": "AIMD", "nep": "MLIP"}[key]
                    if (row, col) == (0, 0) else None)

for ax in axes[-1, :]:
    ax.set_xlabel(r"Time $t$ [ps]")
    ax.set_xticks([0.0, 0.2, 0.4, 0.6])

fig.text(0.055, 0.50, r"Dipole Moment $\mu_x$ [a.u.]", rotation=90,
         va="center", ha="center")
axes[0, 0].legend(frameon=False, loc="upper left", ncol=2, fontsize=9,
                  handlelength=1.6, borderaxespad=0.2, columnspacing=1.0)

fig.savefig(OUTDIR / "figS2_dipole_dynamics.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figS2_dipole_dynamics.pdf", bbox_inches="tight")
print("figS2_dipole_dynamics written")
