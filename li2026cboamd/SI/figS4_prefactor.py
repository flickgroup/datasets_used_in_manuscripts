"""SI Fig. S4: time evolution of the polarizability prefactor (1+l.chi.l)^-1.

PBE0 regeneration of the earlier LDA-era prefactor figure, restyled to match the
main-text figures. One panel per coupling strength (chi-included NEP-MD runs),
first 0.6 ps. The prefactor is computed from the stored polarizability.dat
(chi_xx column) as 1/(1 + lambda^2 chi_xx); NO new simulations.

Data: PBE0/aug-cc-pVDZ NEP runs in data/single_molecule/nep_md/lambda=*_polar.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
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
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA_NEP = DATA / "single_molecule" / "nep_md"

AU_T_PS = 2.4188843265857e-5
T_MAX = 0.6
LAMBDAS = [0.3, 0.25, 0.2, 0.15, 0.1, 0.05, 0.02, 0.008, 0]   # rows, top to bottom

COLOR = "#1177b0"   # MLIP blue, as in the main-text figures

fig, axes = plt.subplots(len(LAMBDAS), 1, figsize=(6.5, 7.5), sharex=True,
                         gridspec_kw={"hspace": 0.14})

for row, lam in enumerate(LAMBDAS):
    ax = axes[row]
    ax.set_xlim(0, T_MAX)
    ax.tick_params(direction="in", top=True, right=True, pad=2)
    name = "lambda=0" if lam == 0 else f"lambda={lam:g}_polar"
    f = DATA_NEP / name / "polarizability.npz"
    if f.exists():
        d = co2io.load_trace_columns(f)
        t = d[:, 1] * AU_T_PS
        m = t <= T_MAX
        pref = 1.0 / (1.0 + lam**2 * d[m, 2])
        ax.plot(t[m], pref, color=COLOR, lw=1.0)
    else:
        # lambda = 0: the prefactor is identically 1
        print(f"missing: {f} (plotting constant 1)")
        ax.plot([0, T_MAX], [1.0, 1.0], color=COLOR, lw=1.0)
    if lam == 0:
        ax.set_ylim(0.95, 1.05)
    ax.yaxis.set_major_locator(MaxNLocator(3))
    fmt = ScalarFormatter(useOffset=False)
    ax.yaxis.set_major_formatter(fmt)
    ax.text(0.975, 0.82, rf"$\lambda={lam:g}$", transform=ax.transAxes,
            ha="right", va="center", fontsize=9,
                bbox=dict(facecolor="white", edgecolor="none", pad=1.2))

axes[-1].set_xlabel(r"Time $t$ [ps]")
axes[-1].set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
fig.text(0.035, 0.50,
         r"$\left(1+\lambda_{\alpha}\cdot\chi_0\cdot\lambda_{\alpha}\right)^{-1}$",
         rotation=90, va="center", ha="center")

fig.savefig(OUTDIR / "figS4_prefactor.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figS4_prefactor.pdf", bbox_inches="tight")
print("figS4_prefactor written")
