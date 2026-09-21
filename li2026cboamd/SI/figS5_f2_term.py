"""SI Fig. S5: effect of the polarizability-gradient force term on mu_x(t).

PBE0 regeneration of the earlier LDA-era f2-term figure, restyled to match the
main-text figures. Compares the dressed x-dipole of two otherwise identical
chi-included NEP-MD trajectories at lambda = 0.3:
  - dchi/dR included:  nep_md/lambda=0.3_polar        (production)
  - dchi/dR neglected: nep_md/lambda=0.3_polar_nof2   (polar_force
    false, short 2 ps rerun with identical initial conditions)
Single-column figure (3.4 in wide) for the achemso twocolumn SI layout.
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
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ROOT = Path(__file__).resolve().parent
OUTDIR = ROOT
DATA_NEP = DATA / "single_molecule" / "nep_md"

AU_T_PS = 2.4188843265857e-5
T_MAX = 0.6

RUNS = [
    ("lambda=0.3_polar",
     r"$\partial\chi_0/\partial R_I$ included", "#cc163a", "-", 1.2),
    ("lambda=0.3_polar_nof2",
     r"$\partial\chi_0/\partial R_I$ neglected", "#1177b0", "--", 1.0),
]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
for name, label, color, ls, lw in RUNS:
    f = DATA_NEP / name / "dipole.npz"
    d = co2io.load_trace_columns(f)
    t = d[:, 1] * AU_T_PS
    m = t <= T_MAX
    ax.plot(t[m], d[m, 2], color=color, ls=ls, lw=lw, label=label)

ax.set_xlim(0, T_MAX)
ax.set_ylim(-0.027, 0.027)
ax.set_xlabel(r"Time $t$ [ps]")
ax.set_ylabel(r"Dipole Moment $\mu_x$ [a.u.]")
ax.tick_params(direction="in", top=True, right=True)
ax.legend(frameon=False, loc="upper right", fontsize=8, handlelength=1.5,
          borderaxespad=0.2, labelspacing=0.3)
ax.text(0.03, 0.95, r"$\lambda=0.3$, $\chi$ included", transform=ax.transAxes,
        ha="left", va="top", fontsize=8)

fig.savefig(OUTDIR / "figS5_f2_term.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "figS5_f2_term.pdf", bbox_inches="tight")
print("figS5_f2_term written")
