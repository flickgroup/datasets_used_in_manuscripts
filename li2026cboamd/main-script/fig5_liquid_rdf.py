"""Manuscript Fig. 5: liquid CO2 radial distribution functions in the cavity.

Style/setup matches fig3_liquid_spectrum_lambdas.py / fig4_liquid_rabi_peak_and_
splitting.py (fonts, sizes, output naming). Three panels: g_CC, g_CO, g_OO for
  - out of cavity (lambda = 0),                       red, solid
  - chi neglected, fixed omega_c, lambda = 0.06,      blue, dashed
  - chi included, omega_c/n_eff, lambda = 0.06,       green, dotted
RDF data are the rdf_s1_*.dat files produced by postprocess/plot_rdf_s1.py
(columns: r[A], g_CC, g_CO, g_OO; identical r grids).
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
OFF = DATA / "liquid" / "rdf" / "offresonant"
RES = DATA / "liquid" / "rdf" / "resonant"
OUTDIR = ROOT

colors = {"nocav": "#cc163a", "neg": "#1177b0", "res": "#1b7837"}
styles = {"nocav": ("-", 1.8), "neg": ("--", 1.3), "res": (":", 1.6)}

DATASETS = [
    ("nocav", OFF / "rdf_s1_10_out_of_cavity_ir_40traj_newmodels.npz",
     r"no cavity"),
    ("neg", OFF / "rdf_s1_22_cavity_ir_40traj_newmodels_nopolar_lam006.npz",
     r"$\chi$ neglected, $\lambda=0.06$"),
    ("res", RES / "rdf_s1_28_cavity_ir_40traj_newmodels_polar_lam006.npz",
     r"$\chi$ included, $\omega_c/n_{\mathrm{eff}}$," "\n" r"$\lambda=0.06$"),
]
PAIRS = [(r"C$-$C", 1), (r"C$-$O", 2), (r"O$-$O", 3)]

data = {key: co2io.load_rdf_columns(path) for key, path, _ in DATASETS}
ref = data["nocav"]

fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.289), sharex=True, sharey=True,
                         gridspec_kw={"wspace": 0.10})

for icol, (pair, col) in enumerate(PAIRS):
    ax = axes[icol]
    for key, path, label in DATASETS:
        d = data[key]
        ls, lw = styles[key]
        ax.plot(d[:, 0], d[:, col], color=colors[key], lw=lw, ls=ls,
                label=label if icol == 2 else None)
    ax.set_xlim(2.0, 7.0)
    ax.set_ylim(-0.05, 2.1)
    ax.set_xticks([2, 3, 4, 5, 6, 7])
    ax.tick_params(direction="in", top=True, right=True)
    ax.set_xlabel(r"Distance $r$ [$\mathring{\mathrm{A}}$]")
    ax.text(0.95, 0.90, pair, transform=ax.transAxes, ha="right", va="center",
            fontsize=10)

axes[0].set_ylabel(r"RDF $g(r)$")
# legend inside panel c along the bottom left: anchored just right of the
# intramolecular O-O spike (r ~ 2.3) and below the main O-O structure,
# which stays above g ~ 0.9 for r > 3
axes[2].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.16, 0.02),
               fontsize=8, handlelength=1.1, borderaxespad=0.3,
               labelspacing=0.3, handletextpad=0.4)

# panel letters inside the top-left corner; "c" shifted right past the
# intramolecular O-O spike at r ~ 2.3 A
for ax, letter, x in zip(axes, "abc", (0.04, 0.04, 0.13)):
    ax.text(x, 0.94, letter, transform=ax.transAxes, fontsize=14,
            fontfamily="cmb10", ha="left", va="top")

fig.savefig(OUTDIR / "fig5_liquid_rdf.png", dpi=600, bbox_inches="tight")
fig.savefig(OUTDIR / "fig5_liquid_rdf.pdf", bbox_inches="tight")

print("max |Delta g(r)| (r in [2, 7] A)")
m = (ref[:, 0] >= 2.0) & (ref[:, 0] <= 7.0)
for key, path, _ in DATASETS[1:]:
    d = data[key]
    for pair, col in PAIRS:
        dev = np.abs(d[m, col] - ref[m, col]).max()
        print(f"  {key:5s} {pair}: {dev:.4f}")
