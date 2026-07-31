#!/usr/bin/env python3
"""Fig 1: Ar dimer interaction energy outside and inside the cavity.

Panel (a): PES for lambda = 0 and for cavity polarisation along x, y, z, at
           coupling strengths 0.05 (opaque) and 0.07/0.09/0.11 (faded).
Panel (b): cavity-induced part of the interaction energy at lambda = 0.05,
           pMBD against QED-CCSD-2, with the MBD and MBD+GA references.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

P_Har = 2.0 * 13.60569193                    # Hartree -> eV
DATA = Path(os.environ.get("PMBD_DATA", "data"))
IMG = Path(os.environ.get("PMBD_IMAGES", "images"))

# ------------------------------------------------------------------- style
# The published figure was typeset with LaTeX. Set USETEX=off to fall back to
# matplotlib's own mathtext (curves identical, text metrics slightly different).
plt.rcParams.update({"font.size": 6})
if os.environ.get("USETEX", "on") != "off":
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

fig_width = 246.0 / 72.27                    # PRL single column, in inches
fig_height = fig_width * (np.sqrt(5) - 1.0) / 2.0
plt.rcParams.update({
    "axes.labelsize": 8, "font.size": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.figsize": [fig_width, fig_height],
})

# -------------------------------------------------------------------- data
def pmbd(lam):
    """R, E_outside, E_pol_x, E_pol_y, E_pol_z (total energies, Hartree)."""
    d = np.loadtxt(DATA / f"pmbd_ar_dimer_lambda_{lam}_summary.txt", skiprows=2)
    return d[:, 0], d[:, 1], d[:, 2], d[:, 3], d[:, 4]

def dE(y, ref=None):
    """Interaction energy in meV, referenced to the last point (R = 25 A)."""
    ref = y if ref is None else ref
    return (y - ref[-1]) * 1000 * P_Har

R, out, px, py, pz = pmbd("5e-2")

# QED-CCSD-2 total energies. The scan runs out to R = 500 A; the reference
# point used in the paper is row -6 (R = 25 A), so drop the longer tail.
cc = np.loadtxt(DATA / "qedcc_ar_dimer_summary.txt", skiprows=2)
R_cc, cc_ref = cc[:-6, 0], cc[-6]
cc_out, cc_x, cc_z = cc[:-6, 1], cc[:-6, 2], cc[:-6, 4]

# PBE0 / MBD / GA decomposition: R, E0, MBD, pMBD_x, pMBD_y, pMBD_z, GA
ga = np.loadtxt(DATA / "pmbd_ar_dimer_summary.txt", skiprows=2)
R_ga = ga[:, 0]
mbd_tot = ga[:, 1] + ga[:, 2]
ga_tot = mbd_tot + ga[:, 6]

# ------------------------------------------------------------------ figure
gs1 = gridspec.GridSpec(1, 2)
gs1.update(hspace=0.15, wspace=0.45, left=0.14, right=0.97, bottom=0.16, top=0.96)

# ---- (a) potential energy surfaces
ax1 = plt.subplot(gs1[0, 0])
ax1.hlines(y=0, xmin=3, xmax=25, linewidth=1, color="grey", linestyle="dashed")

ax1.plot(R, dE(out), color="black")
ax1.plot(R, dE(pz), color="red")
ax1.plot(R, dE(px), color="blue")
ax1.plot(R, dE(py), color="limegreen", dashes=[2, 2])

for lam in ("7e-2", "9e-2", "1.1e-1"):
    Rl, _, lx, ly, lz = pmbd(lam)
    ax1.plot(Rl, dE(lz), color="red", alpha=0.3)
    ax1.plot(Rl, dE(lx), color="blue", dashes=[0, 2, 2, 0], alpha=0.3)
    ax1.plot(Rl, dE(ly), color="limegreen", dashes=[2, 2], alpha=0.3)

ax1.set_xlim(3.2, 8)
ax1.set_ylim(-22, 15)
ax1.set_xticks([4, 5, 6, 7, 8])
ax1.set_xlabel(r"R$_z$ [$\AA$]", labelpad=0)
ax1.set_ylabel(r"$\Delta E(R)$ [meV]", labelpad=0)

# ---- (b) cavity-induced contribution
ax2 = plt.subplot(gs1[0, 1])
ax2.hlines(y=0, xmin=3, xmax=25, linewidth=1, color="grey", linestyle="dashed")

ax2.plot(R_ga, np.zeros_like(R_ga), label=r"PBE0-MBD", color="black")
ax2.plot(R_ga, dE(ga_tot) - dE(mbd_tot), label=r"PBE0-MBD+GA", color="grey")

ax2.plot(R, dE(pz) - dE(out), label=r"PBE0-pMBD: $\lambda_z$", color="red")
ax2.plot(R_cc, (cc_z - cc_ref[4] - cc_out + cc_ref[1]) * 1000 * P_Har,
         label=r"QED-CC:$\lambda_z$", color="red", ls="--")
ax2.plot(R, dE(px) - dE(out), label=r"PBE0-pMBD:$\lambda_x$", color="blue")
ax2.plot(R_cc, (cc_x - cc_ref[2] - cc_out + cc_ref[1]) * 1000 * P_Har,
         label=r"QED-CC:$\lambda_x$", color="blue", ls="--")

ax2.legend(loc="lower right", borderpad=0.5, fontsize=6)
ax2.set_xlim(3.2, 8)
ax2.set_ylim(-8, 7)
ax2.set_xticks([4, 5, 6, 7, 8])
ax2.set_xlabel(r"R$_z$ [$\AA$]", labelpad=0)
ax2.set_ylabel(r"$\Delta E(R) - \Delta E_{\lambda = 0}(R) $ [meV]", labelpad=0)

# ---- inset schematic and annotations
axin = ax1.inset_axes((2.5, 4.5, 10, 10), transform=ax1.transData)
axin.imshow(plt.imread(str(IMG / "ardimer3.png"), format="png"))
axin.axis("off")

ax1.text(7.3, 12.5, "Ar", zorder=10, fontsize=6)
ax1.text(7.3, 5.5, "Ar", zorder=10, fontsize=6)
ax1.annotate("", xy=(7.5, 12), xycoords="data", xytext=(7.5, 7), textcoords="data",
             arrowprops=dict(arrowstyle="<->", color="black", lw=0.7, ls="-"), zorder=10)
ax1.text(6.8, 9, r"$R_z$", zorder=10, fontsize=6)

ax1.text(4, 12, "(a)")
ax2.text(4.0, 6, r"(b) $\lambda = 0.05$")

shift = 6
ax1.annotate("", xy=(6, -12 - shift), xycoords="data", xytext=(7, -12 - shift),
             textcoords="data", arrowprops=dict(arrowstyle="<-", color="blue", lw=0.7))
ax1.annotate("", xy=(6, -12 - shift), xycoords="data", xytext=(6, -7 - shift),
             textcoords="data", arrowprops=dict(arrowstyle="<-", color="red", lw=0.7))
ax1.annotate("", xy=(6, -12 - shift), xycoords="data", xytext=(6.8, -10 - shift),
             textcoords="data", arrowprops=dict(arrowstyle="<-", color="limegreen", lw=0.7))
ax1.text(7.0, -12.4 - shift, "x", color="blue")
ax1.text(6.8, -10.4 - shift, "y", color="limegreen")
ax1.text(5.9, -6.9 - shift, "z", color="red")

ax1.text(4, -8.5, r"$\lambda=0$", rotation=55, fontsize=6)
ax1.annotate("", xy=(5, -6), xycoords="data", xytext=(5.5, -10), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="blue", lw=0.7))
ax1.annotate("", xy=(5, 0), xycoords="data", xytext=(5.5, 10), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="red", lw=0.7))
ax1.text(5.45, 10.1, r"$\lambda$", fontsize=6, color="red")
ax1.text(5.5, -10.1, r"$\lambda$", fontsize=6, color="blue")

OUT = Path("fig_1")
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    plt.savefig(OUT / f"fig_1.{ext}", dpi=200)
print("Saved -> fig_1/fig_1.pdf")
