#!/usr/bin/env python3
"""Fig 2: benzene dimer and graphene bilayer interaction energies in a cavity."""

import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from pathlib import Path

P_Har = 2.0 * 13.60569193          # Hartree -> eV
DATA = Path(os.environ.get("PMBD_DATA", "data"))
IMG = Path(os.environ.get("PMBD_IMAGES", "images"))

plt.rcParams.update({"font.size": 6})
# Set USETEX=off to fall back to matplotlib mathtext when LaTeX is unavailable.
if os.environ.get("USETEX", "on") != "off":
    plt.rc("text", usetex=True)
    plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

fig_width_pt = 246.0
fig_width = fig_width_pt / 72.27
fig_height = fig_width * (np.sqrt(5) - 1.0) / 2.0

plt.rcParams.update({
    "axes.labelsize": 8, "font.size": 8, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "figure.figsize": [fig_width, fig_height],
})

# columns of data.dat, per polarisation v1/v2 and direction x/y/z:
#   0 R(A) | 1 E0 2 vdwene 3 pt (v1x) | 4,5,6 (v2x) | 7,8,9 (v1y) | 10,11,12 (v2y)
#          | 13,14,15 (v1z) | 16,17,18 (v2z)
data = np.loadtxt(DATA / "benzene_dimer_pmbd.dat")

nocav = data[:, 1] + data[:, 2] - data[-1, 1] - data[-1, 2]
cavxv1 = data[:, 1] + data[:, 2] + data[:, 3] - data[-1, 1] - data[-1, 2] - data[-1, 3]
cavyv1 = data[:, 7] + data[:, 8] + data[:, 9] - data[-1, 7] - data[-1, 8] - data[-1, 9]
cavzv1 = data[:, 13] + data[:, 14] + data[:, 15] - data[-1, 13] - data[-1, 14] - data[-1, 15]

gs1 = gridspec.GridSpec(1, 2)
gs1.update(hspace=0.02, wspace=0.3, left=0.16, right=0.97, bottom=0.16, top=0.96)

# ---------------------------------------------------------- (a) benzene dimer
ax1 = plt.subplot(gs1[0, 0])
ax1.hlines(y=0, xmin=0, xmax=25, linewidth=1, color="grey", linestyle="dashed")

ax1.plot(data[:, 0], nocav * P_Har, label=r"$\lambda_0$", color="black", lw=1.5)
ax1.plot(data[:, 0], cavxv1 * P_Har, label=r"$\lambda_x$", color="blue", lw=1.5)
ax1.plot(data[:, 0], cavyv1 * P_Har, label=r"$\lambda_y$", color="limegreen", lw=1.5)
ax1.plot(data[:, 0], cavzv1 * P_Har, label=r"$\lambda_z$", color="red", lw=1.5)

ax1.set_xlim(0, 8)
ax1.set_xlabel(r"$R_x$ [$\AA$]", labelpad=0)
ax1.set_ylabel(r"$\Delta E(R)$ [eV]", labelpad=0)
ax1.set_xticks([0, 2, 4, 6, 8])

arr_image = plt.imread(str(IMG / "benzene_gimp.png"), format="png")
axin = ax1.inset_axes((0.5, 0.5, 0.45, 0.45))
axin.imshow(arr_image)
axin.axis("off")

ax1.annotate("", xy=(5.2, 0.120), xycoords="data", xytext=(6.5, 0.120), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="black", lw=1, ls="-"), zorder=20)
ax1.text(6.6, 0.118, r"$R_x$", zorder=10, fontsize=6)
ax1.annotate("", xy=(5.7, 0.04), xycoords="data", xytext=(5.7, 0.085), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="black", lw=1, ls="-"), zorder=10)
ax1.text(4.5, 0.0650, r"$R_z$", zorder=10, fontsize=6)

# ------------------------------------------------------- (b) graphene bilayer
# 0 R | 1 E0 2 vdw 3 xcav | 4 E0 5 vdw 6 ycav | 7 E0 8 vdw 9 zcav
datag = np.loadtxt(DATA / "graphene_bilayer_pmbd.dat")

nocav_g = datag[:, 1] + datag[:, 2] - datag[-1, 1] - datag[-1, 2]
cavx = datag[:, 1] + datag[:, 2] + datag[:, 3] - datag[-1, 1] - datag[-1, 2] - datag[-1, 3]
cavy = datag[:, 1] + datag[:, 5] + datag[:, 6] - datag[-1, 1] - datag[-1, 5] - datag[-1, 6]
cavz = datag[:, 1] + datag[:, 8] + datag[:, 9] - datag[-1, 1] - datag[-1, 8] - datag[-1, 9]

ax2 = plt.subplot(gs1[0, 1])
ax2.plot(datag[:, 0], nocav_g * P_Har, label=r"$\lambda_0$", color="black", lw=1.5)
ax2.plot(datag[:, 0], cavz * P_Har, label=r"$\lambda_z$", color="red", lw=1.5)
ax2.plot(datag[:, 0], cavx * P_Har, label=r"$\lambda_x$", color="blue", lw=1.5)
ax2.plot(datag[:, 0], cavy * P_Har, "--", label=r"$\lambda_y$", color="limegreen",
         dashes=[2, 2], lw=1.5)

ax2.set_xlim(3, 6)
ax2.set_ylim(-2.3, 1)
ax2.set_xlabel(r"$R_z$ [$\AA$]", labelpad=0)
ax2.hlines(y=0, xmin=0, xmax=25, linewidth=1, color="grey", linestyle="dashed")

ax1.text(0.6, 0.120, r"(a) $\lambda = 0.05$")

ax1.annotate("", xy=(2, 0.01), xycoords="data", xytext=(3.8, 0.01), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="blue", lw=0.7))
ax1.annotate("", xy=(2, 0.01), xycoords="data", xytext=(2, 0.06), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="red", lw=0.7))
ax1.annotate("", xy=(3.2, 0.035), xycoords="data", xytext=(2, 0.01), textcoords="data",
             arrowprops=dict(arrowstyle="->", color="limegreen", lw=0.7))

arr_image = plt.imread(str(IMG / "bilayer_aa_25_plot.png"), format="png")
axin = ax2.inset_axes((0.48, -0.11, 0.54, 0.54), zorder=-10)
axin.imshow(arr_image)
axin.axis("off")

ax2.annotate("", xy=(4.6, -1.575), xycoords="data", xytext=(4.6, -1.975), textcoords="data",
             arrowprops=dict(arrowstyle="->", color="black", lw=1, ls="-"), zorder=20)
ax2.text(4.15, -2, r"$R_z$", zorder=20, fontsize=6)

ax1.text(3.6, 0.008, "x", color="blue")
ax1.text(1.9, 0.061, "z", color="red")
ax1.text(3.2, 0.035, "y", color="limegreen")

ax1.text(5.0, -0.13, r"$\lambda=0$", rotation=0, fontsize=6)
ax1.annotate("", xy=(4, -0.1), xycoords="data", xytext=(5, -0.125), textcoords="data",
             arrowprops=dict(arrowstyle="->", color="black", lw=1, ls="-"), zorder=20)

ax2.text(3.8, -0.5, r"$\lambda=0$", rotation=0, fontsize=6)
ax2.annotate("", xy=(4.2, -0.58), xycoords="data", xytext=(4.7, -1.16), textcoords="data",
             arrowprops=dict(arrowstyle="<-", color="black", lw=1, ls="-"), zorder=20)

ax2.text(3.4, 0.75, r"(b) $\lambda = 0.05$")

OUT = Path("fig_2")
OUT.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    plt.savefig(OUT / f"fig_2.{ext}", dpi=200)
print("Saved -> fig_2/fig_2.pdf")
