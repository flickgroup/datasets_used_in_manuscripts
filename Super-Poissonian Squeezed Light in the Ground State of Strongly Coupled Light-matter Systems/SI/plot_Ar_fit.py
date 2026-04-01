#!/usr/bin/env python3
"""Ar chain scaling: energy & photon number up to N=1000."""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt

BASE = 14
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix", "figure.dpi": 150, "savefig.dpi": 600,
    "font.size": BASE, "axes.labelsize": BASE+2, "legend.fontsize": BASE,
    "xtick.labelsize": BASE, "ytick.labelsize": BASE,
    "lines.linewidth": 2.8, "axes.linewidth": 1.8,
    "xtick.major.size": 6, "ytick.major.size": 6,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
})

PLOT_NS   = set(range(1, 101, 10)) | set(range(100, 1001, 20))
EXCLUDE_NS = {160, 660, 800}

# ── Load data ──────────────────────────────────────────────────────────────────
ns, enes, phns, rpa1s = [], [], [], []
with h5py.File("Ar_scaling_fit.h5") as f:
    for key in sorted(f.keys(), key=lambda k: int(k.split("_")[1])):
        n = int(key.split("_")[1])
        if n not in PLOT_NS or n in EXCLUDE_NS:
            continue
        g = f[key]
        ns.append(n)
        enes.append(float(np.real(g["ene"][()])))
        phns.append(float(np.real(g["phn"][()])))
        if "pt_rpa_orders" in g:
            o = np.real(g["pt_rpa_orders"][()])
            rpa1s.append(float(o[0] + o[1]))
        else:
            rpa1s.append(np.nan)

x    = np.array(ns,   float)
y_e  = np.array(enes, float)
y_p  = np.array(phns, float)
y_r1 = np.array(rpa1s, float)

ok_r1 = ~np.isnan(y_r1)
x_r1, y_r1 = x[ok_r1], y_r1[ok_r1]

# ── Linear fit (forced through origin, N≤100) ──────────────────────────────────
def fit_slope(xd, yd): return np.dot(xd, yd) / np.dot(xd, xd)

slope_e = fit_slope(x_r1[x_r1 <= 100], y_r1[x_r1 <= 100])   # from RPA-1
slope_p = fit_slope(x[x <= 100],       y_p[x <= 100])

x_fit = np.linspace(0, x.max(), 800)

# ── Figure ────────────────────────────────────────────────────────────────────
fig, (ax_e, ax_p) = plt.subplots(1, 2, figsize=(7.0, 4.5))
RPA_COL = "#e07b00"

for ax, xd, yd, slope, ylabel, color, label in [
    (ax_e, x,    y_e, slope_e, r"$E(N)_\lambda - E(N)_{\lambda=0}$ [a.u.]", "crimson",  "(a)"),
    (ax_p, x,    y_p, slope_p, "Photon Number",                              "#17becf",  "(b)"),
]:
    ax.scatter(xd, yd, color=color, s=60, zorder=3,
               edgecolors="k", linewidths=0.6, label="pMBD")

    if label == "(b)":
        ax.plot(x_fit[x_fit <= 100], slope * x_fit[x_fit <= 100],
                "k-", lw=1.8, zorder=2,
                label=f"Linear fit ($N\\leq100$)\nslope = {slope:.3e}")
        ax.plot(x_fit[x_fit >= 100], slope * x_fit[x_fit >= 100],
                color="gray", ls="--", lw=1.5, zorder=2, alpha=0.7,
                label="Linear extrapolation")

    # sublinear annotation at N≈800
    i800 = np.argmin(np.abs(xd - 800))
    xa, ya = xd[i800], yd[i800]
    ax.annotate("", xy=(xa, ya), xytext=(xa, slope*xa),
                arrowprops=dict(arrowstyle="<->", color="navy", lw=1.5))
    ax.text(xa+20, 0.5*(ya + slope*xa), "sublinear\ndeviation",
            color="navy", fontsize=10, va="center", ha="left")

    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_xlabel("Number of atoms ($N$)")
    ax.minorticks_off()
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.set_xlim(1, xd.max()*1.05)
    ax.set_ylim(0, None)
    ax.set_xticks([1] + list(range(200, int(xd.max())+1, 200)))
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.text(0.97, 0.05, label, transform=ax.transAxes,
            fontweight="bold", fontsize=BASE+2, ha="right", va="bottom")

    # inset: N≤100
    ax_in = ax.inset_axes([0.50, 0.08, 0.32, 0.18])
    m100  = xd <= 100
    ax_in.scatter(xd[m100], yd[m100], color=color, s=25,
                  edgecolors="k", linewidths=0.4)
    ax_in.plot(np.linspace(0, 100, 300), slope * np.linspace(0, 100, 300),
               "k-", lw=2.0)
    if label == "(a)":
        m100_r1 = x_r1 <= 100
        ax_in.scatter(x_r1[m100_r1], y_r1[m100_r1], color=RPA_COL, s=25,
                      edgecolors="k", linewidths=0.4)
        ax_in.plot(np.linspace(0, 100, 300), slope_e * np.linspace(0, 100, 300),
                   color=RPA_COL, lw=2.0)
    ax_in.set_xlim(0, 105); ax_in.set_xticks([0, 50, 100])
    ax_in.tick_params(labelsize=8)
    for sp in ax_in.spines.values(): sp.set_linewidth(1.0)

# energy panel: add RPA-1 on top (needs its own ylim and legend)
ax_e.set_ylim(0, y_r1.max() * 1.08)
ax_e.scatter(x_r1, y_r1, color=RPA_COL, s=60, edgecolors="k",
             linewidths=0.6, zorder=4, label="1st order RPA")
ax_e.plot(x_fit, slope_e * x_fit, color=RPA_COL, lw=2.8, zorder=5,
          label=f"Linear fit ($N\\leq100$)\nslope = {slope_e:.3e}")
ax_e.legend(frameon=False, fontsize=11, loc="upper left")

plt.tight_layout(w_pad=0.5)
for ext in ("pdf", "png"):
    plt.savefig(f"Ar_scaling_publication.{ext}", bbox_inches="tight",
                **({"dpi": 300} if ext == "png" else {}))
print("Saved: Ar_scaling_publication.pdf / .png")
