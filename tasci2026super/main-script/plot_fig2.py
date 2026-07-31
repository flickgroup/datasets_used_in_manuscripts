#!/usr/bin/env python3
"""Fig 2: Quantum uncertainty & Wigner functions — Ar chains (λ=0.025)"""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
from pathlib import Path

PMBD_FILE = Path("pmbd_lam0025.h5")
OUTDIR    = Path("fig_2")
BASE      = 14

OMEGA  = 2.0 * 0.03674930495120813   # 2 eV in Hartree
COLORS = {"prod": "#1505FC", "dx": "#44BA09", "dp": "#FC0505", "r": "k"}

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

def setup_ax(ax):
    ax.minorticks_off()
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.set_xticks([1, 25, 50, 75, 100]); ax.set_xlim(1, 100)

def wigner(dx, dp, omega=OMEGA, lim=3.0, npts=200):
    """Gaussian Wigner function in dimensionless coordinates (fixed box)."""
    sig_q, sig_p = dx * np.sqrt(omega), dp / np.sqrt(omega)
    q = np.linspace(-lim, lim, npts)
    Q, P = np.meshgrid(q, q)
    W = np.exp(-0.5 * ((Q/sig_q)**2 + (P/sig_p)**2))
    return W / W.max(), [-lim, lim, -lim, lim]

# ── Load data ──────────────────────────────────────────────────────────────────
with h5py.File(PMBD_FILE) as f:
    idx   = np.argsort(f["N"])
    Ns    = np.array(f["N"])[idx]
    Dx    = np.sqrt(np.real(np.array(f["pt_xx"]))[idx])
    Dp    = np.sqrt(np.real(np.array(f["pt_pp"]))[idx])

mask = np.isfinite(Dx) & np.isfinite(Dp) & (Dx > 0) & (Dp > 0)
Ns, Dx, Dp = Ns[mask], Dx[mask], Dp[mask]
Prod = Dx * Dp
r    = 0.5 * np.log(Dx * OMEGA / Dp)

every10 = [i for i, n in enumerate(Ns) if n % 10 == 0 and n > 0]

OUTDIR.mkdir(parents=True, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.33),
                                gridspec_kw={"wspace": 0.4})

# ── Panel A: uncertainty ───────────────────────────────────────────────────────
setup_ax(ax1)
ax1.set_yscale("log")
ax1.plot(Ns, Prod, color=COLORS["prod"], marker="o", ms=7, markevery=every10, label=r"$\Delta q \Delta p$")
ax1.plot(Ns, Dx,   color=COLORS["dx"],   marker="s", ms=7, markevery=every10, label=r"$\Delta q$")
ax1.plot(Ns, Dp,   color=COLORS["dp"],   marker="^", ms=7, markevery=every10, label=r"$\Delta p$")
ax1.axhline(0.5, color="0.35", ls=(0, (4, 2)), lw=1.8, label="0.5")

ax1.set_xlabel("number of atoms")
ax1.set_ylabel("Uncertainty [a.u.] (log scale)")
ax1.legend(frameon=False, loc="center right", bbox_to_anchor=(0.5, 0.6), fontsize=12)

ax1.set_yticks([0.15, 0.2, 0.5, 1, 2, 3, 6])
ax1.yaxis.set_major_formatter(FuncFormatter(
    lambda x, _: {0.15: "0.15", 0.2: "0.2"}.get(round(x, 2), f"{x:g}")))
ax1.minorticks_off()
ax1.text(0.05, 0.95, "(a)", transform=ax1.transAxes, fontweight="bold", va="top")

# ── Panel B: squeezing + Wigner insets ────────────────────────────────────────
setup_ax(ax2)
ax2.plot(Ns, r, color=COLORS["r"], marker="o", ms=7, markevery=every10, label="pMBD")
ax2.axhline(0.0, color="0.35", ls=(0, (4, 2)), lw=1.8, label="photon-GA")
ax2.set_xlabel("number of atoms")
ax2.set_ylabel(r"Squeezing parameter ($r$)")
ax2.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.99), fontsize=12)
ax2.text(0.75, 0.98, "(b)", transform=ax2.transAxes, fontweight="bold", va="top")

inset_positions = [[0.25, 0.053, 0.30, 0.30],
                   [0.50, 0.28,  0.30, 0.30],
                   [0.69, 0.53,  0.30, 0.30]]

for k, (pos, N_target) in enumerate(zip(inset_positions, [1, 50, 100])):
    i = int(np.argmin(np.abs(Ns - N_target)))
    W, ext = wigner(Dx[i], Dp[i])

    axw = ax2.inset_axes(pos)
    im = axw.imshow(W, extent=ext, origin="lower", cmap="viridis",
                    aspect="equal", vmin=0, vmax=1)
    axw.set_title(rf"$N={int(Ns[i])}$", fontsize=12, pad=2)
    for sp in axw.spines.values():
        sp.set_linewidth(0.8)

    if k == 0:
        axw.set_xlabel(r"$\tilde{q}$", fontsize=12, labelpad=-1)
        axw.set_ylabel(r"$\tilde{p}$", fontsize=12, labelpad=-5)
        axw.tick_params(labelsize=9, pad=1, length=2)
        axw.set_yticks([-2, 0, 2])
        axw.xaxis.set_major_locator(MaxNLocator(nbins=3))
    else:
        axw.set_xticks([]); axw.set_yticks([])

cax = ax2.inset_axes([1.03, 0.2, 0.04, 0.6])
cbar = fig.colorbar(im, cax=cax)
cbar.set_ticks([0, 0.5, 1])
cbar.ax.tick_params(labelsize=12)
cbar.set_label(r"$W(\tilde{q}, \tilde{p})$", fontsize=13, labelpad=1)

plt.subplots_adjust(left=0.12, right=0.90, top=0.95, bottom=0.15, wspace=0.4)

for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"fig_2.{ext}")
print(f"Saved → {OUTDIR}/fig_2.pdf")
