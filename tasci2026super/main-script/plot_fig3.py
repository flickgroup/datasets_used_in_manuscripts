#!/usr/bin/env python3
"""Fig 3: Photon statistics & entropy — Ar chains (λ=0.025)"""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

PMBD_FILE = Path("pmbd_lam0025.h5")
VV10_FILE = Path("ar_chain_lambda_0025_vv10_R4_N100_GAUGE_VG.h5")
OUTDIR    = Path("fig_3")
BASE      = 14

OMEGA = 2.0 * 0.03674930495120813   # 2 eV in Hartree

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "stix", "figure.dpi": 150, "savefig.dpi": 600,
    "font.size": BASE, "axes.labelsize": BASE+1, "legend.fontsize": 10,
    "xtick.labelsize": BASE, "ytick.labelsize": BASE,
    "lines.linewidth": 2.5, "axes.linewidth": 1.5,
    "xtick.direction": "out", "ytick.direction": "out", "axes.grid": False,
})

def entropy_gaussian(xx, pp, omega=OMEGA):
    """Von Neumann entropy from quadrature variances (Gaussian state)."""
    nu = 2.0 * np.sqrt(xx * omega * pp / omega)
    S  = np.where(nu > 1 + 1e-9,
                  ((nu+1)/2)*np.log((nu+1)/2) - ((nu-1)/2)*np.log((nu-1)/2), 0.0)
    return S

def entropy_photon_ga(n):
    """Photon-GA Von Neumann entropy from the 2x2 reduced density matrix.
       S = -(1-n)*ln(1-n) - n*ln(n)
       Only valid for n in [0, 1]; returns NaN outside this range.
    """
    n = np.asarray(n, float)
    S = np.full_like(n, np.nan)
    valid = (n >= 0) & (n <= 1.0)
    nv = n[valid]
    s  = np.zeros(nv.shape)
    m2 = nv > 1e-12
    s[m2] -= nv[m2] * np.log(nv[m2])
    S[valid] = s
    return S

def make_markevery(nbar, nbar_arr, dense_thresh=0.3, dense_step=1, sparse_step=5):
    """Return index list: every dense_step point where nbar < dense_thresh,
    every sparse_step elsewhere."""
    indices = []
    for i, n in enumerate(nbar_arr):
        if n < dense_thresh:
            if i % dense_step == 0:
                indices.append(i)
        else:
            if i % sparse_step == 0:
                indices.append(i)
    return indices

# ── Load p-MBD ────────────────────────────────────────────────────────────────
with h5py.File(PMBD_FILE) as f:
    idx    = np.argsort(f["N"])
    Ns_p   = np.array(f["N"])[idx]
    nbar_p = np.real(np.array(f["pt_number"]))[idx]
    adaa_p = np.real(np.array(f["pt_adadaa"]))[idx]
    xx_p   = np.real(np.array(f["pt_xx"]))[idx]
    pp_p   = np.real(np.array(f["pt_pp"]))[idx]

ok = np.isfinite(nbar_p) & np.isfinite(adaa_p)
Ns_p, nbar_p, adaa_p, xx_p, pp_p = Ns_p[ok], nbar_p[ok], adaa_p[ok], xx_p[ok], pp_p[ok]

varn_p = adaa_p + nbar_p - nbar_p**2
Q_p    = np.where(nbar_p > 1e-9, (adaa_p - nbar_p**2) / nbar_p, 0.0)
S_p    = entropy_gaussian(xx_p, pp_p)

# ── Load VV10 (photon-GA, <a†a†aa>=0 assumption) ─────────────────────────────
Ns_v, nbar_v = [], []
with h5py.File(VV10_FILE) as f:
    for k in sorted((k for k in f if k.startswith("n_")),
                    key=lambda s: int(s.split("_")[1])):
        g = f[k]
        if "pnumber" in g:
            Ns_v.append(int(k.split("_")[1]))
            nbar_v.append(float(g["pnumber"][()].flat[0]))

Ns_v, nbar_v = np.array(Ns_v), np.array(nbar_v)
varn_v = nbar_v - nbar_v**2
Q_v    = -nbar_v
S_v    = entropy_photon_ga(nbar_v)   # NaN where nbar_v > 1

# ── Figure ────────────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.33),
                                gridspec_kw={"wspace": 0.35})

# Panel A: markers every 5 atoms
every5_p = [i for i, n in enumerate(Ns_p) if n % 5 == 0 and n > 0]
every5_v = [i for i, n in enumerate(Ns_v) if n % 5 == 0 and n > 0]

# Panel B: adaptive marker density — every point below n<0.3, every 5 above
valid_v   = nbar_v <= 1.0
nbar_v_pl = nbar_v[valid_v]
S_v_pl    = S_v[valid_v]
mark_v    = make_markevery(None, nbar_v_pl, dense_thresh=0.3, dense_step=3, sparse_step=5)
mark_p    = make_markevery(None, nbar_p,    dense_thresh=0.3, dense_step=3, sparse_step=5)

# ── Panel A: statistics ───────────────────────────────────────────────────────
ax1.plot(Ns_p, Q_p,    color="#1f77b4", marker="s", ms=6, markevery=every5_p, label="Q (pMBD)")
ax1.plot(Ns_v, Q_v,    color="#6baed6", ls="--", lw=2.4,                      label="Q (photon-GA)")
ax1.plot(Ns_p, varn_p, color="#d62728", marker="o", ms=6, markevery=every5_p, label=r"$\Delta n^2$ (pMBD)")
ax1.plot(Ns_v, varn_v, color="#ff6b6b", ls="--", lw=2.4,                      label=r"$\Delta n^2$ (photon-GA)")
ax1.axhline(0, color="0.4", ls=":", lw=1.8, label="Poisson")

ax1.set_xlabel("number of atoms")
ax1.set_ylabel(r"Mandel $Q$, $\Delta n^2$")
ax1.set_xticks([1, 25, 50, 75, 100]); ax1.set_xlim(1, 100); ax1.set_ylim(-2, 5)
ax1.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.75, 0.98),
           fontsize=10, handletextpad=0.4, labelspacing=0.1)
ax1.text(0.96, 0.96, "(a)", transform=ax1.transAxes, fontweight="bold", va="top", ha="right")

# ── Panel B: entropy vs photon number ────────────────────────────────────────
# Thermal reference curve (guard n=0 to avoid 0*log(0) = nan)
n_grid = np.linspace(0, max(nbar_p.max(), 1.05), 300)
S_thermal = np.zeros_like(n_grid)
m = n_grid > 1e-9
S_thermal[m] = (n_grid[m]+1)*np.log(n_grid[m]+1) - n_grid[m]*np.log(n_grid[m])
ax2.plot(n_grid, S_thermal, "k--", lw=1.8, label="Thermal limit")

# photon-GA: valid domain only (nbar <= 1), denser markers at low n
ax2.plot(nbar_v_pl, S_v_pl, "-o", color="red", ms=5, alpha=0.8,
         markevery=mark_v, lw=1.8, label="photon-GA")

# pMBD: Gaussian entropy, no restriction, denser markers at low n
ax2.plot(nbar_p, S_p, "-D", color="black", ms=5,
         markevery=mark_p, lw=1.8, label="pMBD")

ax2.set_xlabel(r"Photon number")
ax2.set_ylabel(r"Entropy $S_{vN}$")
ax2.set_xlim(left=0); ax2.set_ylim(bottom=0)
ax2.legend(frameon=False, loc="upper left", fontsize=12)
ax2.text(0.88, 0.96, "(b)", transform=ax2.transAxes, fontweight="bold", va="top", ha="right")

for ext in ("pdf", "png"):
    fig.savefig(OUTDIR / f"fig_3_final_ticks.{ext}", bbox_inches="tight")
print(f"Saved → {OUTDIR}/fig_3_final_ticks.pdf")
