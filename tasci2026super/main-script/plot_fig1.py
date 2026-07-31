#!/usr/bin/env python3
"""Fig 1: p-MBD vs RPA orders vs VV10 — Ar chains (λ=0.025)"""

import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path

VV10_FILE = Path("ar_chain_lambda_0025_vv10_R4_N100_GAUGE_VG.h5")
PMBD_FILE = Path("pmbd_lam0025.h5")
OUTDIR    = Path("fig_1")

SKIP_N     = {6, 77, 95}
MARK_EVERY = set(range(10, 10001, 10))
BASE       = 14

COLORS = {"vv10": "#7d411b", "pmbd": "#000000",
          1: "#0d1bda", 2: "#f72525", 10: "#1e8728"}
STYLES = {1: dict(ls="-",           lw=2.8, zorder=3),
          2: dict(ls=(0, (8, 2.5)), lw=3.6, zorder=4),
         10: dict(ls="--",          lw=3.8, zorder=6)}

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

def clean(Ns, Ys):
    Ns, Ys = np.asarray(Ns), np.asarray(Ys)
    mask = ~np.isin(Ns, list(SKIP_N))
    return Ns[mask], Ys[mask]

def markers(ax, Ns, Ys, color, hollow=False, size=50, zorder=6):
    Ns, Ys = np.asarray(Ns), np.asarray(Ys)
    m = np.isin(Ns, list(MARK_EVERY))
    if hollow:
        ax.scatter(Ns[m], Ys[m], s=size, marker='o', facecolors="none",
                   edgecolors=color, linewidths=1.2, zorder=zorder)
    else:
        ax.scatter(Ns[m], Ys[m], s=size, marker='o', color=color,
                   edgecolors="black", linewidths=0.2, zorder=zorder)

def read_vv10(path):
    exc, pn = {}, {}
    with h5py.File(path) as f:
        for k in (k for k in f if k.startswith("n_")):
            n, g = int(k.split("_")[1]), f[k]
            if "exc"     in g: exc[n] = float(g["exc"][()].flat[0])
            if "pnumber" in g: pn[n]  = float(g["pnumber"][()].flat[0])
    for d in (exc, pn):
        ks = sorted(d); d["Ns"] = np.array(ks); d["Vs"] = np.array([d.pop(k) for k in ks])
    for d in (exc, pn):
        d["Ns"], d["Vs"] = clean(d["Ns"], d["Vs"])
    return exc, pn

def read_pmbd(path, orders=(1, 2, 10)):
    with h5py.File(path) as f:
        idx    = np.argsort(f["N"])
        Ns     = np.array(f["N"])[idx]
        pt_exc = np.array(f["pt_exc"])[idx]
        ene    = np.array(f["ene"])[idx]
        pt_rpa = np.array(f["pt_rpa_orders"])[idx]   # (nN, max_order)
        rpa_s  = np.array(f["rpa_orders"])[idx]
        pn     = np.array(f["pt_number"])[idx] if "pt_number" in f else None

    eph_Ns, eph_Vs = clean(Ns, pt_exc - ene)

    rpa = {}
    for o in orders:
        diffs = pt_rpa[:, :o+1].sum(axis=1) - rpa_s[:, :o+1].sum(axis=1)
        rpa[o] = clean(Ns, diffs)

    pn_data = clean(Ns, pn) if pn is not None else ([], [])

    # debug table
    print(f"\n{'N':<5}  {'ene':>14}  {'ΣRPA-1':>14}  {'ΣRPA-2':>14}  {'ΣRPA-10':>14}")
    for n in [1, 25, 50, 75, 100]:
        i = np.searchsorted(Ns, n)
        if i < len(Ns) and Ns[i] == n:
            sums = [rpa_s[i, :o+1].sum() for o in orders]
            print(f"{n:<5}  {ene[i]:14.6e}  {sums[0]:14.6e}  {sums[1]:14.6e}  {sums[2]:14.6e}")

    return (eph_Ns, eph_Vs), rpa, pn_data

def setup_ax(ax):
    ax.minorticks_off()
    ax.tick_params(which="both", direction="out", top=False, right=False)
    ax.set_xticks([1, 25, 50, 75, 100]); ax.set_xlim(1, 100)

# ── Main ───────────────────────────────────────────────────────────────────────
OUTDIR.mkdir(parents=True, exist_ok=True)
vv10_exc, vv10_pn = read_vv10(VV10_FILE)
(pNs, pVs), rpa, (pnNs, pnVs) = read_pmbd(PMBD_FILE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 4.33),
                                gridspec_kw={"wspace": 0.4})

# ── Panel A: energy ────────────────────────────────────────────────────────────
setup_ax(ax1)
ax1.plot(*vv10_exc.values(), color=COLORS["vv10"], zorder=4, label="VV10")
markers(ax1, *vv10_exc.values(), COLORS["vv10"])

for o in (1, 2, 10):
    Ns_o, Vs_o = rpa[o]
    st = STYLES[o]
    ln, = ax1.plot(Ns_o, Vs_o, color=COLORS[o], label=f"RPA-{o}", **st)
    if o == 2:
        ln.set_path_effects([pe.Stroke(linewidth=st["lw"]+0.8,
                                       foreground="white", alpha=0.9), pe.Normal()])
    markers(ax1, Ns_o, Vs_o, COLORS[o], hollow=(o == 10), zorder=st["zorder"]+1)

ax1.plot(pNs, pVs, color=COLORS["pmbd"], lw=3.8, zorder=5, label="p-MBD")
markers(ax1, pNs, pVs, COLORS["pmbd"], zorder=7)

ax1.set_ylim(bottom=0)
ax1.set_xlabel("number of atoms")
ax1.set_ylabel(r"$E(N)_\lambda - E(N)_{\lambda=0}$ [a.u.]", labelpad=2)

# legend in custom order
lmap = dict(zip(*ax1.get_legend_handles_labels()[::-1]))
order = ["p-MBD", "RPA-1", "RPA-2", "RPA-10", "VV10"]
names = {"p-MBD": "pMBD", "RPA-1": "1st", "RPA-2": "2nd", "RPA-10": "10th", "VV10": "photon-GA"}
ax1.legend([lmap[k] for k in order], [names[k] for k in order],
           frameon=False, loc="lower left", bbox_to_anchor=(0.02, 0.52),
           handlelength=1.4, handletextpad=0.5, fontsize=12)

# inset
ax_in = inset_axes(ax1, width="40%", height="40%", loc=3,
                   bbox_to_anchor=(0.68, 0.13, 0.5, 0.5), bbox_transform=ax1.transAxes)
ax_in.minorticks_off(); ax_in.tick_params(which="both", direction="out", labelsize=12)
for o in (2, 10):
    Ns_o, Vs_o = rpa[o]; m = Ns_o >= 80
    st = STYLES[o]
    ln, = ax_in.plot(Ns_o[m], Vs_o[m], color=COLORS[o], **st)
    if o == 2:
        ln.set_path_effects([pe.Stroke(linewidth=st["lw"]+0.8,
                                       foreground="white", alpha=0.9), pe.Normal()])
    markers(ax_in, Ns_o[m], Vs_o[m], COLORS[o], hollow=(o==10), size=35, zorder=st["zorder"]+1)
ax_in.plot(pNs[pNs >= 80], pVs[pNs >= 80], color=COLORS["pmbd"], lw=3.0, zorder=5)
markers(ax_in, pNs[pNs >= 80], pVs[pNs >= 80], COLORS["pmbd"], size=35, zorder=7)
ax_in.set_xlim(90, 100); ax_in.set_ylim(0.09, 0.10)

# ── Panel B: photon number ─────────────────────────────────────────────────────
setup_ax(ax2)
ax2.plot(*vv10_pn.values(), color=COLORS["vv10"], label="VV10")
markers(ax2, *vv10_pn.values(), COLORS["vv10"])
ax2.plot(pnNs, pnVs, color=COLORS["pmbd"], label="p-MBD")
markers(ax2, pnNs, pnVs, COLORS["pmbd"])
ax2.set_ylim(bottom=0)
ax2.set_xlabel("number of atoms")
ax2.set_ylabel("Photon number", labelpad=6)

# panel labels
for ax, lbl in [(ax1, "a"), (ax2, "b")]:
    ax.text(0.06, 0.95, f"({lbl})", transform=ax.transAxes,
            ha="left", va="top", fontsize=BASE+2, fontweight="bold")

for ext in ("pdf", "svg", "png"):
    fig.savefig(OUTDIR / f"fig_1.{ext}", bbox_inches="tight")
plt.close(fig)
print(f"\nSaved → {OUTDIR}/fig_1.pdf")
