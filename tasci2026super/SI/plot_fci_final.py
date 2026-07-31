#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- Dimensions & Style (matches main text figure exactly) ----------------
FIG_WIDTH  = 7.0
FIG_HEIGHT = 9.5
BASE = 14

mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "font.size": BASE,
    "axes.labelsize": BASE + 2,
    "legend.fontsize": BASE,
    "xtick.labelsize": BASE,
    "ytick.labelsize": BASE,
    "lines.linewidth": 2.8,
    "axes.linewidth": 1.8,
    "xtick.major.size": 6.0, "ytick.major.size": 6.0,
    "xtick.major.width": 1.5, "ytick.major.width": 1.5,
    "xtick.direction": "out", "ytick.direction": "out",
    "axes.grid": False,
})

# --- Physics Constants ---
OMEGA_EV = 2.0
EV_TO_HA = 0.03674932217
OMEGA_HA = OMEGA_EV * EV_TO_HA

# --- COLORS ---
COL = {
    "ref":  "#000000",  # Black  (FCI Reference)
    "dq":   "#1f77b4",  # Blue
    "dp":   "#d62728",  # Red
    "prod": "#2ca02c",  # Green
    "xi":   "#6a3d9a",  # Purple
    "ene":  "#e07b00",  # Orange
    "phn":  "#17becf",  # Teal
}

# ---------------- Hard-Coded Reference Data (FCI) ----------------
# Parsed from table.tsv: columns -> Natoms, E_out, E_in, <b†b>, ..., Q², P², Dq*Dp
N_REF    = np.array([1, 2, 3, 4])

E_OUT    = np.array([-2.8873650277470, -5.7747315797320,
                     -8.6620981550780, -11.5494647324720])
E_IN     = np.array([-2.8872180206850, -5.7744373290470,
                     -8.6616566455060, -11.5488759711690])
ENE_REF  = E_IN - E_OUT          # polaritonic energy shift (Ein - Eout)

PHN_REF  = np.array([0.001943961262, 0.003890742052,
                     0.005837499763, 0.007783946338])   # <b†b>

Q2_REF   = np.array([6.856507565,  6.910247719,  6.963987002,  7.017717159])
P2_REF   = np.array([0.03674519975, 0.0367410642, 0.03673692997, 0.03673279931])
PROD_REF = np.array([0.5019399765,  0.5038748407, 0.5058018415,  0.5077207855])

DQ_REF   = np.sqrt(Q2_REF)
DP_REF   = np.sqrt(P2_REF)
XI_REF   = 0.5 * np.log((DQ_REF * OMEGA_HA) / DP_REF)

# ---------------- Data Loading (p-MBD) ----------------
def load_pmbd_data():
    h5_path = "He_dist4_lam_0.025_results_4.h5"
    if not Path(h5_path).exists():
        return None

    res = {'n': [], 'dq': [], 'dp': [], 'prod': [], 'xi': [],
           'ene': [], 'phn': []}
    with h5py.File(h5_path, "r") as f:
        keys = sorted([k for k in f.keys() if k.startswith('N_')],
                      key=lambda x: int(x.split('_')[1]))
        for k in keys:
            res['n'].append(int(k.split('_')[1]))

            q2    = np.real(f[k]['pt_xx'][()])
            p2    = np.real(f[k]['pt_pp'][()])
            dq, dp = np.sqrt(q2), np.sqrt(p2)

            ene   = np.real(f[k]['ene'][()])
            ptexc = np.real(f[k]['ptexc'][()])
            phn   = np.real(f[k]['pt_number'][()])

            res['dq'].append(dq)
            res['dp'].append(dp)
            res['prod'].append(dq * dp)
            res['xi'].append(0.5 * np.log((dq * OMEGA_HA) / dp))
            res['ene'].append(ptexc - ene)   # polaritonic energy shift
            res['phn'].append(phn)

    return {k: np.array(v) for k, v in res.items()}

# ---------------- Helper: draw one panel ----------------
def draw_panel(ax, ref, pb, n_pb, label, color, marker, panel_idx,
               legend_loc="lower right"):
    ax.set_xticks(N_REF)
    ax.set_xlim(0.8, 4.2)
    ax.minorticks_off()
    ax.tick_params(which="both", direction="out", top=False, right=False)

    # FCI reference — solid line + filled black circles
    ax.plot(N_REF, ref, color=COL["ref"], ls="-", lw=2.8, zorder=1)
    ax.scatter(N_REF, ref, s=50, marker='o', color=COL["ref"],
               edgecolors="black", linewidths=0.2, label="QED-FCI", zorder=2)

    # p-MBD — solid line + filled colored circles
    ax.plot(n_pb, pb, color=color, ls="-", lw=2.8, zorder=3)
    ax.scatter(n_pb, pb, s=50, marker='o', color=color,
               edgecolors="black", linewidths=0.2, label="pMBD", zorder=4)

    ax.set_ylabel(label, labelpad=2)
    ax.set_xlabel("number of atoms")
    ax.legend(frameon=False, loc=legend_loc, fontsize=BASE)

    ax.text(0.06, 0.95, f"({chr(97 + panel_idx)})", transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=BASE + 2)

    y_min = min(np.min(ref), np.min(pb))
    y_max = max(np.max(ref), np.max(pb))
    margin = (y_max - y_min) * 0.5 if y_max != y_min else 0.001
    ax.set_ylim(y_min - margin, y_max + margin)

    # Force (0,0) origin for energy, photon number, squeezing
    if panel_idx in (0, 1, 5):
        ax.set_xlim(0, 4.2)
        ax.set_ylim(0, y_max * 1.15)
        ax.set_xticks([0, 1, 2, 3, 4])
        # Add (0,0) point to both series
        ref_with_zero = np.concatenate([[0], ref])
        pb_with_zero  = np.concatenate([[0], pb])
        n_ref_zero    = np.concatenate([[0], N_REF])
        n_pb_zero     = np.concatenate([[0], n_pb])
        ax.lines[0].set_xdata(n_ref_zero)
        ax.lines[0].set_ydata(ref_with_zero)
        ax.lines[1].set_xdata(n_pb_zero)
        ax.lines[1].set_ydata(pb_with_zero)
        # Add (0,0) scatter point for FCI and p-MBD
        ax.scatter([0], [0], s=50, marker='o', color=COL["ref"],
                   edgecolors="black", linewidths=0.2, zorder=2)
        ax.scatter([0], [0], s=50, marker='o', color=color,
                   edgecolors="black", linewidths=0.2, zorder=4)

# ---------------- Main Plotting ----------------
def main():
    dp = load_pmbd_data()
    if dp is None:
        print("Error: H5 result file not found.")
        return

    fig, axes = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
    axes = axes.flatten()

    # Order: energy, photon number, Dq, Dp, DqDp, squeezing
    panel_map = [
        (ENE_REF,  dp['ene'],
         r"$E(N)_\lambda - E(N)_{\lambda=0}$ [a.u.]", COL["ene"], 'p'),

        (PHN_REF,  dp['phn'],
         r"Photon number",    COL["phn"], 'h'),

        (DQ_REF,   dp['dq'],
         r"$\Delta q$ [a.u.] ",                               COL["dq"],  'o'),

        (DP_REF,   dp['dp'],
         r"$\Delta p$ [a.u.]",                               COL["dp"],  's'),

        (PROD_REF, dp['prod'],
         r"$\Delta q\,\Delta p$ [a.u.] ",                     COL["prod"],'D'),

        (XI_REF,   dp['xi'],
         r"Squeezing parameter ($r$)",                COL["xi"],  '^'),
    ]

    # Custom legend locations per panel (default: "lower right")
    legend_locs = {
        3: "lower center",   # panel (d): shift legend left
    }

    for i, (ref, pb, label, color, marker) in enumerate(panel_map):
        draw_panel(axes[i], ref, pb, dp['n'], label, color, marker, i,
                   legend_loc=legend_locs.get(i, "lower right"))

    plt.tight_layout()
    plt.savefig("benchmark_split_diagnostics.pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
