"""SI Fig. S6: parity plots for the liquid CO2 DP and Deep Tensor models.

Reads the `dp test` output of the three PBE0-D3 models shipped under
models/liquid_dp/, re-encoded as float32 .npz (the force and atomic-dipole
tables are 56 MB of ASCII each in their original form). Table S2 of the SI
quotes the RMSE and MAE printed at the end of this script.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import co2io  # noqa: E402

DATA = REPO / "data"


plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

PARITY = DATA / "liquid" / "dp_parity"
out_dir = Path(__file__).resolve().parent

N_ATOMS = 192


def rmse(pred, ref):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.sqrt(np.mean((pred - ref) ** 2)))


def mae(pred, ref):
    pred = np.asarray(pred)
    ref = np.asarray(ref)
    return float(np.mean(np.abs(pred - ref)))


def set_parity_limits(ax, x, y, pad_fraction=0.06):
    lo = min(np.nanmin(x), np.nanmin(y))
    hi = max(np.nanmax(x), np.nanmax(y))
    pad = pad_fraction * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad
    (diag_line,) = ax.plot([lo, hi], [lo, hi], color="0.25", lw=1.0, ls="--")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    return diag_line


def parity_panel(ax, x, y, xlabel, ylabel, title, color="#2f6f9f", label=None):
    ax.scatter(x, y, s=5, alpha=0.28, color=color, edgecolors="none", rasterized=True, label=label)
    set_parity_limits(ax, x, y)
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, pad=4)
    ax.tick_params(direction="in", pad=2)


def set_matching_ticks(ax, ticks, limit_pad=0.0):
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0], min(ticks) - limit_pad)
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1], max(ticks) + limit_pad)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    if ax.lines:
        ax.lines[0].set_data([lo, hi], [lo, hi])
    ax.tick_params(direction="in", length=4, pad=2)


def component_panel(ax, ref, pred, components, colors, xlabel, ylabel, title, scale=1.0):
    all_x = []
    all_y = []
    for name, idx in components:
        x = np.ravel(ref[..., idx]) * scale
        y = np.ravel(pred[..., idx]) * scale
        all_x.append(x)
        all_y.append(y)
        ax.scatter(
            x,
            y,
            s=5,
            alpha=0.25,
            color=colors[name],
            edgecolors="none",
            rasterized=True,
            label=rf"${name}$",
        )
    set_parity_limits(ax, np.concatenate(all_x), np.concatenate(all_y))
    ax.set_xlabel(xlabel, labelpad=2)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.set_title(title, pad=4)
    ax.tick_params(direction="in", pad=2)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.2, borderaxespad=0.2)


# Energy and force model.
energy_raw = co2io.load_parity(PARITY / "ener_energy.npz")
force_raw = co2io.load_parity(PARITY / "ener_force.npz")
energy_shift = np.mean(energy_raw[:, 0])
energy_ref = (energy_raw[:, 0] - energy_shift) * 1000 / N_ATOMS
energy_pred = (energy_raw[:, 1] - energy_shift) * 1000 / N_ATOMS
force_ref = force_raw[:, 0:3] * 1000
force_pred = force_raw[:, 3:6] * 1000

# Deep Tensor true-dipole model. The first 64 atoms are C atoms with zero labels;
# the O-site labels sum to the molecular dipole.
dipole_raw = co2io.load_parity(PARITY / "dipole_atomic.npz")
n_dipole = dipole_raw.shape[1] // 2
dipole_ref_all = dipole_raw[:, :n_dipole].reshape(dipole_raw.shape[0], -1, 3)
dipole_pred_all = dipole_raw[:, n_dipole:].reshape(dipole_raw.shape[0], -1, 3)
dipole_ref = dipole_ref_all[:, 64:, :]
dipole_pred = dipole_pred_all[:, 64:, :]
global_dipole_ref = dipole_ref_all.sum(axis=1)
global_dipole_pred = dipole_pred_all.sum(axis=1)

# Global polarizability model.
polar_raw = co2io.load_parity(PARITY / "polar_global.npz")
n_polar = polar_raw.shape[1] // 2
polar_ref = polar_raw[:, :n_polar].reshape(polar_raw.shape[0], 3, 3)
polar_pred = polar_raw[:, n_polar:].reshape(polar_raw.shape[0], 3, 3)

metrics = {
    "energy_rmse_mev_atom": rmse(energy_pred, energy_ref),
    "energy_mae_mev_atom": mae(energy_pred, energy_ref),
    "force_rmse_mev_a": rmse(force_pred, force_ref),
    "force_mae_mev_a": mae(force_pred, force_ref),
    "atomic_dipole_rmse_ea": rmse(dipole_pred, dipole_ref),
    "atomic_dipole_mae_ea": mae(dipole_pred, dipole_ref),
    "global_dipole_rmse_ea": rmse(global_dipole_pred, global_dipole_ref),
    "global_dipole_mae_ea": mae(global_dipole_pred, global_dipole_ref),
    "polar_rmse_a3": rmse(polar_pred, polar_ref),
    "polar_mae_a3": mae(polar_pred, polar_ref),
}

component_colors = {
    "x": "#2f6f9f",
    "y": "#c44e52",
    "z": "#55a868",
    "xx": "#2f6f9f",
    "yy": "#c44e52",
    "zz": "#55a868",
    "xy": "#2f6f9f",
    "xz": "#c44e52",
    "yz": "#55a868",
}

fig, axes = plt.subplots(3, 2, figsize=(7.2, 8.6))
axes = axes.ravel()

parity_panel(
    axes[0],
    energy_ref,
    energy_pred,
    "DFT relative energy [meV/atom]",
    "DP relative energy [meV/atom]",
    "Energy",
)
set_matching_ticks(axes[0], [-20, 0, 20])

component_panel(
    axes[1],
    force_ref,
    force_pred,
    [("x", 0), ("y", 1), ("z", 2)],
    component_colors,
    r"DFT force [meV $\mathrm{\AA}^{-1}$]",
    r"DP force [meV $\mathrm{\AA}^{-1}$]",
    "Forces",
)
set_matching_ticks(axes[1], [-5000, 0, 5000])

component_panel(
    axes[2],
    dipole_ref,
    dipole_pred,
    [("x", 0), ("y", 1), ("z", 2)],
    component_colors,
    r"DFT O-site dipole [$e\,\mathrm{\AA}$]",
    r"DP O-site dipole [$e\,\mathrm{\AA}$]",
    "O-site atomic dipole",
)
set_matching_ticks(axes[2], [-1, 0, 1], limit_pad=0.06)

parity_panel(
    axes[3],
    polar_ref[:, 0, 0],
    polar_pred[:, 0, 0],
    r"DFT $\chi_{xx}$ [$\mathrm{\AA}^3$]",
    r"DP $\chi_{xx}$ [$\mathrm{\AA}^3$]",
    r"$\chi_{xx}$",
)
set_matching_ticks(axes[3], [140, 170, 200], limit_pad=2.0)

component_panel(
    axes[4],
    polar_ref.reshape(-1, 9),
    polar_pred.reshape(-1, 9),
    [("yy", 4), ("zz", 8)],
    component_colors,
    r"DFT $\chi_{yy},\chi_{zz}$ [$\mathrm{\AA}^3$]",
    r"DP $\chi_{yy},\chi_{zz}$ [$\mathrm{\AA}^3$]",
    r"$\chi_{yy}$ and $\chi_{zz}$",
)
set_matching_ticks(axes[4], [140, 170, 200], limit_pad=2.0)

component_panel(
    axes[5],
    polar_ref.reshape(-1, 9),
    polar_pred.reshape(-1, 9),
    [("xy", 1), ("xz", 2), ("yz", 5)],
    component_colors,
    r"DFT off-diagonal $\chi$ [$\mathrm{\AA}^3$]",
    r"DP off-diagonal $\chi$ [$\mathrm{\AA}^3$]",
    r"Off-diagonal $\chi$",
)
set_matching_ticks(axes[5], [-20, 0, 20], limit_pad=1.0)

for label, ax in zip(["a", "b", "c", "d", "e", "f"], axes):
    ax.text(
        -0.20,
        1.12,
        label,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        clip_on=False,
    )

fig.tight_layout(w_pad=0.8, h_pad=0.25)
fig.savefig(out_dir / "figS6_liquid_dp_parity.png", dpi=600, bbox_inches="tight")
fig.savefig(out_dir / "figS6_liquid_dp_parity.pdf", bbox_inches="tight")

for key, value in metrics.items():
    print(f"{key}: {value:.10g}")
