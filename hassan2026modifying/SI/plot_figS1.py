#!/usr/bin/env python3
"""Fig. S1: unit cell on an N x N x 1 q-grid against an N x N x 1 supercell at Γ.

Cavity-mediated energy ΔE_xc = ptexc - ene of AA bilayer graphene at
R = 3.6425 Å, ω = 2 eV, one out-of-plane mode, in meV/atom. The supercell runs
couple at λ = 0.1 a.u., the unit-cell runs at sqrt(N_c) x 0.1 a.u. The two
coincide for every N.

Run from inside SI/:  python plot_figS1.py  ->  fig_S1/fig_S1.{png,pdf}
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main-script"))
import pmbd_bilayer as pb  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "unitcell_supercell.h5"

# ptexc - ene is in Hartree per simulation cell: 4 atoms for the unit cell,
# 4 N² for the supercell.
with h5py.File(DATA, "r") as f:
    n = f["N"][:]
    unit = ((f["unitcell/ptexc"][:] - f["unitcell/ene"][:])
            * pb.HARTREE_EV * 1000 / f["unitcell/n_atoms"][:])
    sup = ((f["supercell/ptexc"][:] - f["supercell/ene"][:])
           * pb.HARTREE_EV * 1000 / f["supercell/n_atoms"][:])

pb.use_style()
fig, ax1 = plt.subplots(figsize=(8, 9))
line1, = ax1.plot(n, unit, "bo-", linewidth=2, markersize=7,
                  label=r"Unit cell with $\lambda_{\alpha, \mathrm{eff}} = \sqrt{N_c} \times 0.1$ a.u.")
ax1.set_xlabel(r"N $\times$ N $\times$ 1 $\bf{q}$-points", color="blue", fontsize=20)
ax1.set_xticks(n)
ax1.tick_params(axis="x", labelcolor="blue", labelsize=17)
ax1.set_ylabel(r"$\Delta E_\mathrm{xc}$ [meV/atom]", fontsize=20)
ax1.tick_params(axis="y", labelsize=17)

ax2 = ax1.twiny()
line2, = ax2.plot(n, sup, "ro--", linewidth=2, markersize=4,
                  label=r"Supercell with $\lambda_{\alpha, \mathrm{eff}} = 0.1$ a.u.")
ax2.set_xticks(n)
ax2.tick_params(axis="x", labelcolor="red", labelsize=17)
ax2.set_xlabel(r"N $\times$ N $\times$ 1 Supercell", color="red", fontsize=20)

ax1.legend([line1, line2], [line1.get_label(), line2.get_label()], loc="best",
           fontsize=17)

# Insets and the arrows pointing both at the N = 5 point, in data coordinates.
pb.add_schematic(ax1, "graphene_5x5_supercell", (10.6, 157.35), 0.25)
ax1.annotate("", xy=(5.3, 132.0), xytext=(7.5, 151.79),
             arrowprops=dict(arrowstyle="->, head_width=0.2", color="black", lw=2))
pb.add_schematic(ax2, "graphene_unitcell", (3.5, 98.11), 0.14)
ax2.annotate("", xy=(4.7, 128.0), xytext=(2.8, 114.77),
             arrowprops=dict(arrowstyle="->, head_width=0.2", color="black", lw=2))

pb.save(fig, "fig_S1", "fig_S1")
plt.close(fig)

print(f"{'N':>3}{'unit cell':>14}{'supercell':>14}   (meV/atom)")
for k, u, s in zip(n, unit, sup):
    print(f"{k:3d}{u:14.4f}{s:14.4f}")
print(f"max |unit - super| = {np.max(np.abs(unit - sup)):.2e} meV/atom")
