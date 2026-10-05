#!/usr/bin/env python3
"""Fig. S4: AB bilayer graphene binding curves at equal λ/ω.

E(R) - E(10 Å) in meV/atom at λ = 0.1 a.u., ω = 10 eV and at λ = 0.3 a.u.,
ω = 30 eV, against the curve outside the cavity, λ = 0.

Run from inside SI/:  python plot_figS4.py  ->  fig_S4/fig_S4.{png,pdf}
"""

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "main-script"))
import pmbd_bilayer as pb  # noqa: E402

DATA = Path(__file__).resolve().parent / "data" / "lambda_omega_ratio.h5"

# These runs sample a 37 x 37 x 1 q-grid, so `ene` is already the converged MBD
# energy and `ptexc - ene` is the Γ-point cavity term divided by N_q = 37².
# Rescaling by 37² / 15 divides it by N_c = 15 instead, as in the main-text runs.
PT_SCALE = 37**2 / 15

CURVES = ((10, "royalblue", r"$\lambda_\alpha$=0.1 a.u., $\omega$=10eV"),
          (30, "red", r"$\lambda_\alpha$=0.3 a.u., $\omega$=30eV"))


def total_ev(e_dft, ene, ptexc):
    return e_dft + ene * pb.HARTREE_EV + PT_SCALE * (ptexc - ene) * pb.HARTREE_EV


curves = []
with h5py.File(DATA, "r") as f:
    r = f["r"][:]
    for omega, color, label in CURVES:
        g, ref = f[f"omega{omega}"], f[f"omega{omega}/reference"]
        e = total_ev(g["E_dft"][:], g["ene"][:], g["ptexc"][:])
        e_ref = total_ev(ref["E_dft"][()], ref["ene"][()], ref["ptexc"][()])
        curves.append((r, (e - e_ref) * 1000 / 4, color, label))

bare = pb.load("graphene_ab")
curves.append((bare.r, bare.curve_mev_per_atom(0.0), "black", r"$\lambda_\alpha$=0 a.u."))

pb.use_style()
fig, ax = plt.subplots(figsize=(5.2, 7))
for x, y, color, label in curves:
    ax.plot(x, y, "-", color=color, linewidth=2.2, label=label)

ax.set_xlabel(r"Interlayer Distance R [$\mathrm{\AA}$]", fontsize=20)
ax.set_ylabel("Energy E(R) [meV/atom]", fontsize=20)
ax.set_xlim(3.0, 5.0)
ax.set_xticks([3.0, 3.5, 4.0, 4.5, 5.0])
ax.tick_params(labelsize=17)
ax.legend(loc="upper right", frameon=False, fontsize=14, handlelength=1.6,
          labelspacing=0.4, borderaxespad=0.6)

pb.add_schematic(ax, "graphene_ab", (3.58, -10.7), 0.13)

pb.save(fig, "fig_S4", "fig_S4")
plt.close(fig)

for x, y, _, label in curves:
    i = y.argmin()
    print(f"{label:45s} minimum {y[i]:7.2f} meV/atom at R = {x[i]:.4f} A")
