#!/usr/bin/env python3
"""Identify the resonant IR-active mode (largest |mode effective charge|) and
its polarization axis for each system, to set the cavity photon parameters."""
import os
import numpy as np
from vasp_reader import adiabatic_dat_from_vaspout
from qedmft import adiabatic as qad
from qedmft.units import EV_PER_HARTREE

DATA = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../matter-data/vasp-dfpt"))

# note: this one deliberately looks at the *relaxed* CO2 run as well, which is
# why CO2_relax is shipped; the figures all use the unrelaxed geometry.
SYSTEMS = {
    "CO2 (norelax, used by the figures)": "CO2",
    "CO2 (relax)":                        "CO2_relax",
    "FeCO5":                              "FeCO5",
    "hBN":                                "hBN",
    "HfS2":                               "HfS2",
}

for name, sub in SYSTEMS.items():
    path = os.path.join(DATA, sub)
    dat = adiabatic_dat_from_vaspout(path)
    # free-matter normal modes
    freqs, eigdispls = qad.hmat_to_freqs(dat.cmat, dat.masses_flat,
                                         freq_only=False)
    # mode effective charge vector per mode: sum_I Z*_I . e_k(I)
    # eigdispls: (nmode, 3N) ; zmat: (3N, 3)
    Zmode = eigdispls @ dat.zmat            # (nmode, 3)
    Znorm = np.sqrt((Zmode ** 2).sum(-1))
    fmeV = freqs * EV_PER_HARTREE * 1e3
    print(f"\n=== {name}  ({sub})  natoms={dat.natoms} ===")
    print("species:", dat.species)
    order = np.argsort(-Znorm)
    print(" idx   freq(meV)    |Z*|      Z*_x      Z*_y      Z*_z")
    for k in order[:8]:
        print(f" {k:3d}  {fmeV[k]:9.2f}  {Znorm[k]:8.3f}  "
              f"{Zmode[k,0]:8.3f}  {Zmode[k,1]:8.3f}  {Zmode[k,2]:8.3f}")
    kres = order[0]
    print(f" -> resonant mode {kres}: {fmeV[kres]:.2f} meV "
          f"({freqs[kres]:.6f} Ha), dominant axis "
          f"{['x','y','z'][np.argmax(np.abs(Zmode[kres]))]}")
