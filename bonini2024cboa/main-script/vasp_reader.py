#!/usr/bin/env python3
"""
Faithful drop-in replacement for qedmft.io.adiabatic_dat_from_p4vsp.

The original reader uses py4vasp (a custom fork that is not installable here)
to pull the structure, force-constant matrix, Born effective charges and the
clamped-ion (electronic) dielectric tensor out of a VASP calculation, and then
applies a fixed set of unit conversions to build a qedmft.adiabatic.AdiabaticMatter.

py4vasp simply reads these arrays out of `vaspout.h5`:
    results/positions/*                          -> structure (-> ASE)
    results/linear_response/force_constants      -> force_constants
    results/linear_response/born_charges         -> charge_tensors
    results/linear_response/electron_dielectric_tensor -> dielectric "clamped_ion"

This module reads exactly those datasets with h5py + ASE and reproduces the
identical conversions found in qedmft/io.py::adiabatic_dat_from_p4vsp. Only the
reader is replaced; the physics comes from the qedmft package itself, which is
not shipped here. See the folder README for where to get it.
"""
import os
import sys
import h5py
import numpy as np
from ase import Atoms

HERE = os.path.dirname(os.path.abspath(__file__))

QEDMFT_CLONE_HINT = """
The qedmft package (John Bonini) is not shipped with this dataset. Get it from

    git clone https://github.com/jrbp/qedmft

and make it importable in one of three ways:

    pip install ./qedmft                 # or  pip install -e ./qedmft
    export QEDMFT_SRC=/path/to/qedmft/src
    place the clone next to this dataset folder, as ../../qedmft

Only qedmft.adiabatic and qedmft.units are used. qedmft.io is never imported:
it pulls in py4vasp, which this reader exists to avoid.
"""


def _add_qedmft_to_path():
    """Make `qedmft` importable, from an install or from a source checkout."""
    try:
        import qedmft  # noqa: F401
        return
    except ImportError:
        pass
    candidates = []
    env = os.environ.get("QEDMFT_SRC")
    if env:
        candidates.append(env)
    # a clone sitting next to the dataset folder, or inside it
    candidates += [os.path.join(HERE, "../../../qedmft/src"),
                   os.path.join(HERE, "../../qedmft/src"),
                   os.path.join(HERE, "../qedmft/src")]
    for c in candidates:
        c = os.path.abspath(c)
        if os.path.isdir(os.path.join(c, "qedmft")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return
    raise ImportError("cannot import qedmft." + QEDMFT_CLONE_HINT)


_add_qedmft_to_path()

from qedmft.adiabatic import AdiabaticMatter            # noqa: E402
from qedmft.units import ANG_PER_BOHR, EV_PER_HARTREE, MASS_AMU_FACT  # noqa: E402


def _structure_from_h5(h5):
    """Rebuild the ASE Atoms object exactly as py4vasp's structure.to_ase()."""
    g = h5["results/positions"]
    scale = g["scale"][()]
    cell = scale * g["lattice_vectors"][()]
    frac = g["position_ions"][()]
    ion_types = [t.decode().strip() for t in g["ion_types"][()]]
    counts = g["number_ion_types"][()]
    symbols = []
    for sym, n in zip(ion_types, counts):
        symbols += [sym] * int(n)
    atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
    return atoms


def adiabatic_dat_from_vaspout(path, enforce_asr=False):
    """Mirror of qedmft.io.adiabatic_dat_from_p4vsp using h5py instead of py4vasp."""
    h5_path = os.path.join(path, "vaspout.h5")
    with h5py.File(h5_path, "r") as h5:
        ase_struct = _structure_from_h5(h5)
        lr = h5["results/linear_response"]
        force_constants = lr["force_constants"][()]
        born = lr["born_charges"][()]
        eps_clamped = lr["electron_dielectric_tensor"][()]

    vol_au = ase_struct.cell.volume / ANG_PER_BOHR ** 3
    res = {}
    res["calc_dir"] = path
    res["species"] = ase_struct.get_chemical_symbols()
    res["cell"] = np.asarray(ase_struct.cell) / ANG_PER_BOHR
    res["coords"] = ase_struct.get_scaled_positions()
    res["masses"] = ase_struct.get_masses() * MASS_AMU_FACT
    # same sign + unit conversion as adiabatic_dat_from_p4vsp
    cmat_raw = -force_constants * (ANG_PER_BOHR ** 2 / EV_PER_HARTREE)
    if enforce_asr:
        from qedmft.io import apply_asr_correction  # noqa
        res["cmat"] = apply_asr_correction(cmat_raw)
    else:
        res["cmat"] = cmat_raw
    res["zs"] = born
    res["eps_3D"] = eps_clamped
    res["chi0"] = vol_au * (eps_clamped - np.eye(3)) / (4 * np.pi)
    return AdiabaticMatter(**res)


if __name__ == "__main__":
    from qedmft import adiabatic as qad
    from qedmft.units import EV_PER_HARTREE as EVH
    path = sys.argv[1] if len(sys.argv) > 1 else (
        os.path.join(HERE, "../matter-data/vasp-dfpt/CO2_relax"))
    dat = adiabatic_dat_from_vaspout(path)
    freqs = qad.hmat_to_freqs(dat.cmat, dat.masses_flat)
    print("species:", dat.species)
    print("phonon freqs (meV):", np.round(freqs * EVH * 1e3, 3))
