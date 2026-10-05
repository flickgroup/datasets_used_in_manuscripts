#!/usr/bin/env python3
"""Consolidate the raw periodic-pMBD run trees into the HDF5 files the figures read.

The raw runs are one JSON file per (stacking, coupling strength, interlayer
distance): about 56 000 files for the six binding curves. Everything the figures
need is a handful of scalars from each, so this collapses them into

    data/<system>.h5                    one per bilayer, Figs. 1, 2, S2, S3, S4
    SI/data/unitcell_supercell.h5       Fig. S1
    SI/data/lambda_omega_ratio.h5       Fig. S4

Every value written is asserted bit-identical to the JSON it came from, and every
value dropped because it repeats across coupling strengths is asserted to repeat.

Usage:

    python build_dataset.py RAW_DIR

RAW_DIR is a directory holding the expanded run trees

    hbn_aa/  hbn_ab/  hbn_aap/  hbn_ab1/  graphene_aa/  graphene_ab/
    hbn_aa_offset/ ... graphene_ab_offset/
    unitcell_pmbd/  supercell_pmbd/  lambda_omega_ratio/

as provided by the authors. Each
`<system>/lam<λ>/` directory holds `pmbd_om_ev2_r<R>.json` for every interlayer
distance, and `<system>/lam0.0/` additionally holds `mbd_energy_converged.txt`.
The `_offset` trees are the same bilayer cell with the two layers pulled 10 Å
apart, one single distance each, and provide the E(R = 10 Å) reference of the
binding curves.
"""

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np

HERE = Path(__file__).resolve().parent
MAIN_DATA = HERE.parent / "data"
SI_DATA = HERE.parent / "SI" / "data"

# Interlayer distances scanned, in Angstrom. The grid is uniform, so it is
# regenerated here rather than trusted from the file names.
R_GRID = np.round(np.arange(3.0, 5.1 + 0.001, 0.0025), 4)

# Coupling strengths in atomic units.
LAMBDAS = np.round(np.arange(0.0, 0.2 + 0.001, 0.02), 2)

REFERENCE_R_ANG = 10.0  # layer separation of the `_offset` runs

SYSTEMS = {
    "hbn_aa":      dict(material="hBN",      stacking="AA"),
    "hbn_ab":      dict(material="hBN",      stacking="AB"),
    "hbn_aap":     dict(material="hBN",      stacking="AA'"),
    "hbn_ab1":     dict(material="hBN",      stacking="AB1"),
    "graphene_aa": dict(material="graphene", stacking="AA"),
    "graphene_ab": dict(material="graphene", stacking="AB"),
}

# Fig. S1: N x N q-grids on the unit cell against N x N supercells at Γ.
SUPERCELL_N = np.arange(1, 15 + 1, 2)

# Fig. S4: AB graphene at two (ω in eV, λ in a.u.) pairs of equal ratio.
OMEGA_LAMBDA = ((10, 0.10), (30, 0.30))

# `mbd_energy_converged.txt` lists `r: <R>, mbd_energy: <E>` lines, with a
# handful of distances repeated from restarted runs.
MBD_LINE = re.compile(r"r:\s*(\d+\.?\d*)\s*,\s*mbd_energy:\s*(-?\d+\.?\d*)")

GZIP = dict(compression="gzip", compression_opts=4)


# The run directories and files were named from Python's float repr, so
# `str(float(...))` is what reproduces them: 'lam0.0', 'lam0.1', 'r3.0',
# 'r3.0025'. Anything smarter (%g, %.4f) gets the trailing zeros wrong.
def lam_dir(lam):
    """Directory name for a coupling strength: 0.0 -> 'lam0.0', 0.2 -> 'lam0.2'."""
    return f"lam{float(lam)}"


def r_name(r):
    """File name for an interlayer distance: 3.5 -> 'pmbd_om_ev2_r3.5.json'."""
    return f"pmbd_om_ev2_r{float(r)}.json"


def read_converged_mbd(path, r_grid):
    """MBD energies on `r_grid`, in Hartree, from a mbd_energy_converged.txt.

    The file is appended to by every run, so a later line supersedes an earlier
    one for the same R.
    """
    table = {}
    for r, e in MBD_LINE.findall(path.read_text()):
        r, e = float(r), float(e)
        if table.get(r, e) != e:
            print(f"  {path.parent.parent.name}: R = {r}, superseded "
                  f"{table[r]!r} by the later {e!r}")
        table[r] = e
    missing = [r for r in r_grid if float(r) not in table]
    if missing:
        raise ValueError(f"{path}: no MBD energy for R = {missing[:5]} ...")
    return np.array([table[float(r)] for r in r_grid])


def read_pmbd_json(path):
    """The numbers the figures need out of one pMBD output file."""
    d = json.loads(path.read_text())
    return dict(
        ene=d["ene"],                            # uncoupled MBD correlation energy, Ha
        ptexc=d["ptexc"],                        # coupled pMBD xc energy, Ha
        # `AT_energy` and `pt_AT_energy` are also present in the raw output but
        # are identically zero everywhere, so the Axilrod-Teller three-body term
        # is not carried into the dataset.
        at=(d["AT_energy"], d["pt_AT_energy"]),
        e_dft=d["json"]["total_energy"],         # VASP PBE total energy, eV
        ratio=d["json"]["ratio"],                # Hirshfeld volume ratios, per atom
        species=d["json"]["atomic_species"],
    )


def read_all(paths):
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(read_pmbd_json, paths))


def check_lossless(recs, paths, ene, ptexc, e_dft, ratio, species):
    """Every carried value equals its source; every dropped one is redundant."""
    for rec, path, a, b, c, d in zip(recs, paths, ene, ptexc, e_dft, ratio):
        if rec["ene"] != a or rec["ptexc"] != b:
            raise AssertionError(f"{path}: pMBD energies did not survive the copy")
        if rec["e_dft"] != c or rec["ratio"] != list(d):
            raise AssertionError(f"{path}: DFT energy or Hirshfeld ratios differ "
                                 "from the lam0.0 run they were collapsed onto")
        if rec["species"] != species:
            raise AssertionError(f"{path}: atomic species differ")
        if rec["at"] != (0, 0):
            raise AssertionError(f"{path}: nonzero Axilrod-Teller energy")


def build_bilayer(raw, name, meta):
    run = raw / name
    ref_run = raw / f"{name}_offset"
    for d in (run, ref_run):
        if not d.is_dir():
            raise FileNotFoundError(d)

    e_mbd = read_converged_mbd(run / "lam0.0" / "mbd_energy_converged.txt", R_GRID)
    e_mbd_ref = read_converged_mbd(
        ref_run / "lam0.0" / "mbd_energy_converged.txt", np.array([0.0])
    )[0]

    n_lam, n_r = len(LAMBDAS), len(R_GRID)
    paths = [run / lam_dir(lam) / r_name(r) for lam in LAMBDAS for r in R_GRID]
    recs = read_all(paths)
    ene = np.array([rec["ene"] for rec in recs]).reshape(n_lam, n_r)
    ptexc = np.array([rec["ptexc"] for rec in recs]).reshape(n_lam, n_r)

    # The DFT total energy and the Hirshfeld ratios come out of VASP and do not
    # depend on the coupling, so only the lam0.0 sweep of them is kept.
    species = recs[0]["species"]
    e_dft = np.array([rec["e_dft"] for rec in recs[:n_r]])
    ratio = np.array([rec["ratio"] for rec in recs[:n_r]])

    ref_paths = [ref_run / lam_dir(lam) / "pmbd_om_ev2_r0.json" for lam in LAMBDAS]
    ref = read_all(ref_paths)
    ene_ref = np.array([rec["ene"] for rec in ref])
    ptexc_ref = np.array([rec["ptexc"] for rec in ref])
    e_dft_ref = ref[0]["e_dft"]
    ratio_ref = np.array(ref[0]["ratio"])

    out = MAIN_DATA / f"{name}.h5"
    with h5py.File(out, "w") as f:
        f.attrs["material"] = meta["material"]
        f.attrs["stacking"] = meta["stacking"]
        f.attrs["atomic_species"] = species
        f.attrs["n_atoms"] = len(species)
        f.attrs["xc"] = "PBE"
        f.attrs["kpoints"] = [31, 31, 1]
        f.attrs["omega_ev"] = 2.0
        f.attrs["polarization"] = "z (out of plane)"
        f.attrs["reference_r_ang"] = REFERENCE_R_ANG
        f.attrs["units"] = ("r in Angstrom, lambda in a.u., E_dft in eV, "
                            "all pMBD energies in Hartree")

        f.create_dataset("r", data=R_GRID, **GZIP)
        f.create_dataset("lambda", data=LAMBDAS, **GZIP)
        f.create_dataset("E_dft", data=e_dft, **GZIP)
        f.create_dataset("E_mbd", data=e_mbd, **GZIP)
        f.create_dataset("ene", data=ene, **GZIP)
        f.create_dataset("ptexc", data=ptexc, **GZIP)
        f.create_dataset("hirshfeld_ratio", data=ratio, **GZIP)

        g = f.create_group("reference")
        g.attrs["r_ang"] = REFERENCE_R_ANG
        g.create_dataset("E_dft", data=e_dft_ref)
        g.create_dataset("E_mbd", data=e_mbd_ref)
        g.create_dataset("ene", data=ene_ref, **GZIP)
        g.create_dataset("ptexc", data=ptexc_ref, **GZIP)

    with h5py.File(out, "r") as f:
        e_dft_all = np.tile(f["E_dft"][:], n_lam)
        ratio_all = np.tile(f["hirshfeld_ratio"][:], (n_lam, 1))
        check_lossless(recs, paths, f["ene"][:].ravel(), f["ptexc"][:].ravel(),
                       e_dft_all, ratio_all, species)
        g = f["reference"]
        check_lossless(ref, ref_paths, g["ene"][:], g["ptexc"][:],
                       np.full(n_lam, g["E_dft"][()]),
                       np.tile(ratio_ref, (n_lam, 1)), species)
        if not (np.array_equal(f["E_mbd"][:], e_mbd)
                and g["E_mbd"][()] == e_mbd_ref):
            raise AssertionError(f"{out}: converged MBD energies did not survive")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB, "
          f"{n_lam} couplings x {n_r} distances, lossless)")


def build_unitcell_supercell(raw):
    unit_paths = [raw / "unitcell_pmbd" / f"pmbd_om_ev2_unitcell{n}x{n}.json"
                  for n in SUPERCELL_N]
    super_paths = [raw / "supercell_pmbd" / f"pmbd_om_ev2_supercell{n}x{n}.json"
                   for n in SUPERCELL_N]
    unit = [json.loads(p.read_text()) for p in unit_paths]
    sup = [json.loads(p.read_text()) for p in super_paths]

    out = SI_DATA / "unitcell_supercell.h5"
    with h5py.File(out, "w") as f:
        f.attrs["material"] = "graphene"
        f.attrs["stacking"] = "AA"
        f.attrs["r_ang"] = 3.6425
        f.attrs["omega_ev"] = 2.0
        f.attrs["lambda_supercell"] = 0.1
        f.attrs["units"] = "pMBD energies in Hartree, per simulation cell"
        f.create_dataset("N", data=SUPERCELL_N)
        for key, recs in (("unitcell", unit), ("supercell", sup)):
            g = f.create_group(key)
            g.create_dataset("ene", data=[d["ene"] for d in recs])
            g.create_dataset("ptexc", data=[d["ptexc"] for d in recs])
            g.create_dataset("n_atoms",
                             data=[len(d["json"]["atomic_species"]) for d in recs])

    with h5py.File(out, "r") as f:
        for key, recs, paths in (("unitcell", unit, unit_paths),
                                 ("supercell", sup, super_paths)):
            g = f[key]
            for i, (d, p) in enumerate(zip(recs, paths)):
                if g["ene"][i] != d["ene"] or g["ptexc"][i] != d["ptexc"]:
                    raise AssertionError(f"{p}: pMBD energies did not survive")
                if (d["AT_energy"], d["pt_AT_energy"]) != (0, 0):
                    raise AssertionError(f"{p}: nonzero Axilrod-Teller energy")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB, "
          f"N = {SUPERCELL_N.tolist()}, lossless)")


def build_lambda_omega_ratio(raw):
    out = SI_DATA / "lambda_omega_ratio.h5"
    sources = {}
    with h5py.File(out, "w") as f:
        f.attrs["material"] = "graphene"
        f.attrs["stacking"] = "AB"
        f.attrs["qgrid"] = [37, 37, 1]
        f.attrs["reference_r_ang"] = REFERENCE_R_ANG
        f.attrs["units"] = ("r in Angstrom, lambda in a.u., E_dft in eV, "
                            "all pMBD energies in Hartree")
        f.create_dataset("r", data=R_GRID, **GZIP)
        for omega, lam in OMEGA_LAMBDA:
            run = raw / "lambda_omega_ratio" / f"om{omega}" / f"lam{lam:.2f}"
            paths = [run / f"pmbd_om_ev{omega}_r{float(r)}.json" for r in R_GRID]
            ref_path = run / f"pmbd_om_ev{omega}_r10.json"
            recs, ref = read_all(paths), read_pmbd_json(ref_path)
            sources[omega] = (recs, paths, ref, ref_path)

            g = f.create_group(f"omega{omega}")
            g.attrs["omega_ev"] = float(omega)
            g.attrs["lambda"] = lam
            g.create_dataset("E_dft", data=[d["e_dft"] for d in recs], **GZIP)
            g.create_dataset("ene", data=[d["ene"] for d in recs], **GZIP)
            g.create_dataset("ptexc", data=[d["ptexc"] for d in recs], **GZIP)
            gr = g.create_group("reference")
            for key, value in (("E_dft", ref["e_dft"]), ("ene", ref["ene"]),
                               ("ptexc", ref["ptexc"])):
                gr.create_dataset(key, data=value)

    with h5py.File(out, "r") as f:
        for omega, (recs, paths, ref, ref_path) in sources.items():
            g = f[f"omega{omega}"]
            for i, (d, p) in enumerate(zip(recs, paths)):
                if (g["E_dft"][i], g["ene"][i], g["ptexc"][i]) != (d["e_dft"], d["ene"], d["ptexc"]):
                    raise AssertionError(f"{p}: energies did not survive the copy")
                if d["at"] != (0, 0):
                    raise AssertionError(f"{p}: nonzero Axilrod-Teller energy")
            gr = g["reference"]
            if (gr["E_dft"][()], gr["ene"][()], gr["ptexc"][()]) != (ref["e_dft"], ref["ene"], ref["ptexc"]):
                raise AssertionError(f"{ref_path}: energies did not survive the copy")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} kB, "
          f"omega/lambda = {OMEGA_LAMBDA}, lossless)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("raw_dir", type=Path, help="directory holding the expanded run trees")
    args = p.parse_args()

    MAIN_DATA.mkdir(parents=True, exist_ok=True)
    SI_DATA.mkdir(parents=True, exist_ok=True)
    for name, meta in SYSTEMS.items():
        build_bilayer(args.raw_dir, name, meta)
    build_unitcell_supercell(args.raw_dir)
    build_lambda_omega_ratio(args.raw_dir)


if __name__ == "__main__":
    main()
