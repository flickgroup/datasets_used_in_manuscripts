#!/usr/bin/env python3
"""
Build the shipped `matter-data/vasp-dfpt/<system>/` tree from a raw VASP DFPT
run directory.

Two things happen per run:

1. `vaspout.h5` is copied with the single dataset `input/potcar/content`
   dropped. That dataset holds the verbatim text of the POTCAR files, which are
   licensed VASP material and may not be redistributed. Nothing in
   `main-script/` reads it. Everything else in the file is copied bit for bit,
   and the arrays the reader actually uses are asserted identical afterwards.

2. `POTCAR.info` is written from the `TITEL` lines of that same dataset, so the
   pseudopotential identity, the part that matters for reproducing the
   calculation, survives in text form.

The text inputs (`INCAR`, `KPOINTS`, `POSCAR`, `CONTCAR`) are copied unchanged.

Usage:
    python3 reduce_vaspout.py <raw_run_dir> <out_dir>
"""
import os
import shutil
import sys

import h5py
import numpy as np

# datasets vasp_reader.py reads; these are verified after the copy
VERIFY = [
    "results/positions/scale",
    "results/positions/lattice_vectors",
    "results/positions/position_ions",
    "results/positions/ion_types",
    "results/positions/number_ion_types",
    "results/linear_response/force_constants",
    "results/linear_response/born_charges",
    "results/linear_response/electron_dielectric_tensor",
]

DROP = "input/potcar/content"
TEXT_INPUTS = ["INCAR", "KPOINTS", "POSCAR", "CONTCAR"]


def _copy_without_potcar(src, dst):
    """Copy every object of `src` into `dst` except the POTCAR text blob."""
    with h5py.File(src, "r") as fi, h5py.File(dst, "w") as fo:
        fo.attrs.update(fi.attrs)

        def visit(name, obj):
            if name == DROP:
                return
            if isinstance(obj, h5py.Group):
                g = fo.require_group(name)
                g.attrs.update(obj.attrs)
            else:
                fo.create_dataset(name, data=obj[()], dtype=obj.dtype)
                fo[name].attrs.update(obj.attrs)

        fi.visititems(visit)


def _titel_lines(src):
    with h5py.File(src, "r") as f:
        if DROP not in f:
            return []
        content = f[DROP][()]
        if isinstance(content, bytes):
            content = content.decode("utf-8", "replace")
        return [ln.strip() for ln in content.splitlines() if "TITEL" in ln]


def _verify(src, dst):
    with h5py.File(src, "r") as fi, h5py.File(dst, "r") as fo:
        assert DROP not in fo, f"{DROP} survived the copy"
        for key in VERIFY:
            a, b = fi[key][()], fo[key][()]
            assert np.array_equal(a, b), f"{key} changed"
        # nothing else went missing either
        names_in, names_out = set(), set()
        fi.visititems(lambda n, o: names_in.add(n))
        fo.visititems(lambda n, o: names_out.add(n))
        missing = names_in - names_out - {DROP}
        assert not missing, f"unexpectedly dropped: {sorted(missing)}"


def reduce_run(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    src = os.path.join(raw_dir, "vaspout.h5")
    dst = os.path.join(out_dir, "vaspout.h5")

    titels = _titel_lines(src)
    _copy_without_potcar(src, dst)
    _verify(src, dst)

    with open(os.path.join(out_dir, "POTCAR.info"), "w") as f:
        f.write("# Pseudopotentials used in this run, from the TITEL lines of\n"
                "# the POTCAR. The POTCAR itself is licensed VASP material and\n"
                "# is not redistributed; see the folder README.\n")
        for ln in titels:
            f.write(ln + "\n")

    for name in TEXT_INPUTS:
        p = os.path.join(raw_dir, name)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(out_dir, name))

    a, b = os.path.getsize(src), os.path.getsize(dst)
    print(f"{raw_dir}\n  -> {out_dir}  "
          f"vaspout.h5 {a/1e6:.2f} MB -> {b/1e6:.2f} MB, "
          f"{len(titels)} pseudopotentials, verified")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    reduce_run(sys.argv[1], sys.argv[2])
