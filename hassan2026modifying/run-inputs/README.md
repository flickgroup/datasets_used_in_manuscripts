# Run inputs

Everything needed to regenerate the raw data behind `data/`. Each point of
a binding curve is one VASP ground-state calculation followed by one pMBD post-processing
step, repeated over 841 interlayer distances and 11 coupling strengths per stacking.

## Files

| File | What it is |
| --- | --- |
| `INCAR` | VASP settings, identical for every point of every scan |
| `KPOINTS` | 31 x 31 x 1 Gamma-centred electronic k-grid |
| `POSCAR_hbn_aa`, `POSCAR_hbn_ab`, `POSCAR_hbn_aap`, `POSCAR_hbn_ab1` | bilayer hBN, the four stackings, at R = 3.0 Å |
| `POSCAR_graphene_aa`, `POSCAR_graphene_ab` | bilayer graphene, at R = 3.0 Å |
| `make_poscar.py` | rewrites a POSCAR to any interlayer distance, including the R = 10 Å reference |
| `jobscript-pmbd-rome.sbatch` | Slurm script for the bilayer pMBD runs |
| `jobscript-pmbd-reference-rome.sbatch` | the same for the R = 10 Å reference runs |

`POTCAR` files are not included: they are licensed VASP material and cannot be
redistributed. The calculations use the standard PBE PAW potentials for B, N and C.

## Geometries

All six cells are the relaxed monolayer in-plane lattice, two layers, and 20 Å of
cell height along z. The layers sit symmetrically about the mid-plane at z = 10 Å,
so the shipped POSCARs (z = 11.5 and 8.5) are R = 3.0 Å. `make_poscar.py` moves them
to any other R; the in-plane lattice and the cell height never change, which is what
makes E(R) - E(10 Å) a well-defined binding energy.

The scan is R = 3.0 to 5.1 Å in steps of 0.0025 Å, plus the single reference point at
R = 10 Å.

```bash
for R in $(seq 3.0 0.0025 5.1); do
    python make_poscar.py POSCAR_hbn_aa $R -o scan/R$R/POSCAR
done
python make_poscar.py POSCAR_hbn_aa 10.0 -o reference/POSCAR
```

## pMBD settings

The job scripts call `run_pmbd.py` from the authors' pMBD driver, which wraps the
periodic pMBD implementation in `libphotonmbd`. The driver is not part of this
dataset; the settings it was called with are recorded here because the two energies
it returns are combined by hand in the plotting scripts.

| Setting | Value |
| --- | --- |
| MBD damping parameter `beta` | 0.83 (the PBE value) |
| Cavity frequency `omega` | 0.07349865 Ha (2 eV) |
| Cavity modes | 1, polarised along z (out of plane) |
| Coupling strengths `lambda` | 0 to 0.2 a.u. in steps of 0.02 |
| q-grid, coupled run | 5 x 3 x 1 (15 q-points) |
| q-grid, uncoupled MBD run | 37 x 37 x 1 |

**The coupling strength is rescaled by the q-grid.** A run requested at
`lambda = 0.14` a.u. is executed at `lambda * sqrt(N_q) = 0.14 * sqrt(15) = 0.5422`,
because the cavity mode couples to all `N_q` periodic images at once. This is the
unit-cell/supercell equivalence derived in the Supplemental Material. The coupling
strengths quoted in the manuscript and stored in the HDF5 files are the *unscaled*
ones.

**The two energies come off different q-grids.** The cavity-induced shift converges
quickly with `N_q` and is taken from the 5 x 3 x 1 run, while the bare MBD dispersion
energy needs a much denser grid and is taken from a separate 37 x 37 x 1 run at
`lambda = 0`. The total energy assembled in `main-script/pmbd_bilayer.py` therefore
mixes the two:

```
E(lambda, R) = E_DFT(R) + [ E_MBD(R, 37x37x1) + shift(lambda, R, 5x3x1) ] * Hartree
```

where `shift = ptexc - ene` is the difference of the coupled and uncoupled pMBD
exchange-correlation energies on the coarse grid.
