# Modifying van der Waals Materials via Cavity Vacuum Fluctuations

Data and plotting scripts for:

> Mohammad Hassan, Cankut Tasci, Leon Orlov-Sullivan, Leonardo A. Cunha, and Johannes Flick,
> *Modifying van der Waals materials via cavity vacuum fluctuations* (2026).

Preprint: [arXiv:2608.28521](https://arxiv.org/abs/2608.28521).

## Layout

```
hassan2026modifying/
├── data/                    the six binding curves, one HDF5 file each
├── images/                  structure renders used as figure insets
├── main-script/
│   ├── pmbd_bilayer.py      shared loader, binding curves, cubic fit, figure helpers
│   ├── plot_fig1.py         Fig. 1
│   └── plot_fig2.py         Fig. 2
├── SI/
│   ├── plot_figS1.py        Fig. S1
│   ├── plot_figS2.py        Fig. S2
│   ├── plot_figS3.py        Fig. S3
│   ├── plot_figS4.py        Fig. S4
│   └── data/                unit cell against supercell for Fig. S1, equal λ/ω for Fig. S4
├── run-inputs/              INCAR, KPOINTS, POSCARs, job scripts, POSCAR generator
└── tools/
    └── build_dataset.py     rebuilds data/ and SI/data/ from the raw run trees
```

The binding curves and the inset renders sit at the top level because the main text and the SI
share them. The SI scripts import `pmbd_bilayer` from `main-script/`.

| Script | Figure | Reads |
| --- | --- | --- |
| `main-script/plot_fig1.py` | Fig. 1 | all six binding curves |
| `main-script/plot_fig2.py` | Fig. 2 | the four hBN curves |
| `SI/plot_figS1.py` | Fig. S1 | `SI/data/unitcell_supercell.h5` |
| `SI/plot_figS2.py` | Fig. S2 | the two graphene curves |
| `SI/plot_figS3.py` | Fig. S3 | all six binding curves |
| `SI/plot_figS4.py` | Fig. S4 | `SI/data/lambda_omega_ratio.h5`, and graphene AB at λ = 0 |

## Data

`data/{hbn_aa,hbn_ab,hbn_aap,hbn_ab1,graphene_aa,graphene_ab}.h5`, one per material
and stacking, all on the same grid of 11 coupling strengths by 841 interlayer distances.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `r` | (841,) | interlayer distance in Å, 3.0 to 5.1 |
| `lambda` | (11,) | coupling strength in a.u., 0 to 0.2 |
| `E_dft` | (841,) | VASP PBE total energy of the cell in eV; independent of λ |
| `E_mbd` | (841,) | bare MBD energy in Ha, from the converged 37 × 37 × 1 q-grid |
| `ene` | (11, 841) | uncoupled pMBD exchange-correlation energy in Ha |
| `ptexc` | (11, 841) | its coupled counterpart; `ptexc - ene` is the cavity-induced shift |
| `hirshfeld_ratio` | (841, 4) | per-atom V<sub>eff</sub>/V<sub>free</sub> from the DFT run |
| `reference/E_dft`, `reference/E_mbd` | scalar | the same cell with the layers 10 Å apart |
| `reference/ene`, `reference/ptexc` | (11,) | |

File attributes record `material`, `stacking`, `atomic_species`, `n_atoms`, `xc`, `kpoints`,
`omega_ev`, `polarization`, `reference_r_ang` and `units`. The stored coupling strengths are the
unscaled ones; `run-inputs/README.md` gives the √N<sub>q</sub> rescaling the solver applies.
`pmbd_bilayer.py` assembles the total energy from its DFT, MBD and cavity pieces and derives the
binding curve, R<sub>0</sub> and ω<sub>LBM</sub>; its module docstring carries the formulas.

`SI/data/unitcell_supercell.h5` holds AA bilayer graphene at R = 3.6425 Å: `N` (1, 3, ..., 15),
and `unitcell/{ene,ptexc,n_atoms}` and `supercell/{ene,ptexc,n_atoms}`, energies in Ha per
simulation cell, for the unit cell on an N × N × 1 q-grid and the N × N × 1 supercell at Γ.

`SI/data/lambda_omega_ratio.h5` holds AB bilayer graphene on the same `r` grid, from runs on a
37 × 37 × 1 q-grid: groups `omega10` (λ = 0.1 a.u.) and `omega30` (λ = 0.3 a.u.), each with
`E_dft` in eV, `ene` and `ptexc` in Ha, and the same three under `reference/` for the layers 10 Å
apart. `SI/plot_figS4.py` documents how the q-grid enters the cavity term.

The raw runs, one JSON file per (stacking, λ, R), are available from the authors on request. Given
those expanded into one directory, `python tools/build_dataset.py /path/to/raw` rewrites both data
folders, asserting each value against the file it came from. `run-inputs/` holds what is needed to
regenerate the raw binding-curve runs themselves.

Note: Unlike the default JSON file format specified in the libphotonmbd repository, in which all 
energies are in units of hartrees, the JSON files included here feature the keys "E0", 
"total_energy", and "mbd_energy" in units of eV.
