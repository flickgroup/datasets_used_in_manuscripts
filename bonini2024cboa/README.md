# Cavity Born-Oppenheimer Approximation for Molecules and Materials via Electric Field Response

Data and plotting scripts for:

> John Bonini, Iman Ahmadabadi, and Johannes Flick, *Cavity Born-Oppenheimer approximation for
> molecules and materials via electric field response*, J. Chem. Phys. **161**, 154104 (2024).

Journal: [10.1063/5.0230983](https://doi.org/10.1063/5.0230983).
Preprint: [arXiv:2407.14613](https://arxiv.org/abs/2407.14613).

## Layout

```
bonini2024cboa/
├── main-script/
│   ├── vasp_reader.py       reads the DFPT arrays out of vaspout.h5
│   ├── reproduce.py         vaspout.h5 -> data/*.dat  (the spectra)
│   ├── make_table1.py       vaspout.h5 -> data/Table_I.dat
│   ├── plot_figures.py      data/*.dat -> figures/*.{png,pdf}
│   ├── analyze_systems.py   optional: per-system normal modes and Z*
│   └── data/                the .dat panel files the plots read
├── matter-data/
│   ├── vasp-dfpt/           the DFPT runs: vaspout.h5 + INCAR, KPOINTS,
│   │                        POSCAR, CONTCAR, POTCAR.info
│   └── octopus/             the explicit-CBOA reference points for Fig. 2
└── tools/
    └── reduce_vaspout.py    how matter-data/vasp-dfpt/ was assembled
```

`reproduce.py` then `plot_figures.py` gives Figs. 2-6; `make_table1.py` then `plot_figures.py`
gives Table I. Fig. 1 is a schematic and is not reproduced.

## Data

`matter-data/vasp-dfpt/` holds five Γ-point VASP DFPT runs, one per directory: `CO2`, `FeCO5`,
`hBN`, `HfS2` and `CO2_relax`. The figures use the first four. `CO2_relax` backs no figure; it is
the relaxed CO<sub>2</sub> geometry, giving 292 meV against the 312 meV of Table I. The `INCAR`
files are the full record of each run. `POTCAR.info` names the pseudopotentials: POTCARs are
licensed VASP material and are not redistributed, so `tools/reduce_vaspout.py` drops the copy VASP
embeds in `vaspout.h5` and keeps their `TITEL` lines instead.

`matter-data/octopus/octopus_CO2_CBOA_FD.txt`, the explicit-CBOA QEDFT reference for Fig. 2, in
cm<sup>-1</sup>: row 0 is λ, row 1 the symmetric stretch, rows 2 and 3 the lower and upper
polariton.

`main-script/data/` holds one file per figure panel, `np.savetxt` with a three-line `#` header
giving the system, ω<sub>res</sub>, the cavity axis and `nmodes`. With `N = nmodes`:

```python
d = np.loadtxt("CO2_Fig3_panelA.dat")
lam = d[:, 0]
N = (d.shape[1] - 1) // 3
freqs, chars, irns = d[:, 1:1+N], d[:, 1+N:1+2*N], d[:, 1+2*N:1+3*N]
```

frequencies in meV, photon character, IR intensity in arbitrary units. Fig. 2 is two files, the
electric-field-response branches and the Octopus points, both λ plus frequency columns in meV.
`Table_I.dat` holds the reproduced Table I; `_all_results.npz` bundles every spectrum array.

## Running it

Python 3 with `numpy`, `scipy`, `h5py`, `ase` and `matplotlib`, plus the `qedmft` package, which
carries the CBOA solver:

```bash
git clone https://github.com/jrbp/qedmft
```

`vasp_reader.py` finds it as an installed package, at `$QEDMFT_SRC`, or as a clone at `../../qedmft`
beside this folder. Then:

```bash
cd main-script
python3 reproduce.py        # ~2 s
python3 make_table1.py
python3 plot_figures.py     # -> figures/
```

Rendered figures are not committed; to go straight to them from the shipped data, run only
`plot_figures.py`.

Verified 2026-09-03 with `qedmft` at `eab1ef7`, numpy 2.2.6, scipy 1.13.1, h5py 3.15.1, ase 3.28.0
and matplotlib 3.10.8: from an empty `data/`, the generating scripts rewrite every shipped `.dat`
file byte for byte.
