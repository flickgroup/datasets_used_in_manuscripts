# Photon Many-Body Dispersion: Exchange-Correlation Functional for Strongly Coupled Light-Matter Systems

Data and plotting scripts for:

> Cankut Tasci, Leonardo A. Cunha, and Johannes Flick,
> *Photon Many-Body Dispersion: Exchange-Correlation Functional for Strongly Coupled Light-Matter
> Systems*, Phys. Rev. Lett. **134**, 073002 (2025).

Journal: [10.1103/PhysRevLett.134.073002](https://doi.org/10.1103/PhysRevLett.134.073002).
Preprint: [arXiv:2404.04765](https://arxiv.org/abs/2404.04765).

## What is in the paper

pMBD is an electron-photon exchange-correlation functional for quantum electrodynamical
density-functional theory (QEDFT), generalising the many-body dispersion (MBD) method to treat
electronic and photonic degrees of freedom on the same footing. The paper shows that pMBD captures
anisotropic electron-photon interactions, beyond-single-photon effects, and cavity-modulated van
der Waals (c-vdW) interactions, benchmarked against QED-CCSD-2.

The central observable throughout is the interaction energy

```
ΔE(R) = E(R) - E(R = 25 Å)
```

evaluated outside the cavity (`λ = 0`) and inside it for cavity polarisation along `x`, `y`, `z`.
The c-vdW contribution changes sign with polarisation: repulsive along `z`, attractive along `x`
and `y`.

## Physical setup

| Parameter | Value |
| --- | --- |
| Cavity frequency `ω_cav` | 2 eV |
| Coupling strength `λ` | 0.05 a.u. (main results); 0.07, 0.09, 0.11 also shown in Fig. 1(a) |
| Electronic structure | PBE0, aug-cc-pVDZ |
| Reference method | QED-CCSD-2 |

Systems: Ar dimer, benzene dimer, AA-stacked graphene bilayer flakes (main text); benzene-Ar and
the water dimer (Supplemental Material).

**Energies in the data files are total energies in Hartree.** The scripts convert to meV or eV.

## Layout

```
tasci2025photon/
├── main-script/            Figures 1 and 2 of the main text
│   ├── plot_fig1.py
│   ├── plot_fig2.py
│   ├── data/
│   ├── images/             schematics used as figure insets
│   ├── fig_1/  fig_2/      (figure output)
└── SI/                     Supplemental Material data (no scripts, see below)
    ├── bz_ar/
    └── water_dimer/
```

## Scripts and the figures they produce

| Script | Figure | Reads | Writes |
| --- | --- | --- | --- |
| `main-script/plot_fig1.py` | Fig. 1 (Ar dimer PES and cavity-induced contribution) | the six `data/*ar_dimer*` files, `images/ardimer3.png` | `fig_1/fig_1.{pdf,png}` |
| `main-script/plot_fig2.py` | Fig. 2 (benzene dimer and graphene bilayer) | `data/benzene_dimer_pmbd.dat`, `data/graphene_bilayer_pmbd.dat`, two images | `fig_2/fig_2.{pdf,png}` |

Both were extracted from the original `plot.ipynb` notebooks. Two changes were needed to make them
run standalone: the `from atomic_constants import P_Har` dependency is now an inline constant
(`P_Har = 2 × 13.60569193`, Hartree to eV), and the absolute `/home/jflick/...` paths to the inset
images are now relative.

`plot_fig1.py` additionally draws the `PBE0-MBD` and `PBE0-MBD+GA` curves in panel (b). The
notebook predates them; they were added for the published version of the figure, and are built
here from the `MBD Correlation` and `GA corr` columns of `pmbd_ar_dimer_summary.txt`.

### Supplemental Material

The SM contains four figures. Fig. S1 is a hand-drawn schematic of the pMBD Hamiltonian. Figs. S2
(Ar dimer c-vdW fits), S3 (benzene-Ar) and S4 (water dimer) are data figures, but **no plotting
scripts for them survive**, so `SI/` ships the underlying data only. Fig. S2 is built from the Ar
dimer data in `main-script/data/`.

## Data files

### `main-script/data/pmbd_ar_dimer_lambda_{5e-2,7e-2,9e-2,1e-1}_summary.txt`

PBE0+pMBD total energies for the Ar dimer, one file per coupling strength. Two header lines, then
five columns:

```
R (Å)    E_outside    E_pol_x    E_pol_y    E_pol_z
```

102 rows, `R` = 3.00 to 8.00 Å in 0.05 Å steps plus a final row at `R` = 25 Å used as the
reference point.

> **Note on `pmbd_ar_dimer_lambda_1e-1_summary.txt`:** despite its filename and its header line,
> this file holds the **λ = 0.11 a.u.** calculation, not λ = 0.1. Verified by comparing against
> the raw output directories, where it matches `04_augccvz_ar_pbe0_0.11` exactly and differs from
> `04_augccvz_ar_pbe0_0.1`. It is the outermost faded curve in Fig. 1(a), so the published figure
> shows λ = 0.05, 0.07, 0.09 and 0.11. The file is shipped under its original name to preserve
> provenance.

### `main-script/data/pmbd_ar_dimer_summary.txt`

The energy decomposition for the Ar dimer at λ = 0.05. Two header lines, then seven columns:

```
R (Å)    E0 (PBE0)    MBD corr    pMBD corr (pol x)    pMBD corr (pol y)    pMBD corr (pol z)    GA corr
```

101 rows on a different grid from the files above: `R` = 1.0 to 8.0 Å (100 points) plus `R` = 25 Å.
Used for the `PBE0-MBD` and `PBE0-MBD+GA` curves in Fig. 1(b) as `E0 + MBD` and `E0 + MBD + GA`.

### `main-script/data/qedcc_ar_dimer_summary.txt`

QED-CCSD-2 total energies for the Ar dimer at λ = 0.05. Two header lines, then

```
R (Å)    E_outside    E_pol_x    E_pol_y    E_pol_z
```

90 rows running out to `R` = 500 Å. The script uses rows up to and including `R` = 25 Å (row `-6`)
as the reference and discards the longer tail.

### `main-script/data/benzene_dimer_pmbd.dat` and `graphene_bilayer_pmbd.dat`

Raw QEDFT scan output, no header. The benzene file has 19 columns: `R` followed by six groups of
(`E0`, `vdwene`, `pt`) for polarisations `v1x, v2x, v1y, v2y, v1z, v2z`. The graphene file has 10
columns: `R` followed by three groups of (`E0`, `vdw`, `cav`) for `x`, `y`, `z`. The exact column
map is repeated as a comment at the top of `plot_fig2.py`.

For the benzene dimer the scan variable is `R_x` at fixed `R_z` = 3.3 Å; for the graphene bilayer
it is the interlayer separation `R_z`.

### `SI/bz_ar/` and `SI/water_dimer/`

`pmbd_*_summary.txt` uses the same seven-column decomposition as `pmbd_ar_dimer_summary.txt`;
`qedcc_*_summary.txt` uses the same five-column layout as `qedcc_ar_dimer_summary.txt`. The scan
coordinate is the benzene-Ar distance and the O-H distance respectively.

`geometries/` holds the xyz structure at each scan point (`*_n0.xyz` through `*_n102.xyz`), indexed
in the same order as the rows of the summary files.

## Reproducing the figures

Python 3 with `numpy` and `matplotlib`. Run each script from inside `main-script/`:

```bash
cd main-script
python plot_fig1.py     # -> fig_1/fig_1.pdf
python plot_fig2.py     # -> fig_2/fig_2.pdf
```

The published figures were typeset with LaTeX (`text.usetex`). If you do not have a LaTeX
installation, set `USETEX=off` to fall back to matplotlib's own mathtext:

```bash
USETEX=off python plot_fig1.py
```

The curves are identical either way; only the text metrics differ, which in Fig. 1(b) makes the
legend box wide enough to overlap the blue curves slightly.
