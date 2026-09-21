# Datasets used in manuscripts

Raw data and plotting scripts behind the figures of manuscripts from the
[Flick group](https://github.com/flickgroup) (City College of New York / CUNY Graduate Center /
Center for Computational Quantum Physics, Flatiron Institute).

The goal is simple: every published figure should be reproducible from the files in this
repository with a single script run, without access to the original compute cluster.

## Contents

| Folder | Manuscript | Systems / methods |
| --- | --- | --- |
| [`flick2022simple`](flick2022simple) | Flick, *Simple Exchange-Correlation Energy Functionals for Strongly Coupled Light-Matter Systems Based on the Fluctuation-Dissipation Theorem*, [Phys. Rev. Lett. **129**, 143201 (2022)](https://doi.org/10.1103/PhysRevLett.129.143201) ([arXiv:2104.06980](https://arxiv.org/abs/2104.06980)) | Be atom and LiH in a single-mode cavity; C20, C60 and C180 in a dissipative cavity of 400 000 photon modes; QEDFT with the gradient approximation (GA), KLI and OEP |
| [`bonini2024cboa`](bonini2024cboa) | Bonini, Ahmadabadi, Flick, *Cavity Born-Oppenheimer Approximation for Molecules and Materials via Electric Field Response*, [J. Chem. Phys. **161**, 154104 (2024)](https://doi.org/10.1063/5.0230983) ([arXiv:2407.14613](https://arxiv.org/abs/2407.14613)) | CO2 and Fe(CO)5 molecules, h-BN and HfS2 monolayers in an optical cavity; vibro-polariton and phonon-polariton spectra from VASP DFPT within the cavity Born-Oppenheimer approximation (CBOA) |
| [`tasci2025photon`](tasci2025photon) | Tasci, Cunha, Flick, *Photon Many-Body Dispersion: Exchange-Correlation Functional for Strongly Coupled Light-Matter Systems*, [Phys. Rev. Lett. **134**, 073002 (2025)](https://doi.org/10.1103/PhysRevLett.134.073002) ([arXiv:2404.04765](https://arxiv.org/abs/2404.04765)) | Ar dimer, benzene dimer, graphene bilayer, benzene-Ar, water dimer in an optical cavity; QEDFT with pMBD, MBD+GA, QED-CCSD-2 |
| [`tasci2026super`](tasci2026super) | Tasci, Hassan, Orlov-Sullivan, Cunha, Flick, *Super-Poissonian Squeezed Light in the Ground State of Strongly Coupled Light-matter Systems* ([arXiv:2512.18242](https://arxiv.org/abs/2512.18242)) | Ar and He chains in an optical cavity; QEDFT with the photon many-body dispersion (pMBD) functional, photon-GA, QED-FCI |
| [`li2026cboamd`](li2026cboamd) | Li, Car, Flick, *cboamd: A Machine Learning Molecular Dynamics Framework for Vibrational Strong Coupling* ([arXiv:2609.22022](https://arxiv.org/abs/2609.22022)) | Single CO2 molecule and liquid CO2 (64 molecules) in an optical cavity; machine learning molecular dynamics in the cavity Born-Oppenheimer approximation (CBOA), with NEP and DeePMD-kit models for the energy, dipole and polarizability; polariton IR spectra, Rabi splittings and radial distribution functions |

## Layout convention

One folder per manuscript, named `<first-author><year><keyword>`, all lowercase, no spaces
(for example `tasci2026super`). Each folder contains:

* a `README.md` describing the manuscript, the physical setup, and every file in the folder,
* subfolders mirroring the manuscript structure (`main-script/`, `SI/`, ...),
* the raw data (HDF5 preferred) next to the script that consumes it.

The figures themselves are not stored here. The data and the scripts are the record; run the
scripts to produce the figures.

Data are stored in atomic units unless a folder README states otherwise.

## Using the data

Each folder is self-contained. Clone the repository, change into the relevant subfolder, and run
the plotting scripts from there (they resolve data paths relative to the working directory):

```bash
git clone https://github.com/flickgroup/datasets_used_in_manuscripts.git
cd datasets_used_in_manuscripts/tasci2026super/main-script
python plot_fig1.py
```

The scripts need Python 3 with `numpy` and `matplotlib`, plus `h5py` for the datasets stored as
HDF5.

## Citing

If you use data from this repository, please cite the corresponding manuscript. The relevant
reference is given in each folder's `README.md`.

## Contact

Johannes Flick, Department of Physics, City College of New York (`jflick@ccny.cuny.edu`).
