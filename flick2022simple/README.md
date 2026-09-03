# Simple Exchange-Correlation Energy Functionals for Strongly Coupled Light-Matter Systems Based on the Fluctuation-Dissipation Theorem

Data and plotting scripts for:

> Johannes Flick, *Simple exchange-correlation energy functionals for strongly coupled
> light-matter systems based on the fluctuation-dissipation theorem*, Phys. Rev. Lett. **129**,
> 143201 (2022).

Journal: [10.1103/PhysRevLett.129.143201](https://doi.org/10.1103/PhysRevLett.129.143201).
Preprint: [arXiv:2104.06980](https://arxiv.org/abs/2104.06980).

## How this folder is organized

```
flick2022simple/
├── main-script/
│   ├── plot_fig1.py           data/ -> figures/Fig1_atoms.{png,pdf}
│   ├── plot_fig2.py           data/ + images/ -> figures/Fig2_dissipation.{png,pdf}
│   ├── make_photon_modes.py   rewrites the octopus mode files, which are not shipped
│   ├── data/                  the five .dat files the plots read
│   └── images/                C20/C60/C180 renderings, inset into Fig. 2(b)
├── matter-data/
│   ├── octopus-atoms/         Fig. 1: the Be and LiH runs
│   └── octopus-fullerenes/    Fig. 2: the C20, C60 and C180 runs
└── tools/
    └── collect_photon_ex.py   matter-data/ -> main-script/data/C*_photon-ex.dat
```

There is no `SI/`: the SI has no figures or tables. `main-script/data/` sits under the script that
consumes it for the same reason.

## The data files

`{berillium,LiH}_energy-sort.dat` are five columns: cavity frequency in Hartree, then the KLI, OEP,
GA and OEP-ss total energies, also in Hartree. Row 0 is the omega = 0 baseline that `plot_fig1.py`
subtracts.

`C{20,60,180}_photon-ex.dat` are two columns: the dissipation constant in eV, and the octopus
`Photon ex.` energy in Hartree. The last row is the single-mode run, recorded as kappa = 0.
`plot_fig2.py` scales by `P_Har/N_at`.

## `matter-data/`

File-level selections from the original run directories, copied verbatim: each run keeps its `inp`,
`geometry.xyz` and `static/info`. The `restart/` trees, density output, and the 37.6 MiB photon
mode files are omitted.

Under `octopus-atoms/<system>/`, directories `1`..`20` are omega = 0.1 to 2.0 Ha in steps of 0.1
and `f1`..`f20` are omega = 0.02 to 0.4 Ha in steps of 0.02; `OEP/` is the omega = 0 baseline. The
four method subdirectories map onto the figure's curves as `KLI-pt`, `OEP-pt`, `OEP-gga` (the GA
curve) and `OEP-pt-single-shot`. `grep_etot.sh` and `data_shuffle.py` are the original scripts that
reduce these to `energy-sort.dat`.

Under `octopus-fullerenes/<system>/`, `LDA-relax/` holds the geometry relaxation and each
`OEP-gga-*` one production run. The original tree's intermediate `production_runs` level, named
inconsistently between systems, is flattened away.

## Running it

Python 3 with `numpy` and `matplotlib`, nothing else:

```bash
cd main-script
python3 plot_fig1.py        # instant
python3 plot_fig2.py        # ~2 s
```

To rebuild the shipped data from `matter-data/` rather than trust it:

```bash
python3 tools/collect_photon_ex.py --check          # Fig. 2(b); drop --check to rewrite
cd matter-data/octopus-atoms/berillium              # Fig. 1, per system
bash grep_etot.sh && python3 data_shuffle.py
```

Verified 2026-09-03 with Python 3.12.7, numpy 2.2.6 and matplotlib 3.10.8: from an empty
`figures/`, both figures render; `grep_etot.sh` plus `data_shuffle.py` rewrite `energy.dat` and
`energy-sort.dat` byte for byte for both systems; `collect_photon_ex.py --check` passes; and
`make_photon_modes.py` reproduces the kappa = 0.01 and kappa = 0 mode files byte for byte against
the ones the published runs used. Against the published PNGs both figures give identical curves,
axes and insets, differing only in text glyph rasterization and in the resampling of the three
inset molecule images.

Data in atomic units. Generated figures are not committed.
