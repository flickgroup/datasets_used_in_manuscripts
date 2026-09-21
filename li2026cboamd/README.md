# cboamd: A Machine Learning Molecular Dynamics Framework for Vibrational Strong Coupling

Data, trained models and plotting scripts for:

> Yifan Li, Roberto Car, and Johannes Flick, *cboamd: A Machine Learning Molecular Dynamics
> Framework for Vibrational Strong Coupling* (2026).

Preprint: [arXiv:2609.22022](https://arxiv.org/abs/2609.22022).
Code: [github.com/flickgroup/cboamd](https://github.com/flickgroup/cboamd), v1.0.0 at `89bc6dd`.

## Layout

```
li2026cboamd/
├── data/                            all figure inputs
│   ├── single_molecule/
│   │   ├── lambda=*/                17 CBOA QEDFT (AIMD) runs, 10 ps
│   │   └── nep_md/lambda=*/         18 MLIP runs, 100 ps
│   └── liquid/
│       ├── spectra/{offresonant,resonant}/<case>/   56 averaged spectra
│       ├── rdf/{offresonant,resonant}/              16 RDF summaries
│       ├── dp_parity/                               4 DeePMD test tables
│       └── fig6_scatter_cache_40traj.npz            Fig. 6b descriptors
├── main-script/                     Figures 1-6 of the main text
├── SI/                              Figures S1-S6 and the Table S1/S2 numbers
├── models/
│   ├── single_molecule_nep/         3 NEP models (GPUMD v3.9.5) + train_data/
│   └── liquid_dp/                   3 DeePMD models + train_data/
├── md-inputs/
│   ├── single_molecule/             43 in.json + co2-disp.xyz
│   ├── liquid/                      32 in.lmp, 40 starting configs, postprocessing
│   └── cboamd-lammps/               the `fix cboamd` LAMMPS source
├── reference-calculations/          CP2K (liquid) and PySCF (single molecule) inputs
└── tools/                           the .npz reader and every conversion script
```

`data/` sits at the top level, rather than beside each script, because the main text and the SI
read the same trajectory traces.

## Scripts and the figures they produce

Every script writes its own `.pdf` and `.png` next to itself and prints the numbers quoted in the
paper.

| Figure / Table | Script | Reads |
| --- | --- | --- |
| Fig. 1 | `main-script/fig1_spectrum_lambdas.py` | single-molecule AIMD + NEP dipole traces |
| Fig. 2 | `main-script/fig2_single_molecule_rabi_peak_and_splitting.py` | 18 NEP-MD dipole traces |
| Fig. 3 | `main-script/fig3_liquid_spectrum_lambdas.py` | liquid averaged spectra, both tunings |
| Fig. 4 | `main-script/fig4_liquid_rabi_peak_and_splitting.py` | all 30 cavity spectra |
| Fig. 5 | `main-script/fig5_liquid_rdf.py` | 3 of the 16 RDF summaries |
| Fig. 6a | `main-script/fig6_rabi_scaling.py` | NEP-MD dipole traces + liquid spectra |
| Fig. 6b | `main-script/fig6_rabi_scaling.py` | `fig6_scatter_cache_40traj.npz` |
| Fig. S1, Table S1 | `SI/figS1_nep_parity.py` | NEP parity files in `models/single_molecule_nep/` |
| Fig. S2 | `SI/figS2_dipole_dynamics.py` | single-molecule AIMD + NEP dipole traces |
| Fig. S3 | `SI/figS3_spectrum_lambdas_full.py` | single-molecule AIMD + NEP dipole traces |
| Fig. S4 | `SI/figS4_prefactor.py` | NEP-MD polarizability traces (chi_xx) |
| Fig. S5 | `SI/figS5_f2_term.py` | 2 NEP-MD dipole traces (lambda=0.3 polar, and `_nof2`) |
| Fig. S6, Table S2 | `SI/figS6_liquid_dp_parity.py` | `data/liquid/dp_parity/` |
| (main-text omega values) | `main-script/co2_harmonic_freqs.py` | PySCF, no stored input |
| (Fig. 6 gas-phase slope, SI Z*) | `main-script/co2_dipole_derivative.py` | PySCF, no stored input |
| (SI text values) | `SI/measure_nu2*.py`, `SI/satellite_analysis*.py` | single-molecule traces |

## Data

Traces, spectra, RDFs and DeePMD parity tables ship as compressed `float32` `.npz` rather than the
original ASCII. `tools/co2io.py` hides the encoding: its loaders return exactly the arrays the
ASCII files held, and document every column layout.

```python
import sys; sys.path.insert(0, "tools")
import co2io

trace = co2io.load_trace("data/single_molecule/nep_md/lambda=0.3_polar/dipole.npz")
trace["mu_x"]          # dressed x-dipole, a.u.
trace["mu_bare_x"]     # bare x-dipole, a.u.
trace["dt"]            # time step, a.u.

freq, cols, meta = co2io.load_spectrum(
    "data/liquid/spectra/offresonant/12_cavity_ir_40traj_newmodels_polar_lam010/"
    "spectrum_direct_dressed_x_avg40_hamming11.npz")
```

The liquid spectra carry the frequency labels their producing code wrote. Two small systematic
offsets in those labels are recorded in each file's `meta`; see `tools/convert_spectra.py`.

`tools/` holds every converter, so each shipped array can be regenerated from the raw data:
`convert_traces.py`, `convert_spectra.py`, `convert_rdf.py`, `convert_parity.py` and
`convert_dp_trainset.py`. Each asserts its reduction lossless per file.

## Running it

Python 3 with `numpy` and `matplotlib`, plus `pyscf` for `co2_harmonic_freqs.py` and
`co2_dipole_derivative.py`. Run each script from inside its own folder:

```bash
cd main-script && python3 fig1_spectrum_lambdas.py
cd ../SI       && python3 figS1_nep_parity.py
```

## Retraining the models

**Single molecule (NEP, GPUMD v3.9.5).** All three models train from the same pair of files,
`models/single_molecule_nep/train_data/{train,test}.xyz` (1360 and 680 structures), whose
extended-XYZ comment lines carry every label. The models differ only in `nep.in`, through
`model_type` 0 (energy/force), 1 (dipole) and 2 (polarizability). Copy `train.xyz`, `test.xyz` and
the model's `nep.in` into a directory and run `nep`.

`train_data/` also holds the three scripts that built the set and the two dataset JSONs they
exchange, so `02_recompute_labels.py` -> `03_assemble_xyz.py` reruns end to end from this
repository (step 02 needs PySCF). `01_extract_dataset.py` is the one step that does not rerun
here: it reads the raw AIMD trajectories, and its output is exactly the shipped JSONs.

**Liquid (DeePMD-kit).** Each of `models/liquid_dp/{ener,dipole,polar}_model/` holds the
`input.json` used, whose `systems` path resolves to the matching subset of
`models/liquid_dp/train_data/`. Run `dp train input.json` in the model directory and compare
against the shipped `lcurve.out`. The production `in.lmp` files loaded `dp compress`-ed models,
which are not shipped because they regenerate exactly:

```bash
dp compress -i frozen_model.pb -o frozen_model_compressed.pb
```
