# Super-Poissonian Squeezed Light in the Ground State of Strongly Coupled Light-matter Systems

Data and plotting scripts for all figures of:

> Cankut Tasci, Mohammad Hassan, Leon Orlov-Sullivan, Leonardo A. Cunha, and Johannes Flick,
> *Super-Poissonian Squeezed Light in the Ground State of Strongly Coupled Light-matter Systems* (2026).

Preprint and journal reference: to be added.

## What is in the paper

Ground-state quantum-optical observables of strongly coupled light-matter systems are computed
with quantum electrodynamical density-functional theory (QEDFT) using the photon many-body
dispersion (pMBD) functional, which captures higher-order electron-photon correlations and
multi-photon processes. The reference point throughout is the photon-Gaussian-approximation
functional (photon-GA, built on VV10), which is first order in the light-matter coupling and
therefore misses squeezing and photon-number correlations entirely.

The main result: in the collective strong-coupling regime the cavity mode is squeezed and its
photon statistics become super-Poissonian (Mandel `Q > 0`) already in the ground state.

## Physical setup

All results use the same cavity and geometry:

| Parameter | Value |
| --- | --- |
| Photon frequency `ω_α` | 2 eV (0.0734986 Hartree) |
| Light-matter coupling `λ_α` | 0.025 a.u. |
| Interatomic distance `d` | 4 Å |
| Geometry | linear chain along `z`, single cavity mode polarized along `z` |
| Gauge | velocity gauge |

Main text: chains of Ar atoms, `N = 1 ... 100`.
SI: He chains `N = 1 ... 4` (benchmark against QED-FCI) and Ar chains up to `N = 1000`
(large-`N` scaling).

**All quantities are in atomic units.**

## Layout

```
tasci2026super/
├── main-script/           Figures 1-3 of the main text
│   ├── plot_fig1.py
│   ├── plot_fig2.py
│   ├── plot_fig3.py
│   ├── pmbd_lam0025.h5
│   ├── ar_chain_lambda_0025_vv10_R4_N100_GAUGE_VG.h5
│   ├── lam_0.025_rpa_len_N100_R4_z_alpha_dft_free.h5
│   ├── fig_1/  fig_2/  fig_3/
└── SI/                    Figures S1-S2 of the Supplemental Material
    ├── plot_fci_final.py
    ├── plot_Ar_fit.py
    ├── He_dist4_lam_0.025_results_4.h5
    ├── Ar_scaling_fit.h5
    ├── benchmark_split_diagnostics.pdf
    └── Ar_scaling_publication.pdf / .png
```

## Scripts and the figures they produce

| Script | Figure | Reads | Writes |
| --- | --- | --- | --- |
| `main-script/plot_fig1.py` | Fig. 1 (electron-photon xc energy and photon number vs `N`) | `pmbd_lam0025.h5`, `ar_chain_..._vv10_....h5` | `fig_1/fig_1.{pdf,svg,png}` |
| `main-script/plot_fig2.py` | Fig. 2 (quadrature uncertainties, squeezing parameter, Wigner insets) | `pmbd_lam0025.h5` | `fig_2/fig_2.pdf` |
| `main-script/plot_fig3.py` | Fig. 3 (Mandel `Q`, photon-number variance, von Neumann entropy) | `pmbd_lam0025.h5`, `ar_chain_..._vv10_....h5` | `fig_3/fig_3_final_ticks.pdf` |
| `SI/plot_fci_final.py` | Fig. S1 (pMBD vs QED-FCI on He chains) | `He_dist4_lam_0.025_results_4.h5` | `benchmark_split_diagnostics.pdf` |
| `SI/plot_Ar_fit.py` | Fig. S2 (Ar chain scaling to `N = 1000`) | `Ar_scaling_fit.h5` | `Ar_scaling_publication.{pdf,png}` |

The QED-FCI reference values in Fig. S1 are small enough to be hard-coded as arrays at the top of
`plot_fci_final.py` rather than stored in HDF5.

## Data files

### `main-script/pmbd_lam0025.h5`

pMBD results for the Ar chain, one entry per system size, arrays of length 100 indexed by `N`
(`N = 1 ... 100`, sort by the `N` dataset before use). File attributes record the cavity
setup: `gauge = "velocity"`, `lambda = 0.025`, `omega_ev = 2`, `omega_hartree = 0.0734986...`.

| Dataset | Shape | Meaning |
| --- | --- | --- |
| `N` | (100,) | number of atoms |
| `E0` | (100,) | total DFT ground-state energy of the chain (order `-527 N` Ha) |
| `ene` | (100,) | uncoupled (`λ = 0`) MBD correlation energy; zero at `N = 1` |
| `pt_exc` | (100,) | coupled (`λ = 0.025`) counterpart of `ene` |
| `pt_number` | (100,) complex | ground-state photon number `⟨a†a⟩` |
| `pt_adadaa` | (100,) complex | second-order correlation `⟨a†a†aa⟩` |
| `pt_xx`, `pt_pp` | (100,) complex | quadrature variances `⟨q²⟩`, `⟨p²⟩` |
| `pt_rpa_orders` | (100, 10) | order-by-order RPA contributions, coupled |
| `rpa_orders` | (100, 10) | order-by-order RPA contributions, uncoupled |
| `AT_energy`, `pt_AT_energy` | (100,) | Axilrod-Teller three-body term, uncoupled / coupled |

Quantities derived in the scripts:

```
E(N)_λ - E(N)_{λ=0}  =  pt_exc - ene
Δq, Δp               =  sqrt(pt_xx), sqrt(pt_pp)
squeezing r          =  0.5 * ln(Δq * ω / Δp)
Mandel Q             =  (⟨a†a†aa⟩ - ⟨n⟩²) / ⟨n⟩
photon variance Δn²  =  ⟨a†a†aa⟩ + ⟨n⟩ - ⟨n⟩²
RPA at order o       =  sum(pt_rpa_orders[:, :o+1]) - sum(rpa_orders[:, :o+1])
```

Sizes `N = 6, 77, 95` are excluded from Fig. 1 via the `SKIP_N` set in `plot_fig1.py`.

### `main-script/ar_chain_lambda_0025_vv10_R4_N100_GAUGE_VG.h5`

photon-GA (VV10) reference for the same Ar chains, stored as one group per size, `n_1 ... n_100`,
plus a top-level `N_list`. Each group holds scalars: `E0`, `exc` (coupled energy), `pnumber`
(photon number `⟨a†a⟩`), `dip_proj` (dipole projection on the mode polarization), and the
perturbative terms `npt2`, `npt3_int`.

Because photon-GA is a first-order theory it implies `⟨a†a†aa⟩ = 0`, so its Mandel parameter and
photon variance in Fig. 3 are evaluated analytically as `Q = -⟨n⟩` and `Δn² = ⟨n⟩ - ⟨n⟩²`.

### `main-script/lam_0.025_rpa_len_N100_R4_z_alpha_dft_free.h5`

Raw per-size pMBD/RPA output for the Ar chain (groups `n_1 ... n_100`), kept for completeness.
It is **not** read by any of the plotting scripts, whose inputs were consolidated into
`pmbd_lam0025.h5`. Each group holds `At_energy`, `Eph`, `MandelQ`, `pnumber`, `adadaa`,
`RPA_orders` (15 orders), the free-atom polarizabilities `alpha_0_rsscs` (length `N`), and the
RPA eigenvalue spectrum `eigs_rpa` (shape `(30, 3N)`).

### `SI/He_dist4_lam_0.025_results_4.h5`

pMBD results for He chains, `N = 1 ... 4`, groups `N_1 ... N_4`. Same cavity parameters as the Ar
runs. Each group holds `ene`, `ptexc`, `pt_number`, `pt_xx`, `pt_pp`, `pt_rpa_orders`,
`rpa_orders`, `AT_energy`, `pt_AT_energy`, and `kpoints`. The naming follows
`pmbd_lam0025.h5`: `ene` is the uncoupled MBD energy, `ptexc` the coupled one, and the plotted
shift is `ptexc - ene`. These are the sizes for which the QED-FCI reference is affordable.

### `SI/Ar_scaling_fit.h5`

pMBD results for Ar chains out to `N = 1000`, 66 sizes, groups zero-padded as
`N_00001 ... N_01000`. Each group holds `ene`, `phn`, `pt_xx`, `pt_pp`, `pt_adadaa`,
`pt_rpa_orders`, `rpa_orders`, `AT_energy`, `pt_AT_energy`, and `kpoints`.

Note the differing convention in this file: `ene` is already the polaritonic energy **shift**
`E(N)_λ - E(N)_{λ=0}` (it matches `pt_exc - ene` from `pmbd_lam0025.h5` where the two overlap),
and `phn` is the photon number. `plot_Ar_fit.py` therefore plots both directly, with no
subtraction.

The script plots `N ∈ {1, 11, 21, ..., 91} ∪ {100, 120, ..., 1000}` (`PLOT_NS`) and excludes
`N = 160, 660, 800` (`EXCLUDE_NS`).

## Reproducing the figures

Python 3 with `numpy`, `h5py`, and `matplotlib`. The scripts resolve data paths relative to the
current working directory, so run them from inside their own folder:

```bash
cd main-script
python plot_fig1.py     # -> fig_1/fig_1.pdf, .svg, .png
python plot_fig2.py     # -> fig_2/fig_2.pdf
python plot_fig3.py     # -> fig_3/fig_3_final_ticks.pdf

cd ../SI
python plot_fci_final.py   # -> benchmark_split_diagnostics.pdf
python plot_Ar_fit.py      # -> Ar_scaling_publication.pdf, .png
```

The published figures use the STIX / Times New Roman serif fonts. If those are not installed,
matplotlib falls back to DejaVu Serif and the text metrics will differ slightly from the versions
committed here; the data curves are unaffected.
