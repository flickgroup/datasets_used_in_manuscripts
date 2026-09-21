#!/bin/bash
# Drive the two IR postprocess scripts over the johannes_runs newmodels cases.
#   (A) reconstruct_dressed_dipole_spectra.py : 8 cavity cases, correct lambda
#       -> spectrum_direct_dressed_x_avg40{,_hamming11}.dat + per-traj files
#   (B) ir_from_cboamd.py (bare x dipole, direct FFT) : all 9 cases, per traj,
#       then averaged inline -> spectrum_direct_x_avg40{,_hamming11}.dat
# NOTE: no `set -u` -- some environment activation scripts use unbound vars
# (e.g. ADDR2LINE) and would abort activation under nounset.
set -o pipefail
# Activate the environment that provides numpy here.
# Activate an environment providing numpy here.
cd "$(dirname "$0")"                       # johannes_runs/postprocess
JR=".."
PY=python

# case -> lambda (cavity cases only)
CAV_CASES=(
  "11_cavity_ir_40traj_newmodels_nopolar_lam010 0.1  nopolar"
  "13_cavity_ir_40traj_newmodels_nopolar_lam001 0.01 nopolar"
  "16_cavity_ir_40traj_newmodels_nopolar_lam002 0.02 nopolar"
  "17_cavity_ir_40traj_newmodels_nopolar_lam003 0.03 nopolar"
  "12_cavity_ir_40traj_newmodels_polar_lam010   0.1  polar"
  "14_cavity_ir_40traj_newmodels_polar_lam001   0.01 polar"
  "18_cavity_ir_40traj_newmodels_polar_lam002   0.02 polar"
  "19_cavity_ir_40traj_newmodels_polar_lam003   0.03 polar"
)
ALL_CASES=(
  10_out_of_cavity_ir_40traj_newmodels
  11_cavity_ir_40traj_newmodels_nopolar_lam010
  13_cavity_ir_40traj_newmodels_nopolar_lam001
  16_cavity_ir_40traj_newmodels_nopolar_lam002
  17_cavity_ir_40traj_newmodels_nopolar_lam003
  12_cavity_ir_40traj_newmodels_polar_lam010
  14_cavity_ir_40traj_newmodels_polar_lam001
  18_cavity_ir_40traj_newmodels_polar_lam002
  19_cavity_ir_40traj_newmodels_polar_lam003
)

echo "########## (A) reconstruct dressed-dipole spectra (cavity cases) ##########"
for row in "${CAV_CASES[@]}"; do
  set -- $row; case_dir=$1; lam=$2; chi=$3
  echo "--- $case_dir  lambda=$lam chi=$chi ---"
  $PY reconstruct_dressed_dipole_spectra.py "$JR/$case_dir" \
      --lambda-value "$lam" --chi "$chi" --dt-ps 0.0005 --ntraj 40 \
      || echo "  [FAIL] reconstruct $case_dir"
done

echo "########## (B) ir_from_cboamd bare x spectra (all cases) + average ##########"
for case_dir in "${ALL_CASES[@]}"; do
  echo "--- $case_dir : per-traj bare spectra ---"
  for i in $(seq -w 0 39); do
    $PY ir_from_cboamd.py \
        --input  "$JR/$case_dir/traj_${i}/cboamd_output.dat" \
        --output "$JR/$case_dir/traj_${i}/spectrum_direct_x_corrected.dat" \
        --dt-ps 0.0005 --components x --method direct --window none \
        || echo "  [FAIL] ir traj_${i}"
  done
  # average the 40 per-traj bare spectra on the common grid
  $PY - "$JR/$case_dir" <<'PY'
import sys, glob, numpy as np
case = sys.argv[1]
def hamming_smooth(v, w=11):
    p = np.r_[v[w-1:0:-1], v, v[-2:-w-1:-1]]
    k = np.hamming(w); s = np.convolve(k/k.sum(), p, mode="valid")
    return s[w//2-1:-w//2]
files = sorted(glob.glob(f"{case}/traj_*/spectrum_direct_x_corrected.dat"))
freq=None; specs=[]
for fn in files:
    d=np.loadtxt(fn, comments="#"); f=d[:,0]; s=d[:,1]
    if freq is None: freq=f
    elif len(f)!=len(freq) or not np.allclose(f,freq): raise SystemExit(f"grid mismatch {fn}")
    specs.append(s)
st=np.vstack(specs); mean=st.mean(0); std=st.std(0,ddof=1); sem=std/np.sqrt(len(specs))
ms,ss,es=hamming_smooth(mean),hamming_smooth(std),hamming_smooth(sem)
np.savetxt(f"{case}/spectrum_direct_x_avg40.dat", np.column_stack([freq,mean,std,sem]),
           header=f"freq_cm-1 mean std sem components=x method=direct window=none n_spectra={len(specs)} dipole=bare")
np.savetxt(f"{case}/spectrum_direct_x_avg40_hamming11.dat",
           np.column_stack([freq,mean,std,sem,ms,ss,es]),
           header=f"freq_cm-1 mean std sem mean_h11 std_h11 sem_h11 components=x method=direct window=none n_spectra={len(specs)} dipole=bare")
# report the asymmetric-stretch peak (1800-3200 cm-1)
m=(freq>=1800)&(freq<=3200); pk=freq[m][np.argmax(ms[m])]
print(f"  {case.split('/')[-1]}: bare stretch peak (hamming) = {pk:.1f} cm^-1, {len(specs)} spectra")
PY
done

echo "########## DONE ##########"
