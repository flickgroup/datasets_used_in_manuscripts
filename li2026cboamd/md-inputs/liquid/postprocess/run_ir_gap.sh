#!/bin/bash
# Postprocess the gap-fill cavity cases (lambda=0.04..0.09, nopolar+polar):
# (A) dressed-dipole reconstruction + averaged spectra, (B) bare x spectra +
# average. Mirrors run_ir.sh (which covered cases 10-19). Idempotent.
set -o pipefail
# Activate the environment that provides numpy here.
# Activate an environment providing numpy here.
cd "$(dirname "$0")"
JR=".."
PY=python

CAV_CASES=(
  "20_cavity_ir_40traj_newmodels_nopolar_lam004 0.04 nopolar"
  "21_cavity_ir_40traj_newmodels_nopolar_lam005 0.05 nopolar"
  "22_cavity_ir_40traj_newmodels_nopolar_lam006 0.06 nopolar"
  "23_cavity_ir_40traj_newmodels_nopolar_lam007 0.07 nopolar"
  "24_cavity_ir_40traj_newmodels_nopolar_lam008 0.08 nopolar"
  "25_cavity_ir_40traj_newmodels_nopolar_lam009 0.09 nopolar"
  "26_cavity_ir_40traj_newmodels_polar_lam004   0.04 polar"
  "27_cavity_ir_40traj_newmodels_polar_lam005   0.05 polar"
  "28_cavity_ir_40traj_newmodels_polar_lam006   0.06 polar"
  "29_cavity_ir_40traj_newmodels_polar_lam007   0.07 polar"
  "30_cavity_ir_40traj_newmodels_polar_lam008   0.08 polar"
  "31_cavity_ir_40traj_newmodels_polar_lam009   0.09 polar"
)

echo "########## (A) reconstruct dressed-dipole spectra ##########"
for row in "${CAV_CASES[@]}"; do
  set -- $row; case_dir=$1; lam=$2; chi=$3
  echo "--- $case_dir  lambda=$lam chi=$chi ---"
  $PY reconstruct_dressed_dipole_spectra.py "$JR/$case_dir" \
      --lambda-value "$lam" --chi "$chi" --dt-ps 0.0005 --ntraj 40 \
      || echo "  [FAIL] reconstruct $case_dir"
done

echo "########## (B) bare x spectra + average ##########"
for row in "${CAV_CASES[@]}"; do
  set -- $row; case_dir=$1
  echo "--- $case_dir : per-traj bare spectra ---"
  for i in $(seq -w 0 39); do
    $PY ir_from_cboamd.py \
        --input  "$JR/$case_dir/traj_${i}/cboamd_output.dat" \
        --output "$JR/$case_dir/traj_${i}/spectrum_direct_x_corrected.dat" \
        --dt-ps 0.0005 --components x --method direct --window none \
        || echo "  [FAIL] ir traj_${i}"
  done
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
m=(freq>=1800)&(freq<=3200); pk=freq[m][np.argmax(ms[m])]
print(f"  {case.split('/')[-1]}: bare stretch peak (hamming) = {pk:.1f} cm^-1, {len(specs)} spectra")
PY
done
echo "########## DONE ##########"
