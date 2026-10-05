#!/bin/bash
#SBATCH --job-name="pmbd"
#SBATCH --output="pmbd.%j.%N.out"
#SBATCH --partition=ccq
#SBATCH --nodes=1
#SBATCH --export=ALL
#SBATCH --constraint='rome'
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=128
#SBATCH --ntasks-per-core=1

export OMP_NUM_THREADS=1
ulimit -s unlimited

# loading modules for VASP 6.*
module purge
module load slurm
module load python/3.11.11 
module load gcc openmpi fftw intel-oneapi-mkl hdf5/mpi git
module list

export OMP_NUM_THREADS=1
ulimit -s unlimited

source ~/ceph/python_projects/pmbd/.venv/bin/activate

python ~/ceph/python_projects/pmbd/run_pmbd_monolayer.py $lambda
