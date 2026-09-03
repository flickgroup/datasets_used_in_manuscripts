#!/bin/bash
#SBATCH --job-name="octopus"
#SBATCH --output="octopus.%j.%N.out"
#SBATCH --partition=gen,ccq #ccq
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=28
#SBATCH --export=ALL
#SBATCH -t 2:30:00
#SBATCH --constraint=skylake

module purge
module load intel/compiler/2017-4 intel/mpi/2017-4 intel/mkl/2017-4 lib/fftw3/3.3.8-intelmpi lib/hdf5/1.12.0-intelmpi 


OCTOPUS=/mnt/home/jflick/opt/octopus/octopus-flick/octopus-density/bin/octopus

cp -r ../LDA/restart .
mpiexec $OCTOPUS &> oct.out

