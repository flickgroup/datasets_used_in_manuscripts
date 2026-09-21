#!/bin/bash
# One reference PBE0-D3 single point, as used for the liquid CO2 labels.
# The production runs used 128 MPI ranks; adapt the launcher to your machine.

export CP2K=/path/to/cp2k_psmp.sif   # CP2K 2025.1, OpenMPI generic psmp
export OMP_NUM_THREADS=1

mpiexec -n 128 apptainer exec --bind $PWD:/workdir --pwd /workdir $CP2K cp2k.popt -i input > output
