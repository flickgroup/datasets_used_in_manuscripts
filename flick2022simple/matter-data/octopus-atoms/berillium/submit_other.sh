#!/bin/bash
# create dirs script for an octopus 
# (c) Johannes Flick

module load gcc python3

for ii in {17..20} #35
do
   echo "Step $ii "
cd $ii

cd KLI-pt
sbatch jobscript.sbatch
cd ..

cd OEP-pt
sbatch jobscript.sbatch
cd ..

cd OEP-gga
sbatch jobscript.sbatch
cd ..

cd OEP-pt-single-shot
sbatch jobscript.sbatch
cd ..


cd ..
done

