#!/bin/bash
# create dirs script for an octopus 
# (c) Johannes Flick

mkdir LDA
cd LDA
cp ../inp_common inp
cp ../geometry.xyz .
cp ../jobscript_LDA.sbatch jobscript.sbatch
sed -e "s/XCFunctional = oep_x/XCFunctional = lda_x + lda_c_pz_mod/g"  inp >  inp2
sed -e "s/MixingScheme = linear/ /g"  inp2 >  inp
rm -rf inp2
cd ..

mkdir KLI
cd KLI
cp ../inp_common inp
cp ../geometry.xyz .
cp ../jobscript_KLI.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
mv inp2 inp
cd ..

mkdir OEP
cd OEP
cp ../inp_common inp
cp ../geometry.xyz .
cp ../jobscript_KLI.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
sed -e "s/OEPLevel = 3/OEPLevel = 5/g"  inp2 >  inp
rm -rf inp2
cd ..

for ii in {1..20}
do
   echo "Step $ii "
   omega=$(python3 -c "print($ii/50)")
   echo "omega: $omega"

  mkdir f$ii
  cd f$ii

mkdir KLI-pt
cd KLI-pt
cp ../../inp_common inp
cp ../../geometry.xyz .
cp ../../jobscript_KLI_inside.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
sed -e "s/PhotonsOEP = no/PhotonsOEP = yes/g"  inp2 >  inp
sed -i "s/0.288611332750 | 0.1 | 1 | 0 | 0/$omega | 0.1 | 1 | 0 | 0/g" inp
rm -rf inp2
cd ..

mkdir OEP-pt
cd OEP-pt
cp ../../inp_common inp
cp ../../geometry.xyz .
cp ../../jobscript_KLI_inside.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
sed -e "s/OEPLevel = 3/OEPLevel = 5/g"  inp2 >  inp3
sed -e "s/PhotonsOEP = no/PhotonsOEP = yes/g"  inp3 >  inp
sed -i "s/0.288611332750 | 0.1 | 1 | 0 | 0/$omega | 0.1 | 1 | 0 | 0/g" inp
rm -rf inp2 inp3
cd ..

mkdir OEP-gga
cd OEP-gga
cp ../../inp_common inp
cp ../../geometry.xyz .
cp ../../jobscript_KLI_inside.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
sed -e "s/OEPLevel = 3/OEPLevel = 5/g"  inp2 >  inp3
sed -e "s/Photons = no/Photons = yes/g"  inp3 >  inp
rm -rf inp2 inp3
sed -i "s/PhotonDens = no/PhotonDens = yes/g" inp
sed -i "s/PhotonDensGradCorr = no/PhotonDensGradCorr = yes/g" inp
sed -i "s/PhotonSkipVxc = no/PhotonSkipVxc = yes/g" inp
sed -i "s/0.288611332750 | 0.1 | 1 | 0 | 0/$omega | 0.1 | 1 | 0 | 0/g" inp
cd ..

mkdir OEP-pt-single-shot
cd OEP-pt-single-shot
cp ../../inp_common inp
cp ../../geometry.xyz .
cp ../../jobscript_KLI_inside.sbatch jobscript.sbatch
sed -e "s/FromScratch = yes/FromScratch = no/g"  inp >  inp2
sed -e "s/OEPLevel = 3/OEPLevel = 5/g"  inp2 >  inp3
sed -e "s/PhotonsOEP = no/PhotonsOEP = yes/g"  inp3 >  inp
rm -rf inp2 inp3
sed -i "s/PhotonSkipVxc = no/PhotonSkipVxc = yes/g" inp
sed -i "s/0.288611332750 | 0.1 | 1 | 0 | 0/$omega | 0.1 | 1 | 0 | 0/g" inp
cd ..

cd ..
done

