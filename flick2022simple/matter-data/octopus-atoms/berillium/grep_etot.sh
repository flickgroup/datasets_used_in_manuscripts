#!/bin/bash
# create dirs script for an octopus 
# (c) Johannes Flick

rm energy.dat

module load gcc python3

energy=$(grep 'Total       =' OEP/static/info | sed 's/Total       =//g')
energy_OEPpt=$(echo $energy_OEP $energy)
energy_KLIpt=$(echo $energy_OEP $energy)
energy_GGA=$(echo $energy_OEP $energy)
energy_OEPptss=$(echo $energy_OEP $energy)


omega_list=$(echo "0")


for ii in {1..20} #35
do
   echo "Step $ii "
   cd $ii
   omega=$(python3 -c "print($ii/10)")
   echo "omega: $omega"
   omega_list=$(echo $omega_list $omega)


   energy=$(grep 'Total       =' KLI-pt/static/info | sed 's/Total       =//g')
   energy_KLIpt=$(echo $energy_KLIpt $energy)

   energy=$(grep 'Total       =' OEP-pt/static/info | sed 's/Total       =//g')
   energy_OEPpt=$(echo $energy_OEPpt $energy)

   energy=$(grep 'Total       =' OEP-gga/static/info | sed 's/Total       =//g')
   energy_GGA=$(echo $energy_GGA $energy)

   energy=$(grep 'Total       =' OEP-pt-single-shot/static/info | sed 's/Total       =//g')
   energy_OEPptss=$(echo $energy_OEPptss $energy)
   cd ..
done

for ii in {1..20} #35
do
   echo "Step $ii "
   cd f$ii
   omega=$(python3 -c "print($ii/50)")
   echo "omega: $omega"
   omega_list=$(echo $omega_list $omega)


   energy=$(grep 'Total       =' KLI-pt/static/info | sed 's/Total       =//g')
   energy_KLIpt=$(echo $energy_KLIpt $energy)

   energy=$(grep 'Total       =' OEP-pt/static/info | sed 's/Total       =//g')
   energy_OEPpt=$(echo $energy_OEPpt $energy)

   energy=$(grep 'Total       =' OEP-gga/static/info | sed 's/Total       =//g')
   energy_GGA=$(echo $energy_GGA $energy)

   energy=$(grep 'Total       =' OEP-pt-single-shot/static/info | sed 's/Total       =//g')
   energy_OEPptss=$(echo $energy_OEPptss $energy)
   cd ..
done


echo $omega_list >> energy.dat
echo $energy_KLIpt >> energy.dat
echo $energy_OEPpt >> energy.dat
echo $energy_GGA >> energy.dat
echo $energy_OEPptss >> energy.dat



