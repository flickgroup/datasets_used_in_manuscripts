"""Figure 1 of Flick, Phys. Rev. Lett. 129, 143201 (2022).

Reproduces the figure as published, which is the proof-stage revision: panels (b) and (d)
show the difference from OEP as a percentage. The submitted version showed it in eV with two
zoomed insets, so this will not match the figure in the arXiv v1 PDF.

Run from this directory:  python3 plot_fig1.py
"""

from matplotlib import pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec

# Hartree in eV, as octopus defines it: 2 * P_Ry with P_Ry = 13.60569193
# (src/basic/global.F90, src/basic/unit_system.F90). The original scripts imported this
# from a personal constants module; inlined so this folder depends on nothing outside
# numpy and matplotlib.
P_Har = 2.0 * 13.60569193

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})

hartreeev = P_Har

gs1 = gridspec.GridSpec(2,2)
gs1.update(hspace=0.1, wspace = 0.4,left = 0.12, right = 0.97,bottom=0.15,top=0.94)   
    
data = np.loadtxt('data/berillium_energy-sort.dat')
sortaa = np.argsort(data[:,0])
data = data[sortaa,:]
data = data*hartreeev

ax1 = plt.subplot(gs1[0,0])

ax1.plot(data[1:,0], data[1:,1]-data[0,1],'k',label='KLI')
ax1.plot(data[1:,0], data[1:,2]-data[0,2],'b:',label='OEP')
ax1.plot(data[1:,0], data[1:,3]-data[0,3],'r--',label='GA')
ax1.plot(data[1:,0], data[1:,4]-data[0,4],'g-.',label='OEP-ss')

ax1.legend(loc='upper right')
#ax1.set_xlabel(r"$\omega_\alpha$ [eV]")
ax1.set_ylabel(r"$E_x$ [eV]")
ax1.set_title('Absolute')
ax1.set_ylim(0,0.7)
###

ax2 = plt.subplot(gs1[0,1])
ax2.plot(data[1:,0], 100.*abs((data[1:,1]-data[0,1]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2])),'k',label='KLI')
ax2.plot(data[1:,0], 100.*abs((data[1:,3]-data[0,3]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2])),'r--',label='GA')
ax2.plot(data[1:,0], 100.*abs((data[1:,4]-data[0,4]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2])),'g-.',label='OEP-ss')
ax2.legend(loc='upper right')
#ax2.set_xlabel(r"$\omega_\alpha$ [eV]")
ax2.set_ylabel(r"$\Delta E_x$ [%]")
ax2.set_title('Relative to OEP')


###

data = np.loadtxt('data/LiH_energy-sort.dat')
sortaa = np.argsort(data[:,0])
data = data[sortaa,:]

data = data*hartreeev

ax3 = plt.subplot(gs1[1,0])

ax3.plot(data[1:,0], data[1:,1]-data[0,1],'k',label='KLI')
ax3.plot(data[1:,0], data[1:,2]-data[0,2],'b:',label='OEP')
ax3.plot(data[1:,0], data[1:,3]-data[0,3],'r--',label='GA')
ax3.plot(data[1:,0], data[1:,4]-data[0,4],'g-.',label='OEP-ss')

#ax3.legend(loc='upper right')
ax3.set_xlabel(r"$\omega_\alpha$ [eV]"+'\n'+'cavity frequency')
ax3.set_ylabel(r"$E_x$ [eV]")
#ax3.set_title('LiH Absolute')
ax3.set_ylim(0,0.6)

###

ax4 = plt.subplot(gs1[1,1])
ax4.plot(data[1:,0], abs((data[1:,1]-data[0,1]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2]))*100.,'k',label='KLI')
ax4.plot(data[1:,0], abs((data[1:,3]-data[0,3]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2]))*100.,'r--',label='GA')
ax4.plot(data[1:,0], abs((data[1:,4]-data[0,4]-data[1:,2]+data[0,2])/(data[1:,2]-data[0,2]))*100.,'g-.',label='OEP-ss')

#ax4.plot(data[1:,0], data[1:,1]-data[0,1]-data[1:,2]+data[0,2],'k',label='ptKLI')
#ax4.plot(data[1:,0], data[1:,3]-data[0,3]-data[1:,2]+data[0,2],'r--',label='ptGGA')
#ax4.plot(data[1:,0], data[1:,4]-data[0,4]-data[1:,2]+data[0,2],'g-.',label='ptOEP-ss')
#ax4.legend(loc='lower right')
ax4.set_xlabel(r"$\omega_\alpha$ [eV]"+'\n'+'cavity frequency')
ax4.set_ylabel(r"$\Delta E_x$ [%]")
#ax4.set_title('LiH Relative to OEP')

ax1.set_xlim([0,56])
ax2.set_xlim([0,56])
ax3.set_xlim([0,56])
ax4.set_xlim([0,56])

ax1.set_xticklabels([])
ax2.set_xticklabels([])

ax1.set_xticks([0,10,20,30,40,50])
ax2.set_xticks([0,10,20,30,40,50])
ax3.set_xticks([0,10,20,30,40,50])
ax4.set_xticks([0,10,20,30,40,50])

ax1.text(3,0.04,'(a) Beryllium')
ax2.text(25,1,'(b) Beryllium')
ax3.text(3,0.03,'(c) LiH')
ax4.text(35,10.5,'(d) LiH')

plt.savefig('figures/Fig1_atoms.png',dpi=200)
plt.savefig('figures/Fig1_atoms.pdf',dpi=200)

