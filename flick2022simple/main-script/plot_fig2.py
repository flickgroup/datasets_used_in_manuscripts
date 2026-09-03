"""Figure 2 of Flick, Phys. Rev. Lett. 129, 143201 (2022).

Spectral densities and electron-photon energy for C20, C60 and C180.

Extracted from dissipation-wang-lambdas2.ipynb (cells 0, 1, 2, 6, 7), which is the
notebook that produced the published 02-dissipation-c60.pdf. Changes made during
extraction, none of which touch the numerics:

  * added the `import matplotlib.pyplot as plt` that the notebook only ever executed
    in cell 3, a scratch diagnostic plot that is not part of the figure
  * dropped the mode-file writer at the end of cell 2 -> make_photon_modes.py
  * dropped the single-kappa `lambda_array` loop from cell 2: cells 6 and 7 read only
    `lambda_arrayjj`, so it was 400000 iterations of dead work
  * dropped cells 3, 4 and 5 (a scratch plot and two sum-rule prints) and cell 8, which
    is an earlier version of Fig. 1 superseded at proof stage by plot_fig1.py
  * pinned figure.figsize to the Jupyter inline backend's 6 x 4 in, see below
  * repointed the data, image and output paths at data/, images/ and figures/

Do not "simplify" nom: dom enters the Lorentzian normalisation in `lorentzian`, so the
per-mode lambda amplitude scales with the grid spacing. nom = 400000 is the 400 000
photon modes quoted in the paper and it sets the panel (a) y scale.

Run from this directory:  python3 plot_fig2.py
"""

import numpy as np
# Hartree in eV, as octopus defines it: 2 * P_Ry with P_Ry = 13.60569193
# (src/basic/global.F90, src/basic/unit_system.F90). The original scripts imported this
# from a personal constants module; inlined so this folder depends on nothing outside
# numpy and matplotlib.
P_Har = 2.0 * 13.60569193

import matplotlib.pyplot as plt

# The notebook ran under the Jupyter inline backend, whose figure.figsize default is
# 6.0 x 4.0 in, not matplotlib's own 6.4 x 4.8. The published 02-dissipation-c60.png is
# 1200x800 px at dpi=200, i.e. 6 x 4 in. Nothing in the notebook states this, so it has
# to be set explicitly here or the panel proportions and inset placement come out wrong.
plt.rcParams['figure.figsize'] = [6.0, 4.0]

omegac = 3/P_Har
lambdac = 0.1
kappac = 0.01/P_Har
nom = 400000

def lorentzian(omega, kappa, omegac, dom):
    return (dom*1/2/np.pi*kappa/((omega-omegac)**2+(kappa/2)**2)) #multiply with dom?

omega_array = np.linspace(0,0.3,nom)
dom = omega_array[1] - omega_array[0]

lambda_arrayjj = np.zeros((len(omega_array),5))

jj = 0
for kappac in [0.01,0.05,0.1,0.5,1]:
    print (kappac)
    kappac = kappac/P_Har

    for ii in range(0,nom):
        lambda_arrayjj[ii,jj] = np.sqrt(lambdac**2*lorentzian(omega_array[ii], kappac, omegac, dom))
    jj+=1

import matplotlib.gridspec as gridspec

plt.rcParams.update({'font.size': 12})

gs1 = gridspec.GridSpec(1,2)
gs1.update(hspace=0.1, wspace = 0.45,left = 0.08, right = 0.95,bottom=0.18,top=0.9)

ax1 = plt.subplot(gs1[0,0])

ax1.plot(omega_array*P_Har, lambda_arrayjj[:,0], 'k:',label='0.01')
#ax1.plot(omega_array*P_Har, lambda_arrayjj[:,1])
ax1.plot(omega_array*P_Har, lambda_arrayjj[:,2], 'b--',label='0.1')
#ax1.plot(omega_array*P_Har, lambda_arrayjj[:,3])
ax1.plot(omega_array*P_Har, lambda_arrayjj[:,4], 'r',label='1')

ax1.set(xlabel=r'$\omega_\alpha$ [eV]'+'\n'+'cavity frequency', ylabel=r'$\lambda_\alpha$ [a.u,]',
       title='')
ax1.legend(loc='upper right',title=r'$\kappa$ [eV]')

##
ax2 = plt.subplot(gs1[0,1])

data20 = np.loadtxt('data/C20_photon-ex.dat')
ax2.plot(data20[:,0], data20[:,1]*P_Har/20,'b')

data60 = np.loadtxt('data/C60_photon-ex.dat')
ax2.plot(data60[:,0], data60[:,1]*P_Har/60,'k:')

data180 = np.loadtxt('data/C180_photon-ex.dat')
ax2.plot(data180[:,0], data180[:,1]*P_Har/180,'r--')



ax2.set(xlabel=r'$\kappa$ [eV]'+'\n'+'dissipation constant', ylabel=r'$E^{(GA)}_{x}/N_{at}$ [eV]',
       title='')
             
ax2.set_title('(b) Electron-photon energy')
ax1.set_title('(a) Spectral density')    

#ax1.text(0,0.035)
ax1.set_xlim([0,7])
ax1.set_xticks([0,2,4,6,6])

ax1.set_ylim([0,0.004])
ax1.set_yticks([0,0.001,0.002,0.003,0.004])
ax1.set_yticklabels([0,1,2,3,4])
ax1.text(0.3,0.0035,r'x$10^{-3}$')

#import matplotlib as mpl
#image = plt.imread(file, format='png')

# Draw image
#axin = ax2.inset_axes([0,0,4,4],transform=ax.transData)    # create new inset axes in data coordinates
#axin.imshow(image)
#axin.axis('off')

from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import matplotlib.image as mpimg

arr_lena = mpimg.imread('images/C20.png')
imagebox = OffsetImage(arr_lena, zoom=0.03)
ab = AnnotationBbox(imagebox, (0.5, 0.305),  xybox=(0.54, 0.3143),
                    xycoords='data',
                    boxcoords="data", pad=0.1,
                   arrowprops=dict(
                        arrowstyle="->", color='blue'
                        ))
ax2.add_artist(ab)

arr_lena = mpimg.imread('images/C60.png')
imagebox = OffsetImage(arr_lena, zoom=0.03)
ab = AnnotationBbox(imagebox, (0.2, 0.31),  xybox=(0.16, 0.302),
                    xycoords='data',
                    boxcoords="data", pad=0.1,
                   arrowprops=dict(
                        arrowstyle="->", color='black'
                        ))
ax2.add_artist(ab)


arr_lena = mpimg.imread('images/C180.png')
imagebox = OffsetImage(arr_lena, zoom=0.03)
ab = AnnotationBbox(imagebox, (0.8, 0.2925),  xybox=(0.4, 0.2925),
                    xycoords='data',
                    boxcoords="data", pad=0.1,
                   arrowprops=dict(
                        arrowstyle="->", color='red'
                        ))
ax2.add_artist(ab)


ax2.text(0.55,0.307,r'$C_{20}$',color='blue')
ax2.text(0.02,0.307,r'$C_{60}$',color='black')
ax2.text(0.65,0.290,r'$C_{180}$',color='red')

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

axins = zoomed_inset_axes(ax2, 26, loc=5) # zoom = 6
axins.plot(data60[:,0], data60[:,1]*P_Har/60,'k--')
axins.plot(data180[:,0], data180[:,1]*P_Har/180,'r:')
x1, x2, y1, y2 = 0.745, 0.755, 0.2955, 0.29575
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
plt.xticks(visible=False)
plt.yticks(visible=False)

# draw a bbox of the region of the inset axes in the parent axes and
# connecting lines between the bbox and the inset axes area
mark_inset(ax2, axins, loc1=2, loc2=4, fc="none", ec="0.5")

#plt.show()
plt.savefig('figures/Fig2_dissipation.png',dpi=200)
plt.savefig('figures/Fig2_dissipation.pdf',dpi=200)
