import numpy as np

data = np.loadtxt('energy.dat')
data = data.T

np.savetxt('energy-sort.dat', data)

