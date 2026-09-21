import time, numpy as np
from pyscf import gto, dft
mol = gto.M(atom=[["O",(-1.170708,0,0)],["C",(0.01,0,0)],["O",(1.170708,0,0)]], basis="aug-cc-pvdz")
mf = dft.RKS(mol); mf.xc="pbe0"; mf.verbose=0; mf.conv_tol=1e-6
t=time.time(); e=mf.kernel(); t1=time.time()-t
t=time.time(); g=mf.nuc_grad_method().kernel(); t2=time.time()-t
import os
print(f"OMP={os.environ.get(chr(39)+chr(79)+chr(77)+chr(80)+chr(95)+chr(78)+chr(85)+chr(77)+chr(95)+chr(84)+chr(72)+chr(82)+chr(69)+chr(65)+chr(68)+chr(83)+chr(39),chr(63))} SCF {t1:.2f}s grad {t2:.2f}s e={e:.6f}")
