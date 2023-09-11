#!/usr/bin/env python
import numpy
from tqdm.auto import tqdm, trange

data = numpy.loadtxt('dihedral.dat', usecols=(1,2), skiprows=1)
state_list = []

for val in data:
    if val[0] >= -180 and val[0] <= -45 and val[1] >= -55 and val[1] <= 30: # Phi/Psi for Alpha Helix
         state_list.append(0)
    elif val[0] >= 165 and val[0] <= 180 and val[1] >= -55 and val[1] <= 30:
         state_list.append(0)
    elif val[0] >= -170 and val[0] <= -55 and val[1] >= 40 and val[1] <= 100: # Phi/Psi for C7eq
         state_list.append(1)
    elif val[0] >= 25 and val[0] <= 90 and val[1] >= -55 and val[1] <=0: # Phi/Psi for C7ax
         state_list.append(2)
    else:
         state_list.append(-1)

n = numpy.asarray(state_list)
numpy.save('states.npy', n)
