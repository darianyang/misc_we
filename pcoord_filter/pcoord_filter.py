import numpy as np

# Step 1: Load data from each file
tt = np.loadtxt('tt_dist.dat', skiprows=1, usecols=1)
m1x1 = np.loadtxt('M1_W184_chi12.dat', skiprows=1, usecols=1)
m2x2 = np.loadtxt('M2_W184_chi12.dat', skiprows=1, usecols=1)

# Step 2: Compare arrays and replace tt values if conditions met
condition_mask = np.logical_and(m1x1 > -95, 
                                np.logical_and(m1x1 < -40, 
                                               np.logical_and(m2x2 > -95, m2x2 < -40)))
tt[condition_mask] = -1

# Step 3: Save tt array as a text file
np.savetxt('tt_dist_pcoord.txt', tt, fmt='%.4f')

print("File saved: tt_dist_pcoord.txt")
