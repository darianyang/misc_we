import h5py
import pandas as pd
import sys

iteration = int(sys.argv[1])
iteration_path = f'iterations/iter_{iteration:08d}/auxdata'

# Step 1: Load the HDF5 file using h5py
with h5py.File('west.h5', 'r') as f:
#with h5py.File('west_5bs.h5', 'r') as f:
    # Step 2: Extract datasets from the HDF5 file
    tt = f[f'{iteration_path}/tt_dist'][:,-1]
    oa1 = f[f'{iteration_path}/o_angle_m1'][:,-1]
    oa2 = f[f'{iteration_path}/o_angle_m2'][:,-1]
    m1x1 = f[f'{iteration_path}/M1_W184_chi1'][:,-1]
    m2x1 = f[f'{iteration_path}/M2_W184_chi1'][:,-1]

# Step 3: Build a DataFrame from the datasets
df = pd.DataFrame({'tt':tt, 'oa1':oa1, 'oa2':oa2, 'm1x1':m1x1, 'm2x1':m2x1})
#print(df.head())

# Step 4: Define value conditions for each column
condition1 = df['tt'] < 5
condition2 = (df['oa1'] < 12) | (df['oa2'] < 12)
condition3 = (df['m1x1'] > -95) & (df['m1x1'] < -40) & (df['m2x1'] > -95) & (df['m2x1'] < -40) 

# Step 5: Find rows that satisfy the conditions
satisfying_rows = df[condition1 & condition2 & condition3]

print("Rows satisfying the conditions:")
print(satisfying_rows)

