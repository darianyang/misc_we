"""
Got through a h5 file and convert to less frames saved.
e.g. from 101 frames to 11.
"""

import h5py
import numpy as np
from tqdm.auto import tqdm

# make new h5 file copy
import shutil
shutil.copy("i500.h5", "WT_v00_11f.h5")

# open, read + write
f = h5py.File("WT_v00_11f.h5", "r+")
# find all aux names along with pcoord
# note that here using 1d pcoord so easier to deal with
auxnames = ["pcoord"] + list(f["iterations/iter_00000001/auxdata/"])

# loop each iteration of h5 file
for iter in tqdm(range(1, 501)):
    # loop each aux dataset / pcoord
    for aux in auxnames:
        # account for aux vs pcoord paths
        if aux != "pcoord":
            aux = "auxdata/" + aux
        data =f[f"iterations/iter_{iter:08d}/{aux}"][:]

        # 2d datasets
        if aux != "pcoord":
            # reshape from n_segs, 101, 1 to n_segs, 11, 1
            data_f = data[:, ::10]
        # 3d pcoord dataset
        elif aux == "pcoord":
            # reshape from n_segs, 101, 1 to n_segs, 11, 1
            data_f = data[:, ::10, :]

        # write/replace with new dataset if not already filtered
        if data.shape[1] == 101 and data_f.shape[1] == 11:
            # resize to smaller size array (takes first n points truncated)
            f[f"iterations/iter_{iter:08d}/{aux}"].resize(data_f.shape)
            # to recover original properly intervaled reduced dataset
            f[f"iterations/iter_{iter:08d}/{aux}"][:] = data_f

# close h5 file
f.close()