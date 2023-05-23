import h5py
import numpy

with h5py.File("standard.h5", "w") as h5file:

    h5file['/'].attrs['wemd_current_iteration'] = 2
    h5file['/'].attrs['west_current_iteration'] = 2

    grp = h5file.create_group("iterations")
    subgrp = grp.create_group("iter_00000001")

    data_d1 = numpy.array(numpy.loadtxt("data.out", skiprows=1, usecols=1))
    data_d2 = numpy.array(numpy.loadtxt("data.out", skiprows=1, usecols=2))

    nframes = data_d1.shape[0]

    data_arr = numpy.zeros((nframes, 1, 2))

    data_d1_rshp = data_d1.reshape(nframes,-1)
    data_d2_rshp = data_d2.reshape(nframes,-1)

    data_arr[:,:,0] = data_d1_rshp
    data_arr[:,:,1] = data_d2_rshp

    weights = numpy.zeros((nframes))

    div_weights = weights + (1/nframes)

    pcoord = subgrp.create_dataset("pcoord", shape=(nframes,1,2), dtype=numpy.float64)

    pcoord[...] = data_arr

    seg_index_dtype = numpy.dtype([('weight', numpy.float64)])

    seg_index_table_ds = subgrp.create_dataset('seg_index', shape=(nframes,), dtype=seg_index_dtype)

    seg_index_table = seg_index_table_ds[...]

    seg_index_table_ds['weight', :] = div_weights
