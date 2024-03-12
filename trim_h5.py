import h5py

auxnames = ['1_75_39_c2', 'M1Oe_M2He1', 'M2Oe_M1He1', 'Mono1_SASA', 'Mono2_SASA', 'Num_Inter_NC', 'Num_Inter_NNC', 'Num_Intra_NC', 'Num_Intra_NNC', 'RMS_Backbone', 'RMS_Dimer_Int', 'RMS_Heavy', 'RMS_Key_Int', 'RMS_Mono1', 'RMS_Mono2', 'RoG', 'Secondary_Struct', 'Total_SASA', 'XTAL_REF_RMS_Heavy', 'fit_m1_rms_heavy_h9m2', 'fit_m1_rms_heavy_m2', 'fit_m2_rms_heavy_h9m1', 'fit_m2_rms_heavy_m1']

auxnames = ['auxdata/' + i for i in auxnames]

auxnames += ['pcoord']

# Open the HDF5 file in read-write mode
with h5py.File('west.h5', 'r+') as file:

    for iter in range(1,201):
    
        iter_prefix = f"iterations/iter_{iter:08d}"
    
        # Loop through the list of dataset names
        for auxname in auxnames:
            # Create a new dataset with the same data
            #print(iter_prefix, auxname)
            new_data = file[f"{iter_prefix}/{auxname}_new"][:,::10]
            #print(new_data.shape)
            #import sys ; sys.exit(0)
            new_dataset = file.create_dataset(f'{iter_prefix}/{auxname}', data=new_data)

            # Optionally, you can copy attributes from the old dataset to the new one
            for attr_name, attr_value in file[iter_prefix][auxname].attrs.items():
                new_dataset.attrs[attr_name] = attr_value

            # Delete the old dataset
            del file[iter_prefix][f"{auxname}_new"]

            # Rename the new dataset back to the original name
            #new_dataset.name = f"{iter_prefix}/{auxname}"
