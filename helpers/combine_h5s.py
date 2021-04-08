import h5py
import numpy as np
import io
from PIL import Image
import matplotlib.pyplot as plt

h5datasets = ['zx_7_d10_inmc_celebA_01.hdf5', 
              'zx_7_d10_inmc_celebA_02.hdf5',
              'zx_7_d10_inmc_celebA_03.hdf5', 
              'zx_7_d10_inmc_celebA_04.hdf5',
              'zx_7_d10_inmc_celebA_05.hdf5',
              'zx_7_d10_inmc_celebA_20.hdf5']

with h5py.File('celebA50k.hdf5',mode='w') as h5fw:
    row1 = 0
    for h5name in h5datasets:
        h5fr = h5py.File(h5name,'r') 
        arr_data = h5fr['zx_7'][:]
        print(arr_data.shape)
        h5fw.require_dataset('data', dtype="<f8",  
                                     shape=(50765, 10, 64, 64))
        h5fw['data'][row1:row1+arr_data.shape[0],:,:,:] = arr_data[:,:,:,:]
        print("Indices %d to %d filled by %s" % (row1, 
                                                 row1+arr_data.shape[0], 
                                                 h5name))
        row1 += arr_data.shape[0]