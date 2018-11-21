
# coding: utf-8

# In[1]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''





import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import xarray as xr
import glob
import datetime
from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
import dask
from dask.distributed import Client
import timeit
import zarr


# In[2]:


E = ed.EsioData.load()
obs_dir = E.obs_dir
obs_dir


# In[16]:


product_list = ['NSIDC_0081' , 'NSIDC_0079', 'NSIDC_0051']

# Loop through each dataset
for c_product in product_list:
    
    # Load in netcdf files as dataset
    ds = xr.open_mfdataset(E.obs[c_product]['sipn_nc']+'_yearly/*.nc', 
                              concat_dim='time', autoclose=True, parallel=True)
    ds = ds.chunk({'time':1}) # Bug in Zarr, remove once fixed (https://github.com/pydata/xarray/pull/2487)

    # Save to Zarr file
    ds.to_zarr(os.path.join(obs_dir,'zarr',c_product), 'w')
    
    print("Done with",c_product)


