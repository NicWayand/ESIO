
# coding: utf-8

# In[ ]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''


# In[ ]:


# Standard Imports



import matplotlib
import scipy
import matplotlib.pyplot as plt
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import os
import re
import glob
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import dask
# dask.set_options(get=dask.threaded.get)
from dask.distributed import Client

# ESIO Imports

from esio import EsioData as ed
from esio import metrics


# In[ ]:


def Update_Model_Aggs():
    
    E = ed.EsioData.load()
    model_dir = E.model_dir
    # Directories
    # Define models to plot
    all_models = list(E.model.keys())
    all_models = [x for x in all_models if x not in ['piomas','MME']] # remove some models
#     all_models = ['uclsipn']
    runType='forecast'
    updateall = False

    ds_region = xr.open_mfdataset(os.path.join(E.grid_dir, 'sio_2016_mask_Update.nc'))

    for model in all_models:
        print(model)

        data_dir = E.model[model][runType]['sipn_nc']
        data_out = os.path.join(model_dir, model, runType, 'sipn_nc_agg')
        if not os.path.exists(data_out):
            os.makedirs(data_out)

        all_files = glob.glob(os.path.join(data_dir, '*.nc'))
        print("Found ",len(all_files)," files.")
        if updateall:
            print("Updating all files...")
        else:
            print("Only updating new files")

        # Remove any "empty" files (sometimes happends with ecmwf downloads)
        all_files_new = []
        for cf in all_files:
            if os.stat(cf).st_size > 0:
                all_files_new.append(cf)
            else:
                print("Found empty file: ",cf,". Consider deleting or redownloading.")
        all_files = sorted(all_files_new) # Replace and sort

        # For each file
        for cf in all_files:
            # Check if already imported and skip (unless updateall flag is True)
            # Always import the most recent two months of files (because they get updated)
            f_out = os.path.join(data_out, os.path.basename(cf)) # netcdf file out 
            if not updateall:
                 if (os.path.isfile(f_out)) & (cf not in all_files[-2:]):
                    print("Skipping ", os.path.basename(cf), " already imported.")
                    continue # Skip, file already imported

            ds = xr.open_mfdataset(cf , chunks={'fore_time':10, 'ensemble': 5, 'init_time': 10, 'nj': 304, 'ni': 448},
                                  parallel=True) # Works but is not eiffecent 5-15 mins wall time
            ds.rename({'nj':'x', 'ni':'y'}, inplace=True)

            # Calc panArctic extent
            da_panE = metrics.calc_extent(da=ds.sic, region=ds_region)
            da_panE['nregions'] = 99
            da_panE['region_names'] = 'panArctic'

            # Calc Regional extents
            da_RegE = metrics.agg_by_domain(da_grid=ds.sic, ds_region=ds_region)

            # Merge
            ds_out = xr.concat([da_panE, da_RegE], dim='nregions')
            ds_out.name = 'Extent'

            ds_out.load() # This prevents many errors in the dask graph (I don't know why)

            # # Save regridded to netcdf file

            ds_out = None # Memory clean up
            da_panE = None
            da_RegE = None
            ds = None
            print('Saved ', f_out)


        print("Finished...")


# In[ ]:


if __name__ == '__main__':
    # Start up Client
    client = Client(processes=12)
    print(client)
    
    # Call function
    Update_Model_Aggs()


