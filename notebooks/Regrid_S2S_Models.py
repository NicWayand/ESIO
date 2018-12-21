
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
# S2S and C3S Model Regrid

# - Loads in all daily forecasts of sea ice extent
# - Regrids to polar stereographic,
# - Saves to netcdf files grouped by year


# In[2]:




# Standard Imports



import matplotlib
import scipy
import matplotlib.pyplot as plt
import datetime
#import cartopy.crs as ccrs
#from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
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

# ESIO Imports
from esio import EsioData as ed
from esio import import_data
from esio import ice_plot


# In[3]:


# from dask.distributed import Client
# client = Client()
# client


# In[4]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[5]:


E = ed.EsioData.load()
# Directories
all_models = ['ecmwfsipn','ukmetofficesipn','bom', 'ncep', 'ukmo', 
              'eccc', 'kma', 'cma', 'ecmwf', 'hcmr', 'isaccnr',
              'jma', 'metreofr'] 
# all_models = [ 'ukmetofficesipn']
updateall = False
cvar = 'sic'
stero_grid_file = E.obs['NSIDC_0051']['grid']


# In[6]:


obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[7]:


# Regridding Options
method='nearest_s2d' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[8]:


# Store dictionary to convert each model variable names to sipn syntax
var_dic = {}

var_dic['cma'] = {'.*ICEC':'sic'}
# 'initial_time0_hours':'init_time',
#                  'lat_0':'lat', 'lon_0':'lon',
#                  'forecast_time0':'fore_time',
# guess it looks like cma
for model in all_models:
    var_dic[model] = var_dic['cma']
# Set models that are different
#var_dic['bom'] = {'ICEC_P11_L1_GGA0_avg24h':'sic'}
# 'initial_time0_hours':'init_time',
#                  'lat_0':'lat', 'lon_0':'lon',
#                  'forecast_time0':'fore_time',
# C3S models
var_dic['ukmetofficesipn'] = {'.*CI_GDS0_SFC':'sic'}
var_dic['ecmwfsipn'] = {'.*CI_GDS0_SFC':'sic'}

# list of models that have month init times
monthly_init_model = ['ecmwfsipn', 'ukmetofficesipn', 'metreofr']


# In[9]:


## TODO
# - Get lat lon bounds 


# In[10]:


def test_plot():
    # Set up plotting info
    cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
    cmap_sic.set_bad(color = 'red')

    # Plot original projection
    plt.figure(figsize=(20,10))
    ax1 = plt.axes(projection=ccrs.PlateCarree())
    ds_p = ds.sic.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
    ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                     vmin=0, vmax=1,
                                     cmap=matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10)),
                        transform=ccrs.PlateCarree());
    ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
    gl = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
    gl.xlabels_bottom = True
    gl.ylabels_left = True
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax1.coastlines(linewidth=0.75, color='black', resolution='50m');
    plt.title(model)
    
    # Plot SIC on target projection
    (f, ax1) = ice_plot.polar_axis()
    f.set_size_inches((10,10))
    ds_p = ds.sic.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
    ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                         transform=ccrs.PlateCarree(),
                                         cmap=cmap_sic)
    ax1.set_title('Orginal Grid')

    # Plot SIC on target projection
    (f, ax1) = ice_plot.polar_axis()
    ds_p2 = var_out.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
    ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                         transform=ccrs.PlateCarree(),
                                         cmap=cmap_sic)
    ax1.set_title('Target Grid')


# In[ ]:


# For both forecast and reforecast
for runType in ['forecast','reforecast']: 
    print('Working on...',runType)
    # For each model
    for model in all_models:
        print('Regridding ', model, '...')

        data_dir = E.model[model][runType]['native']
        data_out = E.model[model][runType]['sipn_nc']
        model_grid_file = E.model[model]['grid']

        all_files = glob.glob(os.path.join(data_dir, '*.grib'))
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

        weights_flag = False # Flag to set up weights have been created



        # Load land/sea mask file
        if model_grid_file.split('/')[-1]!='MISSING':
            ds_mask = xr.open_mfdataset(model_grid_file, autoclose=True)
        else:
            ds_mask = None

        # For each file
        for cf in all_files:
            # Check if already imported and skip (unless updateall flag is True)
            # Always import the most recent two months of files (because they get updated)
            f_out = os.path.join(data_out, os.path.basename(cf).split('.')[0]+'_Stereo.nc') # netcdf file out 
            if not updateall:
                if (os.path.isfile(f_out)) & (cf not in all_files[-2:]):
                    print("Skipping ", os.path.basename(cf), " already imported.")
                    continue # Skip, file already imported

            ds = xr.open_dataset(cf, engine='pynio')

            # Some grib files do not have a init_time dim, because its assumed for the month
            if model in monthly_init_model:
                c_coords = list(ds.coords.dims.keys())
                tar_coords = list(filter(re.compile('.*initial_time').match, c_coords))
                if len(tar_coords)==0: # Check if we have no initial_time* coordinate
                    print('Adding init_time as decoder failed to get it.....')
                    ds.coords['initial_time1_hours'] = datetime.datetime(int(cf.split('.')[0].split('_')[1]), 
                                                                  int(cf.split('.')[0].split('_')[2]), 1)
                    ds = ds.expand_dims('initial_time1_hours')

            # Test we have initial_time0_hours or initial_time1_hours
            if ('initial_time0_hours' not in ds.coords) & ('initial_time1_hours' not in ds.coords):
                print('initial_time... not found in file: ',cf,' Skipping it, need to FIX!!!!!!!!')
                continue

            # Rename variables per sipn guidelines
            #ds.rename(var_dic[model], inplace=True);
            ds = import_data.rename_vars(ds=ds, var_dict=var_dic[model])
            # Rename coords
            ds = import_data.rename_coords(ds)

            # Apply masks (if available)
            if ds_mask:
                # land_mask is the fraction of native grid cell that is land
                # (1-land_mask) is fraction ocean
                # Multiply sic by fraction ocean to get actual native grid cell sic
                # Also mask land out where land_mask==1
                ds[cvar] = ds[cvar] * (1 - ds_mask.land_mask.where(ds_mask.land_mask<0.5)) # Use 50% thresshold for ocean area vs land area


        #     # Set sic below 0 to 0
        #     if X.sic.min().values < 0:
        #         print("Some negative SIC "+str(X.sic.min().values)+" found in input PIOMAS, setting to 0")
        #         X = X.where(X>=0, other=0)

        #     # Apply model mask
        #     X = X.where(X.imask)

            # Check only data from one month (download bug)
            cm = pd.to_datetime(ds.init_time.values).month
            if model not in monthly_init_model:
                if np.diff(cm).max() > 0:
                    fm = int(cf.split('.')[0].split('_')[2]) # Target month in file
                    print("Found dates outside month, removing...")
                    ds = ds.where(xr.DataArray(pd.to_datetime(ds.init_time.values).month==fm,
                                           dims=ds.init_time.dims, coords=ds.init_time.coords), drop=True)

            # Calculate regridding matrix
            regridder = xe.Regridder(ds, obs_grid, method, periodic=True, reuse_weights=weights_flag)
            weights_flag = True # Set true for following loops

            # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
            if method=='conservative':
                print('Removing edge cells that contain source NaN cells, should probably check here')
                regridder = import_data.add_matrix_NaNs(regridder)

            # Regrid variable
            var_out = regridder(ds[cvar])

            # Expand dims
            var_out = import_data.expand_to_sipn_dims(var_out)

            #test_plot()
            #xr.exit()

            # # Save regridded to netcdf file
            var_out.to_netcdf(f_out)
            var_out = None # Memory clean up
            ds = None
            print('Saved ', f_out)


        # End of all files for current model
        # Clean up data from this model
        ds_mask = None

        # Clean up
        if weights_flag:
            regridder.clean_weight_file()  # clean-up    


# In[ ]:


# cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
# cmap_sic.set_bad(color = 'red')


# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# ds_mask.land_mask.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap='Blues')


# In[ ]:


# cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
# cmap_sic.set_bad(color = 'red')


# # # Plot SIC on target projection
# # (f, ax1) = ice_plot.polar_axis()
# # f.set_size_inches((10,10))
# # ds_mask.land_mask.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
# #                                      transform=ccrs.PlateCarree(),
# #                                      cmap='Blues')

# # # Plot SIC on target projection
# # (f, ax1) = ice_plot.polar_axis()
# # f.set_size_inches((10,10))
# # mask_out.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
# #                                      transform=ccrs.PlateCarree(),
# #                                      cmap='Blues')

# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# ds_p = ds.sic.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
# ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Orginal Grid')

# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# ds_p2 = var_out.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
# ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Target Grid')


# In[ ]:




# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# (1 - ds_mask.land_mask.where(ds_mask.land_mask<1)).plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('mask we use')


# In[ ]:





