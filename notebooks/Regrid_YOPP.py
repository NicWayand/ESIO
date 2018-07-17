
# coding: utf-8

# In[ ]:


# # YOPP Forecast

# - Loads in all daily forecasts of sea ice extent
# - Regrids to polar stereographic,
# - Saves to netcdf files grouped by year


# In[ ]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

# Standard Imports



import matplotlib
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
import os
import glob
import seaborn as sns
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# ESIO Imports

from esio import EsioData as ed
from esio import import_data


# In[ ]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[ ]:


E = ed.EsioData.load()
# Directories
model='yopp'
runType='forecast'
updateall = False
data_dir = E.model[model][runType]['native']
data_out = E.model[model][runType]['sipn_nc']
model_grid_file = E.model[model]['grid']
stero_grid_file = E.obs['NSIDC_0051']['grid']


# In[ ]:


obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[ ]:


# Regridding Options
method='nearest_s2d' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[ ]:


## TODO
# - Get mask
# - Get lat lon bounds 


# In[ ]:


all_files = glob.glob(os.path.join(data_dir, 'yopp*ci*.grib'))
print("Found ",len(all_files)," files.")
if updateall:
    print("Updating all files...")
else:
    print("Only updating new files")


# In[ ]:


weights_flag = False # Flag to set up weights have been created

cvar = 'sic'

# Load land/sea mask file
if os.path.basename(model_grid_file)!='MISSING':
    ds_mask = xr.open_dataset(model_grid_file)
else:
    ds_mask = None

for cf in all_files:
    # Check if already imported and skip (unless updateall flag is True)
    f_out = os.path.join(data_out, os.path.basename(cf).split('.')[0]+'_Stereo.nc') # netcdf file out 
    if not updateall:
        if os.path.isfile(f_out):
            print("Skipping ", os.path.basename(cf), " already imported.")
            continue # Skip, file already imported

    ds = xr.open_dataset(cf, engine='pynio')

    # Rename variables per esipn guidelines
    ds.rename({'CI_GDS4_SFC':'sic', 'g4_lat_2':'lat', 'g4_lon_3':'lon', 'initial_time0_hours':'init_time',
              'forecast_time1':'fore_time'}, inplace=True);
    
    # Apply masks (if available)
    if ds_mask:
        print('found mask')
        # land_mask is the fraction of native grid cell that is land
        # (1-land_mask) is fraction ocean
        # Multiply sic by fraction ocean to get actual native grid cell sic
        # Also mask land out where land_mask==1
        ds[cvar] = ds[cvar] * (1 - ds_mask.land_mask.where(ds_mask.land_mask<0.5)) # Using 50% threshold here
        

#     ds.coords['nj'] = model_grid.nj
#     ds.coords['ni'] = model_grid.ni
#     ds.coords['lat'] = model_grid.lat
#     ds.coords['lon'] = model_grid.lon
#     ds.coords['lat_b'] = model_grid.lat_b
#     ds.coords['lon_b'] = model_grid.lon_b
#     ds.coords['imask'] = model_grid.imask
    
#     # Set sic below 0 to 0
#     if X.sic.min().values < 0:
#         print("Some negative SIC "+str(X.sic.min().values)+" found in input PIOMAS, setting to 0")
#         X = X.where(X>=0, other=0)
        
#     # Apply model mask
#     X = X.where(X.imask)
    
    # Calculate regridding matrix
    regridder = xe.Regridder(ds, obs_grid, method, periodic=False, reuse_weights=weights_flag)
    weights_flag = True # Set true for following loops
    
    # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
    if method=='conservative':
        regridder = import_data.add_matrix_NaNs(regridder)
    
    # Regrid variable
    var_out = regridder(ds[cvar])
    
    # Expand dims
    var_out = import_data.expand_to_sipn_dims(var_out)

    # # Save regridded to netcdf file
    
    var_out.to_netcdf(f_out)
    var_out = None # Memory clean up
    print('Saved ', f_out)


# In[ ]:


# Clean up
if weights_flag:
    regridder.clean_weight_file()  # clean-up


# # Plotting

# In[ ]:


# sic_all = xr.open_dataset(f_out)





# # Set up plotting info
# cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
# cmap_sic.set_bad(color = 'red')

# # Plot original projection
# plt.figure(figsize=(20,10))
# ax1 = plt.axes(projection=ccrs.PlateCarree())
# ds_p = ds.sic.isel(init_time=1).isel(fore_time=79)
# ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                  vmin=0, vmax=1,
#                                  cmap=matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10)),
#                     transform=ccrs.PlateCarree());
# ax1.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
# gl = ax1.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
# gl.xlabels_bottom = True
# gl.ylabels_left = True
# gl.xformatter = LONGITUDE_FORMATTER
# gl.yformatter = LATITUDE_FORMATTER
# ax1.coastlines(linewidth=0.75, color='black', resolution='50m');


# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Original Grid')


# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# f.set_size_inches((10,10))
# ds_p2 = sic_all.sic.isel(init_time=1).isel(fore_time=79).isel(ensemble=0)
# ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Target Grid')


