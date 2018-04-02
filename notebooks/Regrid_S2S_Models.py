
# coding: utf-8

# In[1]:


# S2S Model Regrid

# - Loads in all daily forecasts of sea ice extent
# - Regrids to polar stereographic,
# - Saves to netcdf files grouped by year


# In[2]:


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

# ESIO Imports
import esio
import esiodata as ed


# In[3]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[4]:


E = ed.esiodata.load()
# Directories
all_models = ['bom', 'cma', 'ecmwf', 'hcmr', 'isaccnr',
          'jma', 'metreofr', 'ncep', 'ukmo', 'eccc', 'kma']
runType='forecast'
updateall = False

stero_grid_file = E.obs['NSIDC_0051']['grid']


# In[5]:


obs_grid = esio.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)


# In[6]:


# Regridding Options
method='bilinear' # ['bilinear', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[7]:


# Store dictionary to convert each model variable names to sipn syntax
var_dic = {}

var_dic['cma'] = {'initial_time0_hours':'init_time',
                 'lat_0':'lat', 'lon_0':'lon',
                 'forecast_time0':'fore_time',
                 'ICEC_P11_L1_GLL0_avg24h':'sic'}
# guess it looks like cma
for model in all_models:
    var_dic[model] = var_dic['cma']
# Set models that are different
var_dic['bom'] = {'initial_time0_hours':'init_time',
                 'lat_0':'lat', 'lon_0':'lon',
                 'forecast_time0':'fore_time',
                 'ICEC_P11_L1_GGA0_avg24h':'sic'}



# In[8]:


## TODO
# - Get mask
# - Get lat lon bounds 


# In[9]:


def test_plot():
    # Set up plotting info
    cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
    cmap_sic.set_bad(color = 'red')

    # Plot original projection
    plt.figure(figsize=(20,10))
    ax1 = plt.axes(projection=ccrs.PlateCarree())
    ds_p = ds.sic.isel(init_time=0).isel(fore_time=0)
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
    (f, ax1) = esio.polar_axis()
    ds_p2 = var_out.isel(init_time=0).isel(fore_time=0).isel(ensemble=0)
    ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                         transform=ccrs.PlateCarree(),
                                         cmap=cmap_sic)
    ax1.set_title('Target Grid')


# In[10]:


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

    weights_flag = False # Flag to set up weights have been created

    cvar = 'sic'

    for cf in all_files:
        # Check if already imported and skip (unless updateall flag is True)
        f_out = os.path.join(data_out, os.path.basename(cf).split('.')[0]+'_Stereo.nc') # netcdf file out 
        if not updateall:
            if os.path.isfile(f_out):
                print("Skipping ", os.path.basename(cf), " already imported.")
                continue # Skip, file already imported

        ds = xr.open_dataset(cf, engine='pynio')

        # Rename variables per esipn guidelines
        print(ds)
        ds.rename(var_dic[model], inplace=True);

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
        regridder = xe.Regridder(ds, obs_grid, method, periodic=True, reuse_weights=weights_flag)
        weights_flag = True # Set true for following loops

        # Add NaNs to empty rows of matrix (forces any target cell with ANY source cells containing NaN to be NaN)
        regridder = esio.add_matrix_NaNs(regridder)

        # Regrid variable
        var_out = regridder(ds[cvar])

        # Expand dims
        var_out = esio.expand_to_sipn_dims(var_out)
        
        #test_plot()

        # # Save regridded to netcdf file
        var_out.to_netcdf(f_out)
        var_out = None # Memory clean up
        print('Saved ', f_out)
    # Clean up
    if weights_flag:
        regridder.clean_weight_file()  # clean-up    


