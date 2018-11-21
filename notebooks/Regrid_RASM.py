
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
import datetime
warnings.simplefilter(action='ignore', category=FutureWarning)
import dask
from dask.distributed import Client
# ESIO Imports

from esio import EsioData as ed
from esio import import_data
from esio import ice_plot


# In[2]:


dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
# client = Client(n_workers=8)
# client


# In[3]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[4]:


E = ed.EsioData.load()
# Directories
all_models=['rasmesrl']
runType='forecast'
updateall = False


# In[5]:


stero_grid_file = E.obs['NSIDC_0051']['grid']
obs_grid = import_data.load_grid_info(stero_grid_file, model='NSIDC')
# Ensure latitude is within bounds (-90 to 90)
# Have to do this because grid file has 90.000001
obs_grid['lat_b'] = obs_grid.lat_b.where(obs_grid.lat_b < 90, other = 90)
obs_grid.rename({'imask':'mask'}, inplace=True);
obs_grid


# In[6]:


## CAFS SIC (aice) has nan for all non-sea ice covered areas. So use the sst field to create the land mask
# land_mask is the fraction of native grid cell that is land
def get_land_mask_hack(ds):
    ds_land_mask = ds.sst[0,:,:].drop('time')
    ds_land_mask = ds_land_mask.isnull()
    ds_land_mask.name = 'land_mask'
    ds_land_mask.attrs = {'land_mask':'the fraction of native grid cell that is land'}
    return ds_land_mask 


# In[7]:


def fill_NaNOcean_with_Zeros(ds=None, vars=None, ds_land_mask=None):
    ds_out = ds
    for cvar in vars:
        ds_out[cvar] = ds_out[cvar].fillna(0.0).where(~ds_land_mask)
    return ds_out


# In[8]:


# ds_small = ds.sst[:,0:10,0:10].rename({'lat':'nj','lon':'ni'})
# ds_small


# In[9]:


# plt.figure()
# plt.plot(ds_small.ULON.values.flatten(), ds_small.ULAT.values.flatten(),'k*',label='U')
# plt.plot(ds_small.TLON.values.flatten(), ds_small.TLAT.values.flatten(),'ro',label='T')
# plt.legend()


# In[10]:


# plt.figure()
# plt.plot(ds.lat[0:10,0:10].values.flatten(), ds.lon[0:10,0:10].values.flatten(),'ro',label='center')
# plt.plot(ds.lat_b[0:11,0:11].values.flatten(), ds.lon_b[0:11,0:11].values.flatten(),'ko',label='bounds')

# plt.legend()


# In[11]:


# plt.figure()
# plt.plot(obs_grid.lat.values.flatten(), obs_grid.lon.values.flatten(),'mo',label='center_OBS')
# plt.plot(ds.lat.values.flatten(), ds.lon.values.flatten(),'ro',label='center')
# plt.plot(ds.lat_b.values.flatten(), ds.lon_b.values.flatten(),'ko',label='bounds')

# plt.legend()


# In[12]:


# from scipy.interpolate import RegularGridInterpolator

# def get_lat_lon_bounds_from_corner(cen_lat=None, cen_lon=None):
#     ''' Some models only provide lat lon coords or the cell center and the corners. Transform to
#     the bounding N+1 lats'''

#     # Input
#     # Center lat and lon of grid cells (N x M)
#     #
#     # Output
#     # lat_b and lon_b - bounds (N+1 x M+1) for each grid lat lon grid cell center

#     # Add cell bound coords (lat_b and lon_b)
#     n_j = cen_lat.nj.size
#     n_i = cen_lat.ni.size
#     nj_b = np.arange(0, n_j + 1) # indices of edge of cells
#     ni_b = np.arange(0, n_i + 1)

#     nj = np.arange(0, n_j)
#     ni = np.arange(0, n_i)

#     interf_lat = RegularGridInterpolator((nj, ni), cen_lat, bounds_error=False, fill_value=None)
#     interf_lon = RegularGridInterpolator((nj, ni), cen_lon, bounds_error=False, fill_value=None)

#     # Create empty matrix
#     b_grid_lat = np.ones((n_j + 1, n_i + 1))*np.NaN
#     b_grid_lon = np.ones((n_j + 1, n_i + 1))*np.NaN
#     # Interpolate each value (inner only)
#     for ci in ni_b:
#         for cj in nj_b:
#             b_grid_lat[cj,ci] = interf_lat([[cj-0.5, ci-0.5]])
#             b_grid_lon[cj,ci] = interf_lon([[cj-0.5, ci-0.5]])

#     ds_lat_b = xr.DataArray(b_grid_lat, dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b, 'ni_b':ni_b})
#     ds_lon_b = xr.DataArray(b_grid_lon, dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b, 'ni_b':ni_b})

#     return (ds_lat_b, ds_lon_b)


# In[13]:


# (ds_lat_b, ds_lon_b) = get_lat_lon_bounds_from_corner(cen_lat=ds.rename({'lat':'nj','lon':'ni'}).TLAT[0:10,0:10], 
#                                                       cen_lon=ds.rename({'lat':'nj','lon':'ni'}).TLON[0:10,0:10])


# In[14]:


# plt.figure()
# plt.plot(ds.TLON[0:10,0:10].values.flatten(), ds.TLAT[0:10,0:10].values.flatten(),'ro',label='center')
# plt.plot(ds_lon_b.values.flatten(), ds_lat_b.values.flatten(),'ko',label='bounds')
# plt.legend()


# In[15]:


# Regridding Options
method='conservative_normed' # ['bilinear', 'conservative_normed', 'conservative', 'nearest_s2d', 'nearest_d2s', 'patch']


# In[16]:


## TODO
# - Get mask
# - Get lat lon bounds 


# In[17]:


var_dic = {'aice':'sic','lat':'nj','lon':'ni','TLAT':'lat','TLON':'lon'}
var_dic_new = {'aice':'sic'}


# In[18]:


for model in all_models:
    print('Regridding ', model, '...')
    
    data_dir = E.model[model][runType]['native']
    data_out = E.model[model][runType]['sipn_nc']
    model_grid_file = E.model[model]['grid']
    
    # Files are stored as per time step (about 45 per init_time)
    # First parse files to see what unique init_times we have
    # ARCu0.08_121_2018042112_t0300.nc
    prefix = 'RASM-ESRL'
    all_files = sorted(glob.glob(os.path.join(data_dir, prefix+'*.nc')))
    # Remove init times that started on 12 our (only a few at begining of record)
    all_files = [x for x in all_files if '-12_t' not in x]
    init_times = list(set([s.split('_')[1].split('-00')[0] for s in all_files]))
    
    print("Found ",len(init_times)," initialization times.")
    if updateall:
        print("Updating all files...")
    else:
        print("Only updating new files")


    weights_flag = False # Flag to set up weights have been created

    # Load land/sea mask file
    if os.path.basename(model_grid_file)!='MISSING':
        ds_mask = xr.open_mfdataset(model_grid_file)
    else:
        ds_mask = None

    for cf in sorted(init_times):
        new_grid = False # Assume old grid
        
        # Check if already imported and skip (unless updateall flag is True)
        f_out = os.path.join(data_out, prefix+'_'+cf+'_Stereo.nc') # netcdf file out 
        if not updateall:
            # TODO: Test if the file is openable (not corrupted)
            if os.path.isfile(f_out):
                print("Skipping ", cf, " already imported.")
                continue # Skip, file already imported

        c_files = sorted(glob.glob(os.path.join(data_dir, prefix+'*_'+cf+'*.nc')))
        ds = xr.open_mfdataset(c_files, concat_dim='time', decode_times=False, autoclose=True)
                
        # Fill sea ice vars (sic and hi) with zeros where there isn't any ice in ocean (previously NaNs)
        ds_land_mask = get_land_mask_hack(ds) # get land mask from sst field
        ds = fill_NaNOcean_with_Zeros(ds=ds, vars=['aice','hi'], 
                                      ds_land_mask=ds_land_mask)
        
        # Check if its the updated grid
        if 'TLAT' not in ds:
            new_grid = True

        # Rename variables per esipn guidelines
        if new_grid:
            ds.rename(var_dic_new, inplace=True);
        else:
            ds.rename(var_dic, inplace=True);
        
        
        ds = ds.drop('time_bounds')

        # Format times
        ds.coords['init_time'] = np.datetime64(cf)  #np.datetime64(ds.tau.attrs['time_origin'])
        ds.coords['tau'] = ds.tau

        ds.swap_dims({'time':'tau'}, inplace=True)
        ds.rename({'tau':'fore_time'}, inplace=True)
        ds.fore_time.attrs['units'] = 'Forecast offset from initial time'
        ds = ds.drop(['time'])
        ds.coords['fore_time'] = ds.fore_time.astype('timedelta64[h]') 

        # Apply masks (if available)
        if ds_mask:
            print('found mask')
            # land_mask is the fraction of native grid cell that is land
            # (1-land_mask) is fraction ocean
            # Multiply sic by fraction ocean to get actual native grid cell sic
            # Also mask land out where land_mask==1
            ds = ds * (1 - ds_mask.land_mask.where(ds_mask.land_mask<1))

        ds.coords['mask'] = ds.sic.isel(fore_time=0).notnull().drop(['fore_time','init_time'])
        
        if not new_grid:
            # Add lat lon bounds (on fly becuase grid changes with different files (system grid change???))
            n_j = ds.nj.size
            n_i = ds.ni.size
            nj_b = np.arange(0, n_j + 1)
            ni_b = np.arange(0, n_i + 1)
            ds_b = ds.interp(nj=nj_b-0.5, ni=ni_b-0.5, kwargs={'fill_value': None})

            ds_b = ds_b.rename({'nj':'nj_b','ni':'ni_b','lat':'lat_b','lon':'lon_b'})[['lat_b','lon_b']].drop(['ULAT','ULON'])
            ds = xr.merge([ds, ds_b])
           
        # Calculate regridding matrix
        if new_grid: # Use bilinear becuase its regualar grid
            regridder = xe.Regridder(ds, obs_grid, 'bilinear', periodic=False, reuse_weights=weights_flag)
            weights_flag = False
        else:   
            regridder = xe.Regridder(ds, obs_grid, method, periodic=False, reuse_weights=weights_flag)
            weights_flag = True # Set true for following loops

        # Regrid variables
        var_list = []
        for cvar in ds.data_vars:
            
            # offset hack to keep orig missing mask
            offset = 10.0
            ds_coarse = regridder(ds[cvar]+offset)
            ds_coarse = ds_coarse.where(ds_coarse!=0) - offset
            # Bound max and min
            if cvar=='sic':
                c_notmissing = ds_coarse.notnull()
                ds_coarse = ds_coarse.where(ds_coarse>=0, other=0)
                ds_coarse = ds_coarse.where(ds_coarse<=1, other=1)
                ds_coarse = ds_coarse.where(c_notmissing)
            elif cvar=='hi':
                c_notmissing = ds_coarse.notnull()
                ds_coarse = ds_coarse.where(ds_coarse>=0, other=0)
                ds_coarse = ds_coarse.where(c_notmissing)
                
            var_list.append(ds_coarse)
        ds_out = xr.merge(var_list)

        # Expand dims
        ds_out = import_data.expand_to_sipn_dims(ds_out)
        
#         plt.figure(figsize=(12*400/300,12))
#         ds_out.sic[0,0,0,:,:].plot()
#         print(ds.sic.max().values)
#         print(ds.sic.min().values)
#         print(ds_out.sic.max().values)
#         print(ds_out.sic.min().values)

#         # Save regridded to netcdf file
#         xr.exit()
        
        ds_out.to_netcdf(f_out)
        ds_out = None # Memory clean up
        ds = None
        print('Saved ', f_out)


# In[ ]:


# regridder = xe.Regridder(ds, obs_grid, method, periodic=True, reuse_weights=weights_flag)


# In[ ]:



# plt.plot(ds.lon.values.flatten(), ds.lat.values.flatten(),'ro',label='center')
# plt.plot(ds.lon_b.values.flatten(), ds.lat_b.values.flatten(),'ko',label='bounds')


# plt.plot(obs_grid.lon.values.flatten(), obs_grid.lat.values.flatten(),'mo',
#          label='center_OBS')
# plt.plot(obs_grid.lon_b.values.flatten(), obs_grid.lat_b.values.flatten(),'go',
#          label='center_OBS')
# plt.plot(ds.lon.values.flatten(), ds.lat.values.flatten(),'ro',label='center')
# plt.plot(ds.lon_b.values.flatten(), ds.lat_b.values.flatten(),'ko',label='bounds')


# (f, ax1) = ice_plot.polar_axis()
# plt.plot(obs_grid.lon.values.flatten(), obs_grid.lat.values.flatten(),'mo',
#          label='center_OBS', transform=ccrs.PlateCarree())
# # plt.plot(obs_grid.lon_b.values[0:10,0:10].flatten(), obs_grid.lat_b[0:10,0:10].values.flatten(),'go',
# #          label='center_OBS')
# # plt.plot(ds.lon.values.flatten(), ds.lat.values.flatten(),'ro',label='center')
# ax1.plot(ds.lon_b.values.flatten(), ds.lat_b.values.flatten(),'ko',label='bounds',
#         transform=ccrs.PlateCarree())




# ds.sic.isel(fore_time=0).notnull().drop(['fore_time','init_time']).plot()

# ds.mask.plot()

# ds_out

# plt.figure(figsize=(12*400/300,12))
# ds_out.mask[:,:].plot()

# plt.figure(figsize=(12*400/300,12))
# ds.sic[0,:,:].plot()
# print(ds.sic[0,:,:].mean().values)

# plt.figure(figsize=(12*400/300,12))
# ds_out.sic[0,0,0,:,:].plot(vmin=0, vmax=1)
# print(ds_out.sic[0,:,:].mean().values)


# In[20]:


# Clean up
if weights_flag:
    regridder.clean_weight_file()  # clean-up


# # Plotting

# In[ ]:


# sic_all = xr.open_mfdataset(f_out)

# # Set up plotting info
# cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues", 10))
# cmap_sic.set_bad(color = 'red')

# # Plot original projection
# plt.figure(figsize=(20,10))
# ax1 = plt.axes(projection=ccrs.PlateCarree())
# ds_p = ds.sic.isel(fore_time=8)
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
# ds_p.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Original Grid')

# # Plot SIC on target projection
# (f, ax1) = ice_plot.polar_axis()
# ds_p2 = sic_all.sic.isel(init_time=0).isel(fore_time=8).isel(ensemble=0)
# ds_p2.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
#                                      transform=ccrs.PlateCarree(),
#                                      cmap=cmap_sic)
# ax1.set_title('Target Grid')


