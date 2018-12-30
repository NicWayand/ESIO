
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

'''
Plot forecast maps with all available models.
'''




import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, mpld3
from collections import OrderedDict
import itertools
import numpy as np
import numpy.ma as ma
import pandas as pd
import struct
import os
import xarray as xr
import glob
import datetime
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
np.seterr(divide='ignore', invalid='ignore')


from esio import EsioData as ed
from esio import ice_plot

import dask
dask.set_options(get=dask.threaded.get)

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[ ]:


def remove_small_contours(p, thres=10):
    for level in p.collections:
        for kp,path in reversed(list(enumerate(level.get_paths()))):
            # go in reversed order due to deletions!

            # include test for "smallness" of your choice here:
            # I'm using a simple estimation for the diameter based on the
            #    x and y diameter...
            verts = path.vertices # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))

            if diameter<thres: # threshold to be refined for your actual dimensions!
                del(level.get_paths()[kp])  # no remove() for Path objects:(


# In[ ]:


# Plotting Info
runType = 'forecast'
variables = ['sic'] #, 'hi']
metrics_all = {'sic':['anomaly','mean','SIP'], 'hi':['mean']}
# Some models are terrible/have serious issues, so don't include in MME
MME_NO = ['hcmr']


# In[ ]:


# Initialization times to plot
cd = datetime.datetime.now()
cd = datetime.datetime(cd.year, cd.month, cd.day) # Set hour min sec to 0. 
init_slice = np.arange(cd - datetime.timedelta(days=70), cd, datetime.timedelta(days=1))


# In[ ]:


# Forecast times to plot
weeks = pd.to_timedelta(np.arange(0,5,1), unit='W')
months = pd.to_timedelta(np.arange(2,12,1), unit='M')
years = pd.to_timedelta(np.arange(1,2), unit='Y') - np.timedelta64(1, 'D') # need 364 not 365
slices = weeks.union(months).union(years).round('1d')
da_slices = xr.DataArray(slices, dims=('fore_time'))


# In[ ]:


#############################################################
# Load in Data
#############################################################
E = ed.EsioData.load()

# Get median ice edge by DOY
median_ice_fill = xr.open_mfdataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'ice_edge.nc')).sic
# Get mean sic by DOY
mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic


# In[ ]:


import timeit
start_time = timeit.default_timer()
ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'/*.nc', concat_dim='time', autoclose=True)#,
print(timeit.default_timer() - start_time)


# In[ ]:


# print(ds_51.time.min().values, ds_51.time.max().values)
# print(ds_81.time.min().values, ds_81.time.max().values)
# print(ds_79.time.min().values, ds_79.time.max().values)


# In[ ]:


# Define models to plot
models_2_plot = list(E.model.keys())
models_2_plot = [x for x in models_2_plot if x!='piomas'] # remove some models


# In[ ]:



# Get # of models and setup subplot dims
Nmod = len(E.model.keys()) + 2 #(+2 for obs and MME)
Nr = int(np.floor(np.sqrt(Nmod)))
Nc = int(np.ceil(Nmod/Nr))
assert Nc*Nr>=Nmod, 'Need more subplots'

for cvar in variables:
    
    fig_dir = os.path.join(E.fig_dir, 'model', 'all_model', cvar, "maps")
    if not os.path.exists(fig_dir):
        os.makedirs(fig_dir)


             
    print("Starting plots...")
    for it in np.flip(init_slice,axis=0): # Flip it here so plots backwards in time
        print(it)
        
        for (cs, ft) in enumerate(da_slices.values): 
            print(ft.astype('timedelta64[D]'))
            cs_str = format(cs, '02')
            cdoy = pd.to_datetime(it + ft).timetuple().tm_yday
            
            # Loop through variable of interest + any metrics (i.e. SIP) based on that
            for metric in metrics_all[cvar]:


                # Set up plotting info
                if cvar=='sic':
                    if metric=='mean':
                        cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Blues_r", 10))
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Concentration (-)'
                        c_vmin = 0
                        c_vmax = 1
                    elif metric=='SIP':
                        cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"])
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Probability (-)'
                        c_vmin = 0
                        c_vmax = 1
                    elif metric=='anomaly':
#                         cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("coolwarm", 9))
                        cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","blue"])
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'SIC Anomaly to 1980-2010 Mean'
                        c_vmin = -1
                        c_vmax = 1

                elif cvar=='hi':
                    if metric=='mean':
                        cmap_c = matplotlib.colors.ListedColormap(sns.color_palette("Reds_r", 10))
                        cmap_c.set_bad(color = 'lightgrey')
                        c_label = 'Sea Ice Thickness (m)'
                        c_vmin = 0
                        c_vmax = None
                else:
                    raise ValueError("cvar not found.") 


                MME_list = []            
                # New Plot
                (f, axes) = ice_plot.multi_polar_axis(ncols=Nc, nrows=Nr, Nplots=Nmod)
                
                # Plot Obs (if available)
                ax_num = 0
                if ((it + ft) in ds_81.time.values):
                    if metric=='mean':
                        da_obs_c = ds_81.sic.sel(time=(it + ft))
                    elif metric=='SIP':
                        da_obs_c = (ds_81.sic.sel(time=(it + ft)) >=0.15).astype('int').where(ds_81.sic.sel(time=(it + ft)).notnull())
                    elif metric=='anomaly':
                        da_obs_VT = ds_81.sic.sel(time=(it + ft))
                        da_obs_mean = mean_1980_2010_sic.isel(time=cdoy)
                        da_obs_c = da_obs_VT - da_obs_mean
                    else:
                        raise ValueError('Not implemented')
                    da_obs_c.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                                          transform=ccrs.PlateCarree(),
                                          add_colorbar=False,
                                          cmap=cmap_c,
                                          vmin=c_vmin, vmax=c_vmax)

                    # Overlay median ice edge
                    if metric=='mean':
                        po = median_ice_fill.isel(time=cdoy).plot.contour(ax=axes[ax_num], x='xm', y='ym', 
                                                                          colors=('#bc0f60'),
                                                                          linewidths=[0.5],
                                                                          levels=[0.5], label='Median ice edge 1981-2010')
                        remove_small_contours(po, thres=10)
                else: # When were in the future (or obs are missing)
                    if metric=='anomaly': # Still get climatological mean for model difference
                        da_obs_mean = mean_1980_2010_sic.isel(time=cdoy)
                axes[ax_num].set_title('Observed')
                
                
                # Plot all Models
                p = None # initlaize to know if we found any data
                for (i, cmod) in enumerate(models_2_plot):
                    i = i+2 # shift for obs and MME
                    axes[i].set_title(E.model[cmod]['model_label'])

                    # Load in Model
                    model_forecast = os.path.join(E.model[cmod][runType]['sipn_nc'], '*.nc') 

                    # Check we have files 
                    files = glob.glob(model_forecast)
                    if not files:
                        #print("Skipping model", cmod, "no forecast files found.")
                        continue # Skip this model
                    ds_model = xr.open_mfdataset(model_forecast, 
                                chunks={'fore_time': 1, 'init_time': 1, 'nj': 304, 'ni': 448}, concat_dim='init_time')
                    ds_model.rename({'nj':'x', 'ni':'y'}, inplace=True)


                    # Select init and fore_time of interest
                    # TODO: Allow models that are not a midnight (i.e. 12) using tolarance of 13 hours
                    if it in ds_model.init_time.values:
    #                     print('found init_time')
                        ds_model = ds_model.sel(init_time=it)
                    else:
                        continue

                    # Select var of interest (if available)
                    if cvar in ds_model.variables:
    #                     print('found ',cvar)
                        ds_model = ds_model[cvar]
                    else:
                        continue

                    if ft in ds_model.fore_time.values:
    #                     print('found fore_time')
                        ds_model = ds_model.sel(fore_time=ft)
                    else:
                        continue
                    
    #                 print("Found data for model ", cmod, ". Plotting...")    
                    if metric=='mean': # Calc ensemble mean
                        ds_model = ds_model.mean(dim='ensemble')
                    elif metric=='SIP': # Calc probability
                        # Issue of some ensemble members having missing data
#                         ds_model = ds_model.where(ds_model>=0.15, other=0).mean(dim='ensemble')
                        ok_ens = ((ds_model.notnull().sum(dim='x').sum(dim='y'))>0) # select ensemble members with any data
                        ds_model = ((ds_model.where(ok_ens, drop=True)>=0.15) ).mean(dim='ensemble').where(ds_model.isel(ensemble=0).notnull())
                    elif metric=='anomaly': # Calc anomaly in reference to mean observed 1980-2010
                        ds_model = ds_model.mean(dim='ensemble') - da_obs_mean
                        # Add back lat/long (get dropped because of round off differences)
                        ds_model['lat'] = da_obs_mean.lat
                        ds_model['lon'] = da_obs_mean.lon
                    else:
                        raise ValueError('metric not implemented')

                        
  
                        
                        
                    # Build MME
                    if 'ensemble' in ds_model:
                        ds_model = ds_model.drop('ensemble')
                    if cmod not in MME_NO: # Exclude some models (bad) from MME
                        MME_list.append(ds_model)

                    # Plot
                    p = ds_model.plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                                          transform=ccrs.PlateCarree(),
                                          add_colorbar=False,
                                          cmap=cmap_c,
                                          vmin=c_vmin, vmax=c_vmax)


                    # Overlay median ice edge
                    if metric=='mean':
                        po = median_ice_fill.isel(time=cdoy).plot.contour(ax=axes[i], x='xm', y='ym', 
                                                                          colors=('#bc0f60'),
                                                                          linewidths=[0.5],
                                                                          levels=[0.5], label='Median ice edge 1981-2010')
                        remove_small_contours(po, thres=10)

                    axes[i].set_title(E.model[cmod]['model_label'])
                    
                    # Clean up for current model
                    ds_model = None



                # MME
                ax_num = 1
                if MME_list: # If we had any models for this time
                    # Concat over all models
                    ds_MME = xr.concat(MME_list, dim='model')
                    # Take average
                    ds_MME = ds_MME.mean(dim='model')
                    # Mask out using OBSERVED LAND MASK
                    # TODO: should happen if all models have land mask.... fix upstream
                    pmme = ds_MME.plot.pcolormesh(ax=axes[ax_num], x='lon', y='lat', 
                                                  transform=ccrs.PlateCarree(),
                                                  add_colorbar=False, 
                                                  cmap=cmap_c,vmin=c_vmin, vmax=c_vmax)
                    # Overlay median ice edge
                    if metric=='mean':
                        po = median_ice_fill.isel(time=cdoy).plot.contour(ax=axes[ax_num], x='xm', y='ym', 
                                                                          colors=('#bc0f60'),
                                                                          linewidths=[0.5],
                                                                          levels=[0.5], label='Median ice edge 1981-2010')
                        remove_small_contours(po, thres=10)
                axes[ax_num].set_title('MME')

                # Make pretty
                f.subplots_adjust(right=0.8)
                cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
                if p:
                    cbar = f.colorbar(p, cax=cbar_ax, label=c_label)
                    if metric=='anomaly':
                        cbar.set_ticks(np.arange(-1,1.1,0.2))
                    else:
                        cbar.set_ticks(np.arange(0,1.1,0.1))
                    #cbar.set_ticklabels(np.arange(0,1,0.05))

                # Set title of all plots
                init_time =  pd.to_datetime(it).strftime('%Y-%m-%d-%H:%M')
                valid_time = pd.to_datetime(it+ft).strftime('%Y-%m-%d-%H:%M')
                plt.suptitle('Initialization Time: '+init_time+'\n Valid Time: '+valid_time, fontsize=20)
                plt.subplots_adjust(top=0.85)

                # Save to file
                f_out = os.path.join(fig_dir,'panArctic_'+metric+'_'+runType+'_'+init_time+'_'+cs_str+'.png')
                f.savefig(f_out,bbox_inches='tight',dpi=200)
                #print("saved ", f_out)

                # Mem clean up
                plt.close(f)
                p = None
                ds_MME= None
                da_obs_c = None
                da_obs_mean = None
            


# In[ ]:










