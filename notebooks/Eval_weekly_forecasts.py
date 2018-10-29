
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
import matplotlib.pyplot as plt
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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import json

from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
from esio import metrics

import dask
import timeit

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[ ]:


#from dask.distributed import Client
#client = Client()
#client


# In[ ]:


metrics_all = ['anomaly','mean','SIP']
runType = 'forecast'
variables = ['sic']


# In[ ]:


# Get path data
E = ed.EsioData.load()
grid_dir = E.grid_dir

# Load in regional data
ds_region = xr.open_dataset(os.path.join(grid_dir, 'sio_2016_mask_Update.nc'))


# In[ ]:


concat_dim_time = 'fore_time'
drop_coords = ['init_start','valid_start','valid_end']


# In[ ]:


cvar = variables[0]


# In[ ]:


# Define fig dir and make if doesn't exist
fig_dir = os.path.join('/home/disk/sipn/nicway/Nic/figures', 'model', 'MME', cvar, 'BSS')
if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)


# In[ ]:


for metric in [metrics_all[2]]:
    print(metric)
    metric_dir = os.path.join(E.model['MME'][runType]['sipn_nc'], metric)
    
    # Get list of inits (dirs)
    init_dirs = sorted([ name for name in os.listdir(metric_dir) if os.path.isdir(os.path.join(metric_dir, name)) ])
    print(init_dirs)
    
    ds_init_l = []
    for c_init in init_dirs:
        # Open files
        ds_i = xr.open_mfdataset(os.path.join(metric_dir,c_init,'*.nc'), drop_variables=['xm','ym'], 
                                 concat_dim=concat_dim_time, autoclose=True)
        ds_init_l.append(ds_i)
        
    # Drop extra coords becasue of this issue: https://github.com/pydata/xarray/pull/1953
    ds_init_l = [x.drop(drop_coords) for x in ds_init_l]
    ds_m = xr.concat(ds_init_l, dim='init_end')


# In[ ]:


# lat and lon get loaded as different for each file, set to constant except along x and y
ds_m.coords['lat'] = ds_m.sel(model='Observed').isel(init_end=0,fore_time=0).lat.drop([concat_dim_time,'init_end','model'])
ds_m.coords['lon'] = ds_m.sel(model='Observed').isel(init_end=0,fore_time=0).lon.drop([concat_dim_time,'init_end','model'])


# In[ ]:


allModels = [x for x in ds_m.model.values if x not in ['Observed']]
allModels


# In[ ]:


# Add MME
MME_avg = ds_m.sel(model=allModels).mean(dim='model')
MME_avg.coords['model'] = 'MME'
MME_avg


# In[ ]:


ds_ALL = xr.concat([ds_m, MME_avg], dim='model')
ds_ALL.model


# In[ ]:


ds_ALL.init_end


# In[ ]:


ds_ALL.model


# In[ ]:


# %load_ext autoreload
# %autoreload
# from esio import metrics


# In[ ]:


# For SIP, calculate the Brier Skill Score
l = []
for cmod in allModels+['MME']:
    c_SIP_BSS = metrics.BrierSkillScore(da_mod_sip=ds_ALL.sel(model=cmod).SIP, 
                                      da_obs_ip=ds_ALL.sel(model='Observed').SIP, 
                                      region=ds_region, 
                                      testplots=False)
    c_SIP_BSS.coords['model'] = cmod
    l.append(c_SIP_BSS)
SIP_BSS = xr.concat(l, dim='model')


# In[ ]:


SIP_BSS.fore_time.values.astype('timedelta64[D]').astype(int)


# In[ ]:


SIP_BSS.init_end.values


# In[ ]:


SIP_BSS.min().values


# In[ ]:


SIP_BSS.mean().values


# In[ ]:


def add_subplot_title(cmod, E, ax=None, BSS_val=''):
    if cmod in E.model.keys():
        ax.set_title(E.model[cmod]['model_label']+'\n('+BSS_val+')')
    else:
        ax.set_title(cmod)


# In[ ]:


# add missing info for climatology
E.model_color['climatology'] = (0,0,0)
E.model_linestyle['climatology'] = '--'
E.model_marker['climatology'] = '*'
E.model['climatology'] = {'model_label':'Climatology'}


# In[ ]:


# Aggregate over space (x,y)
BSS_agg = SIP_BSS.mean(dim=['x','y'])
BSS_agg


# In[ ]:


BSS_agg.fore_time.values.astype('timedelta64[D]').astype(int)


# In[ ]:


# Agg over init_time
BSS_agg_init = BSS_agg.mean(dim='init_end')

sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
f = plt.figure(figsize=(10,10))
for cmod in BSS_agg_init.model.values:
    # Get model plotting specs
    cc = E.model_color[cmod]
    cl = E.model_linestyle[cmod]
    if cmod=='MME':
        lw=4
    else:
        lw = 2
    plt.plot(BSS_agg_init.fore_time.values.astype('timedelta64[D]').astype(int),
            BSS_agg_init.sel(model=cmod).values, label=E.model[cmod]['model_label'],
            color=cc,
            linestyle=cl,
            linewidth=lw)
plt.legend(loc='lower right', bbox_to_anchor=(1.4, 0))
plt.ylabel('BSS (-)')
plt.xlabel('Lead time (Days)')
# Save to file
f_out = os.path.join(fig_dir,'BSS_by_lead_time.png')
f.savefig(f_out,bbox_inches='tight', dpi=300)


# In[ ]:


fig_dir


# In[ ]:


# Plot BSS by init time for 1 selected fore_time
# 4 = 28 days
BSS_agg_fore = BSS_agg.isel(fore_time=4)

sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})
f = plt.figure(figsize=(10,10))
for cmod in BSS_agg_fore.model.values:
    # Get model plotting specs
    cc = E.model_color[cmod]
    cl = E.model_linestyle[cmod]
    cm = E.model_marker[cmod]
    if cmod=='MME':
        lw=4
    else:
        lw = 2
    plt.plot(BSS_agg_fore.init_end.values,
            BSS_agg_fore.sel(model=cmod).values, label=E.model[cmod]['model_label'],
            color=cc,
            linestyle=cl,
            linewidth=lw,
            marker=cm)
plt.legend(loc='lower right', bbox_to_anchor=(1.4, -0.1))
plt.ylabel('BSS (-)')
plt.xlabel('Initialization date')
f.autofmt_xdate()
# Save to file
f_out = os.path.join(fig_dir,'BSS_by_init_time_'+str(BSS_agg_fore.fore_time.values.astype('timedelta64[D]').astype(int))+'_days.png')
f.savefig(f_out,bbox_inches='tight', dpi=300)


# In[ ]:


BSS_agg


# In[ ]:


# Plot init_time vs. fore_time BSS for select models

f = plt.figure()
plt.pcolormesh(BSS_agg_fore.init_end.values, BSS_agg_init.fore_time.values.astype('timedelta64[D]').astype(int), BSS_agg.sel(model='MME').T.values)
plt.colorbar()
f.autofmt_xdate()
plt.ylabel('Lead time (Days)')


# # SIP

# In[ ]:


# Set up color maps
cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"], N=10)
cmap_c.set_bad(color = 'lightgrey')
c_label = 'Sea Ice Probability (-)'
c_vmin = 0
c_vmax = 1

nrows = np.int(np.ceil(np.sqrt(ds_ALL.model.size)))
ncols = nrows
Nplots = ds_ALL.model.size + 1


        
for ft in ds_ALL.fore_time.values:  

    # New Plot
    central_extent = [-3850000*0.6, 3725000*0.6, -5325000*0.45, 5850000*0.45] # (x0, x1, y0, y1
    (f, axes) = ice_plot.multi_polar_axis(ncols=ncols, nrows=nrows, Nplots=Nplots, 
                                          extent=central_extent, central_longitude=0)

    for (i, cmod) in enumerate(ds_ALL.model.values):
        # Plot
        add_subplot_title(cmod, E, ax=axes[i])
        p = ds_ALL.sel(model=cmod).isel(init_end=0).sel(fore_time=ft).SIP.plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                              transform=ccrs.PlateCarree(),
                              add_colorbar=False,
                              cmap=cmap_c,
                              vmin=c_vmin, vmax=c_vmax)
        add_subplot_title(cmod, E, ax=axes[i])

    # Make pretty
    f.subplots_adjust(bottom=0.05)
    # cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = f.add_axes([0.25, 0.001, .5, 0.04]) #  [left, bottom, width, height] w
    cbar = f.colorbar(p, cax=cbar_ax, label=c_label, orientation='horizontal')
    cbar.set_ticks(np.arange(0,1.1,0.2))

    # Set title of all plots
    lead_time_days = str(ft.astype('timedelta64[D]').astype(int))
    print(lead_time_days)
    plt.suptitle(lead_time_days+' day lead time', fontsize=15)
    plt.subplots_adjust(top=0.85)
    
# # Save to file
# f_out = os.path.join(fig_dir,'panArctic_'+metric+'_'+runType+'_'+cyear+'_'+cmonth+'_withDiff.png')
# f.savefig(f_out,bbox_inches='tight', dpi=300)
# f_out = os.path.join(fig_dir,'panArctic_'+metric+'_'+runType+'_'+cyear+'_'+cmonth+'_withDiff_lowRES.png')
# f.savefig(f_out,bbox_inches='tight', dpi=90)
# print("saved ", f_out)


# # BSS

# In[ ]:


SIP_BSS_init_avg = SIP_BSS.mean(dim='init_end')

sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})

# Set up color maps
cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"], N=10)
cmap_c.set_bad(color = 'lightgrey')
c_label = 'BSS (0=best, 1=worst)'
c_vmin = 0
c_vmax = 1

nrows = np.int(np.ceil(np.sqrt(SIP_BSS_init_avg.model.size)))
ncols = int(np.ceil(SIP_BSS_init_avg.model.size/nrows))
Nplots = SIP_BSS_init_avg.model.size + 1
        
for ft in SIP_BSS_init_avg.fore_time.values:  
    
    # New Plot
    central_extent = [-3850000*0.6, 3725000*0.6, -5325000*0.45, 5850000*0.45] # (x0, x1, y0, y1
    (f, axes) = ice_plot.multi_polar_axis(ncols=ncols, nrows=nrows, Nplots=Nplots, 
                                          extent=central_extent, central_longitude=0)

    for (i, cmod) in enumerate(SIP_BSS_init_avg.model.values):
        # Plot
        add_subplot_title(cmod, E, ax=axes[i])
        p = SIP_BSS_init_avg.sel(model=cmod).sel(fore_time=ft).plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                              transform=ccrs.PlateCarree(),
                              add_colorbar=False,
                              cmap=cmap_c,
                              vmin=c_vmin, vmax=c_vmax)
        add_subplot_title(cmod, E, ax=axes[i], BSS_val='{0:.3f}'.format(SIP_BSS_init_avg.sel(model=cmod).sel(fore_time=ft).mean(dim=['x','y']).load().item()))

    # Make pretty
    f.subplots_adjust(bottom=0.05)
    # cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
    cbar_ax = f.add_axes([0.25, 0.001, .5, 0.04]) #  [left, bottom, width, height] w
    cbar = f.colorbar(p, cax=cbar_ax, label=c_label, orientation='horizontal')
    cbar.set_ticks(np.arange(-1,1.1,0.2))
    
    # Set title of all plots
    lead_time_days = str(ft.astype('timedelta64[D]').astype(int))
    print(lead_time_days)
    plt.suptitle(lead_time_days+' day lead time', fontsize=15)
    plt.subplots_adjust(top=0.93)

    # Save to file
    f_out = os.path.join(fig_dir,'BSS_Avg_all_Inits_'+lead_time_days.zfill(3)+'_day_lead_time.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)


# In[ ]:


sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})

# Set up color maps
cmap_c = matplotlib.colors.LinearSegmentedColormap.from_list("", ["white","orange","red","#990000"], N=10)
cmap_c.set_bad(color = 'lightgrey')
c_label = 'BSS (0=best, 1=worst)'
c_vmin = 0
c_vmax = 1

nrows = np.int(np.ceil(np.sqrt(SIP_BSS.model.size)))
ncols = nrows
Nplots = SIP_BSS.model.size + 1


        

# New Plot
central_extent = [-3850000*0.6, 3725000*0.6, -5325000*0.45, 5850000*0.45] # (x0, x1, y0, y1
(f, axes) = ice_plot.multi_polar_axis(ncols=ncols, nrows=nrows, Nplots=Nplots, 
                                      extent=central_extent, central_longitude=0)

for (i, cmod) in enumerate(SIP_BSS.model.values):
    # Plot
    add_subplot_title(cmod, E, ax=axes[i])
    cBSS_data = SIP_BSS.sel(model=cmod).mean(dim='fore_time')
    p = cBSS_data.plot.pcolormesh(ax=axes[i], x='lon', y='lat', 
                          transform=ccrs.PlateCarree(),
                          add_colorbar=False,
                          cmap=cmap_c,
                          vmin=c_vmin, vmax=c_vmax)
    add_subplot_title(cmod, E, ax=axes[i], BSS_val='{0:.3f}'.format(cBSS_data.mean(dim=['x','y']).load().item()))

# Make pretty
f.subplots_adjust(bottom=0.05)
# cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
cbar_ax = f.add_axes([0.25, 0.001, .5, 0.04]) #  [left, bottom, width, height] w
cbar = f.colorbar(p, cax=cbar_ax, label=c_label, orientation='horizontal')
cbar.set_ticks(np.arange(-1,1.1,0.2))

# Set title of all plots
lead_time_days = str(SIP_BSS.fore_time[-1].values.astype('timedelta64[D]').astype(int))
print(lead_time_days)
plt.suptitle('BSS for all lead times', fontsize=15)
plt.subplots_adjust(top=0.85)

# Save to file
f_out = os.path.join(fig_dir,'BSS_Avg_all_Inits_all_lead_times.png')
f.savefig(f_out,bbox_inches='tight', dpi=300)


