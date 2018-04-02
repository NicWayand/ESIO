
# coding: utf-8

# In[ ]:


'''
Plot observed and modeled sea ice variables of interest.

'''

import matplotlib
matplotlib.use('Agg')
import mpld3
import matplotlib.pyplot as plt
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

import esio
import esiodata as ed

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

#############################################################
# Load in Data
#############################################################

E = ed.esiodata.load()

# Load in Obs
data_dir = E.data_dir
grid_dir = E.grid_dir
fig_dir = os.path.join(E.fig_dir, 'obs', 'NSIDC_0081' , 'standard')
da_81 = xr.open_dataarray(E.obs['NSIDC_0081']['sipn_nc']+'/NSIDC_0081.nc')
ds_ext = xr.open_dataset(os.path.join(E.obs['NSIDC_extent']['sipn_nc'], 'N_seaice_extent_daily_v3.0.nc'))

# Load in regional data
# Note minor -0.000004 degree differences in latitude
ds_region = xr.open_dataset(os.path.join(grid_dir, 'sio_2016_mask.nc'))
ds_region.set_coords(['lat','lon'], inplace=True);
ds_region.rename({'nx':'x', 'ny':'y'}, inplace=True);

#############################################################


# In[ ]:



# Get regional averages
da_81reg = esio.agg_by_domain(da_grid=da_81, ds_region=ds_region)

# Get date 30 days ago
ctime = np.datetime64(datetime.datetime.now())
lag_time_30days = ctime - np.timedelta64(30, 'D')
lag_time_90days = ctime - np.timedelta64(90, 'D')
last_sept = esio.get_season_start_date(ctime)

# Select recent period
da_81_30 = da_81.where(da_81.time >= lag_time_30days, drop=True)
# Aggregate over domain
da_81_30_avg = ((da_81_30>0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)


# Plot regional sea ice extents (last 90 days)
f = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1, 1, 1)
for cd in da_81reg.nregions:
    da_81reg.where(da_81reg.time >= last_sept, 
                   drop=True).sel(nregions=cd).plot(label=da_81reg.region_names.sel(nregions=cd).values)
ax1.set_title('Regional sea ice extents')
ax1.set_ylabel('Millions of square km')
plt.legend(bbox_to_anchor=(1.03, 1.05))
f_name = os.path.join(fig_dir,'panArcticSIC_Forecast_Regional_CurrentSeason')
f.savefig(f_name+'.png',bbox_inches='tight',dpi=200)
mpld3.save_json(f, f_name+'.json')

# Plot pan-Arctic sea ice extent
f = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1, 1, 1) # Observations
da_81_30_avg.plot(ax=ax1, label='NSIDC Near-Real-Time\n (Maslanik et al. 1999)')
ds_ext.sel(datetime=da_81_30_avg.time).Extent.plot(label='NSIDC Offical Extent', color='m')
ax1.set_ylabel('Millions of square km')
plt.legend(loc='lower right') #bbox_to_anchor=(1.03, 1.05))
# Save to png and json (for website)
f_name = os.path.join(fig_dir,'panArcticSIC_Forecast_30days')
f.savefig(f_name+'.png',bbox_inches='tight',dpi=200)
mpld3.save_json(f, f_name+'.json')
mpld3.save_html(f, f_name+'.html')

# Select recent period
da_81_3m = da_81.where(da_81.time >= lag_time_90days, drop=True)
# Aggregate over domain
da_81_3m_avg = ((da_81_3m>0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)

## Plot pan-Arctic sea ice extent
f = plt.figure(figsize=(10,5))
ax1 = plt.subplot(1, 1, 1) # Observations
da_81_3m_avg.plot(ax=ax1, label='NSIDC Near-Real-Time\n (Maslanik et al. 1999)')
ds_ext.sel(datetime=da_81_3m_avg.time).Extent.plot(label='NSIDC Offical Extent', color='m')
ax1.set_ylabel('Millions of square km')
plt.legend(loc='lower right')
f.savefig(os.path.join(fig_dir,'panArcticSIC_Forecast_3months.png'),bbox_inches='tight',dpi=200)

# Set up plotting info
cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues_r", 10))
cmap_sic.set_bad(color = 'lightgrey')
cmap_dif = matplotlib.colors.ListedColormap(sns.color_palette("RdBu", 10))
cmap_dif.set_bad(color = 'lightgrey')


# Plot Obs and model SIC for date
(f, ax1) = esio.polar_axis()
f.set_size_inches(10, 5)
# Obs NSIDC 0051
obs1 = da_81.sel(time=ctime, method='nearest')
obs1.plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                     transform=ccrs.PlateCarree(),
                                     cmap=cmap_sic,
                      vmin=0, vmax=1, cbar_kwargs={'label':'Sea Ice Concentration (-)'})
ax1.set_title('NSIDC 0081\n'+pd.to_datetime(obs1.time.values).strftime('%Y-%m-%d'))
plt.tight_layout()
f.savefig(os.path.join(fig_dir,'panArcticSIC_Forecast_Map.png'),bbox_inches='tight',dpi=200)


# Plot obs change from yesterday
# Plot Obs and model SIC for date
(f, ax1) = esio.polar_axis()
f.set_size_inches(10, 5)

# Obs NSIDC 0051
obs1 = da_81.sel(time=ctime, method='nearest')
ctime_m1 = obs1.time.values - np.timedelta64(1, 'D')
obs2 = da_81.sel(time=ctime_m1, method='nearest')
(obs1-obs2).plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                     transform=ccrs.PlateCarree(),
                                     cmap=cmap_dif,
                      vmin=-1, vmax=1, cbar_kwargs={'label':'Sea Ice Concentration (-)'})
ax1.set_title('NSIDC 0081\n'+pd.to_datetime(obs2.time.values).strftime('%Y-%m-%d')+' to '+ 
             pd.to_datetime(obs1.time.values).strftime('%Y-%m-%d'))
plt.tight_layout()
f.savefig(os.path.join(fig_dir,'panArcticSIC_Forecast_Map_1Day_Change.png'),bbox_inches='tight',dpi=200)



# Plot obs change from last week
(f, ax1) = esio.polar_axis()
f.set_size_inches(10, 5)

# Obs NSIDC 0051
obs1 = da_81.sel(time=ctime, method='nearest')
ctime_m1 = obs1.time.values - np.timedelta64(7, 'D')
obs2 = da_81.sel(time=ctime_m1, method='nearest')
(obs1-obs2).plot.pcolormesh(ax=ax1, x='lon', y='lat', 
                                     transform=ccrs.PlateCarree(),
                                     cmap=cmap_dif,
                      vmin=-1, vmax=1, cbar_kwargs={'label':'Sea Ice Concentration (-)'})
ax1.set_title('NSIDC 0081\n'+pd.to_datetime(obs2.time.values).strftime('%Y-%m-%d')+' to '+ 
             pd.to_datetime(obs1.time.values).strftime('%Y-%m-%d'))
plt.tight_layout()
f.savefig(os.path.join(fig_dir,'panArcticSIC_Forecast_Map_1Week_Change.png'),bbox_inches='tight',dpi=200)




