
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
import matplotlib.pyplot as plt
from collections import OrderedDict
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
warnings.simplefilter(action='ignore', category=RuntimeWarning)

from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
from esio import metrics
import dask
import xskillscore as xs

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[ ]:


# from dask.distributed import Client
# client = Client(n_workers=2)
# client = Client()
# client
dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler
# dask.config.set(scheduler='processes')  # overwrite default with threaded scheduler


# In[ ]:


# Parameters
pred_year = 2018 # Prediction year
Y_Start = 1979
Y_End = 2017


# In[ ]:


#############################################################
# Load in Data
#############################################################

E = ed.EsioData.load()
mod_dir = E.model_dir


# In[ ]:


# Get most recent obs
ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly/*.nc', concat_dim='time', autoclose=True, parallel=True)
doyall = metrics.get_DOY(ds_81.time)
ds_81.coords['doy'] = xr.DataArray(doyall, dims='time', coords={'time':ds_81.time})

# Get mean sic by DOY
print("Need up update to 1979-2017 mean")
mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic

end_date = datetime.datetime.now()
end_date = datetime.datetime(end_date.year, end_date.month, end_date.day) # Set hour min sec to 0. 
start_date = datetime.datetime(end_date.year, 1, 1) # Set hour min sec to 0. 


# In[ ]:


# Load in alphas
alpha_cdoy = xr.open_mfdataset('/home/disk/sipn/nicway/data/model/dampedAnomalyTrend/forecast/param/'+str(pred_year)+'_*.nc',
                              concat_dim='doy', autoclose=True)


# In[ ]:


alpha_cdoy


# # Damped Anomaly from Climatological Mean

# In[ ]:


# cmod = 'dampedAnomaly'
# runType = 'forecast'

# # Loop through current year
# for ctime in ds_81.time.sel(time=slice(start_date,end_date)):
    
#     c_sic = ds_81.sic.sel(time=ctime)
#     cdoy = c_sic.doy.item()

#     # Calculate most recent anomaly
#     c_anomaly = c_sic - mean_1980_2010_sic.sel(time=cdoy)

#     da_l = []
#     # Fore each forecast period (here 7 days)
#     fore_cast_interval = 7
#     for fore_index in np.arange(0,60,1):
#         fore_anomaly = (alpha_cdoy.sel(doy=cdoy).alpha**fore_index) * c_anomaly
        
#         valid_doy = cdoy + fore_index * fore_cast_interval
#         # Wrap around if > 366
#         if valid_doy > mean_1980_2010_sic.time.max().values:
#             valid_doy = valid_doy - mean_1980_2010_sic.time.max().values
         
#         # Get SIC by adding anomaly to the mean SIC
#         fore_sic = fore_anomaly + mean_1980_2010_sic.sel(time=valid_doy)
        
#         # Force prediction SIC to be between 0-1
#         ocnmask = fore_sic.notnull()
#         fore_sic = fore_sic.where(fore_sic >= 0, other=0).where(ocnmask)
#         fore_sic = fore_sic.where(fore_sic <= 1, other=1).where(ocnmask)

#         # Add cords
#         fore_sic.coords['fore_time'] = np.timedelta64(int(fore_index*fore_cast_interval),'D')
#         da_l.append(fore_sic)

#     da_sic = xr.concat(da_l, dim='fore_time')    
#     da_sic.coords['init_time'] = c_sic.time
    
#     da_sic = da_sic.drop(['xm','ym','lag','doy','hole_mask','time'])

#     da_sic.name = 'sic'

#     da_sic = import_data.expand_to_sipn_dims(da_sic)
    
#     da_sic = da_sic.rename({'x':'nj','y':'ni'})
            
#     file_out = os.path.join(mod_dir, cmod, runType, 'sipn_nc', pd.to_datetime(da_sic.init_time.item()).strftime('%Y-%m-%d')+'.nc')
#     da_sic.to_netcdf(file_out)
#     print("Saved file:",file_out)


# # Damped Anomaly from Climatological Trend prediction

# In[ ]:


cmod = 'dampedAnomalyTrend'
runType = 'forecast'


# Climatology model trend
all_files = os.path.join(mod_dir,'climatology',runType,'sipn_nc', str(end_date.year)+'*.nc')
files = glob.glob(all_files)
obs_clim_model = xr.open_mfdataset(sorted(files), 
        chunks={'time': 30, 'x': 304, 'y': 448},  
         concat_dim='time', autoclose=True, parallel=True)
obs_clim_model = obs_clim_model['sic']
obs_clim_model = obs_clim_model.swap_dims({'time':'doy'})
obs_clim_model


# In[ ]:


test_plots = False # Need to uncomment xr.exit() below to stop on first doy


# In[ ]:


UpdateAll = False


# In[ ]:


# Loop through current year
for ctime in ds_81.time.sel(time=slice(start_date,end_date)):
    file_out = os.path.join(mod_dir, cmod, runType, 'sipn_nc', pd.to_datetime(ctime.time.values).strftime('%Y-%m-%d')+'.nc')
    
    # Only calc if it doesn't exist
    if os.path.isfile(file_out) & ~UpdateAll:
        continue
    
    c_sic = ds_81.sic.sel(time=ctime)
    cdoy = c_sic.doy.item()

    # Calculate most recent anomaly from predicted trend
    c_anomaly = c_sic - obs_clim_model.sel(doy=cdoy)
    c_anomaly = c_anomaly.drop(['time','doy']) #mean_1980_2010_sic.sel(time=cdoy)

    da_l = []
    # For each forecast period (here 7 days)
    fore_cast_interval = 1
    alpha_lead_time = 7.0 # days that the alpha corr was calculated for
    for fore_days in np.arange(0,366,1):
        fore_anomaly = (alpha_cdoy.sel(doy=cdoy).alpha**(fore_days/alpha_lead_time)) * c_anomaly
        
        valid_doy = cdoy + fore_days
        # Keep  range 1-365
        valid_doy = ((valid_doy-1)%365)+1

        # Get predicted SIC by adding damped anomaly with SIC predicted by trend through historical period
        fore_sic = fore_anomaly + obs_clim_model.sel(doy=valid_doy).drop('doy')
        
        # Force prediction SIC to be between 0-1
        ocnmask = fore_sic.notnull()
        fore_sic = fore_sic.where(fore_sic >= 0, other=0).where(ocnmask)
        fore_sic = fore_sic.where(fore_sic <= 1, other=1).where(ocnmask)

        # Add cords
        fore_sic.coords['fore_time'] = np.timedelta64(int(fore_days),'D')
        da_l.append(fore_sic)
        
#         xr.exit()

    da_sic = xr.concat(da_l, dim='fore_time')    
    da_sic.coords['init_time'] = c_sic.time
    
    da_sic = da_sic.drop(['xm','ym','lag','doy','hole_mask','time'])

    da_sic.name = 'sic'

    da_sic = import_data.expand_to_sipn_dims(da_sic)
    
    da_sic = da_sic.rename({'x':'nj','y':'ni'})
            
    
    da_sic.to_netcdf(file_out)
    print("Saved file:",file_out)


# In[ ]:


# Test plots for presentations


# In[ ]:


if test_plots:
    fig_dir = '/home/disk/sipn/nicway/Nic/figures/pres/A'

    cmap_diff_2 = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","white","blue"])
    cmap_diff_2.set_bad(color = 'lightgrey')
    cmap_sic = matplotlib.colors.ListedColormap(sns.color_palette("Blues_r", 10))
    cmap_sic.set_bad(color = 'lightgrey')

    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    (f, ax) = ice_plot.polar_axis()
    obs_clim_model.sel(doy=valid_doy).drop('doy').plot(ax=ax, 
                                            x='lon', y='lat', 
                                         transform=ccrs.PlateCarree(),cmap=cmap_sic,
                                                      cbar_kwargs={'label':'Sea Ice Concentration (-)'})
    plt.title('')
    f_out = os.path.join(fig_dir,'Linear_Trend.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)

    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    (f, ax) = ice_plot.polar_axis()
    c_anomaly.plot(ax=ax, 
                        x='lon', y='lat', 
                     transform=ccrs.PlateCarree(),cmap=cmap_diff_2,
                                  cbar_kwargs={'label':'SIC Anomaly'})
    plt.title('')
    f_out = os.path.join(fig_dir,'Anomoly.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)

    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    (f, ax) = ice_plot.polar_axis()
    alpha_cdoy.sel(doy=cdoy).alpha.plot(ax=ax, 
                        x='lon', y='lat', 
                     transform=ccrs.PlateCarree(),cmap=cmap_diff_2,
                                  cbar_kwargs={'label':'Lag-1 correlation'})
    plt.title('')
    f_out = os.path.join(fig_dir,'Alpha.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)



    sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})

    (f, ax) = ice_plot.polar_axis()
    fore_sic.plot(ax=ax, 
                        x='lon', y='lat', 
                     transform=ccrs.PlateCarree(),cmap=cmap_sic,
                                                      cbar_kwargs={'label':'Sea Ice Concentration (-)'})
    plt.title('')
    f_out = os.path.join(fig_dir,'DampedTrend_Forecast.png')
    f.savefig(f_out,bbox_inches='tight', dpi=300)


# ### Compare forecasts to check they make sense

# In[ ]:


# cmod = 'climatology' # 1979-2017
# runType = 'forecast'

# all_files = os.path.join(mod_dir,cmod,runType,'sipn_nc', str(end_date.year)+'*.nc')
# files = glob.glob(all_files)

# climatology = xr.open_mfdataset(sorted(files), 
#         chunks={'time': 30, 'x': 304, 'y': 448},  
#          concat_dim='time', autoclose=True, parallel=True)



# # cmod = 'dampedAnomaly'
# # runType = 'forecast'

# # all_files = os.path.join(mod_dir,cmod,runType,'sipn_nc', str(end_date.year)+'*.nc')
# # files = glob.glob(all_files)

# # dampedAnomaly = xr.open_mfdataset(sorted(files), 
# #         chunks={'nj': 304, 'ni': 448},  
# #          concat_dim='init_time', autoclose=True, parallel=True)

# cmod = 'dampedAnomalyTrend'

# all_files = os.path.join(mod_dir,cmod,runType,'sipn_nc', str(end_date.year)+'*.nc')
# files = glob.glob(all_files)

# dampedAnomalyTrend = xr.open_mfdataset(sorted(files), 
#         chunks={'nj': 304, 'ni': 448},  
#          concat_dim='init_time', autoclose=True, parallel=True)

# # Calc SIE



# climatology_agg = climatology.sic.sum(dim=['x','y'])

# # dampedAnomaly_agg = dampedAnomaly.sic.sum(dim=['nj','ni']).isel(ensemble=0)
# # dampedAnomaly_agg = import_data.get_valid_time(dampedAnomaly_agg)

# dampedAnomalyTrend_agg = dampedAnomalyTrend.sic.sum(dim=['nj','ni']).isel(ensemble=0)
# dampedAnomalyTrend_agg = import_data.get_valid_time(dampedAnomalyTrend_agg)

# # Get 1980-2010 Mean
# currentYearTime = [np.datetime64('2018-01-01') + np.timedelta64(int(x-1),'D') for x in mean_1980_2010_sic.time.values]
# mean_1980_2010_sic.coords['valid_time'] =  xr.DataArray(currentYearTime, dims='time', coords={'time':mean_1980_2010_sic.time})
# mean_1980_2010_sic_modified = mean_1980_2010_sic.swap_dims({'time':'valid_time'})
# mean_1980_2010_sic_agg = mean_1980_2010_sic.sum(dim=['x','y'])

# # Get 2018 obs
# target_year_obs = ds_81.sic.sel(time=slice('2018-01-01',end_date)).sum(dim=['x','y'])
# target_year_obs





# # Get ylims
# vmin = np.min([climatology_agg.min(), mean_1980_2010_sic_agg.min(), target_year_obs.min()])
# vmax = np.max([climatology_agg.max(), mean_1980_2010_sic_agg.max(), target_year_obs.max()])
# vmin = vmin - (vmax-vmin)*.1
# vmax = vmax + (vmax-vmin)*.1

# (vmin, vmax)

# mean_1980_2010_sic_agg

# f = plt.figure(figsize=(10,8))
# # Plot climatological trend
# climatology_agg.plot(color='k', label='1979-2017 trend', linewidth=5)
# # Plot climatological mean
# mean_1980_2010_sic_agg.swap_dims({'time':'valid_time'}).plot(color='m', label='1980-2010 mean', linewidth=5)
# # Plot target year obs
# target_year_obs.plot(color='b', label='2018 Observations')
# addleg = True
# for it in dampedAnomalyTrend_agg.init_time.values[0::30]:

#     Y = dampedAnomalyTrend_agg.sel(init_time=it)
#     plt.plot(Y.valid_time, Y.values, color='g', label='Damped Anomaly to 1979-2017 trend')
    
#     if addleg:
#         plt.legend()
#         addleg = False

# plt.ylim([vmin,vmax])
# plt.ylabel('*Extent* as sum of sic')

# f_out = os.path.join('/home/disk/sipn/nicway/Nic/figures/pres/A','damped_example.png')
# f.savefig(f_out,bbox_inches='tight', dpi=300)

# M1 = dampedAnomalyTrend.sel(init_time=it).isel(fore_time=0, ensemble=0).sic
# M1.plot()

# O1 = ds_81.sic.sel(time=it)
# O1.plot()

# M1 = M1.rename({'ni':'y','nj':'x'})

# M1

# O1

# (M1-O1).plot()


