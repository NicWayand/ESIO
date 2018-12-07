
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

'''
Plot forecast maps with all available models.
'''




import matplotlib
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
warnings.simplefilter(action='ignore', category=RuntimeWarning) # not good to supress but they divide by nan are annoying
#warnings.simplefilter(action='ignore', category=UserWarning) # https://github.com/pydata/xarray/issues/2273
import json
from esio import EsioData as ed
from esio import ice_plot
from esio import import_data
import subprocess
import dask
from dask.distributed import Client
import timeit

# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=.8, rc={"lines.linewidth": 2.5})


# In[2]:


#client = Client()
#client
dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler


# In[3]:


#def Update_PanArctic_Maps():
# Plotting Info
runType = 'forecast'
variables = ['sic']
metrics_all = {'sic':['anomaly','mean','SIP'], 'hi':['mean']}
#metrics_all = {'sic':['SIP']}
updateAll = False

# Define Init Periods here, spaced by 7 days (aprox a week)
# Now
cd = datetime.datetime.now()
cd = datetime.datetime(cd.year, cd.month, cd.day) # Set hour min sec to 0. 
# Hardcoded start date (makes incremental weeks always the same)
start_t = datetime.datetime(1950, 1, 1) # datetime.datetime(1950, 1, 1)
# Params for this plot
Ndays = 7 # time period to aggregate maps to (default is 7)

#Npers = 40 # number of periods agg (from current date)
init_start_date = np.datetime64('2018-01-01')

init_slice = np.arange(start_t, cd, datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
# init_slice = init_slice[-Npers:] # Select only the last Npers of periods (weeks) since current date
init_slice = init_slice[init_slice>=init_start_date] # Select only the inits after init_start_date

print(init_slice[0],init_slice[-1])
print('')

# Forecast times
weeks = pd.to_timedelta(np.arange(0,52,1), unit='W')
#months = pd.to_timedelta(np.arange(2,12,1), unit='M')
#years = pd.to_timedelta(np.arange(1,2), unit='Y') - np.timedelta64(1, 'D') # need 364 not 365
#slices = weeks.union(months).union(years).round('1d')
da_slices = xr.DataArray(weeks, dims=('fore_time'))
da_slices.fore_time.values.astype('timedelta64[D]')
print(da_slices)


# In[4]:


#############################################################
# Load in Observed and non-dynamic model Data
#############################################################

E = ed.EsioData.load()
mod_dir = E.model_dir

# Get median ice edge by DOY
median_ice_fill = xr.open_mfdataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'ice_edge.nc')).sic
# Get mean sic by DOY
mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic
# Get average sip by DOY
mean_1980_2010_SIP = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'hist_SIP_1980_2010.nc')).sip    

# Get recent observations
ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly/*.nc', concat_dim='time', autoclose=True, parallel=True)#,

# Define models to plot
models_2_plot = list(E.model.keys())
models_2_plot = [x for x in models_2_plot if x not in ['piomas','MME','MME_NEW','uclsipn','hcmr']] # remove some models
models_2_plot = [x for x in models_2_plot if E.icePredicted[x]] # Only predictive models
# models_2_plot = ['rasmesrl']
models_2_plot


# In[ ]:


# def is_in_time_range(x):
    
#     if x.sel(init_time=slice(time_bds[0],time_bds[1])).init_time.size>0: # We have some time in the time range
#         return x
#     else:
#         return []
# time_bds = [init_slice[0],init_slice[-1]]


# In[ ]:


###########################################################
#          Loop through each dynamical model              #
###########################################################

# Plot all Models
for cmod in models_2_plot:
    print(cmod)

    # Load in Model
    # Find only files that have current year and month in filename (speeds up loading)
    all_files = os.path.join(E.model[cmod][runType]['sipn_nc'], '*.nc') 

    # Check we have files 
    files = glob.glob(all_files)
    if not files:
        continue # Skip this model

    # Get list of variablse we want to drop
    drop_vars = [x for x in xr.open_dataset(sorted(files)[-1],autoclose=True).data_vars if x not in variables]
    
    # Load in model   
    ds_model_ALL = xr.open_mfdataset(sorted(files), 
                                 chunks={ 'fore_time': 1,'init_time': 1,'nj': 304, 'ni': 448},  
                                 concat_dim='init_time', autoclose=True, 
                                 parallel=True, drop_variables=drop_vars)
                                 # preprocess=lambda x : is_in_time_range(x)) # 'fore_time': 1, ,
    ds_model_ALL.rename({'nj':'x', 'ni':'y'}, inplace=True)
    
    # Sort by init_time
    ds_model_ALL = ds_model_ALL.sortby('init_time')
        
    # Get Valid time
    ds_model_ALL = import_data.get_valid_time(ds_model_ALL)
    
    # For each variable
    for cvar in variables:

        # For each init time period
        for it in init_slice: 
            it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
            # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
            # So we need to add one day, so we don't double count.
            print(it_start,"to",it)

            # For each forecast time we haven't plotted yet
            #ft_to_plot = ds_status.sel(init_time=it)
            #ft_to_plot = ft_to_plot.where(ft_to_plot.isnull(), drop=True).fore_time

            for ft in da_slices.values: 

                cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
                cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time

                # Get datetime64 of valid time start and end
                valid_start = it_start + ft
                valid_end = it + ft

                # Loop through variable of interest + any metrics (i.e. SIP) based on that
                for metric in metrics_all[cvar]:

                    # File paths and stuff
                    out_metric_dir = os.path.join(E.model['MME_NEW'][runType]['sipn_nc'], cvar, metric)
                    if not os.path.exists(out_metric_dir):
                        os.makedirs(out_metric_dir) 
                        
                    out_init_dir = os.path.join(out_metric_dir, pd.to_datetime(it).strftime('%Y-%m-%d'))
                    if not os.path.exists(out_init_dir):
                        os.makedirs(out_init_dir)
                        
                    out_mod_dir = os.path.join(out_init_dir, cmod)
                    if not os.path.exists(out_mod_dir):
                        os.makedirs(out_mod_dir)     
                        
                    out_nc_file = os.path.join(out_mod_dir, pd.to_datetime(it+ft).strftime('%Y-%m-%d')+'_'+cmod+'.nc')

                    # Only update if either we are updating All or it doesn't yet exist
                    # OR, its one of the last 3 init times 
                    if updateAll | (os.path.isfile(out_nc_file)==False) | np.any(it in init_slice[-3:]):
                        #print("    Updating...")

                        # Select init period and fore_time of interest
                        ds_model = ds_model_ALL.sel(init_time=slice(it_start, it))

                        # Check we found any init_times in range
                        if ds_model.init_time.size==0:
                            #print('init_time not found.')
                            continue

                        # Select var of interest (if available)
                        if cvar in ds_model.variables:
                            ds_model = ds_model[cvar]
                        else:
                            #print('cvar not found.')
                            continue

                        # Check if we have any valid times in range of target dates
                        ds_model = ds_model.where((ds_model.valid_time>=valid_start) & (ds_model.valid_time<=valid_end), drop=True) 
                        if ds_model.fore_time.size == 0:
                            #print("no fore_time found for target period.")
                            continue

                        # Average over for_time and init_times
                        ds_model = ds_model.mean(dim=['fore_time','init_time'])

                        if metric=='mean': # Calc ensemble mean
                            ds_model = ds_model.mean(dim='ensemble')

                        elif metric=='SIP': # Calc probability
                            # Remove ensemble members having missing data
                            ok_ens = ((ds_model.notnull().sum(dim='x').sum(dim='y'))>0) # select ensemble members with any data
                            ds_model = ((ds_model.where(ok_ens, drop=True)>=0.15) ).mean(dim='ensemble').where(ds_model.isel(ensemble=0).notnull())

                        elif metric=='anomaly': # Calc anomaly in reference to mean observed 1980-2010
                            # Get climatological mean
                            da_obs_mean = mean_1980_2010_sic.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                            # Calc anomaly
                            ds_model = ds_model.mean(dim='ensemble') - da_obs_mean
                            # Add back lat/long (get dropped because of round off differences)
                            ds_model['lat'] = da_obs_mean.lat
                            ds_model['lon'] = da_obs_mean.lon
                        else:
                            raise ValueError('metric not implemented')

                        # drop ensemble if still present
                        if 'ensemble' in ds_model:
                            ds_model = ds_model.drop('ensemble')

                        ds_model.coords['model'] = cmod
                        if 'xm' in ds_model:
                            ds_model = ds_model.drop(['xm','ym']) #Dump coords we don't use

                        # Add Coords info
                        ds_model.name = metric
                        ds_model.coords['model'] = cmod
                        ds_model.coords['init_start'] = it_start
                        ds_model.coords['init_end'] = it
                        ds_model.coords['valid_start'] = it_start+ft
                        ds_model.coords['valid_end'] = it+ft
                        ds_model.coords['fore_time'] = ft
                        
                        # Save to file
                        ds_model.to_netcdf(out_nc_file)

                        # Clean up for current model
                        ds_model = None


# In[ ]:


cmod = 'climatology'

all_files = os.path.join(mod_dir,cmod,runType,'sipn_nc', str(cd.year)+'*.nc')
files = glob.glob(all_files)

obs_clim_model = xr.open_mfdataset(sorted(files), 
        chunks={'time': 30, 'x': 304, 'y': 448},  
         concat_dim='time', autoclose=True, parallel=True)


# In[ ]:


# from esio import metrics
# ds_region = xr.open_mfdataset(os.path.join(E.grid_dir, 'sio_2016_mask_Update.nc'))

# X = obs_clim_model.sic.sel(time=slice('2018-09-01','2018-09-30'))

# metrics.calc_extent(da=X, region=ds_region).mean(dim='time').values


# In[ ]:


###########################################################
#          climatology  trend                             #
###########################################################

cmod = 'climatology'

all_files = os.path.join(mod_dir,cmod,runType,'sipn_nc', str(cd.year)+'*.nc')
files = glob.glob(all_files)

obs_clim_model = xr.open_mfdataset(sorted(files), 
        chunks={'time': 30, 'x': 304, 'y': 448},  
         concat_dim='time', autoclose=True, parallel=True)

# For each variable
for cvar in variables:

    # For each init time period
    for it in init_slice: 
        it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
        # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
        # So we need to add one day, so we don't double count.
        print(it_start,"to",it)

        for ft in da_slices.values: 

            cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
            cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time

            # Get datetime64 of valid time start and end
            valid_start = it_start + ft
            valid_end = it + ft

            # Loop through variable of interest + any metrics (i.e. SIP) based on that
            for metric in metrics_all[cvar]:

                # File paths and stuff
                out_metric_dir = os.path.join(E.model['MME_NEW'][runType]['sipn_nc'], cvar, metric)
                if not os.path.exists(out_metric_dir):
                    os.makedirs(out_metric_dir) 

                out_init_dir = os.path.join(out_metric_dir, pd.to_datetime(it).strftime('%Y-%m-%d'))
                if not os.path.exists(out_init_dir):
                    os.makedirs(out_init_dir)

                out_mod_dir = os.path.join(out_init_dir, cmod)
                if not os.path.exists(out_mod_dir):
                    os.makedirs(out_mod_dir)     

                out_nc_file = os.path.join(out_mod_dir, pd.to_datetime(it+ft).strftime('%Y-%m-%d')+'_'+cmod+'.nc')

                # Only update if either we are updating All or it doesn't yet exist
                # OR, its one of the last 3 init times 
                if updateAll | (os.path.isfile(out_nc_file)==False) | np.any(it in init_slice[-3:]):
                    #print("    Updating...")

                    # Check if we have any valid times in range of target dates
                    ds_model = obs_clim_model[cvar].where((obs_clim_model.time>=valid_start) & (obs_clim_model.time<=valid_end), drop=True) 
                    if 'time' in ds_model.lat.dims:
                        ds_model.coords['lat'] = ds_model.lat.isel(time=0).drop('time') # Drop time from lat/lon dims (not sure why?)

                    # If we have any time
                    if ds_model.time.size > 0:

                        # Average over time
                        ds_model = ds_model.mean(dim='time')

                        if metric=='mean': # Calc ensemble mean
                            ds_model = ds_model
                        elif metric=='SIP': # Calc probability
                            # Issue of some ensemble members having missing data
                            ocnmask = ds_model.notnull()
                            ds_model = (ds_model>=0.15).where(ocnmask)
                        elif metric=='anomaly': # Calc anomaly in reference to mean observed 1980-2010
                            # Get climatological mean
                            da_obs_mean = mean_1980_2010_sic.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                            # Get anomaly
                            ds_model = ds_model - da_obs_mean
                            # Add back lat/long (get dropped because of round off differences)
                            ds_model['lat'] = da_obs_mean.lat
                            ds_model['lon'] = da_obs_mean.lon
                        else:
                            raise ValueError('metric not implemented')   

                        # Drop un-needed coords to match model format
                        if 'doy' in ds_model.coords:
                            ds_model = ds_model.drop(['doy'])
                        if 'xm' in ds_model.coords:
                            ds_model = ds_model.drop(['xm'])
                        if 'ym' in ds_model.coords:
                            ds_model = ds_model.drop(['ym'])
                    
                        # Add Coords info
                        ds_model.name = metric
                        ds_model.coords['model'] = cmod
                        ds_model.coords['init_start'] = it_start
                        ds_model.coords['init_end'] = it
                        ds_model.coords['valid_start'] = it_start+ft
                        ds_model.coords['valid_end'] = it+ft
                        ds_model.coords['fore_time'] = ft
                        
                        # Save to file
                        ds_model.to_netcdf(out_nc_file)

                        # Clean up for current model
                        ds_model = None


# In[ ]:


############################################################################
#                               OBSERVATIONS                               #
############################################################################

cmod = 'Observed'

updateAll = True # We ALWAYS want to update all observations, because each day we get new obs that can be used to evaluate forecasts from up to 12 months ago

# For each variable
for cvar in variables:

    # For each init time period
    for it in init_slice: 
        it_start = it-np.timedelta64(Ndays,'D') + np.timedelta64(1,'D') # Start period for init period (it is end of period). Add 1 day because when
        # we select using slice(start,stop) it is inclusive of end points. So here we are defining the start of the init AND the start of the valid time.
        # So we need to add one day, so we don't double count.
        print(it_start,"to",it)

        for ft in da_slices.values: 

            cdoy_end = pd.to_datetime(it + ft).timetuple().tm_yday # Get current day of year end for valid time
            cdoy_start = pd.to_datetime(it_start + ft).timetuple().tm_yday  # Get current day of year end for valid time

            # Get datetime64 of valid time start and end
            valid_start = it_start + ft
            valid_end = it + ft

            # Loop through variable of interest + any metrics (i.e. SIP) based on that
            for metric in metrics_all[cvar]:

                # File paths and stuff
                out_metric_dir = os.path.join(E.model['MME_NEW'][runType]['sipn_nc'], cvar, metric)
                if not os.path.exists(out_metric_dir):
                    os.makedirs(out_metric_dir) 

                out_init_dir = os.path.join(out_metric_dir, pd.to_datetime(it).strftime('%Y-%m-%d'))
                if not os.path.exists(out_init_dir):
                    os.makedirs(out_init_dir)

                out_mod_dir = os.path.join(out_init_dir, cmod)
                if not os.path.exists(out_mod_dir):
                    os.makedirs(out_mod_dir)     

                out_nc_file = os.path.join(out_mod_dir, pd.to_datetime(it+ft).strftime('%Y-%m-%d')+'_'+cmod+'.nc')

                # Only update if either we are updating All or it doesn't yet exist
                # OR, its one of the last 3 init times 
                if updateAll | (os.path.isfile(out_nc_file)==False) | np.any(it in init_slice[-3:]):
                    #print("    Updating...")

                    # Check if we have any valid times in range of target dates
                    ds_model = da_obs_c = ds_81[cvar].sel(time=slice(valid_start, valid_end))
                    
                    if 'time' in ds_model.lat.dims:
                        ds_model.coords['lat'] = ds_model.lat.isel(time=0).drop('time') # Drop time from lat/lon dims (not sure why?)

                    # Check we have all observations for this week (7)
                    if ds_model.time.size == 7:

                        if metric=='mean':
                            ds_model = ds_model.mean(dim='time') #ds_81.sic.sel(time=(it + ft))
                        elif metric=='SIP':
                            ds_model = (ds_model >= 0.15).mean(dim='time').astype('int').where(ds_model.isel(time=0).notnull())
                        elif metric=='anomaly':
                            da_obs_VT = ds_model.mean(dim='time')
                            da_obs_mean = mean_1980_2010_sic.isel(time=slice(cdoy_start,cdoy_end)).mean(dim='time')
                            ds_model = da_obs_VT - da_obs_mean
                        else:
                            raise ValueError('Not implemented')

                        # Drop coords we don't need
                        ds_model = ds_model.drop(['hole_mask','xm','ym'])
                        if 'time' in ds_model:
                            ds_model = ds_model.drop('time')

                        # Add Coords info
                        ds_model.name = metric
                        ds_model.coords['model'] = cmod
                        ds_model.coords['init_start'] = it_start
                        ds_model.coords['init_end'] = it
                        ds_model.coords['valid_start'] = it_start+ft
                        ds_model.coords['valid_end'] = it+ft
                        ds_model.coords['fore_time'] = ft

                        # Write to disk
                        ds_model.to_netcdf(out_nc_file)

                        # Clean up for current model
                        ds_model = None


# In[ ]:


# 1:04
# 
# 


# In[ ]:


# from dask.distributed import Client
# client = Client(n_workers=8)
# client


# In[5]:


cvar = 'sic' # hard coded for now
# Load in all data and write to Zarr
# Load in all metrics for given variable
print("Loading in weekly metrics...")
ds_m = import_data.load_MME_by_init_end(E=E, 
                                        runType=runType, 
                                        variable=cvar, 
                                        metrics=metrics_all[cvar])

# Drop models that we don't evaluate (i.e. monthly means)
models_keep = [x for x in ds_m.model.values if x not in ['noaasipn','modcansipns_3','modcansipns_4']]
ds_m = ds_m.sel(model=models_keep)
# Get list of dynamical models that are not observations
dynamical_Models = [x for x in ds_m.model.values if x not in ['Observed','climatology','dampedAnomaly','dampedAnomalyTrend']]
# # Get list of all models
# all_Models = [x for x in ds_m.model.values if x not in ['Observed']]
# Add MME
MME_avg = ds_m.sel(model=dynamical_Models).mean(dim='model') # only take mean over dynamical models
MME_avg.coords['model'] = 'MME'
ds_ALL = xr.concat([ds_m, MME_avg], dim='model')


# In[6]:


ds_ALL


# In[ ]:


# check chunk size


# In[ ]:



###### ADD METADATA #################

## Add coordinate system info
ds_ALL.coords['crs'] = xr.DataArray('crs')
ds_ALL['crs'].attrs = {
    'comment': '(https://nsidc.org/data/polar-stereo/ps_grids.html or https://nsidc.org/data/oib/epsg_3413.html) This is a container variable that describes the grid_mapping used by the data in this file. This variable does not contain any data; only information about the geographic coordinate system',
    'grid_mapping_name': 'polar_stereographic',
    'straight_vertical_longitude_from_pole':'-45',
    'latitude_of_projection_origin': '90.0',
    'standard_parallel':'70',
    'false_easting':'0',
    'false_northing':'0'
    }

# Add time coords
ds_ALL.coords['init_start'] = ds_ALL.init_end - np.timedelta64(Ndays,'D') + np.timedelta64(1,'D')
ds_ALL['init_start'].attrs = {
    'comment':        'Start date for weekly average period',
    'long_name':      'Start date for weekly average period',
    'standard_name':  "start_init_date"}

ds_ALL['init_end'].attrs = {
    'comment':        'End date for weekly average period',
    'long_name':      'End date for weekly average period',
    'standard_name':  "end_init_date"}

ds_ALL['fore_time'].attrs = {
    'comment':        'Forecast lead time',
    'long_name':      'Forecast lead time',
    'standard_name':  "forecast_lead_time"}

# Add Valid time (start and end period)
ds_ALL = import_data.get_valid_time(ds_ALL, init_dim='init_end', fore_dim='fore_time')
ds_ALL.rename({'valid_time':'valid_end'}, inplace=True);
ds_ALL.coords['valid_start'] = ds_ALL.valid_end - np.timedelta64(Ndays,'D') + np.timedelta64(1,'D')

# Add attributes
ds_ALL['valid_end'].attrs = {
    'comment':        'End Valid date for weekly average period',
    'long_name':      'End Valid date for weekly average period',
    'standard_name':  "end_valid_date"}

ds_ALL['valid_start'].attrs = {
    'comment':        'Start Valid date for weekly average period',
    'long_name':      'Start Valid date for weekly average period',
    'standard_name':  "start_valid_date"}

# Add Variable attributes
ds_ALL['SIP'].attrs = {
    'comment':        'Sea ice probability, calculated by averaging across ensemble members predictions of sea ice concentration >= 0.15',
    'grid_mapping':   'crs',
    'long_name':      'Sea ice probability',
    'standard_name':  "sea_ice_probability",
    'units':          'fraction'}

ds_ALL['anomaly'].attrs = {
    'comment':        'Anomaly of the forecasted sea ice concentration mean (ensemble average) compared to the 1980 to 2010 Observed Climatology',
    'grid_mapping':   'crs',
    'long_name':      'Anomaly',
    'standard_name':  "anomaly",
    'units':          'fraction'}

ds_ALL['mean'].attrs = {
    'comment':        'Mean of the forecasted sea ice concentration (ensemble average)',
    'grid_mapping':   'crs',
    'long_name':      'Sea ice concentration',
    'standard_name':  "sea_ice_concentration",
    'units':          'fraction'}

# Dataset Attributes
ds_ALL.attrs = {
'comment':                         'Weekly mean sea ice concentration forecasted by multiple models as well as observed by remotly sensed passive microwave sensors.',
'contact':                         'nicway@uw.edu',
'creator_email':                   'nicway@uw.edu',
'creator_name':                    'Nicholas Wayand, University of Washington',
'creator_url':                     'https://atmos.uw.edu/sipn/',
'date_created':                    '2018-12-03T00:00:00',
'date_modified':                   datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
'geospatial_lat_max':              str(float(ds_ALL.lat.max().values)),
'geospatial_lat_min':              str(float(ds_ALL.lat.min().values)),
'geospatial_lat_resolution':       '~25km',
'geospatial_lat_units':            'degrees_north',
'geospatial_lon_max':              str(float(ds_ALL.lon.max().values)),
'geospatial_lon_min':              str(float(ds_ALL.lon.min().values)),
'geospatial_lon_resolution':       '~25km',
'geospatial_lon_units':            'degrees_east',
'history':                         datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')+': updated by Nicholas Wayand',
'institution':                     'UW, SIPN, ARCUS',
'keywords':                        'Arctic > Sea ice concentration > Prediction',
'product_version':                 '1.0',
'project':                         'Sea Ice Prediction Network Phase II',
'references':                      'Wayand, N.E., Bitz, C.M., and E. Blanchard-Wrigglesworth, (in review). A year-round sub-seasonal to seasonal sea ice prediction portal. Submited to Geophysical Research letters.',
'source':                          'Numerical model predictions and Passive microwave measurments.',
'summary':                         'Dataset is updated daily with weekly sea ice forecasts',
'time_coverage_end':               pd.to_datetime(ds_ALL.valid_end.max().values).strftime('%Y-%m-%dT%H:%M:%S'),
'time_coverage_start':             pd.to_datetime(ds_ALL.init_start.min().values).strftime('%Y-%m-%dT%H:%M:%S'),
'title':                           'SIPN2 Sea ice Concentration Forecasts and Observations.'
}

####################################
print(ds_ALL)

# Save to Zarr
print("Saving to Zarr...")
ds_ALL.to_zarr(os.path.join(E.data_dir,'model/zarr', cvar+'.zarr'), mode='w')
print("Finished updating Weekly SIC metrics and saved to Zar")
ds_ALL=None # Flush memory
MME_avg=None
ds_m=None


