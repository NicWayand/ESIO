
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


# In[2]:


dask.config.set(scheduler='threads')  # overwrite default with threaded scheduler (This is faster for this code)


# In[3]:


# from dask.distributed import Client
# client = Client(n_workers=8)
# client


# In[4]:


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
Npers = 4 # init periods to put into a Zarr chunk

init_start_date = np.datetime64('2018-01-01')

init_slice = np.arange(start_t, cd, datetime.timedelta(days=Ndays)).astype('datetime64[ns]')
# init_slice = init_slice[-Npers:] # Select only the last Npers of periods (weeks) since current date
init_slice = init_slice[init_slice>=init_start_date] # Select only the inits after init_start_date
print(init_slice[0],init_slice[-1])


# In[5]:


init_slice


# In[6]:


#############################################################
# Load in Observed and non-dynamic model Data
#############################################################

E = ed.EsioData.load()
mod_dir = E.model_dir

# # Get median ice edge by DOY
# median_ice_fill = xr.open_mfdataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'ice_edge.nc')).sic
# # Get mean sic by DOY
# mean_1980_2010_sic = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'mean_1980_2010_sic.nc')).sic
# # Get average sip by DOY
# mean_1980_2010_SIP = xr.open_dataset(os.path.join(E.obs_dir, 'NSIDC_0051', 'agg_nc', 'hist_SIP_1980_2010.nc')).sip    

# # Get recent observations
# ds_81 = xr.open_mfdataset(E.obs['NSIDC_0081']['sipn_nc']+'_yearly/*.nc', concat_dim='time', autoclose=True, parallel=True)#,

# # Define models to plot
# models_2_plot = list(E.model.keys())
# models_2_plot = [x for x in models_2_plot if x not in ['piomas','MME','MME_NEW','uclsipn','hcmr']] # remove some models
# models_2_plot = [x for x in models_2_plot if E.icePredicted[x]] # Only predictive models
# # models_2_plot = ['rasmesrl']
# models_2_plot


# In[7]:


cvar = 'sic' # hard coded for now


# For each init chunk
for IS in np.arange(0,len(init_slice),Npers): # Every fourth init date
    start_time_cmod = timeit.default_timer()

    it_start = init_slice[IS]
    if (IS+Npers-1)>=len(init_slice):
        it_end = init_slice[-1]
    else:
        it_end = init_slice[IS+Npers-1]
    
    # Output Zarr dir
    c_zarr_file = os.path.join(E.data_dir,'model/zarr', 'temp', cvar+pd.to_datetime(it_start).strftime('_%Y-%m-%d')+'.zarr')
    
    print("Checking on ",it_start)
    
    # Check if dir exists
    if updateAll | (os.path.isdir(c_zarr_file)==False) | (it_start>init_slice[-1] - np.timedelta64(60,'D')):

        print("Processing",it_start, it_end)

        # Load in all metrics for given variable
        print("Loading in weekly metrics...")
        ds_m = import_data.load_MME_by_init_end(E=E, 
                                                runType=runType, 
                                                variable=cvar, 
                                                metrics=metrics_all[cvar],
                                                init_range=[it_start,it_end])

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


 

        ####################################
#         print(ds_ALL)


        # Rechunk from ~1MB to 100MB chunks
        # Chunk along fore_time and init_end
        ds_ALL = ds_ALL.chunk({'init_end':10,'fore_time': 10, 'model': 1, 'x': 304, 'y': 448})

        # Save to Zarr chunk
        print("Saving to Zarr...")
        ds_ALL.to_zarr(c_zarr_file, mode='w')
        print("Finished ",pd.to_datetime(it_start).strftime('%Y-%m-%d'))
        print("Took ", (timeit.default_timer() - start_time_cmod)/60, " minutes.")
        ds_ALL=None # Flush memory
        MME_avg=None
        ds_m=None


# In[8]:


# xr.open_zarr('/home/disk/sipn/nicway/data/model/zarr/temp/sic_2018-06-24.zarr').SIP.sel(model='MME').notnull().sum().values


# In[9]:


# Combine all Zarr chunks
zarr_dir = '/home/disk/sipn/nicway/data/model/zarr/temp/'
zarr_inits = sorted([ name for name in os.listdir(zarr_dir) if os.path.isdir(os.path.join(zarr_dir, name)) ])
zarr_inits


# In[10]:


zl = []
for c_init in zarr_inits:
    ds = xr.open_zarr(os.path.join(zarr_dir,c_init))
    zl.append(ds)
ds_Zarr = xr.concat(zl,dim='init_end')


# In[11]:


print(ds_Zarr)


# In[12]:


###### ADD METADATA #################

## Add coordinate system info
ds_Zarr.coords['crs'] = xr.DataArray('crs')
ds_Zarr['crs'].attrs = {
    'comment': '(https://nsidc.org/data/polar-stereo/ps_grids.html or https://nsidc.org/data/oib/epsg_3413.html) This is a container variable that describes the grid_mapping used by the data in this file. This variable does not contain any data; only information about the geographic coordinate system',
    'grid_mapping_name': 'polar_stereographic',
    'straight_vertical_longitude_from_pole':'-45',
    'latitude_of_projection_origin': '90.0',
    'standard_parallel':'70',
    'false_easting':'0',
    'false_northing':'0'
    }

# Add time coords
ds_Zarr.coords['init_start'] = ds_Zarr.init_end - np.timedelta64(Ndays,'D') + np.timedelta64(1,'D')
ds_Zarr['init_start'].attrs = {
    'comment':        'Start date for weekly average period',
    'long_name':      'Start date for weekly average period',
    'standard_name':  "start_init_date"}

ds_Zarr['init_end'].attrs = {
    'comment':        'End date for weekly average period',
    'long_name':      'End date for weekly average period',
    'standard_name':  "end_init_date"}

ds_Zarr['fore_time'].attrs = {
    'comment':        'Forecast lead time',
    'long_name':      'Forecast lead time',
    'standard_name':  "forecast_lead_time"}

# Add Valid time (start and end period)
ds_Zarr = import_data.get_valid_time(ds_Zarr, init_dim='init_end', fore_dim='fore_time')
ds_Zarr.rename({'valid_time':'valid_end'}, inplace=True);
ds_Zarr.coords['valid_start'] = ds_Zarr.valid_end - np.timedelta64(Ndays,'D') + np.timedelta64(1,'D')

# Add attributes
ds_Zarr['valid_end'].attrs = {
    'comment':        'End Valid date for weekly average period',
    'long_name':      'End Valid date for weekly average period',
    'standard_name':  "end_valid_date"}

ds_Zarr['valid_start'].attrs = {
    'comment':        'Start Valid date for weekly average period',
    'long_name':      'Start Valid date for weekly average period',
    'standard_name':  "start_valid_date"}

# Add Variable attributes
ds_Zarr['SIP'].attrs = {
    'comment':        'Sea ice probability, calculated by averaging across ensemble members predictions of sea ice concentration >= 0.15',
    'grid_mapping':   'crs',
    'long_name':      'Sea ice probability',
    'standard_name':  "sea_ice_probability",
    'units':          'fraction'}

ds_Zarr['anomaly'].attrs = {
    'comment':        'Anomaly of the forecasted sea ice concentration mean (ensemble average) compared to the 1980 to 2010 Observed Climatology',
    'grid_mapping':   'crs',
    'long_name':      'Anomaly',
    'standard_name':  "anomaly",
    'units':          'fraction'}

ds_Zarr['mean'].attrs = {
    'comment':        'Mean of the forecasted sea ice concentration (ensemble average)',
    'grid_mapping':   'crs',
    'long_name':      'Sea ice concentration',
    'standard_name':  "sea_ice_concentration",
    'units':          'fraction'}

# Dataset Attributes
ds_Zarr.attrs = {
'comment':                         'Weekly mean sea ice concentration forecasted by multiple models as well as observed by remotly sensed passive microwave sensors.',
'contact':                         'nicway@uw.edu',
'creator_email':                   'nicway@uw.edu',
'creator_name':                    'Nicholas Wayand, University of Washington',
'creator_url':                     'https://atmos.uw.edu/sipn/',
'date_created':                    '2018-12-03T00:00:00',
'date_modified':                   datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
'geospatial_lat_max':              str(float(ds_Zarr.lat.max().values)),
'geospatial_lat_min':              str(float(ds_Zarr.lat.min().values)),
'geospatial_lat_resolution':       '~25km',
'geospatial_lat_units':            'degrees_north',
'geospatial_lon_max':              str(float(ds_Zarr.lon.max().values)),
'geospatial_lon_min':              str(float(ds_Zarr.lon.min().values)),
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
'time_coverage_end':               pd.to_datetime(ds_Zarr.valid_end.max().values).strftime('%Y-%m-%dT%H:%M:%S'),
'time_coverage_start':             pd.to_datetime(ds_Zarr.init_start.min().values).strftime('%Y-%m-%dT%H:%M:%S'),
'title':                           'SIPN2 Sea ice Concentration Forecasts and Observations.'
}


# In[13]:


# Hack to decode strings
# ds_Zarr['model'] = [s.decode("utf-8") for s in ds_Zarr.model.values]


# In[ ]:


# Save to one Big Zarr
ds_Zarr.to_zarr(os.path.join(E.data_dir,'model/zarr', cvar+'.zarr'), mode='w')
print("Saved to Large Zarr.")


