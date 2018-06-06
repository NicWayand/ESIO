
# coding: utf-8

# In[ ]:





import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import numpy.ma as ma
import struct
import os
import xarray as xr
import glob
import datetime 
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns

# ESIO Imports

from esio import EsioData as ed


# In[4]:


# from dask.distributed import Client
# client = Client()
# print(client)
import dask
dask.set_options(get=dask.get)


# In[ ]:


# General plotting settings
sns.set_style('whitegrid')
sns.set_context("talk", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[ ]:


#############################################################
# Load in Data
#############################################################

E = ed.EsioData.load()

# Load in Obs
data_dir = E.data_dir
grid_dir = E.grid_dir
temp_dir = r'/home/disk/sipn/nicway/data/model/temp'
# fig_dir = os.path.join(E.fig_dir, 'obs', 'NSIDC_0081' , 'standard')
da_51_in = xr.open_dataarray(E.obs['NSIDC_0051']['sipn_nc']+'/NSIDC_0051.nc')


# In[ ]:


models = ['gfdlsipn']
runType = 'reforecast'
variables = ['sic']


# In[ ]:


c_model = models[0]
cvar = variables[0]


# In[ ]:


# Load in Model
model_forecast = os.path.join(E.model[c_model][runType]['sipn_nc'], '*.nc')
ds_model = xr.open_mfdataset(model_forecast)
ds_model.rename({'nj':'x', 'ni':'y'}, inplace=True)

# Set attributes
ds_model.attrs['model_label'] = E.model[c_model]['model_label']
ds_model.attrs['model_grid_file'] = E.model[c_model]['grid']
ds_model.attrs['stero_grid_file'] = E.obs['NSIDC_0051']['grid']


# In[ ]:


# Select by variable
da_mod_in = ds_model[cvar]


# In[ ]:


da_mod_in


# In[ ]:


# Mask out to common extent (both observations and model have non-null values)
(da_obs, da_mod) = esio.mask_common_extent(da_51_in, da_mod_in)


# In[ ]:


# Aggregate over domain
# TODO: USE the correct area!!
da_obs_avg = da_obs.sum(dim='x').sum(dim='y')*(25*25)/(10**6)
# da_79_avg = da_79.sum(dim='x').sum(dim='y')*(25*25)/(10**6)
# da_81_avg = da_81.sum(dim='x').sum(dim='y')*(25*25)/(10**6)
da_mod_avg = da_mod.sum(dim='x').sum(dim='y')*(25*25)/(10**6)


# In[ ]:


da_mod_avg #.fore_time.values.astype('timedelta64[D]')


# In[ ]:


# Aggreagate Obs to Model temporal time stamp
# ex: gfdl data is monthly, time stamp at beinging of period
da_obs_avg_mon = da_obs_avg.resample(time='MS', label='left').mean()
da_mod_avg_mon = da_mod_avg #.resample(fore_time='d', label='left', keep_attrs=True).mean(dim='fore_time') # Already monthly means, 


# In[ ]:


# Trim to common time periods
(ds_obs_trim, ds_mod_trim) = esio.trim_common_times(da_obs_avg_mon, da_mod_avg_mon)


# In[ ]:


# Temp dump to netcdf then load
os.chdir( temp_dir )
c_e, datasets = zip(*ds_mod_trim.to_dataset(name='sic').groupby('ensemble'))
paths = ['GFDL_extent_esns_%s.nc' % e for e in c_e]
xr.save_mfdataset(datasets, paths)


# In[ ]:


print("Done!")


# In[ ]:


# ds_mod_trim = None # Flush memory


# In[ ]:


# ds_mod_trim = xr.open_mfdataset(os.path.join(temp_dir, 'GFDL_extent_esns_*.nc'), concat_dim='ensemble')

# ds_mod_trim = ds_mod_trim.reindex(ensemble=sorted(ds_mod_trim.ensemble.values))
# # 
# ds_mod_trim.fore_time.values.astype('timedelta64[D]')


# In[ ]:


# Slow way... loop over each init_time and forecast time, calcuate metric


# In[ ]:


# Format obs like model
# da_obs_avg_mon_X = esio.format_obs_like_model(ds_mod_trim, ds_obs_trim)


# In[ ]:


# Get observational mean and sigma
# (mu, sigma) = esio.clim_mu_sigma(da_obs_avg_mon, method='MK')


# In[ ]:


# c_nrmse = esio.NRMSE(ds_mod_trim, da_obs_avg_mon_X, sigma)
# print(c_nrmse)


# In[ ]:


#NRMSE is following the same pattern per months as Hawkins et al. 2016.
# f, ax1 = plt.subplots(1,1)
# f.set_size_inches(20, 10)
# c_nrmse.plot(ax=ax1)


