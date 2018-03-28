
# coding: utf-8

# In[5]:


# Standard Imports
import os
import xarray as xr
import pandas as pd

# ESIO Imports
import esiodata as ed


# In[6]:


E = ed.esiodata.load()
# Directories
model='yopp'
runType='forecast'
updateall = False
file_in = os.path.join(E.obs['NSIDC_extent']['native'], 'N_seaice_extent_daily_v3.0.csv')
file_out = os.path.join(E.obs['NSIDC_extent']['sipn_nc'], 'N_seaice_extent_daily_v3.0.nc')


# In[7]:


dateparse = lambda x: pd.datetime.strptime(x, '%Y     %m   %d')


# In[8]:


df = pd.read_csv(file_in,
                skiprows=1,
                parse_dates={'datetime': ['YYYY', '    MM', '  DD']}, date_parser=dateparse)
df.set_index('datetime', inplace=True)
df.columns = ['Extent','Missing','Source']
ds = xr.Dataset.from_dataframe(df)


# In[9]:


ds.Extent.attrs['units'] = '10^6 sq km'
ds.Extent.attrs['Missing'] = '10^6 sq km'


# In[10]:


ds.to_netcdf(file_out)


