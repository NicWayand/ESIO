
# coding: utf-8

# In[1]:


# Standard Imports
import os
import xarray as xr
import pandas as pd

# ESIO Imports
import esiodata as ed


# In[2]:


E = ed.esiodata.load()

file_in = os.path.join(E.obs['NSIDC_extent']['native'], 'N_seaice_extent_daily_v3.0.csv')
file_out = os.path.join(E.obs['NSIDC_extent']['sipn_nc'], 'N_seaice_extent_daily_v3.0.nc')


# In[3]:


dateparse = lambda x: pd.datetime.strptime(x, '%Y     %m   %d')


# In[4]:


df = pd.read_csv(file_in,
                skiprows=1,
                parse_dates={'datetime': ['YYYY', '    MM', '  DD']}, date_parser=dateparse)
df.set_index('datetime', inplace=True)
df.columns = ['Extent','Missing','Source']
ds = xr.Dataset.from_dataframe(df)


# In[5]:


ds.Extent.attrs['units'] = '10^6 sq km'
ds.Extent.attrs['Missing'] = '10^6 sq km'


# In[6]:


ds.to_netcdf(file_out)


