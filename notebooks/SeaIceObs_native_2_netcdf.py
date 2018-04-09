
# coding: utf-8

# In[ ]:


import numpy as np
import numpy.ma as ma
import os
import xarray as xr
import glob
import loadobservations as lo
import esio
import esiodata as ed

# Dirs
E = ed.esiodata.load()
data_dir = E.obs_dir

# Load in regional data
# Note minor -0.000004 degree differences in latitude
ds_region = xr.open_dataset(os.path.join(E.grid_dir, 'sio_2016_mask.nc'))
ds_region.set_coords(['lat','lon'], inplace=True);
ds_region.rename({'nx':'x', 'ny':'y'}, inplace=True);

# Products to import
product_list = ['NSIDC_0081', 'NSIDC_0051' , 'NSIDC_0079']

# Loop through each product
for c_product in product_list:

    c_data_dir = os.path.join(data_dir, c_product, 'native', '*.bin')
    all_files = sorted(glob.glob(c_data_dir))
    out_dir = os.path.join(data_dir, c_product, 'sipn_nc')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load in 
    ds_sic = lo.load_NSIDC(all_files=all_files, product=c_product)

    # Add lat and lon dimensions
    ds_lat_lon = esio.get_stero_N_grid(grid_dir=E.grid_dir)
    ds_sic.coords['lat'] = ds_lat_lon.lat
    ds_sic.coords['lon'] = ds_lat_lon.lon
    
    # Stereo projected units (m)
    dx = dy = 25000 
    xm = np.arange(-3850000, +3750000, +dx)
    ym = np.arange(+5850000, -5350000, -dy)
    ds_sic.coords['xm'] = xr.DataArray(xm, dims=('x'))
    ds_sic.coords['ym'] = xr.DataArray(ym, dims=('y'))    
    
    # Calculate common values (panArctic extent/area)
    ds_sic['extent'] = ((ds_sic.sic>=0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)
    ds_sic['extent'] = ds_sic['extent'] + (ds_sic.hole_mask.astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # Add hole

    ds_sic['area'] = (ds_sic.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole

    
    # Save to netcdf file
    out_nc = c_product+'.nc'
    ds_sic.to_netcdf(os.path.join(out_dir,out_nc))
    print("Saved ",out_nc)
    print(ds_sic)
    ds_sic = None


