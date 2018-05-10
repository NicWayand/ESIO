
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

# Flags
UpdateAll = False

# Load in regional data
# Note minor -0.000004 degree differences in latitude
ds_region = xr.open_dataset(os.path.join(E.grid_dir, 'sio_2016_mask_Update.nc'))

# Products to import
product_list = ['NSIDC_0081', 'NSIDC_0051', 'NSIDC_0079']

ds_lat_lon = esio.get_stero_N_grid(grid_dir=E.grid_dir)

# Loop through each product
for c_product in product_list:
    print('Importing ', c_product, '...')

    # Find new files that haven't been imported yet
    native_dir = os.path.join(data_dir, c_product, 'native')
    os.chdir(native_dir)
    native_files = sorted(glob.glob('*.bin'))
    nc_dir = os.path.join(data_dir, c_product, 'sipn_nc')
    os.chdir(nc_dir)
    nc_files = sorted(glob.glob('*.nc'))
    if UpdateAll:
        new_files = [x.split('.b')[0] for x in native_files]
        print('Updating all ', len(native_files), ' files...')
    else:
        new_files = np.setdiff1d([x.split('.b')[0] for x in native_files], 
                                 [x.split('.n')[0] for x in nc_files]) # just get file name and compare
        print('Found ', len(new_files), ' new files to import...')

    # Loop through each file
    for nf in new_files:
        
        # Load in 
        ds_sic = lo.load_1_NSIDC(filein=os.path.join(native_dir, nf+'.bin'), product=c_product)

        # Add lat and lon dimensions
        ds_sic.coords['lat'] = ds_lat_lon.lat
        ds_sic.coords['lon'] = ds_lat_lon.lon

        # Stereo projected units (m)
        dx = dy = 25000 
        xm = np.arange(-3850000, +3750000, +dx)
        ym = np.arange(+5850000, -5350000, -dy)
        ds_sic.coords['xm'] = xr.DataArray(xm, dims=('x'))
        ds_sic.coords['ym'] = xr.DataArray(ym, dims=('y'))    
        
        # Calculate extent and area
#         ds_sic['extent'] = ((ds_sic.sic>=0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)
        ds_sic['extent'] = esio.calc_extent(ds_sic.sic, ds_region, fill_pole_hole=True)
#         ds_sic['extent'] = ds_sic['extent'] + (ds_sic.hole_mask.astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # Add hole
        ds_sic['area'] = (ds_sic.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole
    
        # Save to netcdf file
        ds_sic.to_netcdf(os.path.join(nc_dir, nf.split('.b')[0]+'.nc'))
        ds_sic = None
    
#     # Calculate extent and area (saved to separte file)
#     if len(new_files) > 0 : # There were some new files
#         print('Calculating extent and area...')
#         ds_all = xr.open_mfdataset(os.path.join(nc_dir,'*.nc'), concat_dim='time', 
#                                    autoclose=True, compat='no_conflicts',
#                                    data_vars=['sic'])    
#         print('Loaded in all files...')
#         ds_all['extent'] = ((ds_all.sic>=0.15).astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6)
#         ds_all['extent'] = ds_all['extent'] + (ds_all.hole_mask.astype('int') * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # Add hole
#         ds_all['area'] = (ds_all.sic * ds_region.area).sum(dim='x').sum(dim='y')/(10**6) # No pole hole
#         ds_all = ds_all[['extent','area']]
#         # Create new dir to store agg file
#         if not os.path.exists(os.path.join(data_dir, c_product, 'agg_nc')):
#             os.makedirs(os.path.join(data_dir, c_product, 'agg_nc'))
#         ds_all.to_netcdf(os.path.join(data_dir, c_product, 'agg_nc', 'panArctic.nc'))
        
    # For each Product
    print("Finished ", c_product)
    print("")


