import numpy as np
import xarray as xr
import os
import datetime

''' Functions to import and anlayzie sea ice observations '''


############################################################################
# Loading in from native format to sipn netcdf
############################################################################


def parse_NSIDC_date(str1):
    date1 = str1.split('_')[1]
    yyyy = int(date1[0:4])
    mm = int(date1[4:6])
    dd = int(date1[6:8])
    return datetime.datetime(yyyy,mm,dd)

def read_NSIDC_binary(cfile, x, y, product=None):
    n_rows=448
    n_cols=304
    with open(cfile, 'rb') as fr:
        if product=='NSIDC_0051' or product=='NSIDC_0081':
            hdr = fr.read(300) 
            ice = np.fromfile(fr, dtype=np.uint8)
            ice = ice.reshape(n_rows, n_cols)
            ice = ice / 250.
        elif product=='NSIDC_0079':
            ice = np.fromfile(fr, dtype=np.uint16)
            ice = ice.reshape(n_rows, n_cols)
            ice = ice / 1000. 
        else:
            raise ValueError('product name not found')
    
    # Make xarray dataArray
    da_sic = xr.DataArray(ice, coords={'x': x, 'y': y}, dims=('y', 'x'))
    # Add date
    da_sic.coords['time'] = parse_NSIDC_date(os.path.basename(cfile))
    # Mask out non-sic
    da_sic = da_sic.where(da_sic<=1)
    return da_sic

def load_NSIDC(all_files=None, product=None):
    # Define coords
    # Stereo projected units (m?)
    #dx = dy = 25000
    #x = np.arange(-3850000, +3750000, +dx)
    #y = np.arange(+5850000, -5350000, -dy)
    # Indices values
    x = np.arange(0,304,1)
    y = np.arange(0,448,1)
    # Loop through each binary file and read into a Dataarray
    da_l = []
    for cf in all_files:
        da_l.append(read_NSIDC_binary(cf, x, y, product))
    ds_sic = xr.concat(da_l, dim='time')
    return ds_sic


