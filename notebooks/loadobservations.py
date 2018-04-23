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
            #http://nsidc.org/data/nsidc-0051
            #http://nsidc.org/data/nsidc-0081
            hdr = fr.read(300) 
            ice = np.fromfile(fr, dtype=np.uint8)
            ice = ice.reshape(n_rows, n_cols)
            ice_max = 250
            hole_mask = 251
            coast = 253
            land = 254
            missing = 255
        elif product=='NSIDC_0079':
            ice = np.fromfile(fr, dtype=np.uint16)
            ice = ice.reshape(n_rows, n_cols)
            ice_max = 1000.
            hole_mask = 1100
            coast = 9999
            land = 1200
            missing = 9999
        else:
            raise ValueError('product name not found')
    
    # Make xarray dataArray
    da_all = xr.DataArray(ice, coords={'x': x, 'y': y}, dims=('y', 'x'))
    # Scale to (0-1) and mask out non-sic
    ds = (da_all/ice_max)
    ds.name = 'sic'
    ds = ds.where(ds<=1).to_dataset()
    # Add date
    ds.coords['time'] = parse_NSIDC_date(os.path.basename(cfile))
    ds.expand_dims('time')
    #if get_masks:
    # Add other masks
    ds.coords['hole_mask'] = da_all==hole_mask
    #ds.coords['coast'] = da_all==coast # Commented out because makes filse too slow to load, and not used.
    #ds.coords['land'] = da_all==land
    #ds.coords['missing'] = da_all==missing
    
    return ds


# Loads in one file
def load_1_NSIDC(filein=None, product=None):
    # Define coords
    # Indices values
    x = np.arange(0,304,1)
    y = np.arange(0,448,1)
    
    ds_sic = read_NSIDC_binary(filein, x, y, product)
    
    return ds_sic

# Loads in multiple files
def load_NSIDC(all_files=None, product=None):
    # Define coords

    # Indices values
    x = np.arange(0, 304, 1)
    y = np.arange(0, 448, 1)
    # Loop through each binary file and read into a Dataarray
    da_l = []
    for cf in all_files:
        da_l.append(read_NSIDC_binary(cf, x, y, product))
    ds_sic = xr.concat(da_l, dim='time', coords='different')
    
    return ds_sic


