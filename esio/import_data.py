import datetime
import os
import re
import numpy as np
import pandas as pd
import scipy
import xarray as xr
from dateutil.relativedelta import relativedelta


def preprocess_time_monthly(x):
    ''' Preprocesses time variables from GFDL format to SIPN2 format.

    Convert time to initialization and forecast lead time (to fit into orthogonal matrices)
    Input Dims: lat x lon x Time
    Output Dims: lat x lon x init_time x fore_time

    Where we represent fore_time as monthly increments

    '''

    Nmons = x.average_T1.size
    m_i = np.arange(0,Nmons)
    m_dt = ['month' for x in m_i] # list of 'month'

    # Set record dimension of 'time' to the beginning of averaging period 'average_T1'
    x['time'] = x.average_T1

    # Grab forecast times
    xtimes = xr.decode_cf(x).time.values;

    # Get initialization time
    x.coords['init_time'] = xtimes[0] # get first one
    x.coords['init_time'].attrs['comments'] = 'Initilzation time of forecast'

    # Get forecast time (as timedeltas from init_time)
    x.rename({'time':'fore_time'}, inplace=True);
    x.coords['fore_time'] = xr.DataArray(m_i, dims='fore_time')

    # Set time offset for index in fore_time
    x.coords['fore_offset'] = xr.DataArray(m_dt, dims='fore_time', coords={'fore_time':x.fore_time})

    return x

def preprocess_time_monthly_Cansips(x):
    ''' Preprocesses time variables from Cansips format to SIPN2 format.

    Convert time to initialization and forecast lead time (to fit into orthogonal matrices)
    Input Dims: lat x lon x Time
    Output Dims: lat x lon x init_time x fore_time

    Where we represent fore_time as monthly increments

    '''

    Nmons = x.leadtime.size
    m_i = np.arange(0,Nmons)
    m_dt = ['month' for x in m_i] # list of 'month'

    # Set init_time
    x.coords['init_time'] = x.reftime.isel(time=0)
    x.coords['init_time'].attrs['comments'] = 'Initilzation time of forecast'
    x = x.drop(['reftime','leadtime'])

    # Get forecast time (as timedeltas from init_time)
    x.rename({'time':'fore_time'}, inplace=True);
    x.coords['fore_time'] = xr.DataArray(m_i, dims='fore_time')

    # Set time offset for index in fore_time
    x.coords['fore_offset'] = xr.DataArray(m_dt, dims='fore_time', coords={'fore_time':x.fore_time})
    
    return x

def preprocess_time(x):
    ''' Convert time to initialization and foreast lead time (to fit into orthoganal matrices)
    Input Dims: lat x lon x Time
    Output Dims: lat x lon x init_time x fore_time'''

    # Set record dimension of 'time' to the beinging of averaging period 'average_T1'
    x['time'] = x.average_T1

    # Grab forecast times
    xtimes = xr.decode_cf(x).time.values;

    # Get initialization time
    x.coords['init_time'] = xtimes[0] # get first one
    x.coords['init_time'].attrs['comments'] = 'Initilzation time of forecast'

    # Get forecast time (as timedeltas from init_time)
    x.rename({'time':'fore_time'}, inplace=True);
    x.coords['fore_time'] = xtimes - xtimes[0]

    return x

def get_valid_time(ds):
    ''' Given a data set with init_time and fore_time coords, calcuate the valid_time coord.'''
    
    if 'fore_offset' in ds.coords:
        # Then fore_time is just an index for fore_offset (i.e. monthly data)
        # TODO remove hard corded months (get from fore_offset)
        fore_time_offset = np.array([relativedelta(months=+x) for x in ds.fore_time.values])
        # Switch types around so we can add datetime64[ns] with an object of relativedelta, then convert back
        #valid_time = xr.DataArray(np.array([ds.init_time.values.astype('M8[D]').astype('O')]), dims='init_time')  +  xr.DataArray(fore_time_offset, dims='fore_time')
        valid_time = xr.DataArray(ds.init_time.values.astype('M8[D]').astype('O'), dims='init_time') +  xr.DataArray(fore_time_offset, dims='fore_time')
        ds.coords['valid_time'] = valid_time.astype('datetime64[ns]')
    else:
        ds.coords['valid_time'] = ds.init_time + ds.fore_time
        
    return ds


def rename_coords(ds):
    c_cords = list(ds.coords.dims.keys())
    c_dict = {'.*lat':'lat', '.*lon':'lon', '.*forecast_time':'fore_time',
              '.*initial_time':'init_time', '.*ensemble':'ensemble'}

    new_dict = {}
    for key, value in c_dict.items():
        r = re.compile(key)
        newlist = list(filter(r.match, c_cords))
        if len(newlist)>0:
            new_dict[newlist[0]] = value
    ds = ds.rename(new_dict)
    return ds


def open_1_member_monthly(cfiles, e):
    ds = xr.open_mfdataset(cfiles, concat_dim='init_time', decode_times=False,
                           preprocess=lambda x: preprocess_time_monthly(x),
                           autoclose=True)

    # Sort init_time (if more than one)
    if ds.init_time.size>1:
        ds = ds.reindex(init_time=sorted(ds.init_time.values))

    # Add ensemble coord
    ds.coords['ensemble'] = e
    return ds


def open_1_member(cfiles, e):
    ds = xr.open_mfdataset(cfiles, concat_dim='init_time', decode_times=False,
                           preprocess=lambda x: preprocess_time(x),
                           autoclose=True)

    # Sort init_time (if more than one)
    if ds.init_time.size>1:
        ds = ds.reindex(init_time=sorted(ds.init_time.values))
        
    # Some of daily gfdl flor forecast go for 10 years instead of 1, only get 1 year here
    ds = ds.isel(fore_time=slice(0,365))

    # Add ensemble coord
    ds.coords['ensemble'] = e
    return ds


def readbinfile(f, nx, ny):
    with open(f, 'rb') as fid:
        data_array = np.fromfile(fid, np.int32)*1e-5
    return data_array.reshape((nx,ny))


def get_stero_N_grid(grid_dir):
    # Get info about target grid
    flat = os.path.join(grid_dir,'psn25lats_v3.dat')
    flon = os.path.join(grid_dir,'psn25lons_v3.dat')
    NY=304
    NX=448
    lat = readbinfile(flat, NX, NY).T
    lon = readbinfile(flon, NX, NY).T
    # Add cell corner lat/lon
    return xr.Dataset({'lat': (['x', 'y'],  lat), 'lon': (['x', 'y'], lon)})

def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min,ix_min


def cell_bounds_to_corners(gridinfo=None, varname=None):
    # Add cell bound coords (lat_b and lon_b)
    n_j = gridinfo.grid_dims.values[1]
    n_i = gridinfo.grid_dims.values[0]
    nj_b = np.arange(0, n_j + 1) # indices of corner of cells
    ni_b = np.arange(0, n_i + 1)

    # Grab all corners as arrays
    dim_out = tuple(np.flip(gridinfo.grid_dims.T.values,0))
    ul = gridinfo[varname].isel(grid_corners=0).values.reshape(dim_out)
    ll = gridinfo[varname].isel(grid_corners=1).values.reshape(dim_out)
    lr = gridinfo[varname].isel(grid_corners=2).values.reshape(dim_out)
    ur = gridinfo[varname].isel(grid_corners=3).values.reshape(dim_out)

    # Merge together
    m1 = np.concatenate((ul, ur[:,0][:, None]), axis=1) # add on ur at right
    m2 = np.append(ll[-1,:], lr[-1,0])
    m3 = np.concatenate((m1, m2[:, None].T), axis=0) # add ll and lr to bottom
    ds_out = xr.DataArray(m3, dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b, 'ni_b':ni_b})
    ds_out = xr.ufuncs.rad2deg( ds_out ) # rad to deg
    return ds_out


def cell_bounds_to_corners_GFDL(gridinfo=None, varname=None):
    # Add cell bound coords (lat_b and lon_b)
    n_j = gridinfo.grid_dims.values[1]
    n_i = gridinfo.grid_dims.values[0]
    nj_b = np.arange(0, n_j + 1) # indices of corner of cells
    ni_b = np.arange(0, n_i + 1)

    # Grab all corners as arrays
    dim_out = tuple(np.flip(gridinfo.grid_dims.T.values,0))
    ll = gridinfo[varname].isel(grid_corners=0).values.reshape(dim_out)
    lr = gridinfo[varname].isel(grid_corners=1).values.reshape(dim_out)
    ur = gridinfo[varname].isel(grid_corners=2).values.reshape(dim_out)
    ul = gridinfo[varname].isel(grid_corners=3).values.reshape(dim_out)

    # Merge together
    m1 = np.concatenate((ul, ur[:,-1][:, None]), axis=1) # add on ur at right
    m2 = np.append(ll[0,:], lr[0,-1])
    m3 = np.concatenate((m2[:, None].T, m1), axis=0) # add ll and lr to bottom
    ds_out = xr.DataArray(m3, dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b, 'ni_b':ni_b})
    ds_out = xr.ufuncs.rad2deg( ds_out ) # rad to deg
    return ds_out


def add_matrix_NaNs(regridder):
    X = regridder.A
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.A = scipy.sparse.coo_matrix(M)
    return regridder


def load_grid_info(grid_file=None, model=None):
    grid = xr.open_dataset(grid_file)
    n_lat = np.rad2deg(grid.grid_center_lat.values.reshape(tuple(np.flip(grid.grid_dims.T.values,0)))) # Reshape
    n_lon = np.rad2deg(grid.grid_center_lon.values.reshape(tuple(np.flip(grid.grid_dims.T.values,0)))) # Reshape
    grid_imask = grid.grid_imask.values.reshape(tuple(np.flip(grid.grid_dims.T.values,0))) # Reshape

    nj = xr.DataArray(np.arange(0,n_lat.shape[0],1), dims=('nj')) # Make indices
    ni = xr.DataArray(np.arange(0,n_lat.shape[1],1), dims=('ni'))
    lat = xr.DataArray(n_lat, dims=('nj','ni'), coords={'nj':nj, 'ni':ni})
    lon = xr.DataArray(n_lon, dims=('nj','ni'), coords={'nj':nj, 'ni':ni})
    imask = xr.DataArray(grid_imask, dims=('nj','ni'), coords={'nj':nj, 'ni':ni}).astype('bool') # int to bool

    if model=='NSIDC':
        lat_b = cell_bounds_to_corners(gridinfo=grid, varname='grid_corner_lat')
        lon_b = cell_bounds_to_corners(gridinfo=grid, varname='grid_corner_lon')
    elif model=='GFDL' or model=='piomas':
        lat_b = cell_bounds_to_corners_GFDL(gridinfo=grid, varname='grid_corner_lat')
        lon_b = cell_bounds_to_corners_GFDL(gridinfo=grid, varname='grid_corner_lon')
    else:
        raise ValueError('model not found.')

    # Combine
    return xr.Dataset({'lat':lat, 'lon':lon, 'lat_b':lat_b, 'lon_b':lon_b, 'imask':imask})


def split_GFDL(ds_in, varnames=None):

    # GFDL grid split parameters
    j_s = 175
    i_s = 180

    # Subset "top"
    a = ds_in[varnames].isel(nj=slice(j_s,None), ni=slice(None,i_s))
    b = ds_in[varnames].isel(nj=slice(j_s,None), ni=slice(i_s,None))
    b['nj'] = np.flip(a.nj, axis=0) + a.nj.max() - a.nj.min() + 1 # reverse in nj dim (reindexed below)
    b['ni'] = np.flip(a.ni, axis=0) # flip in ni dim to align with a
    ds_top = xr.concat([a, b.T], dim='nj')
    if not hasattr(ds_top, 'data_vars'):  # convert to dataset if not already
        ds_top = ds_top.to_dataset()
    # concat over nj dim
    ds_top = ds_top.reindex({'nj':np.arange(ds_top.nj.min(), ds_top.nj.max()+1, 1)}) # reindex on nj to "flip" b in nj dim

    c = ds_in['lat_b'].isel(nj_b=slice(j_s,ds_in.nj.size), ni_b=slice(None,i_s+1))
    d = ds_in['lat_b'].isel(nj_b=slice(j_s,ds_in.nj_b.size), ni_b=slice(i_s,None))
    d['nj_b'] = np.flip(np.arange(c.nj_b.max()+1, c.nj_b.max()+2+c.nj_b.size), axis=0)
    d['ni_b'] = np.flip(c.ni_b, axis=0)
    ds_top.coords['lat_b'] = xr.concat([c, d], dim='nj_b')
    ds_top = ds_top.reindex({'nj_b':np.arange(ds_top.nj_b.min(), ds_top.nj_b.max()+1, 1)}) # reindex on nj to "flip" b in nj dim

    # Subset "bottom"

    # add overlap
    j_s = j_s + 10 # Here we add 3 poleward cells to the "bottom" sub-grid, to allow overlap with "top" sub-grid.

    ds_bottom = ds_in.isel(nj=slice(None,j_s)).drop(['lat_b','lon_b','nj_b','ni_b'])
    ds_bottom.coords['lat_b'] = ds_in['lat_b'].isel(nj_b=slice(None,j_s+1))

    return (ds_top, ds_bottom)


def regrid_gfdl_split_domain(ds_all, da_top, da_bottom, regridder_top, regridder_bottom):
    # Regrid
    da_out_top = regridder_top(da_top)
    da_out_bottom = regridder_bottom(da_bottom)

    # Mask by latitude
    lat_split = ds_all.lat.isel(nj=175).min() # Get the latitude where model domain was split on
    lat_split_2 = ds_all.lat.isel(nj=175+5).max() #
    da_out_top = da_out_top.where( (da_out_top.lat>=lat_split).values )
    da_out_bottom = da_out_bottom.where( (da_out_bottom.lat<lat_split_2).values )

    # Add dropped coords
    da_out_top['fore_time'] = ds_all.fore_time
    da_out_bottom['fore_time'] = ds_all.fore_time

    # Merge "top" and "bottom"
    da_all_out = da_out_top.combine_first(da_out_bottom)

    return da_all_out


def split_by_lat(ds, latVal=65.0, want=None):
    if want=='above':
        ds_out = ds.drop(['lat_b','lon_b','nj_b','ni_b']).where(ds.lat>latVal, drop=True)
        ds_out.coords['lat_b'] = ds.lat_b.sel(nj_b=np.append(ds_out.nj.values, ds_out.nj.values[-1]+1))
    elif want=='below':
        ds_out = ds.drop(['lat_b','lon_b','nj_b','ni_b']).where(ds.lat<=latVal, drop=True)
        ds_out.coords['lat_b'] = ds.lat_b.sel(nj_b=np.append(ds_out.nj.values, ds_out.nj.values[-1]+1))
    else:
         raise ValueError('Value for want not found. Use above or below.')
    return ds_out


def read_piomas_scalar_monthly(f):
    xDim = 120
    yDim = 360
    yyyy = c_files[0].split('.')[1].split('H')[1] # Get year to split out dates
    with open(f, 'rb') as fid:
        arr = np.fromfile(fid, np.float32).reshape(-1, xDim, yDim)
    # Build dates
    time = np.arange(yyyy+'-01', str(np.int(yyyy)+1)+'-01', dtype='datetime64[M]')
    return xr.DataArray(arr, dims =('time','nj', 'ni'), coords={'time':time})


def read_piomas_scalar_daily(f, varname=None):
    xDim = 120
    yDim = 360
    yyyy = f.split('.')[1].split('H')[1] # Get year to split out dates
    with open(f, 'rb') as fid:
        arr = np.fromfile(fid, np.float32).reshape(-1, xDim, yDim)
    # Build dates
#     time = np.arange(yyyy+'-01', str(np.int(yyyy)+1)+'-01', dtype='datetime64[D]').astype('datetime64[ns]')
    time = pd.date_range(yyyy+'-01-01', yyyy+'-12-31')
    da = xr.DataArray(arr, name=varname, dims =('time','nj', 'ni'), coords={'time':time[0:arr.shape[0]]})
    return da.to_dataset() # push to data set so we can add more coords later


def expand_to_sipn_dims(ds):
    # Force output datasets have ensemble, init_time, and fore_time as dimensions (otherwise add empty ones)
    required_dims = ['ensemble', 'init_time', 'fore_time']
    for d in required_dims:
        if d not in ds.dims:
            ds = ds.expand_dims(d)
    return ds


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


def load_1_NSIDC(filein=None, product=None):
    # Define coords
    # Indices values
    x = np.arange(0,304,1)
    y = np.arange(0,448,1)

    ds_sic = read_NSIDC_binary(filein, x, y, product)

    return ds_sic


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