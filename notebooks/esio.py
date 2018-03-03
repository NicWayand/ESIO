import numpy as np
import xarray as xr
from scipy import stats

''' Functions to process sea ice renalysis, reforecasts, and forecasts.'''

# TODO: make model independent
# TODO: remove hard coded paths

############################################################################
# Regridding/Inporting functions
############################################################################

def preprocess_time(x):
    # TODO: if monthly, use begining of period time step
    ''' Convert time to initialization and foreast lead time (to fit into orthoganal matrices)
    Input Dims: lat x lon x Time
    Output Dims: lat x lon x init_time x fore_time_i'''
    
    # Set record dimension of 'time' to the beinging of averaging period 'average_T1'
    x['time'] = x.average_T1
    
    # Grab forecast times
    xtimes = xr.decode_cf(x).time.values;
    
    # Get initialization time
    x.coords['init_time'] = xtimes[0] # get first one
    x.coords['init_time'].attrs['comments'] = 'Initilzation time of forecast'

    # Get forecast time in days from initilization
    x.rename({'time':'fore_time_i'}, inplace=True);
    x.coords['fore_time_i'] = np.arange(0,12,1)
    x.fore_time_i.attrs['units'] = 'Index of forecast dates'
    
    # Store actual forecast dates
    x.coords['fore_time'] = xr.DataArray(xtimes, dims=('fore_time_i'), coords={'fore_time_i':x.fore_time_i})
    x.fore_time.attrs['comments'] = 'Date of forecast'
    
    return x

# Open a single ensemble member
def open_1_member(cfiles, e):
    ds = xr.open_mfdataset(cfiles, concat_dim='init_time', decode_times=False, 
                           preprocess=lambda x: preprocess_time(x),
                           autoclose=True)
    # Sort init_time
    ds = ds.reindex(init_time=sorted(ds.init_time.values))
    # Add ensemble coord
    ds.coords['ensemble'] = e
    return ds

def readBinFile(f, nx, ny):
    with open(f, 'rb') as fid:
        data_array = np.fromfile(f, np.int32)*1e-5
    return data_array.reshape((nx,ny))

def get_stero_N_grid():
    # Get info about target grid
    flat = r'/home/disk/sipn/nicway/data/grids/psn25lats_v3.dat'
    flon = r'/home/disk/sipn/nicway/data/grids/psn25lons_v3.dat'
    NY=304; 
    NX=448;
    lat = readBinFile(flat, NX, NY).T
    lon = readBinFile(flon, NX, NY).T
    # Add cell corner lat/lon
    return xr.Dataset({'lat': (['x', 'y'],  lat), 'lon': (['x', 'y'], lon)})

# Define naive_fast that searches for the nearest WRF grid cell center
def naive_fast(latvar,lonvar,lat0,lon0):
    # Read latitude and longitude from file into numpy arrays
    latvals = latvar[:]
    lonvals = lonvar[:]
    dist_sq = (latvals-lat0)**2 + (lonvals-lon0)**2
    minindex_flattened = dist_sq.argmin()  # 1D index of min element
    iy_min,ix_min = np.unravel_index(minindex_flattened, latvals.shape)
    return iy_min,ix_min

def mask_common_extent(ds_obs, ds_mod, max_obs_missing=0.1):
    # Mask out areas where either observations or model are missing
    mask_obs = ds_obs.isnull().sum(dim='time') / ds_obs.time.size # Get fraction of missing for each pixel
    mask_mod = ds_mod.sic.isel(fore_time_i=0).isel(init_time=0).isel(ensemble=0).notnull() # Grab one model to get extent
    mask_comb = (mask_obs <= max_obs_missing) & (mask_mod) # Allow 10% missing in observations
    mask_comb = mask_comb.drop(['ensemble','fore_time_i','init_time','fore_time']) # Drop unneeded variables
    # Apply and return
    ds_obs_out = ds_obs.where(mask_comb)
    ds_mod_out = ds_mod.where(mask_comb)
    ds_mod_out.coords['fore_time'] = ds_mod.fore_time # add back coords that were dropped
    return (ds_obs_out, ds_mod_out)

# def cell_bounds_to_corners(gridinfo=None, varname=None):
#     # Add cell bound coords (lat_b and lon_b)
#     n_j = gridinfo.grid_dims.values[1]
#     n_i = gridinfo.grid_dims.values[0]
#     nj_b = np.arange(0, n_j + 1) # indices of corner of cells
#     ni_b = np.arange(0, n_i + 1)

#     # Grab all corners and combine
#     dim_out = tuple(np.flip(gridinfo.grid_dims.T.values,0))
#     ul = xr.DataArray(gridinfo[varname].isel(grid_corners=0).values.reshape(dim_out),
#                  dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b[1:], 'ni_b':ni_b[0:-1]}) 
#     ur = xr.DataArray(gridinfo[varname].isel(grid_corners=1).values.reshape(dim_out),
#                  dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b[1:], 'ni_b':ni_b[1:]}) 
#     lr = xr.DataArray(gridinfo[varname].isel(grid_corners=2).values.reshape(dim_out),
#                  dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b[0:-1], 'ni_b':ni_b[1:]}) 
#     ll = xr.DataArray(gridinfo[varname].isel(grid_corners=3).values.reshape(dim_out),
#                  dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b[0:-1], 'ni_b':ni_b[0:-1]}) 

#     return xr.ufuncs.rad2deg(  ul.combine_first(ur).combine_first(lr).combine_first(ll)  )

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
    m2 = np.append(ll[-1,:], lr[-1,-1])
    m3 = np.concatenate((m1, m2[:, None].T), axis=0) # add ll and lr to bottom
    ds_out = xr.DataArray(m3, dims=('nj_b', 'ni_b'), coords={'nj_b':nj_b, 'ni_b':ni_b}) 
    ds_out = xr.ufuncs.rad2deg( ds_out ) # rad to deg
    return ds_out
#     return (np.rad2deg(ul), np.rad2deg(ll), np.rad2deg(lr), np.rad2deg(ur))



# Load in correct GFDL grid info and format
def load_grid_info(grid_file=None, model=None):
    grid = xr.open_dataset(grid_file)
    n_lat = np.rad2deg(grid.grid_center_lat.values.reshape(tuple(np.flip(grid.grid_dims.T.values,0)))) # Reshape
    n_lon = np.rad2deg(grid.grid_center_lon.values.reshape(tuple(np.flip(grid.grid_dims.T.values,0)))) # Reshape
    
    nj = xr.DataArray(np.arange(0,n_lat.shape[0],1), dims=('nj')) # Make indices
    ni = xr.DataArray(np.arange(0,n_lat.shape[1],1), dims=('ni'))
    lat = xr.DataArray(n_lat, dims=('nj','ni'), coords={'nj':nj, 'ni':ni})
    lon = xr.DataArray(n_lon, dims=('nj','ni'), coords={'nj':nj, 'ni':ni})
    
    if model=='NSIDC':
        lat_b = cell_bounds_to_corners(gridinfo=grid, varname='grid_corner_lat')
        lon_b = cell_bounds_to_corners(gridinfo=grid, varname='grid_corner_lon')
    elif model=='GFDL':
        lat_b = cell_bounds_to_corners_GFDL(gridinfo=grid, varname='grid_corner_lat')
        lon_b = cell_bounds_to_corners_GFDL(gridinfo=grid, varname='grid_corner_lon')
    else:
        raise ValueError('model not found.')
    
    # Combine
    return xr.Dataset({'lat':lat, 'lon':lon, 'lat_b':lat_b, 'lon_b':lon_b})

# Split tripolar grid by 65 N
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
    
    
############################################################################
# Plotting functions
############################################################################

def plot_model_ensm(ds=None, axin=None, labelin=None, color='grey', marker=None):
    # TODO: too slow, need to speed up
    labeled = False
    for e in ds.ensemble:
        for it in ds.init_time:
            if labeled:
                labelin = '_nolegend_'
            axin.plot(ds.fore_time.sel(init_time=it), 
                      ds.sel(ensemble=e).sel(init_time=it), label=labelin, color=color, marker=marker)
            labeled = True
            
############################################################################
# Evaluation functions
############################################################################

def format_obs_like_model(ds_mod, ds_obs):
    ''' Reformats observational dataset to be structured like a model forecast dataset 
    Format obs like model (i.e. ensemble x init_time x forecast_time) '''
    
    ds_obs_X = (ds_mod.copy() * np.nan).load() # Have to call load here to assign ie below
    for (i, e) in enumerate(ds_obs_X.ensemble):
        for (j, it) in enumerate(ds_obs_X.init_time):
            ds_obs_X[i, j, :] = ds_obs.sel(time=ds_mod.fore_time.sel(init_time=it))
    return ds_obs_X

def trim_common_times(ds_obs, ds_mod):
    ''' Trim an observed and modeled dataset to common start and end dates (does not
    insure internal times are the same) ''' 
    
    # Get earliest and latest times
    T_start = np.max([ds_obs.time.values[0], ds_mod.fore_time.min().values])
    T_end = np.min([ds_obs.time.values[-1], ds_mod.fore_time.max().values])
    print(T_start, T_end)
    # Subset
    ds_obs_out = ds_obs.where((ds_obs.time >= T_start) & (ds_obs.time <= T_end), drop=True)
    ds_mod_out = ds_mod.where((ds_mod.fore_time >= T_start) & (ds_mod.fore_time <= T_end) &
                              (ds_mod.init_time >= T_start) & (ds_mod.init_time <= T_end), drop=True)
    return ds_obs_out, ds_mod_out

def clim_mu_sigma(ds_obs, method='MK'):
    ''' Calculate the climatological mean and standard deviation following:
    MK - Maximum knowledge (use all observations times)
    OP - Operatioanl approach (use only observations from past (before initialization time))
    '''
    
    if method=='MK':
        y = ds_obs.values
        x = np.arange(0,ds_obs.time.size,1)
    else:
        raise ValueError('Method not found.')
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    predict_y = intercept + slope * x
    pred_error = y - predict_y
    sigma = np.std(pred_error)
    mu = np.mean(y)
    
    return (mu, sigma)

def NRMSE(ds_mod, ds_obs, sigma):
    ''' Normalized RMSE (NRMSE)'''
    
    #Assert obs has been formated like model
    assert ds_mod.dims==ds_obs.dims
    
    #TODO: Uncertain here to take mean over init_time or fore_time ????
    a = xr.ufuncs.sqrt( ((ds_mod - ds_obs)**2).mean(dim='ensemble').mean(dim='fore_time_i') )
    b = xr.ufuncs.sqrt( 2*(sigma**2) ) # add time variance for OP option
    
    NRMSE =  1 - (a / b)
    return NRMSE



# def PPP(ds_mod, ds_obs):
#     ''' potential prognostic predictability (PPP) '''
    
#     a = 
#     b = 
    
#     PPP = 1 - a / b
#     return PPP









