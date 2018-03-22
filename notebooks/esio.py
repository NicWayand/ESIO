import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import itertools

''' Functions to process sea ice renalysis, reforecasts, and forecasts.'''

# TODO: make model independent
# TODO: remove hard coded paths

############################################################################
# Regridding/Inporting functions
############################################################################

def preprocess_time_OLD(x):
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

# Open a single ensemble member
def open_1_member(cfiles, e):
    ds = xr.open_mfdataset(cfiles, concat_dim='init_time', decode_times=False, 
                           preprocess=lambda x: preprocess_time(x),
                           autoclose=True)
    # Sort init_time (if more than one)
    if ds.init_time.size>1:
        ds = ds.reindex(init_time=sorted(ds.init_time.values))
    # Add ensemble coord
    ds.coords['ensemble'] = e
    return ds

def readBinFile(f, nx, ny):
    with open(f, 'rb') as fid:
        data_array = np.fromfile(fid, np.int32)*1e-5
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

# Add Nans to matrices, which makes any output cell with a weight from a NaN input cell = NaN
def add_matrix_NaNs(regridder):
    X = regridder.A
    M = scipy.sparse.csr_matrix(X)
    num_nonzeros = np.diff(M.indptr)
    M[num_nonzeros == 0, 0] = np.NaN
    regridder.A = scipy.sparse.coo_matrix(M)
    return regridder

# Load in correct GFDL grid info and format
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

# Split GFDL grid into "top":bi-pole north and "bottom":rest
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


# Function that regrides top and bottom of GFLD domain
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


def agg_by_domain(da_grid=None, ds_region=None):
    # TODO: add check for equal dims
    ds_list = []
    for cd in ds_region.nregions:
        # Get name
        region_name = ds_region.region_names.sel(nregions=cd).item(0).decode("utf-8") 
        # Check we want it (exclude some regions)
        if not region_name in ['Ice-free Oceans  ', 'null             ',
                               'land outline     ', 'land             ' ]:
            # Make mask
            cmask = ds_region.mask==cd 
            # Multiple by cell area to get area of sea ice
            da_avg = da_grid.where(cmask) * ds_region.area.where(cmask)
            # Sum up over current domain and convert to millions of km^2
            da_avg = da_avg.sum(dim='x').sum(dim='y') / (10**6)
            # Add domain name
            da_avg['nregions'] = cd
            da_avg['region_names'] = region_name
            ds_list.append(da_avg)
    return xr.concat(ds_list, dim='nregions')

def get_season_start_date(ctime):
    X = ctime.astype(object)
    if X.month<8:
        yyyy = X.year-1
    else:
        yyyy = X.year
    return np.datetime64(str(yyyy)+'-09-01')


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
    
############################################################################
# Plotting functions
############################################################################

def plot_model_ensm(ds=None, axin=None, labelin=None, color='grey', marker=None):
    # TODO: too slow, need to speed up
    labeled = False
    for e in ds.ensemble:
        if labeled:
            labelin = '_nolegend_'
        # Check if it has multiple init_times
        if 'init_time' in ds.dims:
            for it in ds.init_time:
                
                axin.plot(ds.fore_time.sel(init_time=it), 
                          ds.sel(ensemble=e).sel(init_time=it), label=labelin, color=color, marker=marker)
                
        else: # just one init_time
            axin.plot(ds.fore_time, 
                          ds.sel(ensemble=e), label=labelin, color=color, marker=marker)
        labeled = True
        
def plot_reforecast(ds=None, axin=None, labelin=None, color='cycle_init_time', 
                    marker=None, init_dot=True, init_dot_label=True, linestyle='-'):
    labeled = False
    if init_dot:
        init_label = 'Initialization'
    else:
        init_label = '_nolegend_'
        
    if color=='cycle_init_time':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.init_time.size))
    elif color=='cycle_ensemble':
        cmap_c = itertools.cycle(sns.color_palette("GnBu_d", ds.ensemble.size))
    else:
        ccolor = color # Plot all lines with one color
        
    for e in ds.ensemble:    
            cds = ds.sel(ensemble=e)
            
            if color=='cycle_ensemble':
                ccolor = next(cmap_c)
            
            for it in ds.init_time:
                
                if labeled:
                    labelin = '_nolegend_'
                    init_label = '_nolegend_'

                if color=='cycle_init':
                    ccolor = next(cmap_c)        
                  
                # Grab current data
#                 x = cds.fore_time + cds.init_time.sel(init_time=it)
#                 y = cds.sel(init_time=it)
                
                # (optional) plot dot at initialization time
                if init_dot:
                    axin.plot((cds.fore_time + cds.init_time.sel(init_time=it))[0], cds.sel(init_time=it)[0], marker='*', linestyle='None', color='red', label=init_label)
                
                # Plot line
                axin.plot(cds.fore_time + cds.init_time.sel(init_time=it), cds.sel(init_time=it), label=labelin, color=ccolor, marker=marker, linestyle=linestyle)
                
                labeled = True
            
def polar_axis():
    '''cartopy geoaxes centered at north pole'''
    f = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax.coastlines(linewidth=0.75, color='black', resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
    ax.set_extent([0, 359.9, 57, 90], crs=ccrs.PlateCarree())
    return (f, ax)
            
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




# Split GFDL grid into "top":bi-pole north and "bottom":rest
# def split_GFDL(ds_in):

#     # GFDL grid split parameters
#     j_s = 175
#     i_s = 180

#     # Top
#     data_correct = np.r_[ds_in['sic'][j_s:, :i_s], 
#                          np.flipud(np.fliplr(ds_in['sic'][j_s:, i_s:]))]

#     lon_correct = np.r_[ds_in['lon'][j_s:, :i_s], 
#                         np.flipud(np.fliplr(ds_in['lon'][j_s:, i_s:]))]
#     lat_correct = np.r_[ds_in['lat'][j_s:, :i_s], 
#                         np.flipud(np.fliplr(ds_in['lat'][j_s:, i_s:]))]

#     lon_b_correct = xr.DataArray(np.r_[ds_in['lon_b'][j_s:ds_in.nj.size, :(i_s + 1)], 
#                         np.flipud(np.fliplr(ds_in['lon_b'][j_s:ds_in.nj_b.size, i_s:]))], dims=('nj_b','ni_b'))

#     lat_b_correct = xr.DataArray(np.r_[ds_in['lat_b'][j_s:ds_in.nj.size, :(i_s + 1)], 
#                         np.flipud(np.fliplr(ds_in['lat_b'][j_s:ds_in.nj_b.size, i_s:]))], dims=('nj_b','ni_b'))


#     ds_top_C = xr.DataArray(data_correct, dims=('nj','ni'), coords={'lat':( ('nj','ni'), lat_correct), 
#                                                                   'lon':( ('nj','ni'), lon_correct)})

#     ds_top = xr.Dataset(data_vars={'sic':ds_top_C}, coords={'lat_b':lat_b_correct, 'lon_b':lon_b_correct})

#     # Bottom

#     # add overlap
#     j_s = j_s + 3 # Here we add 3 poleward cells to the "bottom" sub-grid, to allow overlap with "top" sub-grid.

#     data_correct_bottom = ds_in['sic'][:j_s, :]

#     lon_bottom = ds_in['lon'][:j_s, :]
#     lat_bottom = ds_in['lat'][:j_s, :]

#     lon_b_bottom = ds_in['lon_b'][:j_s+1, :]
#     lat_b_bottom = ds_in['lat_b'][:j_s+1, :]

#     ds_bottom_C = xr.DataArray(data_correct_bottom, dims=('nj','ni'), coords={'lat':( ('nj','ni'), lat_bottom), 
#                                                                   'lon':( ('nj','ni'), lon_bottom)})

#     ds_bottom = xr.Dataset(data_vars={'sic':ds_bottom_C}, coords={'lat_b':lat_b_bottom, 'lon_b':lon_b_bottom})

#     return (ds_top, ds_bottom)





