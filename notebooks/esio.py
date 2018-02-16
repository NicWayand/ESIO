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
    
############################################################################
# Plotting functions
############################################################################


def plot_model_ensm(ds=None, axin=None, labelin=None, color='grey', marker=None):
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
    # Format obs like model (i.e. ensemble x init_time x forecast_time)
    ds_obs_X = ds_mod.copy() * np.nan
    for (i, e) in enumerate(ds_obs_X.ensemble):
        for (j, it) in enumerate(ds_obs_X.init_time):
            ds_obs_X[i, j, :] = ds_obs.sel(time=ds_mod.fore_time.sel(init_time=it))
    return ds_obs_X

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
    
    #TODO: uncerintayt here to take mean over init_time or fore_time ????
    a = xr.ufuncs.sqrt( ((ds_mod - ds_obs)**2).mean(dim='ensemble').mean(dim='fore_time_i') )
    print(a)
    b = xr.ufuncs.sqrt( 2*(sigma**2) ) # add time variance for OP option
    
    NRMSE =  1 - a / b
    return NRMSE



# def PPP(ds_mod, ds_obs):
#     ''' potential prognostic predictability (PPP) '''
    
#     a = 
#     b = 
    
#     PPP = 1 - a / b
#     return PPP









