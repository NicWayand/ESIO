import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import scipy
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import seaborn as sns
import itertools
import os
import re


''' Functions to process sea ice renalysis, reforecasts, and forecasts.'''

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

# Rename S2S and C3S coord names to sipn standard
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

def get_stero_N_grid(grid_dir):
    # Get info about target grid
    flat = os.path.join(grid_dir,'psn25lats_v3.dat')
    flon = os.path.join(grid_dir,'psn25lons_v3.dat')
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
    mask_mod = ds_mod.isel(fore_time=0).isel(init_time=0).isel(ensemble=0).notnull() # Grab one model to get extent
    mask_comb = (mask_obs <= max_obs_missing) & (mask_mod) # Allow 10% missing in observations
    mask_comb = mask_comb.squeeze() #(['ensemble','fore_time','init_time','fore_time']) # Drop unneeded variables
    
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

def calc_extent(da, region, extent_thress=0.15, fill_pole_hole=False):
    ''' Returns extent in millions of km^2 within all ocean regions (NO LAKES!)'''
    
    # TODO: Need to assert we pass in a DataArray of sic
    extent = (( da.where(region.mask.isin(region.ocean_regions)) >= extent_thress ).astype('int') * region.area).sum(dim='x').sum(dim='y')/(10**6)
    
    # Mask out zero extents (occurs if ensemble changes size)
    extent = extent.where(extent>0)
    
    # Add in pole hole (optional)
    if fill_pole_hole:
        extent = extent + (da.hole_mask.astype('int') * region.area).sum(dim='x').sum(dim='y')/(10**6)
    return extent


def agg_by_domain(da_grid=None, ds_region=None, extent_thress=0.15):
    # TODO: add check for equal dims
    ds_list = []
    for cd in ds_region.nregions.values:
        # Get name
        region_name = ds_region.region_names.sel(nregions=cd).values
        # Check we want it (exclude some regions)
        if not region_name in ['Ice-free Oceans', 'null','land outline', 'land' ]:
            # Make mask
            cmask = ds_region.mask==cd 
            # Multiple by cell area to get area of sea ice
            da_avg = (da_grid.where(cmask==1) >= extent_thress).astype('int') * ds_region.area.where(cmask==1)
            # Sum up over current domain and convert to millions of km^2
            #print((cmask * ds_region.area.where(cmask==1)).sum(dim='x').sum(dim='y') / (10**6))
            da_avg = da_avg.sum(dim='x').sum(dim='y') / (10**6)
            # TODO: Add option to add in pole hole if obs and central arctic
            #print(da_avg.values)
            
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

# Calc NSIDC median sea ice edge between 1981-2010
def get_median_ice_edge(ds, ystart='1981', yend='2012', sic_threshold=0.15):
    median_ice = ds.sel(time=slice(ystart, yend)) #.drop(['coast','land','missing'])
    # Calc "Extent" (1 or 0)
    median_ice['sic'] = (median_ice.sic >= sic_threshold).astype('int')
    DOY = [x.timetuple().tm_yday for x in pd.to_datetime(median_ice.time.values)]
    median_ice['time'] = DOY
    median_ice.reset_coords(['hole_mask'], inplace=True)
    median_ice.load()
    median_ice = median_ice.groupby('time').median(dim='time')
    median_ice_fill = median_ice.where(median_ice.hole_mask==0, other=1).sic # Fill in pole hole with 1 (so contours don't get made around it)
    return median_ice_fill

# Calc the Ice Free Day (frist) by Calender Year
# For observations only
def calc_IFD(da, sic_threshold=0.15):
    ifd = (da < sic_threshold).reduce(np.argmax, dim='time') # Find index of first ice free
    ifd = ifd.where(da.isel(time=0).notnull()) # Apply Orig mask
    return ifd
    
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
                    marker=None, init_dot=True, init_dot_label=True, linestyle='-', 
                    no_init_label=False, linewidth=1.5):
    labeled = False
    if init_dot:
        init_label = 'Initialization'
    else:
        init_label = '_nolegend_'
    if no_init_label:
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
                    
                x = (cds.fore_time + cds.init_time.sel(init_time=it)).values
                y = cds.sel(init_time=it).values
                
                # Check we have data (issue with some models that change number of ensembles, when xarray merges, missing data is filled in (i.e. ukmo)
                # So check y is greater than 0 before plotting)
                if np.sum(y)==0:
                    continue
                
                # (optional) plot dot at initialization time
                if init_dot:
                    axin.plot(x[0], y[0], marker='*', linestyle='None', color='red', label=init_label)
                
                # Plot line
                axin.plot(x, y, label=labelin, color=ccolor, marker=marker, linestyle=linestyle,
                             linewidth=linewidth)
                
                labeled = True
            cds = None
            
def polar_axis():
    '''cartopy geoaxes centered at north pole'''
    f = plt.figure(figsize=(6, 5))
    ax = plt.axes(projection=ccrs.NorthPolarStereo(central_longitude=-45))
    ax.coastlines(linewidth=0.75, color='black', resolution='50m')
    ax.gridlines(crs=ccrs.PlateCarree(), linestyle='-')
    #ax.set_extent([0, 359.9, 57, 90], crs=ccrs.PlateCarree())
    # Full NSIDC extent
    ax.set_extent([-3850000*0.9, 3725000*0.8, -5325000*0.7, 5850000*0.9], crs=ccrs.NorthPolarStereo(central_longitude=-45))
    return (f, ax)

def multi_polar_axis(ncols=4, nrows=4, 
                     Nplots=None, sizefcter=1,
                     extent=None):
    if not Nplots:
        Nplots = ncols*nrows
    # Create a grid of plots
    f, (axes) = plt.subplots(ncols=ncols, nrows=nrows, subplot_kw={'projection': ccrs.NorthPolarStereo(central_longitude=-45)})
    f.set_size_inches(ncols*1.5*sizefcter, nrows*2*sizefcter)
    axes = axes.reshape(-1)
    for (i, ax) in enumerate(axes):  
        axes[i].coastlines(linewidth=0.2, color='black', resolution='50m')
        axes[i].gridlines(crs=ccrs.PlateCarree(), linestyle='--', linewidth=0.20, color='grey')
        #ax.set_extent([0, 359.9, 57, 90], crs=ccrs.PlateCarree())
        # Full NSIDC extent
        if not extent:
            axes[i].set_extent([-3850000*0.9, 3725000*0.8, -5325000*0.7, 5850000*0.9], crs=ccrs.NorthPolarStereo(central_longitude=-45))
        else: # Set Regional extent
             axes[i].set_extent(extent, crs=ccrs.NorthPolarStereo(central_longitude=-45)) 
                
        if i>=Nplots-1:
            f.delaxes(axes[i])  
    return (f, axes)

            
############################################################################
# Evaluation functions
############################################################################

def nanSum(da=None, dim=None):
    return da.sum(dim=dim).where(da.notnull().sum(dim=dim) > 0 )

def format_obs_like_model(ds_mod, ds_obs):
    ''' Reformats observational dataset to be structured like a model forecast dataset 
    Format obs like model (i.e. ensemble x init_time x forecast_time) '''
    
    ds_obs_X = (ds_mod.copy() * np.nan).load() # Have to call load here to assign ie below
    for (i, e) in enumerate(ds_obs_X.ensemble):
        for (j, it) in enumerate(ds_obs_X.init_time):
            ds_obs_X[i, j, :] = ds_obs.sel(time = ( ds_mod.init_time.sel(init_time=it) + ds_mod.fore_time).values )
    
    return ds_obs_X

def trim_common_times(ds_obs=None, ds_mod=None, freq=None):
    ''' Trim an observed and modeled dataset to common start and end dates (does not
    insure internal times are the same) '''
    
    # Get earliest and latest times
    T_start = np.max([ds_obs.time.values[0], ds_mod.init_time.min().values])
    T_end = np.min([ds_obs.time.values[-1], (ds_mod.init_time.max() + ds_mod.fore_time.max()).values])


    # Subset Obs
    ds_obs_out = ds_obs.where((ds_obs.time >= T_start) & (ds_obs.time <= T_end), drop=True)
    # Subset Model
    #ds_mod_out = ds_mod.where(((ds_mod.init_time >= T_start) & 
    #                          ((ds_mod.init_time+ds_mod.fore_time <= T_end).all(dim='fore_time'))), drop=True) # If forecasts times are long, this drops too much data
    
    # For model, we want to drop any valid times before T_start
    ds_mod = ds_mod.where(ds_mod.init_time >= T_start, drop=True)
    
    # AND set to NaN any valid times after T_end (updated)
    valid_time = ds_mod.init_time + ds_mod.fore_time
    ds_mod_out = ds_mod.where( (ds_mod.init_time >= T_start) & 
                          (valid_time <= T_end))
                              
    # Expand obs time to have missing until model valid forecasts
    new_time = pd.date_range(ds_obs_out.time.max().values, valid_time.max().values, freq=freq) # new time we don't have obs yet (future)
    new_obs_time = xr.DataArray(np.ones(new_time.shape)*np.NaN,  dims='time', coords={'time':new_time}) # new dataArray of missing
    ds_obs_out_exp = ds_obs_out.combine_first(new_obs_time) # Merge                         
    T_end = ds_obs_out_exp.time.max().values # new end time
    
    assert (ds_mod_out.init_time+ds_mod_out.fore_time).max().values <= ds_obs_out_exp.time.max().values, 'Model out contains valid times greater then end'

    print(T_start, T_end)
    assert T_start < T_end, 'Start must be before End!'
    
    return ds_obs_out_exp, ds_mod_out

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
    a = xr.ufuncs.sqrt( ((ds_mod - ds_obs)**2).mean(dim='ensemble').mean(dim='fore_time') )
    b = xr.ufuncs.sqrt( 2*(sigma**2) ) # add time variance for OP option
    
    NRMSE =  1 - (a / b)
    return NRMSE



#def PPP(ds_mod, ds_obs):
#    ''' potential prognostic predictability (PPP) '''

#    a = 
#    b = 

#    PPP = 1 - a / b
#    return PPP




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





