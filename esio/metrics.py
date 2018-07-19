import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib

def mask_common_extent(ds_obs, ds_mod, max_obs_missing=0.1):
    ''' Define naive_fast that searches for the nearest WRF grid cell center.'''

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


def calc_extent(da=None, region=None, extent_thress=0.15, fill_pole_hole=False):
    ''' Returns extent in millions of km^2 within all ocean regions (NO LAKES!)'''

    if 'x' not in da.dims:
        raise ValueError('x not found in dims.... might need to rename spatial dims')

    # TODO: Need to assert we pass in a DataArray of sic
    extent = (( da.where(region.mask.isin(region.ocean_regions)) >= extent_thress ).astype('int') * region.area).sum(dim='x').sum(dim='y')/(10**6)

    # Mask out zero extents (occurs if ensemble changes size)
    extent = extent.where(extent>0)

    # Add in pole hole (optional)
    if fill_pole_hole:
        extent = extent + (da.hole_mask.astype('int') * region.area).sum(dim='x').sum(dim='y')/(10**6)

    return extent


def agg_by_domain(da_grid=None, ds_region=None, extent_thress=0.15, fill_pole_hole=False):
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
            da_avg = da_avg.sum(dim='x').sum(dim='y') / (10**6)
            # TODO: Add option to add in pole hole if obs and central arctic
            if fill_pole_hole:
                raise ValueError('Not implemented')

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


def get_median_ice_edge(ds, ystart='1981', yend='2012', sic_threshold=0.15):
    ''' Calc NSIDC median sea ice edge between 1981-2010'''
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


def calc_IFD(da, sic_threshold=0.15, DOY_s=1, time_dim='time'):
    ''' Calc the Ice Free Day (first) by Calender Year. '''
    da = da.rename({time_dim:'time'})
    ifd = (da < sic_threshold).reduce(np.argmax, dim='time') # Find index of first ice free
    ifd = ifd.where(da.isel(time=0).notnull()) # Apply Orig mask
    # Convert to Day of Year by adding the first time
    ifd = ifd + DOY_s
    return ifd

def calc_IFD_10day(da, sic_threshold=0.5, DOY_s=1, time_dim='time', Nday=10, default_ice_free=None):
    ''' Calc the Ice Free Day (first) by Calender Year. 
    Returns day of year (doy) for each pixel when the sic value dropped below the sic_threshold 
    and stayed below that threshold for atleast Nday days.
    '''
    da = da.rename({time_dim:'time'})

    # Here we find pixels WITHOUT (<sic_threshold) Ice!!!
    sip_Xday = (da < sic_threshold).rolling(min_periods=1, center=False, time=Nday).sum().where(da.isel(time=0).notnull()) 

    # Now values range from 0 to Nday, keep only Nday pixel values, rescale them to 1 (end up with 0 with ice and 1 with ocean over past 10 days)
    sip_Xday_adj = sip_Xday.where(sip_Xday == Nday, other=0).where(da.isel(time=0).notnull())
    sip_Xday_adj = sip_Xday_adj.astype('float') / Nday

    # Now reduce over time dime, and find the first "1" value for each pixel (which is the last day of ice presence, with at least 10 days following ice free)
    ifd = sip_Xday_adj.reduce(np.argmax, dim='time') # Find index of first ice free
    #plt.figure()
    #ifd.plot()
    
    # Convert to Day of Year by adding the first time
    ifd = ifd + DOY_s - Nday - 1 # Add DOY_s and subtract the Nday window (rolling returns the right side label "trailing")
    #plt.figure()
    #ifd.plot()
    
    # Classify pixels 
    # 1) that were ice free at first index time to the default ice free date (i.e. June 1st)
    ifd = ifd.where(ifd > DOY_s, other=default_ice_free)
    #plt.figure()
    #ifd.plot()
    
    # 2) that never melted (perenial) to NaN
    # Grab last model time (end of Sept) and get mask of where ice is
    perenial_ice_mask = (da.isel(time=da.time.size-1) >= sic_threshold)
    ifd = ifd.where(~perenial_ice_mask, other=275)
    #plt.figure()
    #ifd.plot()
        
    # Apply Orig mask
    ifd = ifd.where(da.isel(time=0).notnull())    

    return ifd


def calc_hist_sip(ds_sic=None, ystart='2007', yend='2017', sic_threshold=0.15):
    ''' Calc historical SIP for a range of years'''
    
    # Trim by years
    ds_sic = ds_sic.sel(time=slice(ystart, yend))
    
    # Get landmask (where sic is NaN)
    land_mask = ds_sic.drop('hole_mask').isel(time=0).notnull()
    #land_mask.plot()
    
    # Convert sea ice presence
    ds_sp = (ds_sic >= sic_threshold).astype('int') # This unfortunatly makes all NaN -> zeros...
    #ds_sp.isel(time=0).plot()
    
    # Fill in pole hole with 1 (so contours don't get made around it)
    ds_sp = ds_sp.where(ds_sic.hole_mask==0, other=1).drop('hole_mask')
    
    # Add DOY
    DOY = [x.timetuple().tm_yday for x in pd.to_datetime(ds_sp.time.values)]
    ds_sp['time'] = DOY
    
    # Calculate mean SIP
    ds_sip = ds_sp.groupby('time').mean(dim='time')
    
    # Mask land
    ds_sip = ds_sip.where(land_mask)
    
    return ds_sip

def nanSum(da=None, dim=None):
    ''' Return nan sum for pixels where we have atleast 1 NaN value. '''
    return da.sum(dim=dim).where(da.notnull().sum(dim=dim) > 0 )


def format_obs_like_model(ds_mod, ds_obs):
    ''' Reformats observational dataset to be structured like a model forecast dataset
    Format obs like model (i.e. ensemble x init_time x forecast_time) '''

    ds_obs_X = (ds_mod.copy() * np.nan).load() # Have to call load here to assign ie below
    for (i, e) in enumerate(ds_obs_X.ensemble):
        for (j, it) in enumerate(ds_obs_X.init_time):
            ds_obs_X[i, j, :] = ds_obs.sel(time = ( ds_mod.init_time.sel(init_time=it) + ds_mod.fore_time).values )

    return ds_obs_X


def dt64_to_dd(dt64):
    ''' Converts datetime64[ns] into datetime.datetime'''
    return dt64.values.astype('M8[D]').astype('O')


def trim_common_times(ds_obs=None, ds_mod=None, freq=None):
    ''' Trim an observed and modeled dataset to common start and end dates (does not
    insure internal times are the same) '''

    if 'valid_time' not in ds_mod.coords:
        ds_mod.coords['valid_time'] = ds_mod.init_time + ds_mod.fore_time

    # Get earliest and latest times
    print(ds_mod.valid_time.max().values)
    print(ds_obs.time.values[-1])
    T_start = np.max([ds_obs.time.values[0], ds_mod.init_time.min().values])
    T_end = np.min([ds_obs.time.values[-1], (ds_mod.valid_time.max()).values])


    # Subset Obs
    ds_obs_out = ds_obs.where((ds_obs.time >= T_start) & (ds_obs.time <= T_end), drop=True)
    # Subset Model
    #ds_mod_out = ds_mod.where(((ds_mod.init_time >= T_start) &
    #                          ((ds_mod.init_time+ds_mod.fore_time <= T_end).all(dim='fore_time'))), drop=True) # If forecasts times are long, this drops too much data

    # For model, we want to drop any valid times before T_start
    ds_mod = ds_mod.where(ds_mod.init_time >= T_start, drop=True)

    # AND set to NaN any valid times after T_end (updated)

    ds_mod_out = ds_mod.where( (ds_mod.init_time >= T_start) &
                          (ds_mod.valid_time <= T_end))

    # Expand obs time to have missing until model valid forecasts
    #print(ds_obs_out.time.max().values)
    #print(valid_time.max().values)
    #print(freq)
    #offset1 = np.timedelta64(40, 'D') # A shift to handel case of monthly data to have extra missing (NaN) obs
    new_time = pd.date_range(ds_obs_out.time.max().values, ds_mod.valid_time.max().values, freq=freq) # new time we don't have obs yet (future)
    new_obs_time = xr.DataArray(np.ones(new_time.shape)*np.NaN,  dims='time', coords={'time':new_time}) # new dataArray of missing
    ds_obs_out_exp = ds_obs_out.combine_first(new_obs_time) # Merge
    T_end = ds_obs_out_exp.time.max().values # new end time

    print(ds_mod_out.valid_time.max().values)
    print(ds_obs_out_exp.time.max().values)

    assert (ds_mod_out.valid_time).max().values <= ds_obs_out_exp.time.max().values, 'Model out contains valid times greater then end'

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

def IIEE(da_mod=None, da_obs=None, region=None, sic_threshold=0.15, testplots=False):
    ''' The Integrated Ice‐Edge Error Goessling 2016'''
    
    # Input
    # da_mod/da_obs - DataArray of sic from model/observations
    
    # Output
    # IEEE - Area of IEEE in millions of km^2, 
    
    # Should already be formated the same
    assert (sorted(da_mod.dims) == sorted(da_obs.dims)), "Dims should be the same."
    
    # spatial dims in model and obs should be 'x' and 'y' to match regions var names
    assert ('x' in da_mod.dims), "'x' and 'y' should be dims."
    assert ('y' in da_obs.dims), "'x' and 'y' should be dims"
    
    # Reduce to sea ice presence
    mod_sip = (da_mod >= sic_threshold).where(da_mod.notnull())
    obs_sip = (da_obs >= sic_threshold).where(da_obs.notnull())
    
    # Mask to regions of Arctic we are interested in
    mod_sip = mod_sip.where(region.mask.isin(region.ocean_regions))
    obs_sip = obs_sip.where(region.mask.isin(region.ocean_regions))
    
    if testplots:
        plt.figure()
        (abs(mod_sip - obs_sip)).plot(cmap='Reds')
    
    # Calculate both terms (over and under area) in km^2
    IIEE = (abs(mod_sip - obs_sip) * region.area ).sum(dim='x').sum(dim='y')/(10**6)
    
    return IIEE

def _BSS(mod=None, obs=None, time_dim='time'):
    return ((mod-obs)**2).mean(dim=time_dim)

def BrierSkillScore(da_mod_sip=None, da_obs_ip=None, 
                    time_dim='valid_time', region=None, 
                    testplots=False):
    '''
    Brier Skill Score
    ----------
    Parameters:
    da_mod_sip : DataArray
        DataArray of modeled sea ice probabilies (0-1)
    da_obs_ip : DataArray
        DataArray of observed sea ice presence (0 or 1)
    time_dim: String
        Name of time dimension to take mean over when calculating the BSS
    region : DataSet
        DataSet contain spatial location of Arctic regions
    testplots : Boolean
        Flag to turn on test plots    
        
    Returns:
    BSS = Brier Skill Score
    
    '''
    
    # Should already be formated the same
    assert (sorted(da_mod_sip.dims) == sorted(da_obs_ip.dims)), "Dims should be the same."
    
    # spatial dims in model and obs should be 'x' and 'y' to match regions var names
    assert ('x' in da_mod_sip.dims), "'x' and 'y' should be dims."
    assert ('y' in da_obs_ip.dims), "'x' and 'y' should be dims"
        
    # Mask to regions of Arctic we are interested in
    da_mod_sip = da_mod_sip.where(region.mask.isin(region.ocean_regions))
    da_obs_ip = da_obs_ip.where(region.mask.isin(region.ocean_regions))
        
    # Calculate Brier Skill Score
    BSS = _BSS(mod = da_mod_sip, 
               obs = da_obs_ip,
               time_dim = time_dim)
    
    if testplots:
        plt.figure()
        da_mod_sip.plot()
        plt.figure()
        da_obs_ip.plot()   
        plt.figure()
        BSS.plot()
            
    return BSS