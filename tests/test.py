from esio import ice_plot
from esio import import_data
from esio import metrics

import xarray as xr
import numpy as np
import datetime

# Make test model SIP data
da_sip = xr.DataArray(np.ones((1,1,1,3)), 
                      dims=('x','y','valid_time','model'), 
                      coords={'valid_time':[datetime.datetime(2018,1,1)]})

# Make observed sea ice presence data
da_obs = xr.DataArray(np.ones((1,1,1)), 
                      dims=('x','y','valid_time'), 
                      coords={'valid_time':[datetime.datetime(2018,1,1)]})

def test_BSS(da_sip=da_sip, da_obs=da_obs):
    
    # Case Model says SIP=1 and Obs=1, BSS = 0
    BSS = metrics._BSS(mod=da_sip.isel(model=0), 
                       obs=da_obs)
    assert BSS.mean(dim='valid_time') == 0
    
    # Case Model says SIP=1 and Obs=1, BSS = 0
    BSS = metrics._BSS(mod=da_sip.isel(model=0)*0, 
                       obs=da_obs)
    assert BSS.mean(dim='valid_time') == 1
    
    # Case Model says SIP=1 and Obs=1, BSS = 0
    BSS = metrics._BSS(mod=da_sip.isel(model=0)*0.5, 
                       obs=da_obs)
    assert BSS.mean(dim='valid_time') == (0.5)**2