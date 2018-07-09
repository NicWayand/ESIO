import calendar
import datetime
import os
import dask
import numpy as np
import pandas as pd
from ecmwfapi import ECMWFDataServer


@dask.delayed
def download_month(config_dict):
    # Start server
    cserver = ECMWFDataServer()
    cserver.retrieve(config_dict)
    return 1

def download_data_by_month(dataclass=None, main_dir=None,
                           mod_dicts=None, cy=None, cm=None,
                           run_type='forecast'):
    X = 1

    if dataclass=='s2s':
        day_lag = 22
    elif dataclass=='c3':
        day_lag = 16
    else:
        raise ValueError('dataclass not found.')

    DS = datetime.datetime(cy,cm,1)
    DE = datetime.datetime(cy,cm,calendar.monthrange(cy,cm)[1])

    cd = datetime.datetime.now()
    S2S = cd - datetime.timedelta(days=day_lag)

    # Check if current month, insure dates are not within last 16/21 days (embargo)
    DE = np.min([DE, S2S])

    # Check if most recent init date is before current month start
    if DS>DE:
        print('No data avaialble yet for ', str(cy),'-',str(cm))
        print('Re-downloading previous month...')
        download_data_by_month(dataclass=dataclass, main_dir=main_dir,
                               mod_dicts=mod_dicts, cy=cy, cm=cm-1)
        return 0 # Just return an int for dask. Don't continue here.

    # Create date range as string
    
    # Forecast syntax allows the start/to/end syntax
    if run_type=='forecast':
        date_range = DS.strftime('%Y-%m-%d')+'/to/'+ DE.strftime('%Y-%m-%d')
        print(date_range)
    elif run_type=='reforecast':
        pd_dates = pd.date_range(start=DS,end=DE,freq='D')
        date_range = [x.strftime('%Y-%m-%d') for x in pd_dates]
        print(date_range[0],date_range[-1])
        date_range = '/'.join(date_range)
    else:
        raise ValueError('run_type not found.')   
        
    
    for cmod in mod_dicts.keys():
        print(cmod)
        cdict = mod_dicts[cmod]
        # Update cdict
        # If forecast update "date"
        if run_type=='forecast':
            cdict['date'] = date_range
        # else, if a REforecast, update "hdate"
        elif run_type=='reforecast':
            cdict['hdate'] = date_range
        else:
            raise ValueError('run_type not found.')
        target = os.path.join(main_dir, cmod, run_type, 'native',
                                  cmod+'_'+str(cy)+'_'+format(cm, '02')+'.grib')
        cdict['target'] = target
        cdict['expect'] = 'any'
        X = X + download_month(cdict)

    # Call compute to download all models concurently from ecmwf
    X.compute()