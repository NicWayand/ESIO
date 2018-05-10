'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

from ecmwfapi import ECMWFDataServer
import dask
import datetime
import numpy as np
import calendar
import os

@dask.delayed
def download_month(config_dict):
    # Start server
    cserver = ECMWFDataServer()
    cserver.retrieve(config_dict)
    return 1

# Wrapper function to download ECMWF data    
def download_data_by_month(dataclass=None, main_dir=None, 
                           mod_dicts=None, cy=None, cm=None):
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
    
    # Check if current month, insure dates are not within last 21 days (embargo)
    DE = np.min([DE, S2S])
    
    # Check if most recent init date is before current month start
    if DS>DE:
        print('No data avaialble yet for ', str(cy),'-',str(cm))
        print('Re-downloading previous month...')
        download_data_by_month(dataclass=dataclass, main_dir=main_dir, 
                               mod_dicts=mod_dicts, cy=cy, cm=cm-1)
        return 0 # Just return an int for dask. Don't continue here.
    
    # Create date range as string
    date_range = DS.strftime('%Y-%m-%d')+'/to/'+ DE.strftime('%Y-%m-%d')
    
    print(date_range) 
        
    for cmod in mod_dicts.keys():
        print(cmod)
        cdict = mod_dicts[cmod]
        # Update cdict
        print(date_range)
        cdict['date'] = date_range
        target = os.path.join(main_dir, cmod, 'forecast', 'native', 
                                  cmod+'_'+str(cy)+'_'+format(cm, '02')+'.grib')
        cdict['target'] = target
        cdict['expect'] = 'any'
        X = X + download_month(cdict)
    
    # Call compute to download all models concurently from ecmwf
    X.compute()            
