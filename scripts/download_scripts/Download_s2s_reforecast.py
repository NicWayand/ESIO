
# coding: utf-8

# In[ ]:


'''

This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
If you use this code for a publication or presentation, please cite the reference in the README.md on the
main page (https://github.com/NicWayand/ESIO). 

Questions or comments should be addressed to nicway@uw.edu

Copyright (c) 2018 Nic Wayand

GNU General Public License v3.0


'''

from ecmwfapi import ECMWFDataServer
import os
import sys
import dask
import numpy as np
from esio import download
import datetime
from esio import EsioData as ed

'''
Download models with sea ice forecasts within the 2s2 forecast data archive.

https://software.ecmwf.int/wiki/display/S2S/Models

'''

# Check user defined configuraiton file
if len(sys.argv) < 2:
    raise ValueError('Requires either one arguments [recent] \n or two [start_year, end_yaer] [start_month, end_month] (inclusive) ')

# Get name of configuration file/module
timeperiod = sys.argv[1]
if timeperiod=='recent':
    cd = datetime.datetime.now()
    years = [cd.year]
    months = [cd.month]
else:
    year_range_in = list(map(int, sys.argv[1].strip('[]').split(',')))
    month_range_in = list(map(int, sys.argv[2].strip('[]').split(',')))
    if (len(year_range_in)!=2) | (len(month_range_in)!=2):
        raise ValueError('Year range and month range must be two values (inclusive)')
    years = np.arange(year_range_in[0], year_range_in[1]+1, 1)
    months = np.arange(month_range_in[0], month_range_in[1]+1, 1)
    assert np.all(months>=0), 'months must be >=0'
    assert np.all(months<=12), 'months must be <=12'
    


# In[ ]:


# # Testing
# years = np.array([1999,1999])
# months = np.array([1,1])

# reforecast ranges

# start
# 2014-01

# end
# 1993-01-01

print(years)
print(months)


# In[ ]:


E = ed.EsioData.load()
main_dir = E.model_dir


# In[ ]:


# Templet dicts for each model

# Init it
mod_dicts = {}

# # bom
# mod_dicts['bom'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2014-01-01",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "ammc",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440/1440-1464/1464-1488",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }


# # CMA
# mod_dicts['cma'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2014-05-01",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "babj",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }




# # ECMWF
# mod_dicts['ecmwf'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/2018-02-05/2018-02-08/2018-02-12/2018-02-15/2018-02-19/2018-02-22/2018-02-26",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "ecmf",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }

# # HCMR
# mod_dicts['hcmr'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-22/by/7",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "rums",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440/1440-1464",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }

# # ISAC-CNR
# mod_dicts['isaccnr'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-22/by/7",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "isac",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }

# # JMA
# mod_dicts['jma'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-07/to/2018-02-28/by/7",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "rjtd",
# "param": "31",
# "step": "12-36/36-60/60-84/84-108/108-132/132-156/156-180/180-204/204-228/228-252/252-276/276-300/300-324/324-348/348-372/372-396/396-420/420-444/444-468/468-492/492-516/516-540/540-564/564-588/588-612/612-636/636-660/660-684/684-708/708-732/732-756/756-780",
# "stream": "enfo",
# "time": "12:00:00",
# "type": "cf",
# "target": "output",
# }

# Metreo France
mod_dicts['metreofr'] = {
"class": "s2",
"dataset": "s2s",
"date": "2014-12-01",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "lfpw",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
"stream": "enfh",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# NCEP
mod_dicts['ncep'] = {
"class": "s2",
"dataset": "s2s",
"date": "2011-03-01",
"hdate": "1999-01-01",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "kwbc",
"param": "31",
"step": "24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056",
"stream": "enfh",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# # UKMO
# mod_dicts['ukmo'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-28",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "egrr",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
# "stream": "enfh",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }

# # ECCC
# mod_dicts['eccc'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-22/by/7",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "cwao",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }

# # KMA
# mod_dicts['kma'] = {
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-28",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "rksl",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "output",
# }


#mod_dicts.keys()


# In[ ]:


mod_dicts.keys()


# In[ ]:


# Download by month, wait for each month to finish
for cy in years:
    print(cy)
    for cm in months:
        print(cm)
        download.download_data_by_month(dataclass='s2s', main_dir=main_dir, 
                                        mod_dicts=mod_dicts, cy=cy, cm=cm,
                                        run_type='reforecast')


