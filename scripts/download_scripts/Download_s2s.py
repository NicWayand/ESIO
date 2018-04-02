
# coding: utf-8

# In[1]:


from ecmwfapi import ECMWFDataServer
import datetime
import os
import sys
import dask
import numpy as np
import calendar

'''
Download models with sea ice forecasts within the 2s2 forecast data archive.

https://software.ecmwf.int/wiki/display/S2S/Models

'''

# Check user defined configuraiton file
if len(sys.argv) < 2:
    raise ValueError('Requires either one arguments [recent] \n or two [list of years] [list of months (i.e. [2017,2018] [2,3]) ')

# Get name of configuration file/module
timeperiod = sys.argv[1]
if timeperiod=='recent':
    cd = datetime.datetime.now()
    years = [cd.year]
    months = [cd.month]
else:
    years = map(int, sys.argv[1].strip('[]').split(','))
    months = map(int, sys.argv[2].strip('[]').split(','))
print(years)
print(months)


# In[27]:


# Dates to download (by year and month)
# years = [2017, 2018]
# months = np.arange(1,13)
# years = [2018]
# months = [3,4]


# In[28]:


main_dir = '/home/disk/sipn/nicway/data/model'


# In[29]:


# Templet dicts for each model
# To change:
# - date
# - target

# Init it
mod_dicts = {}

# bom
mod_dicts['bom'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/2018-02-04/2018-02-08/2018-02-11/2018-02-15/2018-02-18/2018-02-22/2018-02-25",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "ammc",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440/1440-1464/1464-1488",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}


# CMA
mod_dicts['cma'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-28",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "babj",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}




# ECMWF
mod_dicts['ecmwf'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/2018-02-05/2018-02-08/2018-02-12/2018-02-15/2018-02-19/2018-02-22/2018-02-26",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "ecmf",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# HCMR
mod_dicts['hcmr'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-22/by/7",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "rums",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440/1440-1464",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# ISAC-CNR
mod_dicts['isaccnr'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-22/by/7",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "isac",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# JMA
mod_dicts['jma'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-07/to/2018-02-28/by/7",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "rjtd",
"param": "31",
"step": "12-36/36-60/60-84/84-108/108-132/132-156/156-180/180-204/204-228/228-252/252-276/276-300/300-324/324-348/348-372/372-396/396-420/420-444/444-468/468-492/492-516/516-540/540-564/564-588/588-612/612-636/636-660/660-684/684-708/708-732/732-756/756-780",
"stream": "enfo",
"time": "12:00:00",
"type": "cf",
"target": "output",
}

# Metreo France
mod_dicts['metreofr'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-22/by/7",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "lfpw",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# NCEP
mod_dicts['ncep'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-28",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "kwbc",
"param": "31",
"step": "24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# UKMO
mod_dicts['ukmo'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-28",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "egrr",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# ECCC
mod_dicts['eccc'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-22/by/7",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "cwao",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}

# KMA
mod_dicts['kma'] = {
"class": "s2",
"dataset": "s2s",
"date": "2018-02-01/to/2018-02-28",
"expver": "prod",
"levtype": "sfc",
"model": "glob",
"origin": "rksl",
"param": "31",
"step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440",
"stream": "enfo",
"time": "00:00:00",
"type": "cf",
"target": "output",
}


#mod_dicts.keys()


# In[30]:


def download_month(config_dict):
        # Start server
        cserver = ECMWFDataServer()
        cserver.retrieve(config_dict)
        return 1


# In[31]:


download_month = dask.delayed(download_month)


# In[32]:


# def download_last_30days():
#     X = 1
#     for cmod in mod_dicts.keys():
#         print(cmod)
#         cdict = mod_dicts[cmod]
#         # Update cdict
#         cdict['date'] = date_range
#         print(date_range)
#         target = os.path.join(main_dir, cmod, 'forecast', 'native', 
#                                   cmod+'_'+str(cy)+'_'+format(cm, '02')+'.grib')
#         cdict['target'] = target
#         X = X + download_month(cdict)
#     X.compute()     


# In[41]:


def download_by_month(cy, cm):
    X = 1
    
    DS = datetime.datetime(cy,cm,1)
    DE = datetime.datetime(cy,cm,calendar.monthrange(cy,cm)[1])
    
    cd = datetime.datetime.now()
    S2S = cd - datetime.timedelta(days=21)
    
    # Check if current month, insure dates are not within last 21 days (embargo)
    DE = np.min([DE, S2S])
    
    # Check if most recent init date is before current month start
    if DS>DE:
        print('No data avaialble yet for ', str(cy),'-',str(cm))
        print('Re-downloading previous month...')
        download_by_month(cy, cm-1)
        #return 0 # Just return an int for dask
    
    
    date_range = DS.strftime('%Y-%m-%d')+'/to/'+ DE.strftime('%Y-%m-%d')
    #date_range = str(cy)+'-'+format(cm,'02')+'-01/to/'+str(cy)+'-'+format(cm,'02')+'-'+str(calendar.monthrange(cy,cm)[1])
    
    print(date_range) 
#     if (cd.year==cy) & (cd.month==cm):
#         print('Current month: Excluding last 21 days')
#         # Trim date_range to exclude last 21 days
#         date_end = cd - datetime.timedelta(days=21)
#         date_range = str(cy)+'-'+format(cm,'02')+'-01/to/'+\
#         str(cy)+'-'+format(cm,'02')+'-'+format(date_end.day, '02')
        
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
    X.compute()            


# In[45]:


# Download by month, wait for each month to finish
for cy in years:
    for cm in months:
        download_by_month(cy, cm)


# In[ ]:


# ! Using "2018-02-01/to/2018-02-28", works! if only a subset of dates are available!

# cserver = ECMWFDataServer()
# cserver.retrieve({
# "class": "s2",
# "dataset": "s2s",
# "date": "2018-02-01/to/2018-02-28",
# "expver": "prod",
# "levtype": "sfc",
# "model": "glob",
# "origin": "ammc",
# "param": "31",
# "step": "0-24/24-48/48-72/72-96/96-120/120-144/144-168/168-192/192-216/216-240/240-264/264-288/288-312/312-336/336-360/360-384/384-408/408-432/432-456/456-480/480-504/504-528/528-552/552-576/576-600/600-624/624-648/648-672/672-696/696-720/720-744/744-768/768-792/792-816/816-840/840-864/864-888/888-912/912-936/936-960/960-984/984-1008/1008-1032/1032-1056/1056-1080/1080-1104/1104-1128/1128-1152/1152-1176/1176-1200/1200-1224/1224-1248/1248-1272/1272-1296/1296-1320/1320-1344/1344-1368/1368-1392/1392-1416/1416-1440/1440-1464/1464-1488",
# "stream": "enfo",
# "time": "00:00:00",
# "type": "cf",
# "target": "/home/disk/sipn/nicway/data/model/bom/forecast/native/test.grib",
# "expect": "any"
# })


