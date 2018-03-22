from ecmwfapi import ECMWFDataServer
import datetime
import os

# Configuration info about this model
prefix = "yopp_ci_"
data_dir = r'/home/disk/sipn/nicway/data/model/yopp/forecast'
native_grib = os.path.join(data_dir, 'native')
# Note:
# This dataset is available with 3 day delay.

if not os.path.exists(native_grib):
    os.makedirs(native_grib)

os.chdir(native_grib)

# Download a single day
# This dataset is available with 48 hour delay.
day = datetime.datetime.now() -  datetime.timedelta(days=3)

server = ECMWFDataServer()
server.retrieve({
    "class": "yp",
    "dataset": "yopp",
    "date": day.strftime("%Y-%m-%d"),
    "expver": "1",
    "levtype": "sfc",
    "param": "31.128",
    "step": "0/3/6/9/12/15/18/21/24/27/30/33/36/39/42/45/48/51/54/57/60/63/66/69/72/75/78/81/84/87/90/93/96/99/102/105/108/111/114/117/120/123/126/129/132/135/138/141/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360",
    "stream": "enfo",
    "time": "00:00:00/12:00:00",
    "type": "cf",
    "target": prefix+day.strftime("%Y-%m-%d")+"_00_12.grib",
})
