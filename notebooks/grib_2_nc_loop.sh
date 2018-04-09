#!/bin/bash

# Converts grib to netcdf

set -e

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Need [grib_dir] [nc_dir]"
fi

grib_dir=$1 # Where *.grib files are
nc_dir=$2

cd $grib_dir
for f in yopp_ci_2018-03*.grib; do
    [ -f "$f" ] || break
    echo $f    
    # to netcdf
    ncl_convert2nc $f -o $nc_dir -L #-itime # -th 2000
done

