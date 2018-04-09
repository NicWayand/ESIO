#!/bin/bash

# Converts grib to netcdf

set -e

if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Need [grib_dir] [nc_dir]"
fi

grib_dir=$1 # Where *.grib files are
nc_dir=$2

# Concatenate
cd $grib_dir
echo "ONLY including yopp_ci_2018-03*"
find -type f -name  'yopp_ci_2018-03*.grib' | xargs -n 32 -P 8 cat >> $nc_dir"/temp.grib"

echo "Done concating"

# to netcdf
ncl_convert2nc $nc_dir"/temp.grib" -o $nc_dir -L #-itime # -th 2000

# Clean up
rm -f $nc_dir"/temp.grib"

