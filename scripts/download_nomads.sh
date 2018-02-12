#!/bin/bash

# Downloads historical GFDL FLORB01 sea ice concentration data (~80GB)

set -x  # Echo all lines executed
set -e  # Stop on any error

# FTP locations of data archives
GFDL_MONTHLY=ftp://nomads.gfdl.noaa.gov/NMME/GFDL-FLORB01
# ftp://nomads.gfdl.noaa.gov/NMME/GFDL-FLORB01/FLORB01-P1-ECDA-v3.1-121998/mon/seaIce/OImon/r1i1p1/v20140710/sic/

# Make sure the ACF Data environment variable is set
if [ -z "$GFDL_DATA_DIR" ]; then
     	echo "Need to set GFDL_DATA_DIR"
	exit 1
fi

mkdir -p $GFDL_DATA_DIR

# Download
wget -r -l20 --no-parent --directory-prefix=$GFDL_DATA_DIR -A sic*.nc $GFDL_MONTHLY

echo "Done!"

