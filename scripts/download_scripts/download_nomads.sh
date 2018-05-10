#!/bin/bash

#This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
#If you use this code for a publication or presentation, please cite the reference in the README.md on the
#main page (https://github.com/NicWayand/ESIO). 
#
#Questions or comments should be addressed to nicway@uw.edu
#
#Copyright (c) 2018 Nic Wayand
#
#GNU General Public License v3.0

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

