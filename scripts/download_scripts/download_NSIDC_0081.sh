#!/bin/bash

# Downloads data from nsidc
set -x  # Echo all lines executed
set -e  # Stop on any error

# Source path file
#source ../path_file.sh

# FTP locations of data archives
#data_https=https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0081_nrt_nasateam_seaice/north/
data_ftp=ftp://sidads.colorado.edu/DATASETS/nsidc0081_nrt_nasateam_seaice/north/

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_0081_DATA_DIR" ]; then
     	echo "Need to set GFDL_DATA_DIR"
	exit 1
fi

mkdir -p $NSIDC_0081_DATA_DIR

# Download
#wget --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies --no-check-certificate --auth-no-challenge=on -r -N --reject "index.html*" -np -e robots=off --directory-prefix=$NSIDC_0081_DATA_DIR $data_https

# Download
cd $NSIDC_0081_DATA_DIR
wget -nH --cut-dirs=20 -r -A .bin -N $data_ftp

echo "Done!"

