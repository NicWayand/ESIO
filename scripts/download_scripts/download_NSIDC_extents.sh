#!/bin/bash

# Downloads data from nsidc
set -x  # Echo all lines executed
set -e  # Stop on any error

# Source path file
#source ../path_file.sh

# FTP locations of data archives
data_ftp=ftp://sidads.colorado.edu/DATASETS/NOAA/G02135/north/daily/data/N_seaice_extent_daily_v3.0.csv

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_extent_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_extent_DIR" ]; then
		echo "Need to set NSIDC_extent_DIR"
		exit 1
	fi
fi

mkdir -p $NSIDC_extent_DIR

# Download
cd $NSIDC_extent_DIR
wget -nH --cut-dirs=20 $data_ftp

echo "Done!"

