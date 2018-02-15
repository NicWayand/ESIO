#!/bin/bash

# Downloads data from nsidc
set -x  # Echo all lines executed
set -e  # Stop on any error

# FTP locations of data archives
data_ftp=ftp://sidads.colorado.edu/DATASETS/nsidc0079_gsfc_bootstrap_seaice_v3/final-gsfc/north/daily/

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_0079_DATA_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_0079_DATA_DIR" ]; then
		echo "Need to set NSIDC_0079_DATA_DIR"
		exit 1
	fi
fi

mkdir -p $NSIDC_0079_DATA_DIR

# Download
cd $NSIDC_0079_DATA_DIR
wget -nH --cut-dirs=20 -r -A .bin -N $data_ftp

echo "Done!"

