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

# Downloads data from nsidc
set -x  # Echo all lines executed
set -e  # Stop on any error

# FTP locations of data archives
data_ftp=ftp://sidads.colorado.edu/DATASETS/nsidc0081_nrt_nasateam_seaice/north/

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_0081_DATA_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_0081_DATA_DIR" ]; then
		echo "Need to set NSIDC_0081_DATA_DIR"
		exit 1
	fi
fi

mkdir -p $NSIDC_0081_DATA_DIR

# Download
cd $NSIDC_0081_DATA_DIR
wget -nH --cut-dirs=20 -r -A .bin -N $data_ftp

echo "Done!"

