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

# Source path file
#source ../path_file.sh

# FTP locations of data archives
data_ftp=https://n5eil01u.ecs.nsidc.org/ICEBRIDGE/RDEFT4.001

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_RDEFT4" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_RDEFT4" ]; then
		echo "Need to set NSIDC_RDEFT4"
		exit 1
	fi
fi

mkdir -p $NSIDC_RDEFT4

# Download
cd $NSIDC_RDEFT4

wget -v --cut-dirs=3 --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r -nH -e robots=off $data_ftp

echo "Done!"

