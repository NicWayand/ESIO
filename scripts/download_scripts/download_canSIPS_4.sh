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
data_ftp=ftp://ftp.cccma.ec.gc.ca/pub/wslee/sicn/

# Make sure the ACF Data environment variable is set
if [ -z "$CANSIPS_4_DATA_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$CANSIPS_4_DATA_DIR" ]; then
		echo "Need to set CANSIPS_4_DATA_DIR"
		exit 1
	fi
fi

mkdir -p $CANSIPS_4_DATA_DIR

# Download
cd $CANSIPS_4_DATA_DIR
wget -nH --cut-dirs=20 -r -A "*CanCM4*" -N $data_ftp

echo "Done!"

