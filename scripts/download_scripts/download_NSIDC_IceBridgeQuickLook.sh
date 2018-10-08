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
data_ftp=https://n5eil01u.ecs.nsidc.org/ICEBRIDGE_FTP/Evaluation_Products/IceBridge_Sea_Ice_Freeboard_SnowDepth_and_Thickness_QuickLook

# Make sure the ACF Data environment variable is set
if [ -z "$NSIDC_IBQL" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$NSIDC_IBQL" ]; then
		echo "Need to set NSIDC_IBQL"
		exit 1
	fi
fi

mkdir -p $NSIDC_IBQL

# Download
cd $NSIDC_IBQL

wget --cut-dirs=20 --load-cookies ~/.urs_cookies --save-cookies ~/.urs_cookies --keep-session-cookies -r --reject "index.html*" -nH -e robots=off -A .txt -N $data_ftp

echo "Done!"

