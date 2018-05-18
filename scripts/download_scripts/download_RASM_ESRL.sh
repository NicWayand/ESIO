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
data_ftp=ftp://ftp1.esrl.noaa.gov/RASM-ESRL/ModelOutput/

# Make sure the ACF Data environment variable is set
if [ -z "$RASM_ESRL_DATA_DIR" ]; then
	# Try to source path file
	echo "trying to source path_file.sh"
	source ../path_file.sh
	# Check if its now set
	if [ -z "$RASM_ESRL_DATA_DIR" ]; then
		echo "Need to set NSIDC_0081_DATA_DIR"
		exit 1
	fi
fi

mkdir -p $RASM_ESRL_DATA_DIR

# Download
cd $RASM_ESRL_DATA_DIR
wget -nH --cut-dirs=20 -r -A "RASM-ESRL_4NIC*.gz" -N $data_ftp


cd $REPO_DIR/scripts/download_scripts/
./unzip_file.sh /home/disk/sipn/nicway/data/model/rasmesrl/forecast/native/ RASM-ESRL

echo "Done!"

