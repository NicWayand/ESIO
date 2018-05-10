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

# Downloads NSIDC grids 

set -x  # Echo all lines executed
set -e  # Stop on any error

# FTP locations of data archives
data_tar=ftp://sidads.colorado.edu/pub/DATASETS/seaice/polar-stereo/tools/

# Make sure the ACF Data environment variable is set
if [ -z "$GRID_DIR" ]; then
     	echo "Need to set GRID_DIR"
	exit 1
fi

mkdir -p $GRID_DIR

# Download
cd $GRID_DIR
wget -r -np -nH --cut-dirs=20 $data_tar

echo "Done!"

