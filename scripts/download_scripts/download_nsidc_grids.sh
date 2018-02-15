#!/bin/bash

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

