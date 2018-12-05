#/bin/bash

#This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
#If you use this code for a publication or presentation, please cite the reference in the README.md on the
#main page (https://github.com/NicWayand/ESIO). 
#
#Questions or comments should be addressed to nicway@uw.edu
#
#Copyright (c) 2018 Nic Wayand
#
#GNU General Public License v3.0

#set -x  # Echo all lines executed
#set -e  # Stop on any error

# Set up python paths
source $HOME/.bashrc
source activate esio
which python

failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! Mailing for help."
        mail -s "Error in Daily SIPN2 run." $EMAIL <<< $2
	exit
    fi
}

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh" & # Fast
$REPO_DIR"/scripts/download_scripts/download_NSIDC_extents.sh"  # Fast
failfunction "$?" "download_NSIDC_0081.sh or download_NSIDC_extents.sh had an Error. See log." 

# Model downloads
python $REPO_DIR"/scripts/download_scripts/Download_YOPP_ECMWF.py"  # Slow (30 mins)
failfunction "$?" "Download_YOPP_ECMWF had an Error. See log." 

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Observations to sipn format
which python
python "./Import_NSIDC_Obs.py"
failfunction "$?" "Import_NSIDC_Obs.py had an Error. See log." 
python "./Import_NSIDC_Extents.py"
failfunction "$?" "Import_NSIDC_Extents.py had an Error. See log." 

# Agg Obs to yearly files
python "./Agg_NSIDC_Obs.py"
failfunction "$?" "Agg_NSIDC_Obs.py had an Error. See log." 

# Convert to Zarr
python "./Convert_netcdf_to_Zarr.py"
failfunction "$?" "Convert_netcdf_to_Zarr.py had an Error. See log."

# Upload to GCP
/home/disk/sipn/nicway/data/obs/zarr/update_obs.sh

# Import Models to sipn format
source activate test_nio # Requires new env
python "./Regrid_YOPP.py"
failfunction "$?" "Regrid_YOPP.py had an Error. See log." 

source activate esio

wait # Below depends on above

# Make Plots
# Availblity plots
python "./plot_forecast_availability.py" &
failfunction "$?" "plot_forecast_availability.py had an Error. See log." 

# Observations
python "./plot_observations.py" &
failfunction "$?" "plot_observations.py had an Error. See log." 

echo Finished NRT script.
