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
set -e  # Stop on any error

# Set up python paths
source $HOME/.bashrc
source activate esio
which python

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! Mailing for help."
        mail -s "Error in Daily SIPN2 run."  $EMAIL <<< $2
	exit
    fi
}

# GET NRL model (weekly updated on wedneday) Download on Thursday
$REPO_DIR"/scripts/download_scripts/download_NRL.sh"
failfunction "$?" "download_NRL.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Get SIO forecasts
$REPO_DIR"/scripts/download_scripts/download_NRL_SIO.sh"
failfunction "$?" "download_NRL_SIO.sh had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# Regrid them
wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
python "./Regrid_NESM.py"
failfunction "$?" "Regrid_NESM.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

# GFDL (monthly)
python "./Regrid_GFDL_Forecast.py"
failfunction "$?" "Regrid_GFDL_Forecast.py had an Error. See log (https://atmos.washington.edu/~nicway/sipn/log/)." 

echo Finished Weekly script.


