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


failfunction()
{
    if [ "$1" != 0 ]
    then echo "One of the commands has failed! Mailing for help."
        mail -s "Error in Daily SIPN2 run." $EMAIL <<< $2
	exit
    fi
}

# testing failfunction
#cd /home/asdfsd/ 
#failfunction "$?" "test failed"

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

# Model downloads
python $REPO_DIR"/scripts/download_scripts/Download_s2s.py" "recent" 
failfunction "$?" "Download_s2s.py had an Error. See log." 
python $REPO_DIR"/scripts/download_scripts/Download_C3S.py" "recent" 
failfunction "$?" "Download_C3S.py had an Error. See log." 
$REPO_DIR"/scripts/download_scripts/download_RASM_ESRL.sh" 
failfunction "$?" "download_RASM_ESRL.py had an Error. See log." 
$REPO_DIR"/scripts/download_scripts/download_NRL_GOFS3_1.sh"
failfunction "$?" "download_NRL_GOFS3_1.sh had an Error. See log."

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
source activate test_nio # Requires new env
python "./Regrid_S2S_Models.py"
failfunction "$?" "Regrid_S2S_Models.py had an Error. See log." 

python "./Regrid_RASM.py"
failfunction "$?" "Regrid_RASM.py had an Error. See log." 

python "./Regrid_CFSv2.py"
failfunction "$?" "Regrid_CFSv2.py had an Error. See log." 

wait
source activate esio
wait # Below depends on above

# Calc Aggregate metrics (e.g. extent for different regions)
python "./Calc_Model_Aggregations.py"
failfunction "$?" "Calc_Model_Aggregations.py had an Error. See log." 


# Make Plots
# Availblity plots
which python
python "./plot_forecast_availability.py"
failfunction "$?" "plot_forecast_availability.py had an Error. See log." 

# Extents
python "./plot_Extent_Model_Obs.py"
failfunction "$?" "plot_Extent_Model_Obs.py had an Error. See log." 

python "./plot_Regional_Extent.py"
failfunction "$?" "plot_Regional_Extent.py had an Error. See log." 

# Maps
python "./plot_Maps_Fast.py" 
failfunction "$?" "plot_Maps_Fast.py had an Error. See log." 
#python "./plot_Regional_maps.py"
#failfunction "$?" "plot_Regional_maps.py had an Error. See log." 

echo Finished NRT script.
