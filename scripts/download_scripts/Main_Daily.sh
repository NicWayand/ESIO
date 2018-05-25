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

# Make sure the ACF REPO_DIR environment variable is set
if [ -z "$REPO_DIR" ]; then
    echo "Need to set REPO_DIR"
    exit 1
fi

# Model downloads
python $REPO_DIR"/scripts/download_scripts/Download_s2s.py" "recent" &
python $REPO_DIR"/scripts/download_scripts/Download_C3S.py" "recent" &
$REPO_DIR"/scripts/download_scripts/download_RASM_ESRL.sh" &

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
source activate test_nio # Requires new env
python "./Regrid_S2S_Models.py"
python "./Regrid_RASM.py"
python "./Regrid_CFSv2.py"

wait
source activate esio
wait # Below depends on above

# Calc Aggregate metrics (e.g. extent for different regions)
python "./Calc_Model_Aggregations.py"


# Make Plots
# Availblity plots
which python
python "./plot_forecast_availability.py"

# Extents
python "./plot_Extent_Model_Obs.py" &
python "./plot_Regional_Extent.py"

# Maps
python "./plot_all_model_maps.py" &
python "./plot_Regional_maps.py"

echo Finished NRT script.
