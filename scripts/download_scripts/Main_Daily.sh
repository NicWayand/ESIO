#/bin/bash

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

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Models to sipn format
source activate test_nio # Requires new env
python "./Regrid_S2S_Models.py"

source activate esio

wait # Below depends on above

# Make Plots
# Availblity plots
python "./plot_forecast_availability.py" &

# Models
python "./plot_all_model_maps.py" &
python "./plot_model_forecasts.py" &

# Both
python "./plot_Extent_Model_Obs.py" &

echo Finished NRT script.
