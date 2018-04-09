#/bin/bash

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

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh" & # Fast
$REPO_DIR"/scripts/download_scripts/download_NSIDC_extents.sh" & # Fast

# Model downloads
python $REPO_DIR"/scripts/download_scripts/Download_YOPP_ECMWF.py" & # Slow (30 mins)

wait # Below depends on above

# Move to notebooks
cd $REPO_DIR"/notebooks/" # Need to move here as some esiodata functions assume this

# Import Observations to sipn format
which python
python "./SeaIceObs_native_2_netcdf.py"
python "./Import_NSIDC_Extents.py"

# Import Models to sipn format
source activate test_nio # Requires new env
python "./Regrid_YOPP.py"

source activate esio

wait # Below depends on above

# Make Plots
# Availblity plots
python "./plot_forecast_availability.py" &

# Observations
python "./plot_observations.py" &

# Models
python "./plot_all_model_maps.py" &
python "./plot_model_forecasts.py" &

echo Finished NRT script.
