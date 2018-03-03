#/bin/bash

#set -x  # Echo all lines executed
set -e  # Stop on any error

# Set up python paths
source $HOME/.bashrc
source activate esio

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh" &

# Model downloads
python $REPO_DIR"/scripts/download_scripts/Download_YOPP_ECMWF.py" &

wait

# Call python scripts to convert native format to spin_nc format
# Convert binary to sipn netcdf format
python $HOME"/python/ESIO/notebooks/SeaIceObs_native_2_netcdf.py"
# Make some plots
python $HOME"/python/ESIO/notebooks/plot_pan_arctic_extent_Forecast.py"

echo Finished NRT daily downloads.
