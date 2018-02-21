#/bin/bash

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh"

wait

# Call python scripts to convert native format to spin_nc format
source $HOME/.bashrc
source activate xesmf
# Convert binary to sipn netcdf format
pyhton $HOME"/python/ESIO/notebooks/SeaIceObs_native_2_netcdf.p"
# Make some plots
python $HOME"/python/ESIO/notebooks/plot_pan_arctic_extent_Forecast.py"

echo Finished NRT daily downloads.
