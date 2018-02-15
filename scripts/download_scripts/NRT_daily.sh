#/bin/bash

# Call all download scripts that grab near-real-time data
$REPO_DIR"/scripts/download_scripts/download_NSIDC_0081.sh"

wait

# Call python scripts to convert native format to spin_nc format
# TODO

echo Finished NRT daily downloads.
