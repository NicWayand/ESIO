#!/bin/bash

source activate esio

pyconvert.sh Import_NSIDC_Extents.ipynb
pyconvert.sh Import_NSIDC_Obs.ipynb
pyconvert.sh Regrid_S2S_Models.ipynb
pyconvert.sh Regrid_YOPP.ipynb
pyconvert.sh plot_Extent_Model_Obs.ipynb
pyconvert.sh plot_all_model_maps.ipynb
pyconvert.sh plot_forecast_availability.ipynb
pyconvert.sh plot_observations.ipynb
pyconvert.sh Calc_adjusted_extents.ipynb
pyconvert.sh plot_Regional_maps.ipynb
