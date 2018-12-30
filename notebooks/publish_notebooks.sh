#!/bin/bash

source activate esio

pyconvert.sh Import_NSIDC_Extents.ipynb
pyconvert.sh Import_NSIDC_Obs.ipynb
pyconvert.sh Regrid_S2S_Models.ipynb
pyconvert.sh Regrid_YOPP.ipynb
pyconvert.sh plot_Extent_Model_Obs.ipynb
pyconvert.sh plot_forecast_availability.ipynb
pyconvert.sh plot_observations.ipynb
pyconvert.sh Calc_adjusted_extents.ipynb
pyconvert.sh plot_Regional_maps.ipynb
pyconvert.sh Regrid_NESM.ipynb
pyconvert.sh Regrid_RASM.ipynb
pyconvert.sh plot_Regional_Extent.ipynb
pyconvert.sh Calc_Model_Aggregations.ipynb
pyconvert.sh Regrid_CFSv2.ipynb
pyconvert.sh Agg_NSIDC_Obs.ipynb
pyconvert.sh Model_Damped_Anomaly_Persistence.ipynb
pyconvert.sh Calc_Weekly_Model_Metrics.ipynb
pyconvert.sh plot_Maps_Fast_from_database.ipynb
pyconvert.sh Convert_netcdf_to_Zarr.ipynb
pyconvert.sh Regrid_GFDL_Forecast.ipynb
pyconvert.sh Eval_weekly_forecasts.ipynb
pyconvert.sh Agg_Weekly_to_Zarr.ipynb
