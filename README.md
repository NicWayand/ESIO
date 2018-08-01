# Extended Sea Ice Outlook (ESIO)
## Part of the Sea ice Prediction Network Phase 2 (SIPN2)

[![Build Status](https://travis-ci.org/NicWayand/ESIO.svg?branch=master)](https://travis-ci.org/NicWayand/ESIO) [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/NicWayand/ESIO/master)




Collection of scripts for running the [SIPN2 Portal](http://www.atmos.uw.edu/sipn)

<p align="center">
  <img src="https://atmos.washington.edu/sipn/figures/model/all_model/sic/timeseries/panArctic_extent_forecast_raw_predicted.png?342038402" width="350"/>
</p>

- [notebooks](./notebooks/) contain ipython notebooks and converted .py files for:
  - Regridding Model outputs to the [NSIDC Polar Sterographic 25km grid](https://nsidc.org/data/polar-stereo/ps_grids.html)
  - Calculating pan-Arctic and regional extent, area, etc. 
  - Evaluating model forecasts against observations
  - Creating maps and timeseries plots for website
  
- [scripts](./scripts/) contains a collection of bash shell and python scripts for:
  - Downloading observations
  - Downloading sea ice model forecasts.
  
  #### Project is currently under development (so things will change) and contributions are encouraged! 
