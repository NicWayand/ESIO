#!/bin/bash

inputDir=$1
dateinit=$2 # '2018-03-25'

## declare an array variable
declare -a metrics=("mean" "anomaly" "SIP")

## now loop through the above array
for cm in "${metrics[@]}"
do
	prefix=panArctic_${cm}_forecast_${dateinit}
	convert -delay 60 ${inputDir}/${prefix}*.png ${inputDir}/${prefix}_99.gif
done
