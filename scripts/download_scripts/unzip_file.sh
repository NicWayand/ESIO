#!/bin/bash

ncdir=$1
prefix=$2
zdir=$ncdir'/zipped_files/'$prefix'*'

for f in $zdir
do
    #echo $f
    tar -xzvf $f --strip=2 -C $ncdir
done
