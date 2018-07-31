#!/bin/bash

ncdir=$1
prefix=$2
zdir=$ncdir'/zipped_files/'$prefix'*'
unzippeddir=$ncdir'/download_archive'
mkdir -p $unzippeddir

for f in $zdir
do
    fileN=$(basename $f)
    if [ ! -f ${unzippeddir}'/'${fileN} ]; then
   
        tar --skip-old-files -xzvf $f -C $ncdir

        # Keep track of unzipped files by making empty file in new dir
        touch ${unzippeddir}'/'${fileN}
    fi

done
