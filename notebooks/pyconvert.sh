#!/bin/bash

#This code is part of the SIPN2 project focused on improving sub-seasonal to seasonal predictions of Arctic Sea Ice. 
#If you use this code for a publication or presentation, please cite the reference in the README.md on the
#main page (https://github.com/NicWayand/ESIO). 
#
#Questions or comments should be addressed to nicway@uw.edu
#
#Copyright (c) 2018 Nic Wayand
#
#GNU General Public License v3.0

# Get input jupyter notebook file name
f_ipy=$1

# Convert to .py
jupyter nbconvert --to python $f_ipy

# Strip out magic lines
f_py="${f_ipy%.*}".py
python strip_magic.py $f_py  

