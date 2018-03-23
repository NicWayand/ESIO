#!/bin/bash

# Get input jupyter notebook file name
f_ipy=$1

# Convert to .py
jupyter nbconvert --to python $f_ipy

# Strip out magic lines
f_py="${f_ipy%.*}".py
python strip_magic.py $f_py  

