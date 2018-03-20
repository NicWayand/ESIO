#!/bin/bash

# Assumes were already in the esio conda env

# Get input file name
f_ipy=$1

# Convert to .py
jupyter nbconvert --to python $f_ipy

# Strip out magic lines
which python
f_py="${f_ipy%.*}".py
python strip_magic.py $f_py  

