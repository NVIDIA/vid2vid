#!/bin/bash
cd ./networks/correlation_package
python setup.py install --user
cd ../resample2d_package 
python setup.py install --user
cd ../channelnorm_package 
python setup.py install --user
cd ..
