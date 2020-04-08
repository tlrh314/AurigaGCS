#!/bin/bash

# module unload intel
# module unload impi
# module unload fftw-mpi/2.1.5
# module unload hdf5-mpi
# module unload gsl
# 
# unset LDFLAGS
# unset CFLAGS
# 
# module load gcc/9
# module load gsl
# export LD_LIBRARY_PATH=${GSL_HOME}/lib:${LD_LIBRARY_PATH}
# export CFLAGS="${CFLAGS} $(gsl-config --cflags)"
# export LDFLAGS="${LDFLAGS} $(gsl-config --libs)"

# FC=gfortran F77=gfortran CC=gcc CXX=gcc python setup.py install
export FC=ifort 
export F77=ifort 
export F90=ifort 
export CC=icc 
export CXX=icpc 
python setup.py build --compiler=intelem --fcompiler=intelem
python setup.py install
