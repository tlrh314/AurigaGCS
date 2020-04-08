#!/bin/bash

module load git
module load anaconda/3/2019.03
module load intel
module load mkl
module load impi
module unload fftw-mpi/2.1.5
module load fftw-mpi  # not needed for this repo
module load hdf5-mpi  # not needed for this repo
module load gsl
module load cmake  # not needed for this repo
module load ffmpeg
export LD_LIBRARY_PATH=${GSL_HOME}/lib:${LD_LIBRARY_PATH}
export CFLAGS="${CFLAGS} $(gsl-config --cflags)"
export LDFLAGS="${LDFLAGS} $(gsl-config --libs)"

module list
# Currently Loaded Modulefiles:
#   1) git/2.16             6) fftw-mpi/3.3.8
#   2) anaconda/3/2019.03   7) hdf5-mpi/1.8.21
#   3) intel/19.0.5         8) gsl/2.4
#   4) mkl/2019.5           9) cmake/3.13
#   5) impi/2019.5         10) ffmpeg/3.4

# Alternatively, use gcc -> FC=gfortran F77=gfortran CC=gcc CXX=gcc 
# module load gcc/9
# module load gsl
# export LD_LIBRARY_PATH=${GSL_HOME}/lib:${LD_LIBRARY_PATH}
# export CFLAGS="${CFLAGS} $(gsl-config --cflags)"
# export LDFLAGS="${LDFLAGS} $(gsl-config --libs)"

export FC=ifort 
export F77=ifort 
export F90=ifort 
export CC=icc 
export CXX=icpc 
python setup.py build --compiler=intelem --fcompiler=intelem
python setup.py install
