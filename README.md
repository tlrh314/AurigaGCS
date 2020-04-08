# auriga
- https://ui.adsabs.harvard.edu/abs/2019arXiv190902630H/abstract
- https://arxiv.org/pdf/1909.02630.pdf

# Setup at [Freya](https://www.mpcdf.mpg.de/services/computing/linux/Astrophysics)
## Loaded in bashrc
- `module load git`
- `module load anaconda/3_5.3.0`
- `module load cmake`
- `module load ffmpeg`

## Installation of conda env
- `conda create -n auriga python=3`
- latest: `pip install -r requirements.txt`
- exact: `pip install -r requirements_full.txt`

## Installation of the Arepo-snap-util
- [Get the repo](https://bitbucket.org/federico_marinacci/arepo-snap-util/src/master/)
- We want to namespace the above package to prevent clashing file names the site packages.
- Clone the repo into a folder `path/to/areposnap/src`
- `cp setup_for_areposnap.py path/to/areposnap/setup.py`
- `cp install_areposnap_on_freya.sh path/to/areposnap/install.sh`
- `cp init_for_areposnap.py path/to/areposnap/src/__init__.py`
- `cd path/to/areposnap`
- `./install.sh`
### Alternatively, when using GCC adjust the loaded modules and use something like
- `export CFLAGS="${CFLAGS} $(gsl-config --cflags)"`
- `export LDFLAGS="${LDFLAGS} $(gsl-config --libs)"`
- Change `-qopenmp` to `-fopenmp`
- `export LDFLAGS="$LDFLAGS -undefined dynamic_lookup -bundle"` 
    [on macOS](https://github.com/numpy/numpy/issues/7427)
- `FC=gfortran F77=gfortran CC=gcc CXX=gcc python setup.py install`

We use the following version of the Arepo-snap-util
```
commit 29334e6da69e9e22cb7d3731a9e35c30361d34c0 (HEAD -> master, origin/master, origin/brandon, origin/HEAD)
Author: astrowq <brandon@qwang.org>
Date:   Tue Feb 11 16:42:06 2020 +0100

    update gsl path
```
