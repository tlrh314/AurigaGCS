#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Example script to read in an Auriga simulation snapshot """

import os
import numpy
import matplotlib
from matplotlib import pyplot
pyplot.switch_backend('agg')

from read_auriga_snapshot import eat_snap_and_fof


__author__ = "Timo Halbesma"
__email__ = "halbesma@MPA-Garching.MPG.DE"


ZSUN = 0.0127

ELEMENTS = { 'H':0, 'He':1, 'C':2, 'N':3, 'O':4,
            'Ne':5, 'Mg':6, 'Si':7, 'Fe':8 }

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = { 'H':12.0, 'He':10.98, 'C':8.47, 'N':7.87, 'O':8.73,
                  'Ne':7.97, 'Mg':7.64, 'Si':7.55, 'Fe':7.54 }


def p2(a):
    return ((a)*(a))


def example_plot(s, sf):
    """ Generate arXiv:1708.03635 Fig. 1 (right), but for Auriga. The
        authors use observations of stars in RAVE-TGAS and assign stars
        to the 'halo' if |z| > 1.5 kpc, and to the disk if |z| < 1.5 kpc.
        Here z is the direction along the height of the disk. This plot
        shows a histogram of the [Fe/H] for the disk+halo. """

    # Make a cut of the stars
    istars, = numpy.where(
        # Example to cut out an anulus of |r-rsun| < 2 kpc where rsun=8kpc
        # (s.r() < 10./1000) & (s.r() > 6./1000) & (s.age > 0.)

        # Criteria to select stars within the galactic radius that are
        # associated with the main galaxy
        (s.r() < s.galrad) & (s.r() > 0.) & (s.age > 0.)
        & (s.type == 4) & (s.halo == s.haloid)
    )

    # Criteria to select |x| > 1.5 kpc. The x-direction seems to be
    # the direction along the height of the disk. Note that I use
    # 1.5/1000. This is because the code internal unit length is Mpc.
    outside_disk, = numpy.where(numpy.abs(s.pos[::,0]) > 1.5/1000)
    inside_disk, = numpy.where(numpy.abs(s.pos[::,0]) < 1.5/1000)

    halo = numpy.intersect1d(istars, outside_disk)
    disk = numpy.intersect1d(istars, inside_disk)

    # Clean negative and zero values of gmet to avoid RuntimeErrors later on
    # (e.g. dividing by zero)
    s.data['gmet'] = numpy.maximum( s.data['gmet'], 1e-40 )

    # Here we compute [Fe/H] for the 'halo'. See arXiv:1708.03635 why
    # we adopt this criterion for disk and halo. Anyway, the subgrid model
    # tracks 9 species, see the global variable ELEMENTS. Here we select
    # Fe and H from the data, then scale to Solar.
    metal_halo = numpy.zeros( [numpy.size(halo),2] )
    metal_halo[:, 0] = s.data['gmet'][halo][:, ELEMENTS['Fe']]
    metal_halo[:, 1] = s.data['gmet'][halo][:, ELEMENTS['H']]
    feabund_halo = numpy.log10( metal_halo[:,0] / metal_halo[:,1] / 56. ) - \
        (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
    # Mask to use the bins in the histogram in the range -3, 2.
    mask_halo, = numpy.where((feabund_halo > -3) & (feabund_halo < 2))

    # Repeat, but for a different cut in star particles (the 'disk').
    metal_disk = numpy.zeros( [numpy.size(disk),2] )
    metal_disk[:, 0] = s.data['gmet'][disk][:, ELEMENTS['Fe']]
    metal_disk[:, 1] = s.data['gmet'][disk][:, ELEMENTS['H']]
    feabund_disk = numpy.log10( metal_disk[:,0] / metal_disk[:,1] / 56. ) - \
        (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])
    mask_disk, = numpy.where((feabund_disk > -3) & (feabund_disk < 2))

    # Plot in the same style as the paper.
    fig, ax = pyplot.subplots()

    pyplot.hist(feabund_disk[mask_disk], bins=64, alpha=0.5, normed=True,
        color="blue", label="|x| < 1.5 kpc")
    pyplot.hist(feabund_halo[mask_halo], bins=64, alpha=0.5, normed=True,
        color="red", label="|x| > 1.5 kpc")
    ax.set_xlim(-3.1, 0.95)
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel(r"$f($[Fe/H]$)$")

    ax.text(0.01, 0.94, "Au{0}-{1} Star [Fe/H] Distribution"\
            .format(s.level, s.halo_number), weight="bold",
            fontsize=16, transform=pyplot.gca().transAxes)
    pyplot.legend(loc="center left", frameon=False)
    pyplot.show()
    pyplot.savefig(
        "Au{0}-{1}_{2:03d}_disk-vs-halo_metal_histogram.pdf"\
        .format(s.level, s.halo_number, s.snapnr))
    pyplot.close()


if __name__ == "__main__":
    print("Reading in an Auriga simulation snapshot.\n")

    level = 5
    basedir = "/Volumes/Cygnus"
    basedir += "/hits/universe/GigaGalaxy/level{0}{1}/".format(level, "_MHD" if level == 4 else "")

    if "freya" in os.uname().nodename:
        print("Freya")
        basedir = "/virgo/data/Auriga/"
        basedir += "level{0}_MHD/".format(level)
    print("basedir = {0}".format(basedir))
    for halo_number in [24]:  # range(1, 31):
        if "freya" in os.uname().nodename:
            halodir = basedir+"halo_{0}/".format(halo_number)
        else:
            halodir = basedir+"halo{0}{1}/".format("_" if level == 4 else "", halo_number)
        snappath = halodir+"output/"
        for snapnr in range(63, 64, 1):
            print("level   : {0}".format(level))
            print("halo    : {0}".format(halo_number))
            print("snapnr  : {0}".format(snapnr))
            print("basedir : {0}".format(basedir))
            print("halodir : {0}".format(halodir))
            print("snappath: {0}\n".format(snappath))

            s, sf = eat_snap_and_fof(level, halo_number, snapnr, snappath)
            example_plot(s, sf)
