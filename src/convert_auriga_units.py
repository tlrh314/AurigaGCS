#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Unit Conversions for AREPO/Auriga Simulations """

import numpy

__author__ = "Timo Halbesma"
__email__ = "halbesma@MPA-Garching.MPG.DE"


ZSUN = 0.0127

ELEMENTS = { 'H':0, 'He':1, 'C':2, 'N':3, 'O':4,
            'Ne':5, 'Mg':6, 'Si':7, 'Fe':8 }

# from Asplund et al. (2009) Table 5
SUNABUNDANCES = { 'H':12.0, 'He':10.98, 'C':8.47, 'N':7.87, 'O':8.73,
                  'Ne':7.97, 'Mg':7.64, 'Si':7.55, 'Fe':7.54 }


def compute_iron_over_hydrogen_abundance(s, sf, mask, debug=False):
    metal = numpy.zeros( [numpy.size(mask),2] )
    gmet = numpy.zeros((s.npartall, 9))

    if debug: print("gmet: {0}".format(gmet.shape))
    gas_start = 0
    gas_end = s.nparticlesall[0]
    if debug: print("gas: {0} {1}".format(gas_start, gas_end))
    gmet[gas_start:gas_end] = s.gmet[0:gas_end]

    stars_start = s.nparticlesall[:4].sum()
    stars_end = int(stars_start + s.nparticlesall[4])
    if debug: print("gas: {0} {1}".format(stars_start, stars_end))
    gmet[stars_start:stars_end] = s.gmet[gas_end:]

    metal[:, 0] = gmet[mask][:, ELEMENTS['Fe']]
    metal[:, 1] = gmet[mask][:, ELEMENTS['H']]
    feabund = numpy.log10( metal[:,0] / metal[:,1] / 56. ) - \
        (SUNABUNDANCES['Fe'] - SUNABUNDANCES['H'])

    return feabund


if __name__ == "__main__":
    from read_auriga_snapshot import default_setup

    # gmet is only available for gas and stars. Here we just test if loading
    # arbitrary combinations of particles breaks the function call...
    for loadonlytype in [
                [4], [0, 1, 2, 3, 4, 5], [4, 5], [1, 2, 3, 4, 5],
                [0, 4, 5], [0, 3, 4]
            ]:

        s, sf, istars, idm = default_setup(loadonlytype=loadonlytype)

        istars, = numpy.where(
            (s.type == 4) & (s.halo == 0)
            & (s.r() > 0.) & (s.r() < s.galrad)
        )

        feabund = compute_iron_over_hydrogen_abundance(s, sf, istars)
