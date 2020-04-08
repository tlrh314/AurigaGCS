# -*- coding: utf-8 -*-

"""
Author: James Schombert's Python implementation of Wright's cosmology calculator
Obtained from http://www.astro.ucla.edu/~wright/CC.python at May 10, 2016
Copyright for the original (online) calculator 1999-2016 Edward L. Wright.
If you use this calculator while preparing a paper, please cite Wright (2006, PASP, 118, 1711).

Some additions/alterations were made by TLRH
"""
#!/usr/bin/env python

import sys
import numpy


class CosmologyCalculator(object):
    def __init__(self, z=0.0562, H0=70, WM=0.3, WV=0.7):
        # initialize constants

        WR = 0.        # Omega(radiation)
        WK = 0.        # Omega curvaturve = 1-Omega(total)
        c = 299792.458 # velocity of light in km/sec
        Tyr = 977.8    # coefficent for converting 1/H into Gyr
        DTT = 0.5      # time from z to now in units of 1/H0
        DTT_Gyr = 0.0  # value of DTT in Gyr
        age = 0.5      # age of Universe in units of 1/H0
        age_Gyr = 0.0  # value of age in Gyr
        zage = 0.1     # age of Universe at redshift z in units of 1/H0
        zage_Gyr = 0.0 # value of zage in Gyr
        DCMR = 0.0     # comoving radial distance in units of c/H0
        DCMR_Mpc = 0.0
        DCMR_Gyr = 0.0
        DA = 0.0       # angular size distance
        DA_Mpc = 0.0
        DA_Gyr = 0.0
        kpc_DA = 0.0
        DL = 0.0       # luminosity distance
        DL_Mpc = 0.0
        DL_Gyr = 0.0   # DL in units of billions of light years
        V_Gpc = 0.0
        a = 1.0        # 1/(1+z), the scale factor of the Universe
        az = 0.5       # 1/(1+z(object))

        self.h = H0/100.
        WR = 4.165E-5/(self.h*self.h)   # includes 3 massless neutrino species, T0 = 2.72528
        WK = 1-WM-WR-WV
        az = 1.0/(1+1.0*z)
        age = 0.
        n=1000         # number of points in integrals
        for i in range(n):
            a = az*(i+0.5)/n
            adot = numpy.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
            age = age + 1./adot

        zage = az*age/n
        zage_Gyr = (Tyr/H0)*zage
        DTT = 0.0
        DCMR = 0.0

        # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
        for i in range(n):
            a = az+(1-az)*(i+0.5)/n
            adot = numpy.sqrt(WK+(WM/a)+(WR/(a*a))+(WV*a*a))
            DTT = DTT + 1./adot
            DCMR = DCMR + 1./(a*adot)

        DTT = (1.-az)*DTT/n
        DCMR = (1.-az)*DCMR/n
        age = DTT+zage
        age_Gyr = age*(Tyr/H0)
        DTT_Gyr = (Tyr/H0)*DTT
        DCMR_Gyr = (Tyr/H0)*DCMR
        DCMR_Mpc = (c/H0)*DCMR

      # tangential comoving distance

        ratio = 1.00
        x = numpy.sqrt(abs(WK))*DCMR
        if x > 0.1:
            if WK > 0:
                ratio =  0.5*(numpy.exp(x)-numpy.exp(-x))/x
            else:
                ratio = numpy.sin(x)/x
        else:
            y = x*x
            if WK < 0: y = -y
            ratio = 1. + y/6. + y*y/120.
        DCMT = ratio*DCMR
        DA = az*DCMT
        DA_Mpc = (c/H0)*DA
        kpc_DA = DA_Mpc/206.264806
        DA_Gyr = (Tyr/H0)*DA
        DL = DA/(az*az)
        DL_Mpc = (c/H0)*DL
        DL_Gyr = (Tyr/H0)*DL

        # comoving volume computation

        ratio = 1.00
        x = numpy.sqrt(abs(WK))*DCMR
        if x > 0.1:
            if WK > 0:
                ratio = (0.125*(numpy.exp(2.*x)-numpy.exp(-2.*x))-x/2.)/(x*x*x/3.)
            else:
                ratio = (x/2. - numpy.sin(2.*x)/4.)/(x*x*x/3.)
        else:
            y = x*x
            if WK < 0: y = -y
            ratio = 1. + y/5. + (2./105.)*y*y
        VCM = ratio*DCMR*DCMR*DCMR/3.
        V_Gpc = 4.*numpy.pi*((0.001*c/H0)**3)*VCM

        # TODO: fix this quick'n'dirty fix
        self.H0, self.WM, self.WV, self.z, self.age_Gyr,\
            self.zage_Gyr, self.DTT_Gyr, self.DCMR_Mpc, self.DCMR_Gyr,\
            self.V_Gpc, self.DA_Mpc, self.DA_Gyr, self.kpc_DA, self.DL_Mpc,\
            self.DL_Gyr\
            = H0, WM, WV, z, age_Gyr,\
            zage_Gyr, DTT_Gyr, DCMR_Mpc, DCMR_Gyr,\
            V_Gpc, DA_Mpc, DA_Gyr, kpc_DA, DL_Mpc,\
            DL_Gyr

    # @property
    # def Hubble_of_z(self):
    #     """ Hubble constant as a function of redshift """
    #     units.H0 = units.named('H0', 'km/s/Mpc', (1 | units.km / units.s / units.Mpc).to_unit())

    #     return (self.H0 | units.H0) * numpy.sqrt(self.WM*(1+self.z)**3 + self.WV)

    # def rho_crit(self):
    #     """ Critical density of the Universe as a function of redshift """
    #     rho_crit = 3 * self.Hubble_of_z**2 / (8 * numpy.pi * constants.G)
    #     return rho_crit.value_in(units.g/units.cm**3)

    def __str__(self):
        # TODO: fix this function such that it returns a proper string
        verbose = 1
        if verbose:
            print("Ned Wright's Cosmology Calculator with alterations by TLRH\n")
            print("H_0 = {H0:1.1f}, Omega_M = {WM:1.3f},".format(**{'H0': self.H0, 'WM': self.WM}),)
            print("Omega_vac = {WV:1.3f}, z = {z:1.3f}\n".format(**{'WV': self.WV, 'z': self.z}))
            print("It is now {0:1.3f} Gyr since the Big Bang.".format(self.age_Gyr))
            print("The age at redshift z was {0:1.3f} Gyr.".format(self.zage_Gyr))
            print("The light travel time was {0:1.3f} Gyr.".format(self.DTT_Gyr))
            print("The comoving radial distance, which goes into Hubbles law, is",)
            print("{0:1.1f} Mpc or {1:1.3f} Gly.".format(self.DCMR_Mpc, self.DCMR_Gyr))
            print("The comoving volume within redshift z is {0:1.3f} Gpc^3.".format(self.V_Gpc))
            print("The angular size distance D_A is {0:1.1f} Mpc or {1:1.4f} Gly.".format(self.DA_Mpc, self.DA_Gyr))
            print("This gives a scale of {0:.3f} kpc/\".".format(self.kpc_DA))
            print("The luminosity distance D_L is {0:1.1f} Mpc or {1:1.3f} Gly.".format(self.DL_Mpc, self.DL_Gyr))
            print("The distance modulus, m-M, is {0:1.2f}".format(5*numpy.log10(self.DL_Mpc*1e6)-5))
            # print("Critical density is {0:1.5e} g/cm**3".format(self.rho_crit()))
        else:
            print("Ned Wright's Cosmology Calculator with alterations by TLRH\n")
            print("H_0 = {H0:1.1f}, Omega_M = {WM:1.3f},".format(**{'H0': self.H0, 'WM': self.WM}),)
            print("Omega_vac = {WV:1.3f}, z = {z:1.3f}\n".format(**{'WV': self.WV, 'z': self.z}))
            print("Redshift age             = {0:1.2f} Gyr".format(self.zage_Gyr))
            print("Comoving radial distance = {0:1.2f} Mpc".format(self.DCMR_Mpc))
            print("kpc_DA                   = {0:1.2f} kpc/\"".format(self.kpc_DA))
            print("Distance modulus, m-M    = {0:1.2f}".format((5*numpy.log10(self.DL_Mpc*1e6)-5)))
            # print("Critical density is {0:1.5e g/cm**3}".format(self.rho_crit()))

        return ""


def calculate_rvir(s, sf, z=0.0, delta=200, cosmo="PlanckXVI", verbose=False, debug=False):
    """ Calculate the virial radius of an Auriga galaxy using the
        bisection method to converge to Mvir, rvir.

        There are three 'modes' of calculation: mean, critical, and tophat.
          mean: standard fixed (delta) overdensity with respect to background
          critical: overdensity is delta relative to rho_crit, expressed relative to background
          tophat: overdensity is tophat with respect to background
        Returns: ([Mvir_mean, Mvir_crit, Mvir_tophat], [rvir_mean, rvir_crit, rvir_tophat])

        Note that Auriga uses Planck XVI, but we want to adopt rvir from Patel+2017,
        work based on Illustris where WMAP9 is used, so we swap cosmological parameters. """

    from astropy import constants
    from astropy import units as u
    if verbose: print("Bisecting to find Rvir, Mvir for {0} cosmology".format(cosmo))

    if cosmo == "PlanckXVI":
        cosmo = s.cosmo
    elif cosmo == "WMAP9":
        from areposnap.cosmological_factors import CosmologicalFactors
        WMAP9 = CosmologicalFactors(my_h=70.4/100, my_OmegaMatter=0.2726, my_OmegaLambda=0.7274)
        cosmo = WMAP9
    else:
        print("ERROR: unknown cosmology '{0}'. Please use PlanckXVI or WMAP9 instead.".format(cosmo))
        return

    # https://ned.ipac.caltech.edu/level5/Sept11/Norman/Norman3.html
    omega_0 = cosmo.Get_OmegaM()
    omega_lambda = cosmo.Get_OmegaLambda()
    H0 = cosmo.HubbleConstant_z(0) * cosmo.Get_h() * u.km/u.s/u.Mpc
    rho_background = 3 * omega_0 * H0**2 / (8 * numpy.pi * constants.G)
    omega_z = omega_0 * (1+z)**3 / (omega_0 *(1+z)**3 + (1-omega_0-omega_lambda)*(1+z)**2 + omega_lambda )
    x = omega_z - 1

    delta_mean = delta
    delta_crit = delta / omega_z
    delta_tophat = (18*numpy.pi**2 + 82*x - 39*x**2) / omega_z

    if verbose:
        print("  H0: {0:.2f} km/s/Mpc".format(H0.value))
        print("  rho_crit = {0:.2e} g/cm^3".format(rho_background.to(u.g/u.cm**3).value / omega_0))
        print("  rho_background = {0:.2e} g/cm^3".format(rho_background.to(u.g/u.cm**3).value))
        print("  delta_mean = {0:.2f}".format(delta_mean))
        print("  delta_crit = {0:.2f}".format(delta_crit))
        print("  delta_tophat = {0:.2f}".format(delta_tophat))

    rho_background = rho_background.to(u.M_sun / u.kpc**3).value
    if verbose: print("  rho_background = {0:.2e} Msun/kpc^3\n".format(rho_background))

    Mvirs, rvirs = list(), list()
    for delta in [delta_mean, delta_crit, delta_tophat]:
        # Bisection method to converge to Rvir

        lower = 10    # kpc
        upper = 4000  # kpc
        epsilon = 0.001
        while upper / lower > 1+epsilon:
            rvir = (lower + upper) / 2.  # kpc

            isphere, = numpy.where(1e3*s.r() < rvir)
            Mvir = 1e10 * s.mass[isphere].sum()  # Msun

            overdensity = ( Mvir / (4./3 * numpy.pi * (rvir)**3)) / rho_background

            if debug:
                print("\n  lower: {0} kpc".format(lower))
                print("  upper: {0} kpc".format(upper))
                print("  rvir: {0:.2f} kpc".format(rvir))
                print("  Mvir: {0:.2e} Msun".format(Mvir))
                print("  overdensity: {0:.2f}".format(overdensity))

            if overdensity < delta:
                upper = rvir
            if overdensity > delta:
                lower = rvir

        Mvirs.append(Mvir)
        rvirs.append(rvir)

    return Mvirs, rvirs


if __name__ == '__main__':
    try:
        usage = """Cosmology calculator ala Ned Wright (www.astro.ucla.edu/~wright)
input values = redshift, Ho, Omega_m, Omega_vac
ouput values = age at z, distance in Mpc, kpc/arcsec, apparent to abs mag conversion

Options:   -h for this message
           -v for verbose response """
        if sys.argv[1] == "-h":
            print(usage)
            sys.exit()
        if sys.argv[1] == "-v":
            verbose=1
            length=len(sys.argv)-1
        else:
            verbose=0
            length=len(sys.argv)

        # if no values, assume Benchmark Model input is z
        if length == 2:
            if float(sys.argv[1+verbose]) > 100:
                z=float(sys.argv[1+verbose])/299792.458  # velocity to redshift
            else:
                z=float(sys.argv[1+verbose])             # redshift
            # Benchmark Model (?)
            benchmark = False
            if benchmark:
                H0 = 75                         # Hubble constant
                WM = 0.3                        # Omega(matter)
                WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda
            donnert_2014 = True
            if donnert_2014:
                H0 = 70                         # Hubble constant
                WM = 0.3                        # Omega(matter)
                WV = 0.7                        # Omega(vacuum) or lambda
                s8 = 0.9                        # TODO: find out what this is
            wise_2016_in_prep = False
            if wise_2016_in_prep:
                # Komatsu et al. 2009
                H0 = 71                         # Hubble constant
                WM = 0.27                       # Omega(matter)
                WV = 0.73                       # Omega(vacuum) or lambda
                # z  = 0.0562                   # CygA, Stockton et al. 1994

        # if one value, assume Benchmark Model with given Ho
        elif length == 3:
            z=float(sys.argv[1+verbose])    # redshift
            H0 = float(sys.argv[2+verbose]) # Hubble constant
            WM = 0.3                        # Omega(matter)
            WV = 1.0 - WM - 0.4165/(H0*H0)  # Omega(vacuum) or lambda

        # if Univ is Open, use Ho, Wm and set Wv to 0.
        elif length == 4:
            z=float(sys.argv[1+verbose])    # redshift
            H0 = float(sys.argv[2+verbose]) # Hubble constant
            WM = float(sys.argv[3+verbose]) # Omega(matter)
            WV = 0.0                        # Omega(vacuum) or lambda

        # if Univ is General, use Ho, Wm and given Wv
        elif length == 5:
            z=float(sys.argv[1+verbose])    # redshift
            H0 = float(sys.argv[2+verbose]) # Hubble constant
            WM = float(sys.argv[3+verbose]) # Omega(matter)
            WV = float(sys.argv[4+verbose]) # Omega(vacuum) or lambda

      # or else fail
        else:
            print(usage)
            print("\nError: need some values or too many values")
            sys.exit()

        # All parameters are fine. Calculate stuff
        cc = CosmologyCalculator(z, H0, WM, WV)
        print(cc)

    except IndexError:
        print(usage)
        print("\nError: need some values or too many values")
    except ValueError:
        print(usage)
        print("\nError: nonsense value or option")


    from main import *
    auriga = AurigaOverview(verbose=False)
    run = auriga.getrun(5, 24)
    s, sf = run.load_snapshot(loadonlytype=range(6), verbose=False)

    # fmm2, frm2  = Group_M_Mean200, Group_R_Mean200
    # fmc2, frc2  = Group_M_Crit200, Group_R_Crit200
    # fmt2, frt2  = Group_M_TopHat200, Group_R_TopHat200
    print("\n{0} --> SUBFIND\nmean   M200, r200 = {1:.2e} MSun, {2:.2f} kpc [Planck XVI]".format(
        run.name, 1e10 * sf.data["fmm2"][0], 1e3 * sf.data["frm2"][0]))
    print("crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
        1e10 * sf.data["fmc2"][0], 1e3 * sf.data["frc2"][0]))
    print("tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]\n".format(
        1e10 * sf.data["fmt2"][0], 1e3 * sf.data["frt2"][0]))

    Mvirs, rvirs = calculate_rvir(s, sf, delta=200, cosmo="PlanckXVI", verbose=True, debug=False)
    print("mean   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(Mvirs[0], rvirs[0]))
    print("crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(Mvirs[1], rvirs[1]))
    print("tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]\n".format(Mvirs[2], rvirs[2]))

    Mvirs, rvirs = calculate_rvir(s, sf, delta=200, cosmo="WMAP9", verbose=True, debug=False)
    print("mean   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [WMAP9]".format(Mvirs[0], rvirs[0]))
    print("crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [WMAP9]".format(Mvirs[1], rvirs[1]))
    print("tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [WMAP9]".format(Mvirs[2], rvirs[2]))
