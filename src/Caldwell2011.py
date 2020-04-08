# Paper  II: http://adsabs.harvard.edu/abs/2011AJ....141...61C
#      data: http://vizier.cfa.harvard.edu/viz-bin/Cat?J/AJ/141/61
#        --> 323 GCs

# Vizier Data outdated / not up to date / does not allow to reproduce
# the plots in the paper. The website below provides a more recent version
# https://www.cfa.harvard.edu/oir/eg/m31clusters/M31_Hectospec.html

import os
import numpy
import scipy
from scipy import stats
from matplotlib import pyplot
from astropy import units as u
from astropy import coordinates as coord
from astropy.coordinates import Angle, Distance

from tlrh_util import suppress_stdout
from tlrh_util import calculate_M31_Rgc_Wang2019


def read_caldwell2011_data(verbose=False, debug=False):
    url = \
        "https://www.cfa.harvard.edu/oir/eg/m31clusters/M31_Hectospec_old_catalog.dat"
    fname = "../data/Caldwell2011/M31_Hectospec_old_catalog.dat"
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)

    if not os.path.isfile(fname):
        import urllib.request
        urllib.request.urlretrieve(url, fname)

    Ncols = 20 + 5
    nrows_header = 23
    Nrows = sum(1 for line in open(fname, "r")) - nrows_header

    # Default to shoving a float in the column, but change for known strings
    formats =[numpy.float for i in range(Ncols)]
    formats[0] = "S15"; formats[1] = "S15"; formats[2] = "S15";
    formats[11] = "S3"; formats[12] = "S3"; formats[16] = "S3";
    formats[16] = "S3"; formats[18] = "S3"; formats[19] = "S20"
    formats[-5] = "object"; formats[-4] = "object";

    # Name columns to access 'm by later-on
    names = [
        "Name ", "RA", "DEC", "X", "Y", "VELOCITY", "CZERR",
        "E(B-V)", "V", "V_err", "Ap", "Vel_source", "Mag_source",
        "Age", "[Fe/H]", "e_[Fe/H]", "[Fe/H]_source", "logM",
        "EBV_source", "COMMENTS",
        "SkyCoord", "ICRS", "Rproj", "Rdeproj", "Rgc",
    ]

    # Pack it all up, and initialise empty array
    dtype = { "names": names, "formats": formats }
    data = numpy.empty(Nrows, dtype=dtype)

    with open(fname, "rb") as f:
        for i, row in enumerate(f.readlines()):
            if i < nrows_header: continue
            if debug:
                print(row)
                print(data.dtype)
            # SkyCoord, ICRS, Rproj, Rdeproj not in data
            for j, value in enumerate(row.decode("ascii").split("\t")):
                value = value.strip()
                if debug: print(j, value)
                if j == 13 and value == "(14)": value = ""  # nan b/c len("") is 0
                if formats[j] == numpy.float:
                    if len(value) is 0:
                        value = numpy.nan
                    else:
                        value = float(value)
                if formats[j] == numpy.bytes_:
                    if len(value) is 0:
                        value = ""
                data[names[j]][i-nrows_header] = value

    for x in range(Nrows):
        data["SkyCoord"][x] = coord.SkyCoord(
            data["RA"][x].decode("ascii"),
            data["DEC"][x].decode("ascii"),
            frame="icrs", equinox="J2000", unit=(u.hourangle, u.deg)
        )

    # Here we follow sec 4.1 from Wang, Ma & Liu (2019, sec 4.1). arXiv 1901.11229v1
    # However, what they call deprojected galactocentric radius does not have units
    # kpc, so this must be an angle in units of radian. So we convert this angle
    # on the sky to kpc by using the distance to Andromeda galaxy.
    X, Y, Rproj = calculate_M31_Rgc_Wang2019(data["SkyCoord"],
        deproject=False, debug=False
    )
    data["Rproj"] = Rproj
    X, Y, Rdeproj = calculate_M31_Rgc_Wang2019(data["SkyCoord"],
        deproject=True, debug=False
    )
    data["Rdeproj"] = Rdeproj

    # So if Rproj = Rgc × (π/4), as per Huxor (2014, Fig. 17), then also
    # Rgc = Rproj / (π/4). Thus I can compare MW Rgc = sqrt((X-8)^2 + Y^2 + Z^2)
    # to M31 Rgc calculated from Rproj (that was obtained by plugging in ra, dec,
    # center of M31 and distance to M31). Subsequently the M31 and MW radial
    # distributions can be compared to the Auriga simulations, r() which calculates
    # sqrt(x^2 + y^2 + z^2) :-)
    data["Rgc"] = data["Rproj"] / (numpy.pi/4)

    return data


def print_caldwell2011_data(data, example=True):
    width = 200

    print("\n{0}".format("-"*width))
    print("{:<15s}{:^15s}{:^15s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}".format(
        "Name", "RA", "DEC", "X", "Y", "VELOCITY", "CZERR", "E(B-V)"), end="")
    print("{:^8s}{:^8s}{:^8s}{:^12s}{:^12s}{:^8s}{:^8s}".format(
        "V", "V_err", "Ap", "Vel_source", "Mag_source", "Age", "[Fe/H]"), end="")
    print("{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}{:^8s}".format(
        "e_[Fe/H]", "FeH_source", "logM", "EBV_source",
        "Rproj", "Rdeproj"))
    print("{0}".format("-"*width))
    for i, row in enumerate(data):
        print("{:<15s}{:^15s}{:^15s}{:^8.2f}{:^8.2f}{:^8.2f}{:^8.2f}{:^8.2f}".format(
            row[0].decode("ascii"), row[1].decode("ascii"), row[2].decode("ascii"),
            row[3], row[4], row[5], row[6], row[7]), end="")
        print("{:^8.2f}{:^8.2f}{:^8.2f}{:^12s}{:^12s}{:^8.2f}{:^8.2f}".format(
            row[8], row[9], row[10], row[11].decode("ascii"),
            row[12].decode("ascii"), row[13], row[14]), end="")
        print("{:^8.2f}{:^8s}{:^8.2f}{:^8s}{:^8.2f}{:^8.2f}".format(
            row[15], row[16].decode("ascii"), row[17],
            row[18].decode("ascii"), # row[19].decode("ascii"),
            row[-2], row[-1]))
        if example and i > 3: break
    print("{0}\n".format("-"*width))


def read_caldwell2011_data_from_vizier(fname="../data/Caldwell2011/table1.dat",
        verbose=False, debug=False):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)
    Ncols = 13 + 2 + 2
    Nrows = sum(1 for line in open(fname, "r"))

    # Default to float in column, but change for known strings
    formats =[numpy.float for i in range(Ncols)]
    formats[0] = "S10"; formats[1] = "S1"; formats[2] = "S13"; formats[3] = "S13"
    formats[5] = "S1"; formats[12] = "S1"; formats[13] = "object";
    formats[14] = "object"; formats[15] = "f8"; formats[16] = "f8"

    # Name columns to access 'm by later-on
    names = [
        "Name ", "n_Name", "RA", "DEC", "E(B-V)", ")", "Vel", "e_Vel",
        "logM", "[Fe/H]", "e_[Fe/H]", "Age", "Notes",
        "SkyCoord", "ICRS", "Rproj", "Rdeproj"
    ]

    # Pack it all up, and initialise empty array
    dtype = { "names": names, "formats": formats }
    data = numpy.empty(Nrows, dtype=dtype)

    # Start and end indices from catalog's ReadMe file
    cs = [
        1, 11, 13, 23, 35, 39, 41, 48, 51, 55, 60, 64, 69
    ]
    ce = [
        10, 12, 22, 33, 38, 40, 46, 49, 53, 58, 62, 67, 81
    ]

    with open(fname, "rb") as f:
        for i, row in enumerate(f.readlines()):
            # SkyCoord, ICRS, Rproj, Rdeproj not in data
            for j in range(Ncols - 4):
                value = row[cs[j]-1:ce[j]].strip()
                if formats[j] == numpy.float:
                    if len(value) is 0:
                        value = numpy.nan
                    else:
                        value = float(value)
                if formats[j] == numpy.bytes_:
                    if len(value) is 0:
                        value = ""
                data[names[j]][i] = value

    for x in range(Nrows):
        data["SkyCoord"][x] = coord.SkyCoord(
            data["RA"][x].decode("ascii"),
            data["DEC"][x].decode("ascii"),
            frame="icrs", equinox="J2000", unit=(u.hourangle, u.deg)
        )

    # Here we follow sec 4.1 from Wang, Ma & Liu (2019, sec 4.1). arXiv 1901.11229v1
    # However, what they call deprojected galactocentric radius does not have units
    # kpc, so this must be an angle in units of radian. So we convert this angle
    # on the sky to kpc by using the distance to Andromeda galaxy.
    X, Y, Rproj = calculate_M31_Rgc_Wang2019(data["SkyCoord"],
        deproject=False, debug=False
    )
    data["Rproj"] = Rproj
    X, Y, Rdeproj = calculate_M31_Rgc_Wang2019(data["SkyCoord"],
        deproject=True, debug=False
    )
    data["Rdeproj"] = Rdeproj

    with suppress_stdout():
        has_no_age, = numpy.where( data["Age"] == 14.00 )
        data["Age"][has_no_age] = numpy.nan

    if verbose:
        print("\nWARNING: {0} GC ages were changed".format(len(has_no_age)), end="")

        print(" from 14.00 to numpy.nan (b/c no age estimate available)\n")
        print("Succesfully read: '{0}'".format(fname))
        print("Usage: data = read_caldwell2011_data() ")
        print("You can then access rows using data[0]")
        print("You can access columns using data['colname']")
        print("To find all column names, use 'data.dtype.names'")
        print_caldwell2011_data(data, example=True)

    return data


def print_caldwell2011_data_vizier(data, example=True):
    width = 115

    print("\n{0}".format("-"*width))
    print("{0:<12s}{1:^6s}{2:^16s}{3:^16s}{4:^8s}{5:^3s}".format(
        "Name", "Notes", "RA", "DEC", "E(B-V)", ")"), end="")
    print("{0:>6s}{1:^5s}{2:^8s}{3:>6s}{4:^8s}{5:^8s}{6:^8s}".format(
        "Vel", "+/-", "logM", "[Fe/H]", "+/-", "Age", "Notes"))
    print("{0}".format("-"*width))
    for i, row in enumerate(data):
        print("{0:<12s}{1:^6s}{2:^16}{3:^16}{4:<8.2f}{5:<3s}".format(
            row[0].decode("ascii"), row[1].decode("ascii"),
            row[-4].ra.to_string(u.hour, alwayssign=True, pad=True, precision=2),
            row[-4].dec.to_string(u.degree, alwayssign=True, pad=True, precision=2),
            row[4], row[5].decode("ascii")), end="")
        print("{0: 6.1f}{1: 5.1f}{2:^8.1f}{3: 6.1f}{4:^8.1f}{5:^8.2f}{6:^8s}".format(
            row[6], row[7], row[8], row[9], row[10], row[11], row[12].decode("ascii")))
        if example and i > 3: break
    print("{0}\n".format("-"*width))


def plot_caldwell2011_FeH_hist(do_fit=True):
    (counts, edges), Ngc = get_M31_GCS_FeH()

    fig, ax = pyplot.subplots(figsize=(12, 9))

    ax.fill_between(edges[:-1], counts, lw=0.0, hatch="/",
        step="post", edgecolor="black", facecolor="none")
    pyplot.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k")

    if do_fit:
        from scipy.optimize import leastsq

        fitfunc  = lambda p, x: p[0]*numpy.exp(-0.5*((x-p[1])/p[2])**2)+p[3]
        errfunc  = lambda p, x, y: (y - fitfunc(p, x))

        x = (edges[1:]+edges[:-1])/2
        c, success = leastsq( errfunc, [1, 0.5, 0.5, 0.5], args=(x, counts))
        print("Fitting Gaussian to M31 GCS [Fe/H]")
        print("  g(A,mu,sigma,B) = A*exp(-0.5*((x-mu)/sigma)**2) + B")
        print("  A = {:.3f}\n  mu = {:.3f}\n  sigma = {:.3f}\n  B = {:.3f}".format(
            c[0], c[1], numpy.abs(c[2]), c[3]))
        # x = numpy.linspace(-3.5, 1.0, 0.1)
        pyplot.plot(x, fitfunc(c, x), c="r", ls="--", lw=2)

    ax.text(0.7, 0.7, "N = {0}".format(Ngc), transform=ax.transAxes)

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

    ax.set_xticks(numpy.arange(-3, 1, 0.5))
    ax.set_xticks(numpy.arange(-3.2, 1, 0.1), minor=True)
    ax.set_yticks(numpy.arange(0, 50, 5))
    ax.set_yticks(numpy.arange(0, 50, 1), minor=True)
    pyplot.xlim(-3.2, 0.9)
    pyplot.ylim(0, 45)
    pyplot.xlabel("Cluster Metallicity [Fe/H]")
    pyplot.ylabel("Number per Bin")
    pyplot.show()


def get_M31_GCS_FeH(bins=39, range=(-3.0, 1.0), density=False):
    """ Many empty bins when using 40 bins :-O ... """
    data = read_caldwell2011_data(verbose=False)
    m31gcs, = numpy.where( numpy.isfinite(data["[Fe/H]"]) )
    return numpy.histogram(data["[Fe/H]"][m31gcs], bins=bins, range=range, density=density), len(m31gcs)


def plot_caldwell_mass_metallicity_relation(data, do_fit=True):
    fig, ax = pyplot.subplots(figsize=(12, 9))

    # RuntimeWarning: invalid value encountered in less_equal
    with suppress_stdout():
        rich, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & numpy.isfinite(data["logM"])
            & (data["[Fe/H]"] > -0.8)
        )
        poor, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & numpy.isfinite(data["logM"])
            & (data["[Fe/H]"] <= -0.8)
        )

    ax.plot(data["[Fe/H]"][rich], data["logM"][rich], "ko", ms=8)
    ax.plot(data["[Fe/H]"][poor], data["logM"][poor], "ko", ms=8, mfc="none")

    finite = numpy.union1d(rich, poor)
    print("Sample size: {0}".format(len(finite)))
    # First, calculate Pearson correlation coefficient
    # Here -1 (negative correlation) or +1 (positive correlation)
    # imply an exact linear relationship. 0 means no correlation.
    # The p value is the 2-tailed p-value that an uncorrelated system produces
    # two datasets that have a Pearson correlation at least as extreme as
    # computed from these datasets. Assumes both datasets are normally distributed
    r, p = scipy.stats.pearsonr(data["[Fe/H]"][finite], data["logM"][finite])
    print("Pearson    r: {0:.5f} (p = {1:.5f})".format(r, p))

    # Does not assume normal distribution in both data sets.
    rho, p = scipy.stats.spearmanr(data["[Fe/H]"][finite], data["logM"][finite])
    print("Spearman rho: {0:.5f} (p = {1:.5f})".format(rho, p))

    # https://www.statisticshowto.datasciencecentral.com/
    #    spearman-rank-correlation-definition-calculate/
    FeHrank = scipy.stats.rankdata(data["[Fe/H]"][finite])
    massrank = scipy.stats.rankdata(data["logM"][finite])
    sxy = numpy.sum((FeHrank-numpy.mean(FeHrank)) * (massrank-numpy.mean(massrank)))
    sxy /= len(FeHrank)
    sx = numpy.sum((FeHrank-numpy.mean(FeHrank))**2) / len(FeHrank)
    sy = numpy.sum((massrank-numpy.mean(massrank))**2) / len(massrank)
    rho = sxy / numpy.sqrt(sx * sy)

    # Fit to y|x
    mmr_mean, edges, binnumbers = scipy.stats.binned_statistic(
        data["[Fe/H]"][finite], data["logM"][finite],
        statistic="mean"
    )
    mmr_sem, edges, binnumbers = scipy.stats.binned_statistic(
        data["[Fe/H]"][finite], data["logM"][finite],
        statistic=lambda array: scipy.stats.sem(array)
    )
    mmr_std, edges, binnumbers = scipy.stats.binned_statistic(
        data["[Fe/H]"][finite], data["logM"][finite],
        statistic=lambda array: numpy.std(array)
    )
    ax.errorbar((edges[1:]+edges[:-1])/2, mmr_mean, yerr=mmr_sem,
        ls="none", marker="o", c="r", ms=8)

    if do_fit:
        from tlrh_statistics import fit_to_data
        fitfunc = lambda p, x: p[0]*x + p[1]
        x = (edges[1:]+edges[:-1])/2
        x_ana = numpy.linspace(-3.2, 1.4, 128)
        popt, perr = fit_to_data(ax, x, mmr_mean, mmr_std,
            fitfunc, [0, 5.5], x_ana=x_ana)

    # Fit to x|y
    mmr_mean_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        data["logM"][finite], data["[Fe/H]"][finite],
        statistic="mean"
    )
    mmr_sem_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        data["logM"][finite], data["[Fe/H]"][finite],
        statistic=lambda array: scipy.stats.sem(array)
    )
    mmr_std_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        data["logM"][finite], data["[Fe/H]"][finite],
        statistic=lambda array: numpy.std(array)
    )
    ax.errorbar(mmr_mean_inv, (edges_inv[1:]+edges_inv[:-1])/2, xerr=mmr_sem_inv,
        ls="none", marker="o", c="b", ms=8)
    # TODO: do the fit?

    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel(r"log$_{10}(\rm M/\rm M_{\odot})$")
    ax.set_xticks(numpy.arange(-3.5, 1.5, 0.5))
    ax.set_xticks(numpy.arange(-3.2, 1.5, 0.1), minor=True)
    ax.set_yticks(numpy.arange(3, 9, 1))
    ax.set_yticks(numpy.arange(3, 8.2, 0.2), minor=True)
    pyplot.xlim(-3.2, 1.4)
    pyplot.ylim(4, 7.5)

    pyplot.show()


def plot_caldwell2011_figure23(data):
    fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(10, 11))

    finite, = numpy.where(
       numpy.isfinite(data["[Fe/H]"])
    )

    X, Y, Rproj = calculate_M31_Rgc_Wang2019(data["SkyCoord"],
        deproject=False, debug=False
    )

    ax1.plot(numpy.log10(Rproj[finite]), data["[Fe/H]"][finite], "ko", ms=6)
    with suppress_stdout():  # RuntimeWarning: invalid value encountered in log10
        ax2.plot(numpy.log10(Y[finite]), data["[Fe/H]"][finite], "ko", ms=6)

    # Sanity check: calculate Rproj with the given X and Y in the dataset
    # ax1.plot(numpy.log10(numpy.sqrt(data["X"][finite]**2 + data["Y"][finite]**2)),
    #     data["[Fe/H]"][finite], "ro", ms=6)

    # Sanity check: plot the given Y in the dataset
    # ax2.plot(numpy.log10(data["Y"][finite]), data["[Fe/H]"][finite], "ro", ms=6)

    # Sanity check: compare the given X in the dataset with the calculated X
    # ax2.plot(numpy.log10(X[finite]), data["[Fe/H]"][finite], "ko", ms=6)
    # ax2.plot(numpy.log10(data["X"][finite]), data["[Fe/H]"][finite], "ro", ms=6)

    ax2.axhline(-0.4, ls="--", c="k")

    for ax in [ax1, ax2]:
        ax.set_ylabel("[Fe/H]")
        ax.set_yticks(numpy.arange(-3, 2, 1))
        ax.set_yticks(numpy.arange(-3.2, 1.2, 0.2), minor=True)
        ax.set_ylim(-3, 1)

    ax1.set_xlabel("log(radius) (kpc)")
    ax1.set_xticks(numpy.arange(0, 3, 1))
    ax1.set_xticks(numpy.arange(-1, 2.2, 0.2), minor=True)
    ax1.set_xlim(-0.8, 2.2)

    ax2.set_xlabel("log(minor axis distance) (kpc)")
    ax2.set_xticks(numpy.arange(-2, 3, 1))
    ax2.set_xticks(numpy.arange(-2.2, 2.4, 0.2), minor=True)
    ax2.set_xlim(-2.1, 2.1)

    pyplot.tight_layout()
    pyplot.show()


def plot_caldwell2011_figure24(M31_c11, MW_h96e10, debug=False):
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(16, 8), sharey=True)

    Ryz = numpy.log10(numpy.sqrt(MW_h96e10["Y"]**2 + MW_h96e10["Z"]**2))
    Rproj = numpy.log10(M31_c11["Rproj"])

    # RuntimeWarning: invalid value encountered in less
    with suppress_stdout():
        FeH_high, = numpy.where(
            (M31_c11["[Fe/H]"] > -0.4)
        )
        FeH_med, = numpy.where(
            (M31_c11["[Fe/H]"] < -0.4)
            & (M31_c11["[Fe/H]"] > -0.9)
        )
        FeH_low, = numpy.where(
            (M31_c11["[Fe/H]"] < -0.9)
        )

    r_ana = numpy.linspace(-0.2, 1.3, 9)
    for rmin, rmax in zip(r_ana[:-1], r_ana[1:]):
        rmid = (rmin+rmax)/2
        area = numpy.pi*(10**rmid)**2

        Nmw = len(numpy.where( (Ryz > rmin) & (Ryz < rmax) )[0])
        Nm31 = len(numpy.where( (Rproj > rmin) & (Rproj < rmax) )[0])
        if debug:
            print("{:7.3f}{:7.3f}{:10.3f}{:7d}{:7.3f}{:7d}{:7.3f}".format(
                rmin, rmax, area, Nmw, numpy.log10(Nmw/area),
                Nm31, numpy.log10(Nm31/area)))
        # TODO: calculate the errorbars
        mw = ax1.errorbar(rmid, numpy.log10(Nmw/area),
            yerr=0.2*numpy.log10(numpy.sqrt(Nmw)),
            ls="none", marker="+", c="r", ms=10, capsize=4)
        for ax in [ax1, ax2]:
            m31 = ax.errorbar(rmid, numpy.log10(Nm31/area),
                yerr=0.2*numpy.log10(numpy.sqrt(Nm31)),
                ls="none", marker="o", c="k", ms=10, capsize=4,
            )

        # [Fe/H] > -0.4 --> 'high'
        Nm31_high = len(numpy.where(
            (Rproj[FeH_high] > rmin) & (Rproj[FeH_high] < rmax)
        )[0])
        with suppress_stdout():  # RuntimeWarning: divide by zero encountered in log10
            m31_high = ax.errorbar(rmid, numpy.log10(Nm31_high/area),
                yerr=0.2*numpy.log10(numpy.sqrt(Nm31_high)),
                ls="none", marker="X", c="r", ms=10, capsize=4,
            )

        # -0.9 < [Fe/H] < -0.4 --> 'med'
        Nm31_med = len(numpy.where(
            (Rproj[FeH_med] > rmin) & (Rproj[FeH_med] < rmax)
        )[0])
        m31_med = ax.errorbar(rmid, numpy.log10(Nm31_med/area),
            yerr=0.2*numpy.log10(numpy.sqrt(Nm31_med)),
            ls="none", marker="o", c="orange", ms=10, mfc="none", capsize=4,
        )

        # [Fe/H] < -0.9 --> 'low'
        Nm31_low = len(numpy.where(
            (Rproj[FeH_low] > rmin) & (Rproj[FeH_low] < rmax)
        )[0])
        m31_low = ax.errorbar(rmid, numpy.log10(Nm31_low/area),
            yerr=0.2*numpy.log10(numpy.sqrt(Nm31_low)),
            ls="none", marker="o", c="b", ms=8, capsize=4,
        )

    # Add powerlaw with slope -2.5
    r_ana = numpy.linspace(0, 1, 64)
    ax1.plot(r_ana, -2.5*r_ana , c="r")

    for ax in [ax1, ax2]:
        ax.set_xticks(numpy.arange(-0.5, 2.5, 0.5))
        ax.set_xticks(numpy.arange(-0.25, 1.75, 0.25), minor=True)
        ax.set_yticks(numpy.arange(-2.5, 1.5, 0.5))
        ax.set_yticks(numpy.arange(-3, 1.25, 0.25), minor=True)
        ax.set_xlim(-0.2, 1.4)
        ax.set_ylim(-2.8, 1.2)
        ax.set_xlabel("log R (kpc)")
    ax1.legend([m31, mw], ["M31", "MW"], frameon=True, fontsize=16)
    ax1.set_ylabel(r"log (Clusters kpc$^{-2}$)")

    ax2.legend([m31, m31_high, m31_med, m31_low], ["M31",
        "[Fe/H]$>$-0.4 ({})".format(len(FeH_high)),
        "-0.9$<$[Fe/H]$<$-0.4 ({})".format(len(FeH_med)),
        "[Fe/H]$<$-0.9 ({})".format(len(FeH_low))
        ], frameon=True, fontsize=16)

    # pyplot.tight_layout()
    pyplot.subplots_adjust(wspace=0, hspace=0)
    pyplot.show()


if __name__ == "__main__":
    pyplot.style.use("tlrh")
    data = read_caldwell2011_data()

    print("There are {0: >3d} globular clusters in total".format(len(data)))

    i_has_vel, = numpy.where( numpy.isfinite(data["V"]) )
    print("There are {0: >3d} globular clusters /w velocity measurement".format(len(i_has_vel)))

    i_has_logM, = numpy.where( numpy.isfinite(data["logM"]) )
    print("There are {0: >3d} globular clusters /w logM measurement".format(len(i_has_logM)))


    with suppress_stdout():
        i_has_FeH, = numpy.where( (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4) )
    print("There are {0: >3d} globular clusters /w [Fe/H] measurement".format(len(i_has_FeH)))

    with suppress_stdout():
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4)
            & numpy.isfinite(data["logM"])
        )
    print("There are {0: >3d} globular clusters /w [Fe/H] and logM measurement".format(len(i_has_FeH_and_logM)))

    with suppress_stdout():
        i_has_age, = numpy.where( (data["Age"] < 13.99 ) )
    print("There are {0: >3d} globular clusters /w age measurement".format(len(i_has_age)))

    with suppress_stdout():
        i_has_age_and_logM, = numpy.where(
            (data["Age"] < 13.99 )
            & numpy.isfinite(data["logM"])
        )
    print("There are {0: >3d} globular clusters /w age and logM measurement".format(len(i_has_age_and_logM)))

    with suppress_stdout():
        ihas_age_and_FeH, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4)
            & (data["Age"] < 13.99 )
        )
    print("There are {0: >3d} globular clusters /w age and [Fe/H] measurement".format(
        len(ihas_age_and_FeH)))

    with suppress_stdout():
        i_has_FeH_age_and_logM, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4)
            & numpy.isfinite(data["logM"])
            & (data["Age"] < 13.99 )
        )
    print("There are {0: >3d} globular clusters /w [Fe/H], age, and logM measurement".format(len(i_has_FeH_age_and_logM)))


    import sys; sys.exit(0)


    # [Fe/H] distribution
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))

    counts, edges = numpy.histogram(data["[Fe/H]"][i_has_FeH], bins=24, range=[-2.5, 0.5])
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="k", label="[M31] Globular Clusters /w [Fe/H] measurement")

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=16, frameon=False)

    pyplot.savefig("../out/M31_GlobularClusterSystem_FeH.pdf")
    pyplot.tight_layout()
    pyplot.show()

    # Age distribution
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))

    counts, edges = numpy.histogram(data["Age"][i_has_age], bins=28, range=[0, 14])
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="k", label="[M31] Globular Clusters /w age measurement")

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=16, frameon=False)

    pyplot.tight_layout()
    pyplot.savefig("../out/M31_GlobularClusterSystem_age.pdf")
    pyplot.show()

    # Age vs [Fe/H]
    fig, ax = pyplot.subplots(1, 1, figsize=(12, 9))

    ax.plot(data["[Fe/H]"][ihas_age_and_FeH], data["Age"][ihas_age_and_FeH],
        "ko", label="[M31] Globular Clusters")

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Age [Gyr]")
    ax.legend(fontsize=16, frameon=False)

    pyplot.tight_layout()
    pyplot.savefig("../out/M31_GlobularClusterSystem_age-vs-FeH.pdf")
    pyplot.show()
