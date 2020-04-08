import os
import numpy
import scipy
import matplotlib
from scipy import stats
from matplotlib import pyplot
from astropy import units as u
from astropy import coordinates as coord

from tlrh_util import suppress_stdout
pyplot.style.use("tlrh")

matplotlib.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}",
    "font.size": 28,
})


def read_harris1996_data():
    GCS_in_MW_Harris1996_VersionDec2010 = \
    "http://physwww.mcmaster.ca/%7Eharris/mwgc.dat"
    dirname = os.path.dirname(__file__)
    GCS_of_MW = os.path.join(dirname, "../data/Harris1996_2010edition/mwgc.dat")

    if not os.path.isfile(GCS_of_MW):
        import urllib.request
        urllib.request.urlretrieve(GCS_in_MW_Harris1996_VersionDec2010, GCS_of_MW)

    # PART I: Cluster identifications and positions
    dtype = {
        "names": [
            "ID", "Name", "RADEC", "LONLAT",
            "R_Sun", "R_gc", "X", "Y", "Z"
        ],
        "formats": [
            "S10", "S10", "object_", "object_",
            "float", "float", "float", "float", "float"
        ]
    }
    delimiter = [12, 13, 25, 16, 7, 7, 6, 6, 6]
    eat_RADEC = lambda x: coord.SkyCoord(
        x[:12].decode("ascii"), x[13:].decode("ascii"),
        frame="icrs", equinox="J2000", unit=(u.hourangle, u.deg))
    eat_LONLAT = lambda x: coord.SkyCoord(
        x[:6].decode("ascii"), x[7:].decode("ascii"),
        frame="galactic", unit=(u.deg, u.deg))
    converters =  { 2: eat_RADEC, 3: eat_LONLAT }

    part1 = numpy.genfromtxt(GCS_of_MW, skip_header=72, skip_footer=363, dtype=dtype,
        delimiter=delimiter, converters=converters, autostrip=True)

    # The galactocentric radius of the Milky Way is calculated by Harris
    # as R_gc = sqrt( (X-8)^2 + Y^2 + Z^2 ), where X, Y, Z are in a
    # Sun-centered coordinate system. For a 'fair' comparison of the MW
    # calactocentric radius we multiply the three-dimensional radius by pi/4,
    # i.e. Rproj = Rgc × (π/4), as per Huxor (2014, Fig. 17)
    dtype = numpy.dtype(part1.dtype.descr + [("Rproj", "float")])
    part1_Rproj = numpy.empty(part1.shape, dtype=dtype)
    for name in part1.dtype.names:
        part1_Rproj[name] = part1[name]
    part1_Rproj["Rproj"] = numpy.pi/4 * part1["R_gc"]
    part1 = part1_Rproj

    # PART II: Metallicities, Integrated magnitudes, colors,
    dtype = {
        "names": [
            "ID", "FeH", "wt", "EBminV", "V_HB", "mminMV",
            "V_t", "M_Vt", "UminB", "BminV", "VminR", "VminI", "spt", "ellip"
        ],
        "formats": [
            "S10", "float", "int", "float", "float", "float",
            "float", "float", "float", "float", "float", "float", "S2", "float"
        ]
    }
    delimiter = [12, 7, 4, 6, 5, 6, 6, 7, 8, 6, 7, 6, 6, 6]

    part2 = numpy.genfromtxt(GCS_of_MW, skip_header=252, skip_footer=183, dtype=dtype,
        delimiter=delimiter, autostrip=True)

    # PART III: Radial velocities, velocity dispersions, structural parameters
    dtype = {
        "names": [
            "ID", "v_r", "v_r_err", "v_LSR", "sig_v", "sig_v_err", "c",
            "collapsed", "r_c", "r_h", "mu_V", "rho_0", "lg(tc)", "lg(th)",
        ],
        "formats": [
            "S10", "float", "float", "float", "float", "float", "float",
            "S2", "float", "float", "float", "float", "float", "float"
        ]
    }
    delimiter = [13, 7, 6, 7, 8, 6, 7, 4, 7, 8, 6, 7, 7, 7]

    part3 = numpy.genfromtxt(GCS_of_MW, skip_header=433, skip_footer=2, dtype=dtype,
        delimiter=delimiter, autostrip=True)

    return part1, part2, part3


def print_harris1996_data(part1, part2, part3, example=False):
    width = 115

    # Part 1
    print("\n{0}".format("-"*width))
    print("{0:12s}{1:12s}{2:^16s}{3:^16s}{4:^9s}{5:^10s}{6:>6s}{7:^10s}{8:^8s}{9:^8s}{10:^8s}".format(
        "ID", "Name", "RA", "DEC", "L", "B", "R_Sun", "R_gc", "X", "Y", "Z"))
    print("{0}".format("-"*width))
    for i, row in enumerate(part1):
        print("{0:<12s}{1:<12s}{2:^16s}{3:^16}{4:>8s}{5:>8s}".format(
            row[0].decode("ascii"), row[1].decode("ascii"),
            row[2].ra.to_string(u.hour, alwayssign=True, pad=True, precision=2),
            row[2].dec.to_string(u.degree, alwayssign=True, pad=True, precision=2),
            row[3].l.to_string(u.deg, decimal=True, precision=2),
            row[3].b.to_string(u.deg, decimal=True, precision=2)), end="")
        print("{0: 8.1f}{1: 8.1f}{2: 8.1f}{3: 8.1f}{4: 8.1f}".format(
            row[4], row[5], row[6], row[7], row[8]))
        if example and i > 3: break
    print("{0}\n".format("-"*width))

    # Part 2
    print("\n{0}".format("-"*width))
    print("{:12s}{:<8s}{:<4s}{:<8s}{:<8s}{:<8s}".format(
        "ID", "[Fe/H]", "wt", "E(B-V)", "V_HB", "(m-M)V"), end="")
    print("{:^8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<8s}{:<4s}{:<8s}".format(
        "V_t", "M_V,t", "U-B", "B-V", "V-R", "V-I", "spt", "ellip"))
    print("{0}".format("-"*width))
    for i, row in enumerate(part2):
        print("{0:<12s}{1:<8.2f}{2:<4d}{3:<8.2f}{4:<8.2f}{5:<8.2f}{6: 6.2f}".format(
            row[0].decode("ascii"), row[1], row[2], row[3], row[4], row[5], row[6]), end="")
        print("{0:^10.2f}{1:<8.2f}{2:<8.2f}{3:<8.2f}{4:<8.2f}{5:<4s}{6:<8.2f}".format(
            row[7], row[8], row[9], row[10], row[11], row[12].decode("ascii"), row[13]))
        if example and i > 3: break
    print("{0}\n".format("-"*width))

    # Part 3
    print("\n{0}".format("-"*width))
    print("{:12s}{:<8s}{:<6s}{:<8s}{:<8s}{:<6s}".format(
        "ID", "v_r", "+/-", "v_LSR", "sig_v", "+/-"), end="")
    print("{:8s}{:10s}{:8s}{:<8s}{:<6s}{:<8s}{:<8s}{:<6s}".format(
        "c", "collapsed", "r_c", "r_h", "mu_V", "rho_0 ", "(tc)", "lg(th)"))

    print("{0}".format("-"*width))
    for i, row in enumerate(part3):
        print("{:12s}{:<8.2f}{:<6.2f}{:<8.2f}{:<8.2f}{:<6.2f}".format(
            row[0].decode("ascii"), row[1], row[2], row[3], row[4], row[5]), end="")
        print("{:<8.2f}{:^10s}{:<8.2f}{:<8.2f}{:<6.2f}{:<8.2f}{:<8.2f}{:<6.2f}".format(
            row[6], row[7].decode("ascii"), row[8], row[9], row[10], row[11], row[12], row[13]))
        if example and i > 3: break

    print("{0}\n".format("-"*width))


def combine_part1_part2_part3(p1, p2, p3, Mv_Sun=4.83, mass_to_light=1.7):
    # ID sits in all parts, but we only want it once.
    c1 = [c for c in p1.dtype.names]
    c2 = [c for c in p2.dtype.names if c != "ID"]
    c3 = [c for c in p3.dtype.names if c != "ID"]
    nrows = p1.shape[0]

    # descr makes list of dtype, then + concatenates the lists
    dtype = p1[c1].dtype.descr + p2[c2].dtype.descr + p3[c3].dtype.descr + \
        [("Mass", "float")]
    out = numpy.empty(nrows, dtype=dtype)

    # Copy the data into the new array for all three parts
    for p in [p1, p2, p3]:
        for c in p.dtype.names:
            out[c] = p[c]

    # Sneak the mass in there, too
    isfinite, = numpy.where( numpy.isfinite(p2["M_Vt"]) )
    mass = numpy.power(10, 0.4*(Mv_Sun - p2["M_Vt"][isfinite])) * mass_to_light
    out["Mass"][isfinite] = mass

    return out


def plot_harris1996_figure1_2(part1):
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 9))

    # Left panel
    ax1.plot(part1["Y"], part1["Z"], "ko", ms=5)

    # Bulge
    xcenter, ycenter, xlen, ylen, t = 0, 0, 2, 1.5, numpy.linspace(0.35, numpy.pi-0.35, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    t = numpy.linspace(numpy.pi+0.35, 2*numpy.pi-0.35, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    # Endpoints of disk
    xcenter, ycenter, xlen, ylen, t = -13, -0.025, 0.5, 0.475, numpy.linspace(numpy.pi/2, 3*numpy.pi/2, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    xcenter, ycenter, xlen, ylen, t = 13, -0.025, 0.5, 0.475, numpy.linspace(3*numpy.pi/2, 5*numpy.pi/2, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    # Disk
    for ax in [ax1, ax2]:
        x = numpy.linspace(2, 13, 2); ax.plot(x, [0.5 for i in x], c="k")
        x = numpy.linspace(-2, -13, 2); ax.plot(x, [0.5 for i in x], c="k")
        x = numpy.linspace(2, 13, 2); ax.plot(x, [-0.475 for i in x], c="k")
        x = numpy.linspace(-2, -13, 2); ax.plot(x, [-0.475 for i in x], c="k")

    ax1.set_xticks([-10, 0, 10])
    ax1.set_xticks(range(-18, 20, 2), minor=True)
    ax1.set_yticks([-10, 0, 10])
    ax1.set_yticks(range(-18, 20, 2), minor=True)
    ax1.set_xlabel("Y (kpc)")
    ax1.set_ylabel("Z (kpc)")
    ax1.set_xlim(18, -18)
    ax1.set_ylim(-18, 18)
    ax1.set_aspect("equal")

    # Right panel.. using arbitrary mask
    mask = numpy.where((numpy.abs(part1["Y"]) > 10 ) | (numpy.abs(part1["Z"]) > 5) )
    ax2.plot(part1["Y"][mask], part1["Z"][mask], "ko", ms=5)

    xcenter, ycenter, xlen, ylen, t = 0, -0.5, 2, 1.5, numpy.linspace(0, 2*numpy.pi, 100)
    ax2.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )

    ax2.set_xticks([-100, 0, 100])
    ax2.set_xticks(range(-140, 160, 20), minor=True)
    ax2.set_yticks([-100, 0, 100])
    ax2.set_yticks(range(-180, 160, 20), minor=True)
    ax2.set_xlabel("Y (kpc)")
    ax2.set_xlim(140, -140)
    ax2.set_ylim(-140, 140)
    ax2.set_aspect("equal")

    pyplot.tight_layout()
    pyplot.show()


def plot_harris1996_figure1_2_th(part1, part2):
    fig, (ax1, ax2) = pyplot.subplots(1, 2, figsize=(18, 9))

    finite, = numpy.where( numpy.isfinite(part2["FeH"]) )
    red, = numpy.where( (part2["FeH"][finite] > -1) )
    blue, = numpy.where( (part2["FeH"][finite] <= -1) )

    # Left panel
    ax1.plot(part1["Y"][red], part1["Z"][red], "ro", ms=5)
    ax1.plot(part1["Y"][blue], part1["Z"][blue], "bo", ms=5)

    # Bulge
    xcenter, ycenter, xlen, ylen, t = 0, 0, 2, 1.5, numpy.linspace(0.35, numpy.pi-0.35, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    t = numpy.linspace(numpy.pi+0.35, 2*numpy.pi-0.35, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    # Endpoints of disk
    xcenter, ycenter, xlen, ylen, t = -13, -0.025, 0.5, 0.475, numpy.linspace(numpy.pi/2, 3*numpy.pi/2, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    xcenter, ycenter, xlen, ylen, t = 13, -0.025, 0.5, 0.475, numpy.linspace(3*numpy.pi/2, 5*numpy.pi/2, 100)
    ax1.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )
    # Disk
    for ax in [ax1, ax2]:
        x = numpy.linspace(2, 13, 2); ax.plot(x, [0.5 for i in x], c="k")
        x = numpy.linspace(-2, -13, 2); ax.plot(x, [0.5 for i in x], c="k")
        x = numpy.linspace(2, 13, 2); ax.plot(x, [-0.475 for i in x], c="k")
        x = numpy.linspace(-2, -13, 2); ax.plot(x, [-0.475 for i in x], c="k")

    ax1.set_xticks([-10, 0, 10])
    ax1.set_xticks(range(-18, 20, 2), minor=True)
    ax1.set_yticks([-10, 0, 10])
    ax1.set_yticks(range(-18, 20, 2), minor=True)
    ax1.set_xlabel("Y (kpc)")
    ax1.set_ylabel("Z (kpc)")
    ax1.set_xlim(18, -18)
    ax1.set_ylim(-18, 18)
    ax1.set_aspect("equal")

    # Right panel.. using arbitrary mask
    mask_red = numpy.where((numpy.abs(part1["Y"][red]) > 10 ) | (numpy.abs(part1["Z"][red]) > 5) )
    mask_blue = numpy.where((numpy.abs(part1["Y"][blue]) > 10 ) | (numpy.abs(part1["Z"][blue]) > 5) )
    ax2.plot(part1["Y"][mask_red], part1["Z"][mask_red], "ro", ms=5)
    ax2.plot(part1["Y"][mask_blue], part1["Z"][mask_blue], "bo", ms=5)

    xcenter, ycenter, xlen, ylen, t = 0, -0.5, 2, 1.5, numpy.linspace(0, 2*numpy.pi, 100)
    ax2.plot( xcenter + xlen*numpy.cos(t), ycenter + ylen*numpy.sin(t), c="k" )

    ax2.set_xticks([-100, 0, 100])
    ax2.set_xticks(range(-140, 160, 20), minor=True)
    ax2.set_yticks([-100, 0, 100])
    ax2.set_yticks(range(-180, 160, 20), minor=True)
    ax2.set_xlabel("Y (kpc)")
    ax2.set_xlim(140, -140)
    ax2.set_ylim(-140, 140)
    ax2.set_aspect("equal")

    pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_phi-vs-Rgc_th.png")
    pyplot.show()


def plot_harris1996_figure1_3(part1, part2, use_2d=False):
    fig, ax = pyplot.subplots(figsize=(9, 9))
    ax.text(0.5, 1.02, "Space Distribution (Milky Way)",
        transform=ax.transAxes, ha="center", va="bottom")

    # RuntimeWarning: invalid value encountered in greater
    with suppress_stdout():
        red, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] > -1)
        )
        blue, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] <= -1)
        )

    radii_red = numpy.power(10, numpy.array([0.1, 0.4, 0.6, 0.8, 1, 1.2]))
    volume_red = 4/3 * numpy.pi * radii_red**3
    NredOld  = numpy.zeros(radii_red.shape, dtype="i8")
    for i, (r1, r2) in enumerate(zip(radii_red[:-1], radii_red[1:])):
        nRedOld, = numpy.where(
            (part1["R_gc"][red] > r1) &
            (part1["R_gc"][red] < r2) )
        NredOld[i] = len(nRedOld)
        if use_2d: volume_red[i] = 4*numpy.pi*((r1+r2)/2)**2 * (r2-r1)

    radii_blue = numpy.power(10, numpy.array(
        [0.15, 0.35, 0.65, 0.8, 1, 1.2, 1.3, 1.5, 1.95, 2.5]))
    volume_blue = 4/3 * numpy.pi * radii_blue**3
    NblueOld = numpy.zeros(radii_blue.shape, dtype="i8")
    for i, (r1, r2) in enumerate(zip(radii_blue[:-1], radii_blue[1:])):
        nBlueOld, = numpy.where(
            (part1["R_gc"][blue] > r1) &
            (part1["R_gc"][blue] < r2) )
        NblueOld[i] = len(nBlueOld)
        if use_2d: volume_blue[i] = 4*numpy.pi*((r1+r2)/2)**2 * (r2-r1)

    with suppress_stdout():  # divide by zero encountered in log10
        pyplot.plot(numpy.log10(radii_blue), numpy.log10(NblueOld/volume_blue),
            "ko", ms=8, label=r"[Fe/H] $\leq$ -1, N = {0}".format(len(blue)))
        pyplot.plot(numpy.log10(radii_red), numpy.log10(NredOld/volume_red),
            "ko", ms=10, mfc="none", label=r"[Fe/H] $>$ -1, N = {0}".format(
            len(red)))

    radii_red = numpy.power(10, numpy.linspace(0.5, 2.2, 42))
    radii_blue = numpy.power(10, numpy.linspace(0.0001, 0.7, 42))
    pyplot.plot(numpy.log10(radii_red), numpy.log10(radii_red**(-3.5))+0.75,
        c="k")
    pyplot.plot(numpy.log10(radii_blue), numpy.log10(radii_blue**(-2))-0.25,
        c="k", ls="--")

    ax.text(0.5, 0.2, r"$\phi \sim $R$^{-3.5}$", transform=ax.transAxes,
        fontsize=22)
    ax.set_xticks(numpy.arange(0, 3, 0.5))
    ax.set_xticks(numpy.arange(0, 2.6, 0.1), minor=True)
    ax.set_yticks(range(0, -8, -2))
    ax.set_yticks(numpy.arange(0.5, -8, -0.5), minor=True)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-7.5, 1)
    pyplot.xlabel(r"log R$_{\text{GC}}$ (kpc)")
    pyplot.ylabel(r"log $\phi$ (number per kpc$^3$)")
    pyplot.legend(loc="lower left", frameon=False, fontsize=16)

    # pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_phi-vs-Rgc.png")
    pyplot.show()


def plot_harris1996_figure1_3_th(part1, part2, use_2d=False):
    fig, ax = pyplot.subplots(figsize=(9, 9))
    ax.text(0.5, 1.02, "Space Distribution (Milky Way)",
        transform=ax.transAxes, ha="center", va="bottom")

    # RuntimeWarning: invalid value encountered in greater
    with suppress_stdout():
        red, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] > -1)
        )
        blue, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] <= -1)
        )

    rmax = 10**2.5
    radii = numpy.power(10, numpy.linspace(numpy.log10(1), numpy.log10(rmax), 7))
    volume = 4/3 * numpy.pi * radii**3

    NredOld  = numpy.zeros(radii.shape, dtype="i8")
    NblueOld = numpy.zeros(radii.shape, dtype="i8")

    for i, (r1, r2) in enumerate(zip(radii[:-1], radii[1:])):
        nRedOld, = numpy.where(
            (part1["R_gc"][red] > r1) &
            (part1["R_gc"][red] < r2) )
        nBlueOld, = numpy.where(
            (part1["R_gc"][blue] > r1) &
            (part1["R_gc"][blue] < r2) )
        NredOld[i] = len(nRedOld)
        NblueOld[i] = len(nBlueOld)
        if use_2d: volume[i] = 4*numpy.pi*((r1+r2)/2)**2 * (r2-r1)

    with suppress_stdout():  # divide by zero encountered in log10
        pyplot.plot(numpy.log10(radii), numpy.log10(NblueOld/volume),
            "bo", ms=5, label=r"[Fe/H] $\leq$ -1, N = {0}".format(len(blue)))
        pyplot.plot(numpy.log10(radii), numpy.log10(NredOld/volume),
            "ro", ms=5, label=r"[Fe/H] $>$ -1, N = {0}".format(len(red)))

    radii_red = numpy.power(10, numpy.linspace(0.5, 2.2, 42))
    radii_blue = numpy.power(10, numpy.linspace(0.0001, 0.7, 42))
    pyplot.plot(numpy.log10(radii_red), numpy.log10(radii_red**(-3.5)), c="k")
    pyplot.plot(numpy.log10(radii_blue), numpy.log10(radii_blue**(-2))-0.75,
        c="k", ls="--")

    ax.text(0.7, 0.2, r"$\phi \sim $R$^{-3.5}$", transform=ax.transAxes,
        fontsize=22)
    ax.set_xticks(numpy.arange(0, 3, 0.5))
    ax.set_xticks(numpy.arange(0, 2.6, 0.1), minor=True)
    ax.set_yticks(range(0, -8, -2))
    ax.set_yticks(numpy.arange(0.5, -8, -0.5), minor=True)
    ax.set_xlim(0, 2.5)
    ax.set_ylim(-7.5, 1)
    pyplot.xlabel(r"log R$_{\text{GC}}$ (kpc)")
    pyplot.ylabel(r"log $\phi$ (number per kpc$^3$)")

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    pyplot.legend(loc="lower left", frameon=False, fontsize=16)

    # pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_phi-vs-Rgc_th.png")
    pyplot.show()


def g(x, mu, sigma):
    return 1/(sigma*numpy.sqrt(2*numpy.pi)) * numpy.exp(-0.5*((x-mu)/sigma)**2)


def plot_harris1996_figure1_7(part2):
    fig, ax = pyplot.subplots(figsize=(12, 9))

    mwgcs, = numpy.where( numpy.isfinite(part2["FeH"]) )
    bins, edges = numpy.histogram(part2["FeH"][mwgcs], bins=30, range=(-2.5, 0.5))
    ax.fill_between(edges[:-1], bins, lw=0.0, hatch="/",
        step="post", edgecolor="black", facecolor="none")
    pyplot.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid", c="k")

    FeH_range = numpy.arange(-2.8, 0.5, 0.01)
    ax.plot(FeH_range, 7.5*g(FeH_range, -1.6, 0.3), c="k", ls="dashed")
    ax.plot(FeH_range, 3*g(FeH_range, -0.6, 0.2), c="k", ls="dashed")

    ax.text(0.7, 0.7, "N = {0}".format(len(mwgcs)), transform=ax.transAxes)

    ax.set_xticks(numpy.arange(-2.5, 0.5, 0.5))
    ax.set_xticks(numpy.arange(-2.8, 0.5, 0.1), minor=True)
    ax.set_yticks(numpy.arange(0, 25, 5))
    ax.set_yticks(numpy.arange(0, 25, 1), minor=True)
    pyplot.xlim(-2.8, 0.4)
    pyplot.ylim(0, 20)
    pyplot.xlabel("Cluster Metallicity [Fe/H]")
    pyplot.ylabel("Number per Bin")
    pyplot.show()


def plot_harris1996_figure1_7_th(part2):
    fig, ax = pyplot.subplots(figsize=(12, 9))

    with suppress_stdout():  # invalid value encountered in < and/or >
        red, = numpy.where( (part2["FeH"] < -1) & numpy.isfinite(part2["FeH"]) )
        blue, = numpy.where( (part2["FeH"] >= -1) & numpy.isfinite(part2["FeH"]) )

    bins, edges = numpy.histogram(part2["FeH"][red], bins=30, range=(-2.5, 0.5))
    pyplot.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid", c="r")
    ax.fill_between(edges[:-1], bins, lw=0.0, hatch="/",
        step="post", edgecolor="red", facecolor="none")

    bins, edges = numpy.histogram(part2["FeH"][blue], bins=30, range=(-2.5, 0.5))
    ax.fill_between(edges[:-1], bins, lw=0.0, hatch="/",
        step="post", edgecolor="blue", facecolor="none")
    pyplot.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid", c="b")

    FeH_range = numpy.arange(-2.8, 0.5, 0.01)
    ax.plot(FeH_range, 7.5*g(FeH_range, -1.6, 0.3), c="k", ls="dashed")
    ax.plot(FeH_range, 3*g(FeH_range, -0.6, 0.2), c="k", ls="dashed")

    ax.text(0.7, 0.7, "N = {0}".format(len(red) + len(blue)),
        transform=ax.transAxes)

    ax.set_xticks(numpy.arange(-2.5, 0.5, 0.5))
    ax.set_xticks(numpy.arange(-2.8, 0.5, 0.1), minor=True)
    ax.set_yticks(numpy.arange(0, 25, 5))
    ax.set_yticks(numpy.arange(0, 25, 1), minor=True)
    pyplot.xlim(-2.8, 0.4)
    pyplot.ylim(0, 20)
    pyplot.xlabel("Cluster Metallicity [Fe/H]")
    pyplot.ylabel("Number per Bin")
    pyplot.show()


def plot_harris1996_figure1_8(part1, part2):
    fig, (ax1, ax2) = pyplot.subplots(2, 1, figsize=(9, 14))

    # RuntimeWarning: invalid value encountered in greater
    with suppress_stdout():
        rich, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] > -1.0)
        )
        poor, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & (part2["FeH"] <= -1.0)
        )

    ax1.plot(numpy.log10(part1["R_gc"][rich]), part2["FeH"][rich], "ko", ms=8)
    ax1.plot(numpy.log10(part1["R_gc"][poor]), part2["FeH"][poor],
        "ko", ms=8, mfc="none")

    FeH_mean, edges, binnumbers = scipy.stats.binned_statistic(
        numpy.log10(part1["R_gc"][rich]), part2["FeH"][rich],
        bins=5)
    FeH_sem, edges, binnumbers = scipy.stats.binned_statistic(
        numpy.log10(part1["R_gc"][rich]), part2["FeH"][rich],
        statistic=lambda x: scipy.stats.sem(x), bins=5)
    ax2.errorbar((edges[1:]+edges[:-1])/2, FeH_mean,
        yerr=FeH_sem, ls="none", marker="o", c="k", ms=8 )

    FeH_mean, edges, binnumbers = scipy.stats.binned_statistic(
        numpy.log10(part1["R_gc"][poor]), part2["FeH"][poor],
        statistic="mean")
    # CAUTION, using std err as errorbar in the mean
    FeH_sem, edges, binnumbers = scipy.stats.binned_statistic(
        numpy.log10(part1["R_gc"][poor]), part2["FeH"][poor],
        statistic=lambda x: scipy.stats.sem(x))
    ax2.errorbar((edges[1:]+edges[:-1])/2, FeH_mean,
        yerr=FeH_sem, ls="none", marker="o", c="k", ms=8, mfc="none")

    # Harris' trend line Delta [Fe/H] / Delta log(Rgc) = -0.30
    radii = numpy.power(10, numpy.linspace(0.0001, 1, 42))
    ax2.plot(numpy.log10(radii), numpy.log10(radii**(-0.30))-0.4, c="k")
    ax2.plot(numpy.log10(radii), numpy.log10(radii**(-0.30))-1.2, c="k")

    for ax in [ax1, ax2]:
        ax.set_xticks(numpy.arange(0, 3, 0.5))
        ax.set_xticks(numpy.arange(-0.4, 2.6, 0.1), minor=True)
        ax.set_yticks(numpy.arange(-2.0, 1, 1))
        ax.set_yticks(numpy.arange(-2.4, 0.6, 0.2), minor=True)
        ax.set_xlim(-0.3, 2.5)
        ax.set_ylim(-2.8, 0.4)
    ax1.set_ylabel("[Fe/H]")
    ax2.set_xlabel(r"log R$_{\text{GC}}$ (kpc)")
    ax2.set_ylabel(r"$<$[Fe/H]$>$")
    pyplot.show()


def get_MW_GCS_FeH(bins=30, range=(-2.5, 0.5), density=False):
    part1, part2, part3 = read_harris1996_data()
    mwgcs, = numpy.where( numpy.isfinite(part2["FeH"]) )
    return numpy.histogram(part2["FeH"][mwgcs], bins=bins, range=range, density=density), len(mwgcs)


def plot_MW_mass_distribution(part2, Mv_Sun=4.83, mass_to_light=1.7, nbins=14):
    fig, ax = pyplot.subplots(1, 1, figsize=(6.5, 8))

    isfinite, = numpy.where( numpy.isfinite(part2["M_Vt"]) )
    mass = numpy.power(10, 0.4*(Mv_Sun - part2["M_Vt"][isfinite])) * mass_to_light
    counts, edges = numpy.histogram(numpy.log10(mass), bins=nbins, range=(2, 8))
    # ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k")
    log10_M = (edges[1:]-edges[:-1])  # M already in logspace
    ax.plot((edges[1:]+edges[:-1])/2, counts/log10_M, c="r", ls="dashed")

    ax.set_xticks(numpy.arange(2, 8, 0.5), minor=True)
    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.set_xlabel(r"log$_{10}(M$/M$_\odot$)", fontsize=18)
    ax.set_ylabel(r"$dN / d\log_{10}(M)$", fontsize=18)
    ax.set_yscale("log")
    ax.set_xlim(2, 8)
    ax.set_ylim(1, 5e5)

    pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_MassDistribution.pdf")
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_MassDistribution.png")
    pyplot.show()


def plot_harris_mass_metallicity_relation(part2,
        Mv_Sun=4.83, mass_to_light=1.7, do_fit=True):
    fig, ax = pyplot.subplots(figsize=(12, 9))

    # RuntimeWarning: invalid value encountered in less_equal
    with suppress_stdout():
        rich, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & numpy.isfinite(part2["M_Vt"])
            & (part2["FeH"] > -1.0)
        )
        poor, = numpy.where(
            numpy.isfinite(part2["FeH"])
            & numpy.isfinite(part2["M_Vt"])
            & (part2["FeH"] <= -1.0)
        )

    mass = numpy.power(10, 0.4*(Mv_Sun - part2["M_Vt"])) * mass_to_light
    ax.plot(part2["FeH"][rich], numpy.log10(mass[rich]), "ko", ms=8)
    ax.plot(part2["FeH"][poor], numpy.log10(mass[poor]), "ko", ms=8, mfc="none")

    finite = numpy.union1d(rich, poor)
    print("Sample size: {0}".format(len(finite)))
    # First, calculate Pearson correlation coefficient
    # Here -1 (negative correlation) or +1 (positive correlation)
    # imply an exact linear relationship. 0 means no correlation.
    # The p value is the 2-tailed p-value that an uncorrelated system produces
    # two datasets that have a Pearson correlation at least as extreme as
    # computed from these datasets. Assumes both datasets are normally distributed
    r, p = scipy.stats.pearsonr(part2["FeH"][finite], numpy.log10(mass[finite]))
    print("Pearson    r: {0:.5f} (p = {1:.5f})".format(r, p))

    # Does not assume normal distribution in both data sets.
    rho, p = scipy.stats.spearmanr(part2["FeH"][finite], numpy.log10(mass[finite]))
    print("Spearman rho: {0:.5f} (p = {1:.5f})".format(rho, p))

    # https://www.statisticshowto.datasciencecentral.com/
    #    spearman-rank-correlation-definition-calculate/
    FeHrank = scipy.stats.rankdata(part2["FeH"][finite])
    massrank = scipy.stats.rankdata(numpy.log10(mass[finite]))
    sxy = numpy.sum((FeHrank-numpy.mean(FeHrank)) * (massrank-numpy.mean(massrank)))
    sxy /= len(FeHrank)
    sx = numpy.sum((FeHrank-numpy.mean(FeHrank))**2) / len(FeHrank)
    sy = numpy.sum((massrank-numpy.mean(massrank))**2) / len(massrank)
    rho = sxy / numpy.sqrt(sx * sy)

    mmr_mean, edges, binnumbers = scipy.stats.binned_statistic(
        part2["FeH"][finite], numpy.log10(mass[finite]),
        statistic="mean"
    )
    mmr_sem, edges, binnumbers = scipy.stats.binned_statistic(
        part2["FeH"][finite], numpy.log10(mass[finite]),
        statistic=lambda array: scipy.stats.sem(array)
    )
    mmr_std, edges, binnumbers = scipy.stats.binned_statistic(
        part2["FeH"][finite], numpy.log10(mass[finite]),
        statistic=lambda array: numpy.std(array)
    )
    ax.errorbar((edges[1:]+edges[:-1])/2, mmr_mean, yerr=mmr_sem,
        ls="none", marker="o", c="r", ms=8)

    if do_fit:
        from tlrh_statistics import fit_to_data
        fitfunc = lambda p, x: p[0]*x + p[1]
        x = (edges[1:]+edges[:-1])/2
        x_ana = numpy.linspace(-2.8, 0.4, 128)
        popt, perr = fit_to_data(ax, x, mmr_mean, mmr_std,
            fitfunc, [0, 5], x_ana=x_ana)

    # Plot x|y
    mmr_mean_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        numpy.log10(mass[finite]), part2["FeH"][finite],
        statistic="mean"
    )
    mmr_sem_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        numpy.log10(mass[finite]), part2["FeH"][finite],
        statistic=lambda array: scipy.stats.sem(array)
    )
    mmr_std_inv, edges_inv, binnumbers_inv = scipy.stats.binned_statistic(
        numpy.log10(mass[finite]), part2["FeH"][finite],
        statistic=lambda array: numpy.std(array)
    )
    ax.errorbar(mmr_mean_inv, (edges_inv[1:]+edges_inv[:-1])/2, xerr=mmr_sem_inv,
        ls="none", marker="o", c="b", ms=8)

    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel(r"log$_{10}(\rm M/\rm M_{\odot})$")
    ax.set_xticks(numpy.arange(-2.5, 0.5, 0.5))
    ax.set_xticks(numpy.arange(-2.8, 0.5, 0.1), minor=True)
    ax.set_yticks(numpy.arange(3, 8, 1))
    ax.set_yticks(numpy.arange(3, 7.2, 0.2), minor=True)
    pyplot.xlim(-2.8, 0.4)
    pyplot.ylim(3, 7)

    pyplot.show()


if __name__ == "__main__":
    pyplot.switch_backend("agg")
    pyplot.style.use("tlrh")

    part1, part2, part3 = read_harris1996_data()
    print_harris1996_data(part1, part2, part3, example=True)

    print("There are {0: >3d} globular clusters in total".format(len(part2)))

    i_has_M_vt, = numpy.where( numpy.isfinite(part2["M_Vt"]) )
    print("There are {0: >3d} globular clusters /w M_Vt measurement".format(
        len(i_has_M_vt)))

    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH, = numpy.where( (part2["FeH"] > -6) & (part2["FeH"] < 4) )
    print("There are {0: >3d} globular clusters /w FeH measurement".format(
        len(i_has_FeH)))

    # plot_harris1996_figure1_2(part1)
    # plot_harris1996_figure1_2_th(part1, part2)
    # plot_harris1996_figure1_3(part1, part2)
    # plot_harris1996_figure1_3_th(part1, part2)
    # plot_harris1996_figure1_7(part2)
    # plot_harris1996_figure1_7_th(part2)
    # plot_MW_mass_distribution(part2)
