# Paper  VII: http://adsabs.harvard.edu/abs/2016ApJ...824...42C
#      data: http://vizier.cfa.harvard.edu/viz-bin/Cat?J/ApJ/824/42
#        --> 411 GCs

# Data can also be found at
#   https://www.cfa.harvard.edu/oir/eg/m31clusters/M31_Hectospec.html
# In particular, using
#     https://www.cfa.harvard.edu/oir/eg/m31clusters/CR16_table1.txt
# However, we do not know whether the above url was updated most recently,
# or whether the Vizier data should be used.
# In any case, the Barmy+ 2000, Colucci+ 2014, Peretti+ 2002, Caldwell+ 2011,
# Strader+ 2011, Huxor+ 2014, Galetti+ 2009, Mackey+ 2006, 2007, 2013,
# Perina+ 2011, Rich+ 2005 data is used for this data set.

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


def read_caldwell_romanowsky_2016_data(debug=False):
    url = "https://www.cfa.harvard.edu/oir/eg/m31clusters/CR16_table1.txt"
    fname = "../data/CaldwellRomanowsky2016/CR16_table1.txt"
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)

    if not os.path.isfile(fname):
        import urllib.request
        urllib.request.urlretrieve(url, fname)

    Ncols = 14 + 5
    nrows_header = 55
    Nrows = sum(1 for line in open(fname, "r")) - nrows_header

    # Default to float in column, but change for known strings
    formats =[numpy.float for i in range(Ncols)]
    formats[0] = "S15"; formats[1] = "S15"; formats[2] = "S15";
    formats[5] = "S3"; formats[8] = "S3"; formats[10] = "S1";
    formats[-5] = "object"; formats[-4] = "object";

    # Name columns to access 'm by later-on
    names = [
        "Name ", "RA", "DEC", "RVel", "e_RVel", "r_RVel", "[Fe/H]", "e_[Fe/H]",
        "r_[Fe/H]", "Age", "u_Age", "LogM", "R", "Ra",
        "SkyCoord", "ICRS", "Rproj", "Rdeproj", "Rgc"
    ]

    # Pack it all up, and initialise empty array
    dtype = { "names": names, "formats": formats }
    data = numpy.empty(Nrows, dtype=dtype)

    i_start = [ 1, 12, 24, 35, 42, 47, 51, 56, 60, 65, 69, 71, 75, 81]
    i_end   = [10, 22, 33, 40, 45, 49, 54, 58, 62, 68, 69, 73, 79, 85]

    if debug: print(data.dtype)
    with open(fname, "rb") as f:
        for i, row in enumerate(f.readlines()):
            if i < nrows_header: continue
            if debug:
                print(row)
            # SkyCoord, ICRS, Rproj, Rdeproj not in data
            for j, (i_s, i_e) in enumerate(zip(i_start, i_end)):
                value = row[i_s-1:i_e].strip()
                if debug: print(j, value)
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


def print_caldwell_romanowsky_2016_data(data, example=True):
    width = 135

    print("\n{0}".format("-"*width))
    print("{:<15s}{:^15s}{:^15s}{:^8s}{:^8s}{:^8s}{:^8s}".format(
        "Name", "RA", "DEC", "Rvel", "e_RVel", "r_Rvel", "[Fe/H]"), end="")
    print("{:^8s}{:^8s}{:^8s}{:^8}{:^8s}{:^8s}{:^8s}".format(
        "e_[Fe/H]", "r_[Fe/H]", "Age", "u_Age", "LogM", "R", "Ra"))
    print("{0}".format("-"*width))
    for i, row in enumerate(data):
        print("{:<15s}{:^15s}{:^15s}{:^8.2f}{:^8.2f}{:^8s}{:^8.2f}".format(
            row[0].decode("ascii"), row[1].decode("ascii"), row[2].decode("ascii"),
            row[3], row[4], row[5].decode("ascii"), row[6]), end="")
        print("{:^8.2f}{:^8s}{:^8.2f}{:^8s}{:^8.2f}{:^8.2f}{:^8.2f}".format(
            row[7], row[8].decode("ascii"), row[9], row[10].decode("ascii"),
            row[11], row[12], row[13]))
        if example and i > 3: break
    print("{0}\n".format("-"*width))


if __name__ == "__main__":
    pyplot.style.use("tlrh")
    data = read_caldwell_romanowsky_2016_data()

    print("There are {0: >3d} globular clusters in total".format(len(data)))

    i_has_vel, = numpy.where( numpy.isfinite(data["RVel"]) )
    print("There are {0: >3d} globular clusters /w velocity measurement".format(len(i_has_vel)))

    i_has_logM, = numpy.where( numpy.isfinite(data["LogM"]) )
    print("There are {0: >3d} globular clusters /w logM measurement".format(len(i_has_logM)))


    with suppress_stdout():
        i_has_FeH, = numpy.where( (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4) )
    print("There are {0: >3d} globular clusters /w [Fe/H] measurement".format(len(i_has_FeH)))

    with suppress_stdout():
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(data["[Fe/H]"])
            & (data["[Fe/H]"] > -6) & (data["[Fe/H]"] < 4)
            & numpy.isfinite(data["LogM"])
        )
    print("There are {0: >3d} globular clusters /w [Fe/H] and logM measurement".format(len(i_has_FeH_and_logM)))

    with suppress_stdout():
        i_has_age, = numpy.where( (data["Age"] < 13.99 ) )
    print("There are {0: >3d} globular clusters /w age measurement".format(len(i_has_age)))
    with suppress_stdout():
        i_has_age_and_logM, = numpy.where(
            (data["Age"] < 13.99 )
            & numpy.isfinite(data["LogM"])
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
            & numpy.isfinite(data["LogM"])
            & (data["Age"] < 13.99 )
        )
    print("There are {0: >3d} globular clusters /w [Fe/H], age, and logM measurement".format(len(i_has_FeH_age_and_logM)))

