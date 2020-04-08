# Paper   I: http://adsabs.harvard.edu/cgi-bin/bib_query?2012A&A...543A.156K
#        --> 2 GCs
# Paper  II: http://adsabs.harvard.edu/cgi-bin/bib_query?2013A&A...558A..53K
#            http://adsabs.harvard.edu/abs/2013yCat..35580053K
#      data: ftp://cdsarc.u-strasbg.fr/pub/cats/J/A%2BA/558/A53
#        --> 142 GCs + 5 candidates (90% of Harris 1996, 2010 edition)
# Paper III: http://adsabs.harvard.edu/cgi-bin/bib_query?2014A&A...568A..51S
# Paper  IV: http://adsabs.harvard.edu/abs/2015A%26A...581A..39S
#      data: http://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/581/A39

import numpy
import pandas
from matplotlib import pyplot


def read_kharchenko2013_data(verbose=True, fname="../data/MWSC/catalog.dat"):
    Ncols = 40
    Nrows = sum(1 for line in open(fname, "r"))

    # Default to float in column, but change for known strings
    formats =[numpy.float for i in range(Ncols)]
    formats[0] = "S4"; formats[1] = "S12"; formats[2] = "S2";  formats[3] = "S2"
    formats[35] = "S12"; formats[36] = "S3"

    # Name columns to access 'm by later-on
    names = [
        "MWSC", "Name", "Type", "n_Type", "RAhour", "DEdeg", "GLON",
        "GLAT", "r0", "r1", "r2", "pmRA", "pmDE", "e_pm", "RV", "e_RV",
        "o_RV", "N1sr0", "N1sr1", "N1sr2", "d", "E(B-V)", "MOD", "E(J-Ks",
        "E(J-H)", "dH", "logt", "e_logt", "Nt", "rc", "e_rc", "rt", "e_rt",
        "k", "e_k", "Src", "SType", "[Fe/H]", "e_[Fe/H]", "o_[Fe/H]"
    ]

    # Pack it all up, and initialise empty array
    dtype = { "names": names, "formats": formats}
    data = numpy.empty(Nrows, dtype=dtype)

    # Start and end indices from MWSC catalog's ReadMe file
    cs = [
        1, 6, 22, 24, 26, 35, 43, 51, 59, 66, 73, 80, 87, 94, 101, 109, 117, 123,
        129, 136, 143, 151, 158, 165, 172, 179, 186, 193, 200, 204,
        212, 220, 228, 236, 244, 253, 258, 262, 270, 277
    ]
    ce = [
        4, 22, 24, 25, 34, 42, 50, 58, 65, 72, 79, 86, 93, 100, 108, 116, 122,
        128, 135, 142, 150, 157, 164, 171, 178, 185, 192, 199, 203,
        211, 219, 227, 235, 243, 251, 256, 260, 269, 276, 280
    ]

    with open(fname, "rb") as f:
        for i, row in enumerate(f.readlines()):
            for j in range(Ncols):
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

    if verbose:
        print("Succesfully read: '{0}'".format(fname))
        print("Usage: data = read_kharchenko2013_data() ")
        print("You can then access rows using data[0]")
        print("You can access columns using data['colname']")
        print("To find all column names, use 'data.dtype.names'")

    return data

if __name__ == "__main__":
    data = read_kharchenko2013_data()

    print("There are {0} star clusters in total".format(len(data)))

    i_not_gc, = numpy.where(data["n_Type"] != b"go")
    print("There are {0} not globular clusters".format(len(i_not_gc)))

    i_has_FeH, = numpy.where( (data['[Fe/H]'] > -6) & (data['[Fe/H]'] < 4) )
    print("There are {0} star clusters /w [Fe/H] measurements".format(len(i_has_FeH)))

    i_gc, = numpy.where(data["n_Type"] == b"go")
    print("There are {0} globular clusters".format(len(i_gc)))

    i_gc_candidate, = numpy.where(data["n_Type"] == b"gc")
    print("There are {0} globular clusters candidates".format(len(i_gc_candidate)))

    i_gc_all, = numpy.where( (data["n_Type"] == b"go") | (data["n_Type"] == b"gc") )
    print("There are {0} globular clusters including candidates".format(len(i_gc_all)))

    i_gc_all_has_FeH, = numpy.where( (data['[Fe/H]'] > -6) & (data['[Fe/H]'] < 4)
        & ((data["n_Type"] == b"go") | (data["n_Type"] == b"gc")) )
    print("There are {0} globular clusters including candidates /w [Fe/H] measurements".format(len(i_gc_all_has_FeH)))

    i_has_age, = numpy.where( (data['Nt'] > -1) )
    print("There are {0} star clusters /w age-estimates".format(len(i_has_age)))

    i_gc_all_has_age, = numpy.where( (data['Nt'] > -1)
        & ((data["n_Type"] == b"go") | (data["n_Type"] == b"gc")) )
    print("There are {0} globular clusters including candidates /w age-estimates".format(len(i_gc_all_has_age)))

    i_gc_has_FeH, = numpy.where( (data["n_Type"] == b"go") & (data['[Fe/H]'] > -6) & (data['[Fe/H]'] < 4) )
    # Nt: number of stars used for age-estimate. When Nt == -1, no age estimate available
    i_gc_has_logt, = numpy.where( (data["n_Type"] == b"go") & (data['Nt'] > -1) )

    # [Fe/H] distribution
    pyplot.figure(figsize=(12, 9))

    counts, edges = numpy.histogram(data['[Fe/H]'][i_has_FeH], bins=24, range=[-2.5, 0.5])
    pyplot.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="grey", ls="--", label="Star Clusters /w [Fe/H] measurement")
    counts, edges = numpy.histogram(data['[Fe/H]'][i_gc_has_FeH], bins=12, range=[-2.5, 0.5])
    pyplot.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="k", label="Globular Clusters /w [Fe/H] measurement")

    pyplot.xlabel("[Fe/H]")
    pyplot.ylabel("Count")
    pyplot.legend(frameon=False)

    pyplot.show()

    # Age distribution
    pyplot.figure(figsize=(12, 9))

    counts, edges = numpy.histogram(data['logt'][i_not_gc], bins=28, normed=True, range=[0, 14])
    pyplot.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="grey", ls="--", label="Star Clusters")
    # use i_gc_has_logt, but this is an empty set. This histogram is meaningless.
    counts, edges = numpy.histogram(data['logt'][i_gc], bins=28, normed=True, range=[0, 14])
    pyplot.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="k", label="Globular Clusters")

    pyplot.xlabel("Age")
    pyplot.ylabel("Normed Count")
    pyplot.legend(frameon=False)

    pyplot.show()
