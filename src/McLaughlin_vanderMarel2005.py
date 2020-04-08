import os
import numpy
import matplotlib
from matplotlib import pyplot
from astroquery.vizier import Vizier


def read_McLaughlin_vanderMarel2005_data(verbose=True):
    """ We present a database of structural and dynamical properties for 153
    spatially resolved star clusters in the Milky Way, the Large and Small
    Magellanic Clouds, and the Fornax dwarf spheroidal. This database
    complements and extends others in the literature, such as those of Harris
    and Mackey & Gilmore. Our cluster sample comprises 50 young massive clusters
    in the LMC and SMC, and 103 old globular clusters between the four galaxies """
    # This means that by excluding LMC, SMC data we automatically exclude
    # YMCs, thus, we should be left /w only GCs...
    # Am confused though, as to why catalog contains 216 items (not 153?)
    # Also, there's 0: clusters (10 cols/216 rows), 1: table5 (9 cols/4689 rows),
    # 2: models (17 cols/459 rows), 3: table13 (12 cols/67 rows)

    cat_name = "J/ApJS/161/304"
    if verbose:
        print("Retrieving Vizier catalog: '{0}'".format(cat_name))

    Vizier.ROW_LIMIT = -1  # default 50. Now unlimited :)
    return Vizier.get_catalogs(cat_name)[0]


def plot_milky_way_FeH(data):
    # http://vizier.cfa.harvard.edu/viz-bin/VizieR-n?-source=METAnot&catid=21610304&notid=5&-out=text
    # Note(1): Ages, metallicities, and their uncertainties for LMC, SMC, and
    # Fornax clusters are taken from the compilation and analysis by Mackey &
    # Gilmore (2003MNRAS.338...85M, 2003MNRAS.338..120M, 2003MNRAS.340..175M).
    # All ages given as greater than 13Gyr by Mackey & Gilmore have been re-set
    # here to 13±2Gyr [i.e., log(age)=10.11±0.07] . Milky Way globular cluster
    # ages are set to a uniform 13±2Gyr, while their individual metallicitie
    # are taken from the catalog of Harris (1996, Cat. VII/202) and assigned
    # uncertainties of ±0.2dex.

    # http://vizier.cfa.harvard.edu/viz-bin/VizieR-n?-source=METAnot&catid=21610304&notid=5&-out=text
    # Note (2): Aperture (B-V) colors for LMC, SMC, and Fornax clusters are
    # from the literature summarized in Table 3. (B-V) colors of Milky Way
    # globular clusters are taken from Harris (1996, Cat. VII/202), where
    # available. No corrections for reddening have been applied.

    i_is_MW_and_has_FeH, = numpy.where(
        (data["__Fe_H_"] != 10.11)
    )



def print_vandenberg2013_data(data, example=True):
    width = 115

    print("\n{0}".format("-"*width))
    print("{0:<6s}{1:^7}{2:^8s}{3:^8s}{4:^8s}{5:^8s}{6:^7s}{7:^16s}".format(
        "NGC", "Name", "[Fe/H]", "Age", "fAge", "Method", "Fig", "Range"), end="")
    print("{0:>8s}{1:>8s}{2:>8s}{3:>8s}{4:>15s}".format(
        "HBType", "R_GC", "M_V", "v_e,0", "log(sigma0)"))
    print("{0}".format("-"*width))
    for i, row in enumerate(data):
        print("{0:<6s}{1:^7s}{2:^8.2f}{3:^8.2f}{4:^8.2f}{5:^8s}{6:^7s}{7:^16s}".format(
            str(row[0]), row[1].decode("ascii"), row[2], row[3],
            row[4], row[5].decode("ascii"), row[6].decode("ascii"),
            row[7].decode("ascii")), end="")
        print("{0: 8.1f}{1: 8.1f}{2: 8.1f}{3: 8.1f}{4: 15.1f}".format(
            row[8], row[9], row[10], row[11], row[12]))
        if example and i > 3: break
    print("{0}\n".format("-"*width))


if __name__ == "__main__":
    pyplot.switch_backend("agg")
    pyplot.style.use("tlrh")

    data = read_McLaughlin_vanderMarel2005_data()
