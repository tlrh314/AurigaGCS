import os
import numpy
import matplotlib
from matplotlib import pyplot


def read_vandenberg2013_data(fname="../data/VandenBerg2013_table2.txt"):
    dirname = os.path.dirname(__file__)
    fname = os.path.join(dirname, fname)
    if not os.path.isfile(fname) or not os.path.exists(fname):
        print("ERROR: file not found: {0}".format(fname))

    dtype = [
        "int", "S10", "float", "float", "float", "S1", "S5", "S12",
        "float", "float", "float", "float", "float"
    ]
    names = [
        "NGC", "Name", "FeH", "Age", "fAge", "Method", "Fig", "Range",
        "HBType", "R_GC", "M_V", "v_e,0", "log10sigma0"
    ]
    data = numpy.genfromtxt(fname, delimiter=",", dtype=dtype, names=names)

    # The galactocentric radius of the Milky Way is calculated by Harris
    # as R_gc = sqrt( (X-8)^2 + Y^2 + Z^2 ), where X, Y, Z are in a
    # Sun-centered coordinate system. For a 'fair' comparison of the MW
    # calactocentric radius we multiply the three-dimensional radius by pi/4,
    # i.e. Rproj = Rgc × (π/4), as per Huxor (2014, Fig. 17)
    dtype = numpy.dtype(data.dtype.descr + [("Rproj", "float")])
    data_Rproj = numpy.empty(data.shape, dtype=dtype)
    for name in data.dtype.names:
        data_Rproj[name] = data[name]
    data_Rproj["Rproj"] = numpy.pi/4 * data["R_GC"]
    data = data_Rproj

    return data


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



def plot_vandenberg2013_figure33(data, ax=None):
    if not ax:
        fig, ax = pyplot.subplots(figsize=(12, 9))

    label = r"\begin{{tabular}}{{p{{6cm}}l}}MW: VandenBerg (2013); & Ngc = {0}\end{{tabular}}"\
        .format(len(data))
    ax.errorbar(data["FeH"], data["Age"], xerr=0.15, yerr=data["fAge"],
        marker="o", ls="", c="k", ms=4, elinewidth=2, capsize=4, label=label)

    ax.set_xticks(numpy.arange(-2.5, 0.5, 0.5))
    ax.set_xticks(numpy.arange(-2.5, 0.1, 0.1), minor=True)
    ax.set_yticks(numpy.arange(10, 15, 1))
    ax.set_yticks(numpy.arange(10, 14.2, 0.2), minor=True)
    pyplot.xlim(-2.6, 0)
    pyplot.ylim(9.8, 14)
    pyplot.xlabel("[Fe/H]")
    pyplot.ylabel("Age (Gyr)")
    if not ax:
        pyplot.show()


if __name__ == "__main__":
    pyplot.switch_backend("agg")
    pyplot.style.use("tlrh")

    data = read_vandenberg2013_data()
    plot_vandenberg2013_figure33(data)
