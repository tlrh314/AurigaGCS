import os
import numpy
import scipy
import colorcet
import matplotlib
from matplotlib import pyplot
pyplot.style.use("tlrh")
from scipy.stats import binned_statistic
from scipy.stats import binned_statistic_2d

from main import COLORS
from Caldwell2011 import read_caldwell2011_data
from VandenBerg2013 import read_vandenberg2013_data
from Kharchenko2013 import read_kharchenko2013_data
from Harris1996_2010edition import get_MW_GCS_FeH
from Harris1996_2010edition import read_harris1996_data
from Harris1996_2010edition import print_harris1996_data
from Harris1996_2010edition import plot_MW_mass_distribution
from Harris1996_2010edition import combine_part1_part2_part3
from CaldwellRomanowsky2016 import read_caldwell_romanowsky_2016_data
from McLaughlin_vanderMarel2005 import read_McLaughlin_vanderMarel2005_data

from tlrh_util import suppress_stdout


def compare_harris_and_kharchenko():
    kharchenko2013 = read_kharchenko2013_data(verbose=False)
    i_gc_has_FeH, = numpy.where(
        (kharchenko2013['[Fe/H]'] > -6) & (kharchenko2013['[Fe/H]'] < 4)
        & (  (kharchenko2013["n_Type"] == b"go")  # confirmed
           | (kharchenko2013["n_Type"] == b"gc")  # candidate
          )
    )

    part1, part2, part3 = read_harris1996_data()

    map = {}
    for i in i_gc_has_FeH:
        # print("i = {0}".format(i))
        kh13_name = str(kharchenko2013['Name'][i])\
            .replace("_", " ").replace("Palomar", "Pal")
        # print("Name = {0}".format(kh13_name))

        for j, harris1996_name in enumerate(part1["ID"]):
            if str(harris1996_name) == str(kh13_name):
                # print("Index in harris: {0}".format(j))
                # print("Name: {0}".format(harris1996_name))
                map[i] = j
                break
        else:
            print("Not in Harris1996_2010edition: {0}, type = {1}".format(
                kh13_name, kharchenko2013["n_Type"][i]))
    print("")

    fig, ax = pyplot.subplots(1, 1, figsize=(8, 8))

    xlin = numpy.linspace(-6, 2, 8)
    ax.plot(xlin, xlin, c="k", label="[Fe/H] equal")

    for k, v in map.items():
        print("Harris (2001):     {0:<15s} -> {1}".format(
            part1['ID'][v], part2['FeH'][v]))
        print("Kharchenko (2013): {0:<15s} -> {1}".format(
            kharchenko2013['Name'][k], kharchenko2013['[Fe/H]'][k]))
        ax.plot(part2['FeH'][v], kharchenko2013['[Fe/H]'][k], "rX", ms=5)

    ax.set_xlabel("Harris (2001)")
    ax.set_ylabel("Kharchenko (2013)")
    ax.set_xticks(numpy.arange(-3, 1, 0.5))
    ax.set_yticks(numpy.arange(-3, 1, 0.5))
    ax.set_xlim(-3, 0.5)
    ax.set_ylim(-3, 0.5)

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=16)

    pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_" +
        "Harris1996_vs_Kharchenko2013_FeH.pdf")
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_" +
        "Harris1996_vs_Kharchenko2013_FeH.png")
    pyplot.show()


def plot_milky_way_globular_cluster_system_FeH(
        bins=30, range=(-2.5, 0.5), normed=False):
    fig, ax = pyplot.subplots(figsize=(12, 9))

    harchenko2013 = read_data(verbose=False)
    i_gc_has_FeH, = numpy.where(
        (kharchenko2013['[Fe/H]'] > -6) & (kharchenko2013['[Fe/H]'] < 4)
        & (  (kharchenko2013["n_Type"] == b"go")  # confirmed
           | (kharchenko2013["n_Type"] == b"gc")  # candidate
          )
    )

    counts1, edges = numpy.histogram(kharchenko2013['[Fe/H]'][i_gc_has_FeH],
        bins=bins, range=range, normed=normed)
    label = r"\begin{tabular}{p{6.5cm}l}Kharchenko (2013); & " + \
        "N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_gc_has_FeH))
    ax.plot((edges[1:]+edges[:-1])/2, counts1, drawstyle="steps-mid",
        c="k", label=label)

    # Harris 2001
    (counts, edges), Ngc = get_MW_GCS_FeH(bins=bins, range=range, normed=normed)
    label = r"\begin{tabular}{p{6.5cm}l}MW (H96e10); & " + \
        "N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(Ngc)
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="r", label=label)

    for i, (l, r) in enumerate(zip(edges[1:], edges[:-1])):
        ax.fill_between([l, r], y1=counts[i], y2=counts1[i],
            hatch="////", alpha=0.2,
            edgecolor="black" if counts[i] < counts1[i] else "red",
            facecolor="none")

    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

    ax.set_xlabel("[Fe/H]")
    ax.set_ylabel("Count")
    ax.legend(fontsize=16, frameon=False)

    pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_FeH.pdf")
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_FeH.png")
    pyplot.show()


def calculate_milky_way_globular_cluster_system_total_mass(
        Mv_Sun=4.83, mass_to_light=1.7):
    part1, part2, part3 = read_harris1996_data()

    isfinite, = numpy.where( numpy.isfinite(part2["M_Vt"]) )
    mass = numpy.power(10, 0.4*(Mv_Sun - part2["M_Vt"][isfinite])) * mass_to_light

    print("The total mass of MW's GCS is: {0:.3f} * 1e7 Msun".format(
        mass.sum()/1e7))


def plot_milky_way_globular_cluster_system_mass(
        Mv_Sun=4.83, mass_to_light=1.7, nbins=14):
    part1, part2, part3 = read_harris1996_data()

    fig, ax = pyplot.subplots(figsize=(12, 9))

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


def plot_milky_way_globular_cluster_system_mass_histogram(
        Mv_Sun=4.83, mass_to_light=1.7, nbins=16):
    part1, part2, part3 = read_harris1996_data()

    fig, ax = pyplot.subplots(figsize=(12, 9))

    isfinite, = numpy.where( numpy.isfinite(part2["M_Vt"]) )
    mass = numpy.power(10, 0.4*(Mv_Sun - part2["M_Vt"][isfinite])) * mass_to_light
    counts, edges = numpy.histogram(numpy.log10(mass), bins=nbins, range=(2.5, 6.5))
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c="k", label="Milky Way Globular Cluster System")

    ax.set_xticks(numpy.arange(2, 8, 0.5), minor=True)
    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.set_xlabel(r"log$_{10}(M_{\rm GC}$/M$_\odot$)")
    ax.set_ylabel("Count")
    ax.set_xlim(2.5, 6.5)
    ax.set_ylim(0, 35)
    ax.legend(fontsize=16, frameon=False, loc="upper left")

    pyplot.tight_layout()
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_MassHistogram.pdf")
    pyplot.savefig("../out/MilkyWay_GlobularClusterSystem_MassHistogram.png")
    pyplot.show()


def read_MWandM31_data(Mv_Sun=4.83, mass_to_light=1.7):
    p1, p2, p3 = read_harris1996_data()
    MW_h96e10 = combine_part1_part2_part3(p1, p2, p3,
        Mv_Sun=Mv_Sun, mass_to_light=mass_to_light)
    MW_v13 = read_vandenberg2013_data()
    M31_c11 = read_caldwell2011_data(verbose=False)
    M31_cr16 = read_caldwell_romanowsky_2016_data()

    return MW_h96e10, MW_v13, M31_c11, M31_cr16


def add_age_MWandM31_to_ax(ax, MW_v13, M31_cr16, M31_c11=None,
        bins=28, myrange=(0, 14), density=False, set_ticks=True, grid=True):

    # Milky Way as per vandenBerg+ (2013)
    mwage, = numpy.where(
        (MW_v13["Age"] < 13.99)
    )
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (V13); & " + \
        r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwage))
    print("There are {0: >3d} {1}".format(len(mwage),
        "MW globular clusters /w age measurement"))
    print("Mean: {0:.1f}, stderr: {1:.1f}\nstd: {2:.1f}, min: {3:.1f}\n".format(
        numpy.mean(MW_v13["Age"][mwage]),
        scipy.stats.sem(MW_v13["Age"][mwage]),
        numpy.std(MW_v13["Age"][mwage]),
        numpy.min(MW_v13["Age"][mwage])
    ))

    counts, edges = numpy.histogram(
        MW_v13["Age"][mwage], bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        lw=4, c=COLORS["MW"], label=MW_GCS_label)

    # M31 as per Caldwell & Romanowsky (2016)
    with suppress_stdout():
        i_has_age, = numpy.where(
            (M31_cr16["Age"] < 13.99 )
        )
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); " + \
        r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_age))
    print("There are {0: >3d} {1}".format(len(i_has_age),
         "M31 globular clusters /w age measurement (CR16)"))
    print("Mean: {0:.1f}, stderr: {1:.1f}\nstd: {2:.1f}, min: {3:.1f}\n".format(
        numpy.mean(M31_cr16["Age"][i_has_age]),
        scipy.stats.sem(M31_cr16["Age"][i_has_age]),
        numpy.std(M31_cr16["Age"][i_has_age]),
        numpy.min(M31_cr16["Age"][i_has_age])
    ))

    counts, edges = numpy.histogram(
        M31_cr16["Age"][i_has_age],
        bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        lw=4, c=COLORS["M31"], label=M31_GCS_label)

    # M31 as per Caldwell (2011)
    if M31_c11 is not None:
        with suppress_stdout():  # invalid value encountered in < and/or >
            i_has_age, = numpy.where(
                numpy.isfinite(M31_c11["Age"])
            )
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); " + \
            r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_age))
        print("There are {0: >3d} {1}".format(len(i_has_age),
             "M31 globular clusters /w age measurement (C11)"))
        print("Mean: {0:.1f}, stderr: {1:.1f}\nstd: {2:.1f}, min: {3:.1f}\n".format(
            numpy.mean(M31_c11["Age"][i_has_age]),
            scipy.stats.sem(M31_c11["Age"][i_has_age]),
            numpy.std(M31_c11["Age"][i_has_age]),
            numpy.min(M31_c11["Age"][i_has_age])
        ))

        counts, edges = numpy.histogram(
            M31_c11["Age"][i_has_age],
            bins=bins, range=myrange, density=density
        )
        ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
            lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Count" if not density else "Normalized Count")
    if set_ticks:
        ax.set_xticks(numpy.arange(0, 15, 1))
        ax.set_xticks(numpy.arange(0, 15, 0.5), minor=True)
    #     ax.set_yticks(numpy.arange(0, 1.2, 0.1))
    #     ax.set_yticks(numpy.arange(0, 1.14, 0.02), minor=True)
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_age_logM_MWandM31_to_ax(ax, MW_v13, M31_cr16, M31_c11=None,
        bins=28, myrange=(0, 14), Mv_Sun=4.83, mass_to_light=1.7,
        set_ticks=True, grid=True):

    # M31 as per Caldwell & Romanowsky (2016)
    with suppress_stdout():
        i_has_age_and_logM, = numpy.where(
            (M31_cr16["Age"] < 13.99 )
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); " + \
        r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_age_and_logM))
    print("There are {0: >3d} {1}".format(len(i_has_age_and_logM),
         "M31 globular clusters /w age and logM measurement (CR16)"))

    Msum, Medges, Mcnt = binned_statistic(
        M31_cr16["Age"][i_has_age_and_logM],
        10**M31_cr16["LogM"][i_has_age_and_logM],
        bins=bins, statistic="sum", range=myrange
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum, drawstyle="steps-mid",
        lw=4, c=COLORS["M31"], label=M31_GCS_label)

    # M31 as per Caldwell (2011)
    if M31_c11 is not None:
        with suppress_stdout():  # invalid value encountered in < and/or >
            i_has_age_and_logM, = numpy.where(
                numpy.isfinite(M31_c11["Age"])
                & (M31_c11["[Fe/H]"] > -6) & (M31_c11["[Fe/H]"] < 4)
                & numpy.isfinite(M31_c11["logM"])
            )
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); " + \
            r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_age_and_logM))
        print("There are {0: >3d} {1}".format(len(i_has_age_and_logM),
             "M31 globular clusters /w age and logM measurement (C11)"))

        Msum, Medges, Mcnt = binned_statistic(
            M31_c11["Age"][i_has_age_and_logM],
            10**M31_c11["logM"][i_has_age_and_logM],
            bins=bins, statistic="sum", range=myrange
        )
        ax.plot((Medges[1:]+Medges[:-1])/2, Msum, drawstyle="steps-mid",
            lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    # Milky Way as per vandenBerg+ (2013)
    mwage, = numpy.where(
        (MW_v13["Age"] < 13.99)
        & numpy.isfinite(MW_v13["M_V"])
    )
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_v13["M_V"][mwage])) * mass_to_light
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (V13); & " + \
        r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwage))
    print("There are {0: >3d} {1}".format(len(mwage),
        "MW globular clusters /w age and logM measurement"))

    Msum, Medges, Mcnt = binned_statistic(
        MW_v13["Age"][mwage], MW_GCS_mass,
        bins=bins, statistic="sum", range=myrange
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum, drawstyle="steps-mid",
        lw=4, c=COLORS["MW"], label=MW_GCS_label)

    ax.set_xlabel("Age [Gyr]")
    ax.set_ylabel("Globular Cluster Mass [Msun]")
    if set_ticks:
        ax.set_xticks(numpy.arange(0, 15, 1))
        ax.set_xticks(numpy.arange(0, 15, 0.5), minor=True)
    #     ax.set_yticks(numpy.arange(0, 1.2, 0.1))
    #     ax.set_yticks(numpy.arange(0, 1.14, 0.02), minor=True)
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_FeH_MWandM31_to_ax(ax, MW_h96e10, M31_cr16, M31_c11=None,
        MW_v13=False, bins=39, myrange=(-3, 1.0), density=False,
        set_ticks=False, grid=False, set_labels=False):

    # M31 as per Caldwell & Romanowsky (2016)
    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH, = numpy.where(
            (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH))
    print("There are {0: >3d} {1}".format(len(i_has_FeH),
        "globular clusters /w [Fe/H] measurement (CR16)"))

    counts, edges = numpy.histogram(
        M31_cr16["[Fe/H]"][i_has_FeH], bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        lw=4, c=COLORS["M31"], label=M31_GCS_label)

    # M31 as per Caldwell (2011)
    if M31_c11 is not None:
        with suppress_stdout():  # invalid value encountered in < and/or >
            i_has_FeH, = numpy.where(
                (M31_c11["[Fe/H]"] > -6) & (M31_c11["[Fe/H]"] < 4)
                & numpy.isfinite(M31_c11["logM"])
            )
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); & " + \
            r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH))
        print("There are {0: >3d} {1}".format(len(i_has_FeH),
            "globular clusters /w [Fe/H] measurement (C11)"))

        counts, edges = numpy.histogram(
            M31_c11["[Fe/H]"][i_has_FeH], bins=bins, range=myrange, density=density
        )
        ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
            lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    # Milky Way as per Harris (1996, 2010 ed.)
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["FeH"])
    )
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10) & " + \
        r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwgcs))

    counts, edges = numpy.histogram(
        MW_h96e10["FeH"][mwgcs], bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        lw=4, c=COLORS["MW"], label=MW_GCS_label)

    # Milky Way as per vandenBerg+ (2013)
    if type(MW_v13) == numpy.ndarray:
        mw_FeH_v13, = numpy.where(
            numpy.isfinite(MW_v13["FeH"])
        )
        MW_GCS_label_v13 = r"\begin{tabular}{p{4cm}l}MW (V13); & " + \
            r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_FeH_v13))

        counts_v13, edges_v13 = numpy.histogram(
            MW_v13["FeH"][mw_FeH_v13], bins=bins, range=myrange, density=density
        )
        ax.plot((edges_v13[1:]+edges_v13[:-1])/2, counts_v13,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["MW"],
            label=MW_GCS_label_v13)

    if set_ticks:
        ax.set_xticks(numpy.arange(-4, 1.5, 0.5))
        ax.set_xticks(numpy.arange(-4, 1.25, 0.25), minor=True)
    #     ax.set_yticks(numpy.arange(0, 1.2, 0.1))
    #     ax.set_yticks(numpy.arange(0, 1.14, 0.02), minor=True)
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

    if set_labels:
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel("Count" if not density else "Normalized Count")
        ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_FeH_logM_MWandM31_to_ax(ax, MW_h96e10, M31_cr16, M31_c11=None,
        MW_v13=False, bins=39, myrange=(-3, 1.0),
        Mv_Sun=4.83, mass_to_light=1.7, set_ticks=False,
        grid=False, set_labels=False, mass_denominator=1):

    # M31 as per Caldwell & Romanowsky (2016)
    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(M31_cr16["[Fe/H]"])
            & (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH_and_logM))
    print("There are {0: >3d} {1}".format(len(i_has_FeH_and_logM),
        "globular clusters /w [Fe/H] and logM measurement (CR16)"))

    Msum, Medges, Mcnt = binned_statistic(
        M31_cr16["[Fe/H]"][i_has_FeH_and_logM],
        10**M31_cr16["LogM"][i_has_FeH_and_logM],
        bins=bins, statistic="sum", range=myrange
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
        drawstyle="steps-mid", lw=4, c=COLORS["M31"], label=M31_GCS_label)

    # M31 as per Caldwell (2011)
    if M31_c11 is not None:
        with suppress_stdout():  # invalid value encountered in < and/or >
            i_has_FeH_and_logM, = numpy.where(
                numpy.isfinite(M31_c11["[Fe/H]"])
                & (M31_c11["[Fe/H]"] > -6) & (M31_c11["[Fe/H]"] < 4)
                & numpy.isfinite(M31_c11["logM"])
            )
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); & " + \
            r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH_and_logM))
        print("There are {0: >3d} {1}".format(len(i_has_FeH_and_logM),
            "globular clusters /w [Fe/H] and logM measurement (C11)"))

        Msum, Medges, Mcnt = binned_statistic(
            M31_c11["[Fe/H]"][i_has_FeH_and_logM],
            10**M31_c11["logM"][i_has_FeH_and_logM],
            bins=bins, statistic="sum", range=myrange
        )
        ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    # Milky Way as per Harris (1996, 2010 ed.)
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["M_Vt"])
    )
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_h96e10["M_Vt"][mwgcs])) * mass_to_light
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10) & " + \
        r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwgcs))

    Msum, Medges, Mcnt = binned_statistic(
        MW_h96e10["FeH"][mwgcs], MW_GCS_mass,
        bins=bins, statistic="sum", range=myrange
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
        drawstyle="steps-mid", lw=4, c=COLORS["MW"], label=MW_GCS_label)

    # Milky Way as per vandenBerg+ (2013)
    if type(MW_v13) == numpy.ndarray:
        mw_FeH_v13, = numpy.where(
            numpy.isfinite(MW_v13["FeH"])
        )
        MW_GCS_mass_v13 = numpy.power(10,
            0.4*(Mv_Sun - MW_v13["M_V"][mw_FeH_v13])) * mass_to_light
        MW_GCS_label_v13 = r"\begin{tabular}{p{4cm}l}MW (V13); & " + \
            r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_FeH_v13))

        Msum_v13, Medges_v13, Mcnt_v13 = binned_statistic(
            MW_v13["FeH"][mw_FeH_v13], MW_GCS_mass_v13,
            bins=bins, statistic="sum", range=myrange
        )
        ax.plot((Medges_v13[1:]+Medges_v13[:-1])/2, Msum_v13/mass_denominator,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["MW"],
            label=MW_GCS_label_v13)

    if set_ticks:
        ax.set_xticks(numpy.arange(-4, 1.5, 0.5))
        ax.set_xticks(numpy.arange(-4, 1.25, 0.25), minor=True)
    #     ax.set_yticks(numpy.arange(0, 1.2, 0.1))
    #     ax.set_yticks(numpy.arange(0, 1.14, 0.02), minor=True)
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

    if set_labels:
        ax.set_xlabel("[Fe/H]")
        ax.set_ylabel("Globular Cluster Mass [{0}Msun]".format(
            str(mass_denominator)+" " if mass_denominator != 1 else ""))
        ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_Rgc_MWandM31_to_ax(ax, MW_h96e10, M31_cr16,
        M31_c11=None, MW_v13=False, nbins=32, myrange=(0.1, 500),
        density=False, set_ticks=True, grid=True, pioverfour=True,
        show_cr16_age_subset=False, MW_rvir=1.0, M31_rvir=1.0):

    # Logarithmic bins in x-direction
    bins = numpy.power(10, numpy.linspace(numpy.log10(myrange[0]),
        numpy.log10(myrange[1]), nbins))

    # M31 as per Caldwell & Romanowsky (2016)
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(M31_cr16["Rgc"]))

    counts, edges = numpy.histogram(
        M31_cr16["Rgc"], bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        lw=4, c=COLORS["M31"], label=M31_GCS_label)

    if show_cr16_age_subset:
        with suppress_stdout():
            i_has_age, = numpy.where(
                (M31_cr16["Age"] < 13.99 )
            )
        # M31 as per Caldwell & Romanowsky (2016), but with age data
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
            r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_age))
        counts, edges = numpy.histogram(
            M31_cr16["Rgc"][i_has_age], bins=bins, range=myrange, density=density
        )
        ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
            lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    if M31_c11 is not None:
        # M31 as per Caldwell+ (2011)
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); & " + \
            r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(M31_c11["Rgc"]))

        counts, edges = numpy.histogram(
            M31_c11["Rgc"], bins=bins, range=myrange, density=density
        )
        ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
            lw=4, ls=":", c=COLORS["M31"], label=M31_GCS_label)

    # Milky Way as per Harris (1996, 2010 ed.)
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["R_gc"])
    )
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10);  " + \
        r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwgcs))

    counts, edges = numpy.histogram(
        MW_h96e10["Rproj"][mwgcs] if pioverfour else  MW_h96e10["R_gc"][mwgcs],
        bins=bins, range=myrange, density=density
    )
    ax.plot((edges[1:]+edges[:-1])/2, counts,
        drawstyle="steps-mid", lw=4, c=COLORS["MW"], label=MW_GCS_label)

    # Milky Way as per vandenBerg+ (2013)
    if type(MW_v13) == numpy.ndarray:
        mw_Rgc_v13, = numpy.where(
            numpy.isfinite(MW_v13["R_GC"])
        )
        MW_GCS_label_v13 = r"\begin{tabular}{p{4cm}l}MW (V13);" + \
            r" & N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_Rgc_v13))

        counts_v13, edges_v13 = numpy.histogram(
            MW_v13["Rproj"][mw_Rgc_v13] if pioverfour else MW_v13["R_GC"][mw_Rgc_v13],
            bins=bins, range=myrange, density=density
        )
        ax.plot((edges_v13[1:]+edges_v13[:-1])/2, counts_v13,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["MW"],
            label=MW_GCS_label_v13)

    if MW_rvir > 10 or M31_rvir > 10:
        ax.text(0.98, 0.96, "\\begin{{tabular}}{{r}}$r_{{\\rm vir,MW}} = {0:.0f} {{\\rm \, kpc}}$\\\\$r_{{\\rm vir,M31}} = {1:.0f} {{\\rm \, kpc}}$ \\end{{tabular}}".format(MW_rvir, M31_rvir),
                ha="right", va="top", transform=ax.transAxes)


    ax.set_xlabel("$r_{\\rm gal}$ [kpc]")
    ax.set_ylabel("Count" if not density else "Normalized Count")
    if set_ticks:
        pass
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_Rgc_logM_MWandM31_to_ax(ax, MW_h96e10, M31_cr16,
        M31_c11=None, MW_v13=False, nbins=32, myrange=(0.1, 500),
        Mv_Sun=4.83, mass_to_light=1.7, mass_denominator=1,
        set_ticks=True, grid=True, pioverfour=True):

    # Logarithmic bins in x-direction
    bins = numpy.power(10, numpy.linspace(numpy.log10(myrange[0]),
        numpy.log10(myrange[1]), nbins))

    # M31 as per Caldwell & Romanowsky (2016)
    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH_and_logM))

    Msum, Medges, Mcnt = binned_statistic(
        M31_cr16["Rgc"][i_has_FeH_and_logM],
        10**M31_cr16["LogM"][i_has_FeH_and_logM],
        bins=bins, statistic="sum", range=myrange
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
        drawstyle="steps-mid", lw=4, c=COLORS["M31"], label=M31_GCS_label)

    if M31_c11 is not None:
        # M31 as per Caldwell+ (2011)
        with suppress_stdout():  # invalid value encountered in < and/or >
            i_has_FeH_and_logM, = numpy.where(
                numpy.isfinite(M31_c11["logM"])
            )
        M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (C11); & " + \
            r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH_and_logM))

        Msum, Medges, Mcnt = binned_statistic(
            M31_c11["Rgc"][i_has_FeH_and_logM],
            10**M31_c11["logM"][i_has_FeH_and_logM],
            bins=bins, statistic="sum", range=myrange
        )
        ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["M31"],
            label=M31_GCS_label)

    # Milky Way as per Harris (1996, 2010 ed.)
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["R_gc"])
        & numpy.isfinite(MW_h96e10["M_Vt"])
    )
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_h96e10["M_Vt"][mwgcs])) * mass_to_light
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10);  " + \
        r"& N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mwgcs))

    Msum, Medges, Mcnt = binned_statistic(
        MW_h96e10["Rproj"][mwgcs] if pioverfour else  MW_h96e10["R_gc"][mwgcs],
        MW_GCS_mass,
        bins=bins, statistic="sum"
    )
    ax.plot((Medges[1:]+Medges[:-1])/2, Msum/mass_denominator,
        drawstyle="steps-mid", lw=4, c=COLORS["MW"], label=MW_GCS_label)

    # Milky Way as per vandenBerg+ (2013)
    if type(MW_v13) == numpy.ndarray:
        mw_Rgc_v13, = numpy.where(
            numpy.isfinite(MW_v13["M_V"])
            & numpy.isfinite(MW_v13["R_GC"])
        )
        MW_GCS_mass_v13 = numpy.power(10, 0.4*(Mv_Sun -
            MW_v13["M_V"][mw_Rgc_v13])) * mass_to_light
        MW_GCS_label_v13 = r"\begin{tabular}{p{4cm}l}MW (V13);" + \
            r" & N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_Rgc_v13))

        Msum_v13, Medges_v13, Mcnt_v13 = binned_statistic(
            MW_v13["Rproj"][mw_Rgc_v13] if pioverfour else MW_v13["R_GC"][mw_Rgc_v13],
            MW_GCS_mass_v13,
            bins=bins, statistic="sum"
        )
        ax.plot((Medges_v13[1:]+Medges_v13[:-1])/2, Msum_v13/mass_denominator,
            drawstyle="steps-mid", lw=4, ls=":", c=COLORS["MW"],
            label=MW_GCS_label_v13)

    ax.set_xlabel("$r_{\\rm gal}$ [kpc]")
    ax.set_ylabel("Globular Cluster Mass [{0}Msun]".format(
        str(mass_denominator)+" " if mass_denominator != 1 else ""))
    if set_ticks:
        pass
    if grid:
        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=18)


def add_VandenBerg_FeH_Rgc_logM_MW_to_ax(ax, MW_v13,
        # xbins=[0.1, 1, 3, 8, 15, 30, 125, 1000],
        # ybins=[-4, -2.5, -1.5, -1.0, -0.5, -0.3, 0, 1],
        xbins=[1, 3, 8, 15, 30, 125],
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0],
        Mv_Sun=4.83, mass_to_light=1.7,
        cmap=colorcet.cm["bgy"],
        pioverfour=False, show_bins=True, do_scatter=False,
        debug=False, print_latex=False, uberdebug=False):

    # First get all GCs with age, FeH and Rgc measurements for which we
    # can compute the mass (i.e. it also has Mv measurement)
    mw_age_FeH_Rgc, = numpy.where(
        (MW_v13["Age"] < 13.99)
        & numpy.isfinite(MW_v13["M_V"])
        & numpy.isfinite(MW_v13["FeH"])
        & numpy.isfinite(MW_v13["R_GC"])
    )
    # Then calculate mass
    MW_GCS_mass = numpy.power(10, 0.4*(Mv_Sun -
        MW_v13["M_V"][mw_age_FeH_Rgc])) * mass_to_light
    MW_GCS_label = r"\begin{{tabular}}{{p{{4cm}}l}}MW (V13);" + \
        r" & N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_age_FeH_Rgc))

    if debug:
        print("There are {0: >3d} MW globular clusters /w {1}".format(
            len(mw_age_FeH_Rgc), "age, Rgc and logM measurement"))

        # Sanity check: are we taking all data into account for the chosen binning?
        if pioverfour:
            ismaller_than_xmin, = numpy.where(
                MW_v13["Rproj"][mw_age_FeH_Rgc] < xbins[0] )
            print("  {0} GCs /w Rproj < {1} kpc".format(
                len(ismaller_than_xmin), xbins[0]) )

            ibigger_than_xmax, = numpy.where(
                MW_v13["Rproj"][mw_age_FeH_Rgc] > xbins[-1] )
            print("  {0} GCs /w Rproj > {1} kpc".format(
                len(ibigger_than_xmax), xbins[-1]) )
        else:
            ismaller_than_xmin, = numpy.where(
                MW_v13["R_GC"][mw_age_FeH_Rgc] < xbins[0] )
            print("  {0} GCs /w Rgc < {1} kpc".format(
                len(ismaller_than_xmin), xbins[0]) )

            ibigger_than_xmax, = numpy.where(
                MW_v13["R_GC"][mw_age_FeH_Rgc] > xbins[-1] )
            print("  {0} GCs /w Rgc > {1} kpc".format(
                len(ibigger_than_xmax), xbins[-1]) )

        ismaller_than_ymin, = numpy.where(
            MW_v13["FeH"][mw_age_FeH_Rgc] < ybins[0] )
        print("  {0} GCs /w FeH < {1}".format(
            len(ismaller_than_ymin), ybins[0]) )

        ibigger_than_ymax, = numpy.where(
            MW_v13["FeH"][mw_age_FeH_Rgc] > ybins[-1] )
        print("  {0} GCs /w FeH > {1}".format(
            len(ibigger_than_ymax), ybins[-1]) )

    # Bin the data up
    if pioverfour:
        Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_v13["Rproj"][mw_age_FeH_Rgc],
            MW_v13["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="count",
        )  # because I don't understand the format of Mcnt
        Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_v13["Rproj"][mw_age_FeH_Rgc],
            MW_v13["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="sum",
        )
    else:
        if do_scatter:
            ax.plot(
                MW_v13["R_GC"][mw_age_FeH_Rgc],
                MW_v13["FeH"][mw_age_FeH_Rgc], alpha=0.7,
                ls="none", marker="o", ms=6, mec="w", c=COLORS["MW"]
            )
        Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_v13["R_GC"][mw_age_FeH_Rgc],
            MW_v13["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="count",
        )  # because I don't understand the format of Mcnt
        Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_v13["R_GC"][mw_age_FeH_Rgc],
            MW_v13["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="sum",
        )
    if debug:
        # Mcnt: this assigns to each element of `sample` an integer that
        #       represents the bin in which this observation falls.
        print("\nMsum\n", Msum)
        print("\nMcount\n", Mcount)
        print("\nRgc_edges\n", Rgc_edges)
        print("\nFeH_edges\n", FeH_edges)
        print("\nMcnt\n", Mcnt)

        print("\n{0} of MW GCS taken into account".format(len(Mcnt)))
        print("{0:.2f} % of mass in MW GCS taken into account\n".format(
            100 * Msum.flatten().sum() / (2.96e7)) )

    if uberdebug:  # Just one more sanity check
        print("\nuberdebug\n")
        if not print_latex:
            print("-"*79)
            print("  {0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}{5:<10s}{6:<10s}".format(
                "xmin", "xmax", "ymin", "ymax", "Ngc", "Msum", "log10(Msun)" ))
            print("-"*79)
        else:
            print("        \hline")
            print("        {0:<6s} & {1:<6s} & {2:<6s} & {3:<6s} &"
                  " {4:<6s} & {5:<6s} & {6:<10s} \\\\".format(
                "xmin", "xmax", "ymin", "ymax", "Ngc", "Msum", "log$_{10}$(Msun)" ))
            print("        \hline")
        for xmin, xmax in zip(xbins[:-1], xbins[1:]):
            for ymin, ymax in zip(ybins[:-1], ybins[1:]):
                if pioverfour:
                    ibin, = numpy.where(
                        (MW_v13["Rproj"][mw_age_FeH_Rgc] >= xmin )
                        & (MW_v13["Rproj"][mw_age_FeH_Rgc] < xmax )
                        & (MW_v13["FeH"][mw_age_FeH_Rgc] >= ymin )
                        & (MW_v13["FeH"][mw_age_FeH_Rgc] < ymax )
                    )
                else:
                    ibin, = numpy.where(
                        (MW_v13["R_GC"][mw_age_FeH_Rgc] >= xmin )
                        & (MW_v13["R_GC"][mw_age_FeH_Rgc] < xmax )
                        & (MW_v13["FeH"][mw_age_FeH_Rgc] >= ymin )
                        & (MW_v13["FeH"][mw_age_FeH_Rgc] < ymax )
                    )
                if not print_latex:
                    print("  {0:<10}{1:<10}{2:<10}{3:<10}{4:<10}{5:<10.1e}{6:<10.2f}".format(
                        xmin, xmax, ymin, ymax, len(ibin), MW_GCS_mass[ibin].sum(),
                        numpy.log10( MW_GCS_mass[ibin].sum() )
                    ))
                else:
                    print("        {0:<6} & {1:<6} & {2:<6} & {3:<6} & "
                          "{4:<6} & {5:<6.1e} & {6:<10.2f} \\\\".format(
                        xmin, xmax, ymin, ymax, len(ibin), MW_GCS_mass[ibin].sum(),
                        numpy.log10( MW_GCS_mass[ibin].sum() )
                    ))
            print("-"*79) if not print_latex else print("        \hline")
        print("/uberdebug\n")

    xmid = (Rgc_edges[1:] + Rgc_edges[:-1]) / 2
    xmidlog = ( numpy.log10(Rgc_edges[1:]) + numpy.log10(Rgc_edges[:-1]) ) / 2
    ymid = (FeH_edges[1:] + FeH_edges[:-1]) / 2
    if debug:
        print("{0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}".format(
            "logx", "y", "value", "logvalue", "Ngcs"))
    for i, (logx, rows) in enumerate(zip(xmidlog, Msum)):  # starts at lower left corner?
        for j, (y, value) in enumerate(zip(ymid, rows)):
            # if pioverfour:
            #     Ngcs, = numpy.where(
            #         (MW_v13["Rproj"][mw_age_FeH_Rgc] > Rgc_edges[j] )
            #         & (MW_v13["Rproj"][mw_age_FeH_Rgc] < Rgc_edges[j+1] )
            #         & (MW_v13["FeH"][mw_age_FeH_Rgc] > FeH_edges[i] )
            #         & (MW_v13["FeH"][mw_age_FeH_Rgc] < FeH_edges[i+1] )
            #     )
            # else:
            #     Ngcs, = numpy.where(
            #         (MW_v13["R_GC"][mw_age_FeH_Rgc] > Rgc_edges[j] )
            #         & (MW_v13["R_GC"][mw_age_FeH_Rgc] < Rgc_edges[j+1] )
            #         & (MW_v13["FeH"][mw_age_FeH_Rgc] > FeH_edges[i] )
            #         & (MW_v13["FeH"][mw_age_FeH_Rgc] < FeH_edges[i+1] )
            #     )
            # print("{0:<10}{1:<10.1f}".format(len(Ngcs), MW_GCS_mass[Ngcs].sum() ))
            # Ngcs = len(Ngcs)
            Ngcs = int(Mcount[i][j])
            with suppress_stdout():  # divide by zero encountered in log10
                logvalue = numpy.log10(value)
                logvalue = logvalue if numpy.isfinite(logvalue) else 0
            if debug:
                print("{0:<10.1f}{1:<10.1f}{2:<10.1f}{3:<10.2f}{4:<10d}".format(
                    logx, y, value, logvalue, Ngcs))
            ax.text(10**logx, y, "{0:.2f} ({1})".format(logvalue, Ngcs),
                ha="center", va="center", fontsize=20)

    Msum = numpy.ma.masked_equal(Msum, 0)
    with suppress_stdout():  # divide by zero encountered in log10
        cbar = ax.pcolormesh(Rgc_edges, FeH_edges, numpy.log10(Msum.T),
            cmap=cmap, alpha=0.7)  # , vmin=3, vmax=9.1)
    axcb = pyplot.colorbar(cbar, orientation="horizontal")
    axcb.ax.set_xlabel(r"log$_{10}$( Globular Cluster Mass [Msun] )")

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
    ax.set_xlabel(r"$r_{\rm proj}$ [kpc]" if pioverfour else r"$r_{\rm gal}$ [kpc]")
    ax.set_ylabel("[Fe/H]")
    ax.set_xscale("log")

    if show_bins:
        transx = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for x in xbins:
            ax.axvline(x, c="k", ls="-")
            ax.text(x, 1.01, str(x), fontsize=16, ha="center", transform=transx)

        transy = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        for y in ybins[:-1]:
            ax.axhline(y, c="k", ls="-")
            ax.text(1.01, y, str(y), fontsize=16, va="center", transform=transy)


def add_Harris_FeH_Rgc_logM_MW_to_ax(ax, MW_h96e10,
        # xbins=[0.1, 1, 3, 8, 15, 30, 125, 1000],
        # ybins=[-4, -2.5, -1.5, -1.0, -0.5, -0.3, 0, 1],
        xbins=[1, 3, 8, 15, 30, 125],
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0],
        Mv_Sun=4.83, mass_to_light=1.7, MW_rvir=1.0,
        cmap=colorcet.cm["bgy"],
        pioverfour=False, show_bins=True, do_scatter=False,
        debug=False, uberdebug=False, print_latex=False,
        plot_values_skip=[],
    ):

    # First get all GCs with FeH and Rgc measurements for which we
    # can compute the mass (i.e. it also has Mv measurement)
    mw_age_FeH_Rgc, = numpy.where(
        numpy.isfinite(MW_h96e10["M_Vt"])
        & numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["R_gc"])
    )
    # Then calculate mass
    MW_GCS_mass = numpy.power(10, 0.4*(Mv_Sun -
        MW_h96e10["M_Vt"][mw_age_FeH_Rgc])) * mass_to_light
    MW_GCS_label = r"\begin{{tabular}}{{p{{4cm}}l}}MW (H96e10);" + \
        r" & N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(mw_age_FeH_Rgc))

    if debug:
        print("There are {0: >3d} MW globular clusters /w {1}".format(
            len(mw_age_FeH_Rgc), "age, Rgc and logM measurement"))

        # Sanity check: are we taking all data into account for the chosen binning?
        if pioverfour:
            ismaller_than_xmin, = numpy.where(
                MW_h96e10["Rproj"][mw_age_FeH_Rgc] < xbins[0] )
            print("  {0} GCs /w Rproj < {1} kpc".format(
                len(ismaller_than_xmin), xbins[0]) )

            ibigger_than_xmax, = numpy.where(
                MW_h96e10["Rproj"][mw_age_FeH_Rgc] > xbins[-1] )
            print("  {0} GCs /w Rproj > {1} kpc".format(
                len(ibigger_than_xmax), xbins[-1]) )
        else:
            ismaller_than_xmin, = numpy.where(
                MW_h96e10["R_gc"][mw_age_FeH_Rgc] < xbins[0] )
            print("  {0} GCs /w Rgc < {1} kpc".format(
                len(ismaller_than_xmin), xbins[0]) )

            ibigger_than_xmax, = numpy.where(
                MW_h96e10["R_gc"][mw_age_FeH_Rgc] > xbins[-1] )
            print("  {0} GCs /w Rgc > {1} kpc".format(
                len(ibigger_than_xmax), xbins[-1]) )

        ismaller_than_ymin, = numpy.where(
            MW_h96e10["FeH"][mw_age_FeH_Rgc] < ybins[0] )
        print("  {0} GCs /w FeH < {1}".format(
            len(ismaller_than_ymin), ybins[0]) )

        ibigger_than_ymax, = numpy.where(
            MW_h96e10["FeH"][mw_age_FeH_Rgc] > ybins[-1] )
        print("  {0} GCs /w FeH > {1}".format(
            len(ibigger_than_ymax), ybins[-1]) )

    # Bin the data up
    if pioverfour:
        Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_h96e10["Rproj"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="count",
        )  # because I don't understand the format of Mcnt
        Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_h96e10["Rproj"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="sum",
        )
    else:
        if do_scatter:
            ax.plot(
                MW_h96e10["R_gc"][mw_age_FeH_Rgc],
                MW_h96e10["FeH"][mw_age_FeH_Rgc], alpha=0.7,
                ls="none", marker="o", ms=6, mec="w", c=COLORS["MW"]
            )
        Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_h96e10["R_gc"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="count",
        )  # because I don't understand the format of Mcnt
        Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            MW_h96e10["R_gc"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="sum",
        )
    if debug:
        # Mcnt: this assigns to each element of `sample` an integer that
        #       represents the bin in which this observation falls.
        print("\nMsum\n", Msum)
        print("\nMcount\n", Mcount)
        print("\nRgc_edges\n", Rgc_edges)
        print("\nFeH_edges\n", FeH_edges)
        print("\nMcnt\n", Mcnt)

        print("\n{0} of MW GCS taken into account".format(len(Mcnt)))
        print("{0:.2f} % of mass in MW GCS taken into account\n".format(
            100 * Msum.flatten().sum() / (2.96e7)) )

    if uberdebug:  # Just one more sanity check
        print("\nuberdebug\n")
        if not print_latex:
            print("-"*79)
            print("  {0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}{5:<10s}{6:<10s}".format(
                "xmin", "xmax", "ymin", "ymax", "Ngc", "Msum", "log10(Msun)" ))
            print("-"*79)
        else:
            print("        \hline")
            print("        {0:<6s} & {1:<6s} & {2:<6s} & {3:<6s}"
                  " & {4:<6s} & {5:<6s} & {6:<10s} \\\\".format(
                "xmin", "xmax", "ymin", "ymax", "Ngc", "Msum", "log$_{10}$(Msun)" ))
            print("        \hline")
        for xmin, xmax in zip(xbins[:-1], xbins[1:]):
            for ymin, ymax in zip(ybins[:-1], ybins[1:]):
                if pioverfour:
                    ibin, = numpy.where(
                        (MW_h96e10["Rproj"][mw_age_FeH_Rgc] >= xmin )
                        & (MW_h96e10["Rproj"][mw_age_FeH_Rgc] < xmax )
                        & (MW_h96e10["FeH"][mw_age_FeH_Rgc] >= ymin )
                        & (MW_h96e10["FeH"][mw_age_FeH_Rgc] < ymax )
                    )
                else:
                    ibin, = numpy.where(
                        (MW_h96e10["R_gc"][mw_age_FeH_Rgc] >= xmin )
                        & (MW_h96e10["R_gc"][mw_age_FeH_Rgc] < xmax )
                        & (MW_h96e10["FeH"][mw_age_FeH_Rgc] >= ymin )
                        & (MW_h96e10["FeH"][mw_age_FeH_Rgc] < ymax )
                    )
                if not print_latex:
                    print("  {0:<10}{1:<10}{2:<10}{3:<10}{4:<10}{5:<10.1e}{6:<10.2f}".format(
                        xmin, xmax, ymin, ymax, len(ibin), MW_GCS_mass[ibin].sum(),
                        numpy.log10( MW_GCS_mass[ibin].sum() )
                    ))
                else:
                    print("        {0:<6} & {1:<6} & {2:<6} & {3:<6} &"
                          " {4:<6} & {5:<6.1e} & {6:<10.2f} \\\\".format(
                        xmin, xmax, ymin, ymax, len(ibin), MW_GCS_mass[ibin].sum(),
                        numpy.log10( MW_GCS_mass[ibin].sum() )
                    ))
            print("-"*79) if not print_latex else print("        \hline")
        print("/uberdebug\n")

    xmid = (Rgc_edges[1:] + Rgc_edges[:-1]) / 2
    xmidlog = ( numpy.log10(Rgc_edges[1:]) + numpy.log10(Rgc_edges[:-1]) ) / 2
    ymid = (FeH_edges[1:] + FeH_edges[:-1]) / 2
    if debug:
        print("{0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}".format(
            "logx", "y", "value", "logvalue", "Ngcs"))
    for i, (logx, rows) in enumerate(zip(xmidlog, Msum)):  # starts at lower left corner?
        for j, (y, value) in enumerate(zip(ymid, rows)):
            # if pioverfour:
            #     Ngcs, = numpy.where(
            #         (MW_h96e10["Rproj"][mw_age_FeH_Rgc] > Rgc_edges[j] )
            #         & (MW_h96e10["Rproj"][mw_age_FeH_Rgc] < Rgc_edges[j+1] )
            #         & (MW_h96e10["FeH"][mw_age_FeH_Rgc] > FeH_edges[i] )
            #         & (MW_h96e10["FeH"][mw_age_FeH_Rgc] < FeH_edges[i+1] )
            #     )
            # else:
            #     Ngcs, = numpy.where(
            #         (MW_h96e10["R_gc"][mw_age_FeH_Rgc] > Rgc_edges[j] )
            #         & (MW_h96e10["R_gc"][mw_age_FeH_Rgc] < Rgc_edges[j+1] )
            #         & (MW_h96e10["FeH"][mw_age_FeH_Rgc] > FeH_edges[i] )
            #         & (MW_h96e10["FeH"][mw_age_FeH_Rgc] < FeH_edges[i+1] )
            #     )
            # print("{0:<10}{1:<10.1f}".format(len(Ngcs), MW_GCS_mass[Ngcs].sum() ))
            # Ngcs = len(Ngcs)
            Ngcs = int(Mcount[i][j])
            with suppress_stdout():  # divide by zero encountered in log10
                logvalue = numpy.log10(value)
                logvalue = logvalue if numpy.isfinite(logvalue) else 0
            if debug:
                print("{0:<10.1f}{1:<10.1f}{2:<10.1f}{3:<10.2f}{4:<10d}".format(
                    logx, y, value, logvalue, Ngcs))
            if (i,j) in  plot_values_skip: continue
            ax.text(10**logx, y, "{0:.2f} ({1})".format(logvalue, Ngcs),
                ha="center", va="center", fontsize=20)

    Msum = numpy.ma.masked_equal(Msum, 0)
    with suppress_stdout():  # divide by zero encountered in log10
        cbar = ax.pcolormesh(Rgc_edges, FeH_edges, numpy.log10(Msum.T),
            cmap=cmap, alpha=0.7)  # , vmin=3, vmax=9.1)
    axcb = pyplot.colorbar(cbar, orientation="horizontal")
    axcb.ax.set_xlabel(r"log$_{10}$( Globular Cluster Mass [Msun] )")

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
    ax.set_xlabel(r"$r_{\rm proj}$ [kpc]" if pioverfour else r"$r_{\rm gal}$ [kpc]")
    ax.set_ylabel("[Fe/H]")
    ax.set_xscale("log")

    if show_bins:
        transx = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for x in xbins:
            ax.axvline(x, c="k", ls="-")
            # ax.text(x, 1.01, str(x), fontsize=16, ha="center", transform=transx)

        transy = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        for y in ybins[:-1]:
            ax.axhline(y, c="k", ls="-")
            # ax.text(1.01, y, str(y), fontsize=16, va="center", transform=transy)


def add_CaldwellRomanowsky_FeH_Rgc_logM_M31_to_ax(ax, M31_cr16,
        # xbins=[0.1, 1, 3, 8, 15, 30, 125, 1000],
        # ybins=[-4, -2.5, -1.5, -1.0, -0.5, -0.3, 0, 1],
        xbins=[1, 3, 8, 15, 30, 125],
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0],
        cmap=colorcet.cm["bgy"],
        show_bins=True, do_scatter=False, debug=False,
        plot_values_skip=[],
    ):

    with suppress_stdout():
        m31_FeH_logM_Rgc, = numpy.where(
            numpy.isfinite(M31_cr16["[Fe/H]"])
            & (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label = r"\begin{{tabular}}{{p{{4cm}}l}}M31 (CR16);" + \
        r" & N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(m31_FeH_logM_Rgc))

    if debug:
        print("There are {0: >3d} M31 globular clusters /w {1}".format(
            len(m31_FeH_logM_Rgc), "FeH, Rgc and LogM measurement"))

        # Sanity check: are we taking all data into account for the chosen binning?
        ismaller_than_xmin, = numpy.where(
            M31_cr16["Rgc"][m31_FeH_logM_Rgc] < xbins[0] )
        print("  {0} GCs /w Rgc < {1} kpc".format(
            len(ismaller_than_xmin), xbins[0]) )

        ibigger_than_xmax, = numpy.where(
            M31_cr16["Rgc"][m31_FeH_logM_Rgc] > xbins[-1] )
        print("  {0} GCs /w Rgc > {1} kpc".format(
            len(ibigger_than_xmax), xbins[-1]) )

        ismaller_than_ymin, = numpy.where(
            M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc] < ybins[0] )
        print("  {0} GCs /w [Fe/H] < {1}".format(
            len(ismaller_than_ymin), ybins[0]) )

        ibigger_than_ymax, = numpy.where(
            M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc] > ybins[-1] )
        print("  {0} GCs /w [Fe/H] > {1}".format(
            len(ibigger_than_ymax), ybins[-1]) )

    # Bin the data up
    if do_scatter:
        ax.plot(
            M31_cr16["Rgc"][m31_FeH_logM_Rgc],
            M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc], alpha=0.7,
            ls="none", marker="o", ms=6, mec="w", c=COLORS["M31"],
        )
    Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
        M31_cr16["Rgc"][m31_FeH_logM_Rgc],
        M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc],
        10**M31_cr16["LogM"][m31_FeH_logM_Rgc],
        bins=[xbins, ybins], statistic="count",
    )  # because I don't understand the format of Mcnt
    Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
        M31_cr16["Rgc"][m31_FeH_logM_Rgc],
        M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc],
        10**M31_cr16["LogM"][m31_FeH_logM_Rgc],
        bins=[xbins, ybins], statistic="sum",
    )
    if debug:
        # Mcnt: this assigns to each element of `sample` an integer that
        #       represents the bin in which this observation falls.
        print("\nMsum\n", Msum)
        print("\nMcount\n", Mcount)
        print("\nRgc_edges\n", Rgc_edges)
        print("\nFeH_edges\n", FeH_edges)
        print("\nMcnt\n", Mcnt)

        Msum_obs = (10**M31_cr16["LogM"][numpy.isfinite(M31_cr16["LogM"])]).sum()
        print("\n{0} of M31 GCS taken into account".format(len(Mcnt)))
        print("{0:.2f} % of mass in M31 GCS taken into account\n".format(
            100 * Msum.flatten().sum() / (2.29e8)) )

    xmid = (Rgc_edges[1:] + Rgc_edges[:-1]) / 2
    xmidlog = ( numpy.log10(Rgc_edges[1:]) + numpy.log10(Rgc_edges[:-1]) ) / 2
    ymid = (FeH_edges[1:] + FeH_edges[:-1]) / 2
    if debug:
        print("{0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}".format(
            "logx", "y", "value", "logvalue", "Ngcs"))
    for i, (logx, rows) in enumerate(zip(xmidlog, Msum)):  # starts at lower left corner?
        for j, (y, value) in enumerate(zip(ymid, rows)):
            Ngcs = int(Mcount[i][j])
            with suppress_stdout():  # divide by zero encountered in log10
                logvalue = numpy.log10(value)
                logvalue = logvalue if numpy.isfinite(logvalue) else 0
            if debug:
                print("{0:<10.1f}{1:<10.1f}{2:<10.1f}{3:<10.2f}{4:<10d}".format(
                    logx, y, value, logvalue, Ngcs))
            if (i,j) in plot_values_skip: continue
            ax.text(10**logx, y, "{0:.2f} ({1})".format(logvalue, Ngcs),
                ha="center", va="center", fontsize=20)

    Msum = numpy.ma.masked_equal(Msum, 0)
    with suppress_stdout():  # divide by zero encountered in log10
        cbar = ax.pcolormesh(Rgc_edges, FeH_edges, numpy.log10(Msum.T),
            cmap=cmap, alpha=0.7)  # , vmin=3, vmax=9.1)
    axcb = pyplot.colorbar(cbar, orientation="horizontal")
    axcb.ax.set_xlabel(r"log$_{10}$( Globular Cluster Mass [Msun] )")

    ax.set_xlim(xbins[0], xbins[-1])
    ax.set_ylim(ybins[0], ybins[-1])
    ax.set_xlabel(r"$r_{\rm gal}$ [kpc]")
    ax.set_ylabel("[Fe/H]")
    ax.set_xscale("log")

    if show_bins:
        transx = matplotlib.transforms.blended_transform_factory(
            ax.transData, ax.transAxes)
        for x in xbins:
            ax.axvline(x, c="k", ls="-")
            # ax.text(x, 1.01, str(x), fontsize=16, ha="center", transform=transx)

        transy = matplotlib.transforms.blended_transform_factory(
            ax.transAxes, ax.transData)
        for y in ybins[:-1]:
            ax.axhline(y, c="k", ls="-")
            # ax.text(1.01, y, str(y), fontsize=16, va="center", transform=transy)


def match_Harris_VandenBerg_data(MW_v13, MW_h96e10, debug=False):
    fname = "../out/v13_h10_map.npz"
    if os.path.exists(fname) and os.path.isfile(fname):
        if debug: print("Loading v13_h10_map from file! <3")
        return numpy.load(fname, allow_pickle=True)["v13_h10_map"].item()

    v13_h10_map = dict()
    for i in range(len(MW_v13)):
        ngc_v13 = MW_v13["NGC"][i]
        name_v13 = MW_v13["Name"][i].decode()
        if debug:
            print("ngc_v13: {0}".format(ngc_v13))
            print("name_v13: {0}".format(name_v13))

        for j, id_h10 in enumerate(MW_h96e10["ID"]):
            id_h10 = id_h10.decode().replace("Terzan ", "Ter")\
                .replace("Pal ", "Pal").replace("Arp ", "Arp")
            name_h10 = MW_h96e10["Name"][j].decode()
            if debug:
                print("  id_h10:   {0} (j = {1})".format(id_h10, j))
                print("  name_h10: {0}".format(name_h10, j))

            if (
                id_h10 == "NGC {0}".format(ngc_v13) or
                id_h10 == name_v13  # last three items have no NGC: value is '-1'
            ):
                v13_h10_map[i] = j
                break
        else:
            print("ngc_v13, name_v13 = {0}, {1}"
                  " not in harris catalog!".format(ngc_v13, name_v13))
    numpy.savez(fname, v13_h10_map=v13_h10_map)
    return v13_h10_map


def compare_Harris_Vandenberg_data_sets(MW_h96e10, MW_v13,
        Mv_Sun=4.83, mass_to_light=1.7, debug=False):

    v13_h10_map = dict(match_Harris_VandenBerg_data(MW_v13, MW_h96e10))

    for key_h10, key_v13, verbose_name in zip(
            ["FeH", "M_Vt", "R_GC"],
            ["FeH", "M_V", "R_GC"],
            ["[Fe/H]", r"$M_V$"]
    ):
        fig, ax = pyplot.subplots(figsize=(8, 8))
        for i, (k, v) in enumerate(v13_h10_map.items()):
            if i is 0:
                xmin, xmax = MW_h96e10[key_h10][0], MW_h96e10[key_h10][0]

            if debug:
                print("Harris (1996):     {0:<15s}{1:<15s} -> {2}".format(
                    MW_h96e10['ID'][v].decode(), MW_h96e10['Name'][v].decode(),
                    MW_h96e10[key_h10][v]))
                print("VandenBerg (2013): NGC {0:<11s}{1:<15s} -> {2}\n".format(
                    str(MW_v13['NGC'][k]), MW_v13['Name'][k].decode(),
                    MW_v13[key_v13][k]))

            if MW_h96e10[key_h10][v] < xmin:
                xmin = MW_h96e10[key_h10][v]
            if MW_v13[key_v13][k] < xmin:
                xmin = MW_v13[key_v13][k]

            if MW_h96e10[key_h10][v] > xmax:
                xmax = MW_h96e10[key_h10][v]
            if MW_v13[key_v13][k] > xmax:
                xmax = MW_v13[key_v13][k]

            ax.plot(MW_h96e10[key_h10][v], MW_v13[key_v13][k], "rX", ms=5)

        if debug: print("\nxmin, xmax = {0}, {1}".format(xmin, xmax))
        xlin = numpy.linspace(xmin, xmax, 8)
        ax.plot(xlin, xlin, c="k", label="{0} equal".format(verbose_name))

        ax.set_xlabel("Harris (1996, 2010 ed.)")
        ax.set_ylabel("VandenBerg (2013)")
        # ax.set_xticks(numpy.arange(-3, 1, 0.5))
        # ax.set_yticks(numpy.arange(-3, 1, 0.5))
        # ax.set_xlim(-3, 0.5)
        # ax.set_ylim(-3, 0.5)

        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
        ax.legend(loc="upper left", frameon=False, fontsize=16)


def match_Harris_McLaughlinVandeMarel_data(MW_mc05, MW_h96e10, debug=False):
    fname = "../out/mc05_h10_map.npz"
    if os.path.exists(fname) and os.path.isfile(fname):
        if debug: print("Loading mc05_h10_map from file! <3")
        return numpy.load(fname, allow_pickle=True)["mc05_h10_map"].item()

    i_is_MW, = numpy.where(
        # Include MW clusters only (i.e. not FORNAX, LMC, SMC)
        (numpy.char.find(MW_mc05["Cluster"].astype("str"), "MW") >= 0)
    )

    mc05_h10_map = dict()
    names_mc05 = [n.replace("Cl ", "").replace("NAME ", "").replace(
        "Ruprecht", "Rup").replace("LILLER", "Liller").replace(
        "Tonantzintla", "Ton").replace("Haute-Provence", "HP")
        for n in MW_mc05["SName"]]
    clusters_mc05 = [n.replace("MW-", "").capitalize()
        for n in MW_mc05["Cluster"]]
    for i in range(len(MW_mc05)):
        if i not in i_is_MW:
            # print("Not MW cluster, continue! i={0}".format(i))
            continue

        name_mc05 = names_mc05[i]
        cluster_mc05 = clusters_mc05[i]
        if debug:
            print("name_mc05: {0}".format(name_mc05))
            print("cluster_mc05: {0}".format(cluster_mc05))

        for j, id_h10 in enumerate(MW_h96e10["ID"]):
            id_h10 = id_h10.decode()
            name_h10 = MW_h96e10["Name"][j].decode()
            if debug:
                print("  id_h10:   {0} (j = {1})".format(id_h10, j))
                print("  name_h10: {0}".format(name_h10, j))

            if (
                id_h10 == name_mc05 or id_h10 == cluster_mc05
                or id_h10 == name_h10 or id_h10 == name_h10
            ):
                mc05_h10_map[i] = j
                break
        else:
            print("name_mc05, cluster_mc05 = {0}, {1}"
                  " not in harris catalog!".format(name_mc05, cluster_mc05))
    numpy.savez(fname, mc05_h10_map=mc05_h10_map)
    return mc05_h10_map


def compare_Harris_McLaughlin_vandeMarel2005_data_sets(MW_h96e10, MW_mc05,
        Mv_Sun=4.83, mass_to_light=1.7, debug=False):

    mc05_h10_map = dict(match_Harris_McLaughlinVandeMarel_data(MW_mc05, MW_h96e10))

    for key_h10, key_mc05, verbose_name in zip(
            ["FeH"],
            ["__Fe_H_"],
            ["[Fe/H]"]
    ):
        fig, ax = pyplot.subplots(figsize=(8, 8))
        for i, (k, v) in enumerate(mc05_h10_map.items()):
            if i is 0:
                xmin, xmax = MW_h96e10[key_h10][0], MW_h96e10[key_h10][0]

            if debug:
                print("Harris (1996):   {0:<20s}{1:<20s} -> {2:.2f}".format(
                    MW_h96e10['ID'][v].decode(), MW_h96e10['Name'][v].decode(),
                    MW_h96e10[key_h10][v]))
                print("Mc & vdM (2005): {0:<20s}{1:<20s} -> {2:.2f}\n".format(
                    str(MW_mc05['Cluster'][k]), MW_mc05['SName'][k],
                    MW_mc05[key_mc05][k]))

            if MW_h96e10[key_h10][v] < xmin:
                xmin = MW_h96e10[key_h10][v]
            if MW_mc05[key_mc05][k] < xmin:
                xmin = MW_mc05[key_mc05][k]

            if MW_h96e10[key_h10][v] > xmax:
                xmax = MW_h96e10[key_h10][v]
            if MW_mc05[key_mc05][k] > xmax:
                xmax = MW_mc05[key_mc05][k]

            ax.plot(MW_h96e10[key_h10][v], MW_mc05[key_mc05][k], "rX", ms=5)

        if debug: print("\nxmin, xmax = {0}, {1}".format(xmin, xmax))
        xlin = numpy.linspace(xmin, xmax, 8)
        ax.plot(xlin, xlin, c="k", label="{0} equal".format(verbose_name))

        ax.set_xlabel("Harris (1996, 2010 ed.)")
        ax.set_ylabel("McLaughlin \& van de Marel (2005)")
        # ax.set_xticks(numpy.arange(-3, 1, 0.5))
        # ax.set_yticks(numpy.arange(-3, 1, 0.5))
        # ax.set_xlim(-3, 0.5)
        # ax.set_ylim(-3, 0.5)

        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
        ax.legend(loc="upper left", frameon=False, fontsize=16)


def bin_milkyway_data(MW_h96e10, xbins,ybins,
        Mv_Sun=4.83, mass_to_light=1.7):
    mw_age_FeH_Rgc, = numpy.where(
        numpy.isfinite(MW_h96e10["M_Vt"])
        & numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["R_gc"])
    )
    MW_GCS_mass = numpy.power(10, 0.4*(Mv_Sun -
        MW_h96e10["M_Vt"][mw_age_FeH_Rgc])) * mass_to_light
    Mcount_h96e10, Rgc_edges_h96e10, FeH_edges_h96e10, Mcnt_h96e10 = \
        binned_statistic_2d(
            MW_h96e10["R_gc"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="count",
        )
    Msum_h96e10, Rgc_edges_h96e10, FeH_edges_h96e10, Mcnt_h96e10 = \
        binned_statistic_2d(
            MW_h96e10["R_gc"][mw_age_FeH_Rgc],
            MW_h96e10["FeH"][mw_age_FeH_Rgc],
            MW_GCS_mass, bins=[xbins, ybins], statistic="sum",
        )

    return MW_GCS_mass, Mcount_h96e10, Msum_h96e10, Rgc_edges_h96e10, \
        FeH_edges_h96e10, Mcnt_h96e10


def bin_andromeda_data(M31_cr16, xbins, ybins):
    with suppress_stdout():
        m31_FeH_logM_Rgc, = numpy.where(
            numpy.isfinite(M31_cr16["[Fe/H]"])
            & (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_mass = 10**M31_cr16["LogM"][m31_FeH_logM_Rgc]
    Mcount_cr16, Rgc_edges_cr16, FeH_edges_cr16, Mcnt_cr16 = \
        binned_statistic_2d(
            M31_cr16["Rgc"][m31_FeH_logM_Rgc],
            M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc],
            M31_GCS_mass,
            bins=[xbins, ybins], statistic="count",
        )
    Msum_cr16, Rgc_edges_cr16, FeH_edges_cr16, Mcnt_cr16 = \
        binned_statistic_2d(
            M31_cr16["Rgc"][m31_FeH_logM_Rgc],
            M31_cr16["[Fe/H]"][m31_FeH_logM_Rgc],
            M31_GCS_mass,
            bins=[xbins, ybins], statistic="sum",
        )

    return M31_GCS_mass, Mcount_cr16, Msum_cr16, Rgc_edges_cr16, \
        FeH_edges_cr16, Mcnt_cr16


if __name__ == "__main__":
    pyplot.switch_backend("agg")
    pyplot.style.use("tlrh")

    # plot_milky_way_globular_cluster_system_FeH()
    # plot_milky_way_globular_cluster_system_mass()
    # calculate_milky_way_globular_cluster_system_total_mass()
    # plot_milky_way_globular_cluster_system_mass_histogram()
    # compare_harris_and_kharchenko()

    # part1, part2, part3 = read_harris1996_data()
    # print_harris1996_data(part1, part2, part3)
    # plot_MW_mass_distribution(part2)


    data = read_MWandM31_data()
    MW_h96e10, MW_v13, M31_c11, M31_cr16 = data

    fig, ax = pyplot.subplots(figsize=(12, 9))
    add_age_logM_MWandM31_to_ax(ax, MW_v13, M31_cr16)
    ax.set_xlim(4, 14)
    ax.set_yscale("log")
    # ax.set_ylim(0, 0.7)
    fig.show()


    fig, ax = pyplot.subplots(figsize=(12, 9))
    add_FeH_logM_MWandM31_to_ax(ax, MW_h96e10, M31_cr16, M31_c11,
        set_ticks=True, grid=True, set_labels=True, mass_denominator=1e6)
    ax.set_xlim(-3, 0.6)
    ax.set_yscale("log")
    # ax.set_ylim(0, 0.7)
    fig.show()


    fig, ax = pyplot.subplots(figsize=(12, 9))
    add_Rgc_logM_MW_to_ax(ax, MW_h96e10, M31_c11, MW_v13)
    ax.set_xlim(0.3, 180)
    ax.set_ylim(4e-4, 11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    fig.show()


    fig, ax = pyplot.subplots(figsize=(12, 9))
    add_VandenBerg_FeH_Rgc_logM_MW_to_ax(ax, MW_v13,
        debug=True, uberdebug=True, print_latex=True)
    fig.show()


    fig, ax = pyplot.subplots(figsize=(12, 9))
    add_Harris_FeH_Rgc_logM_MW_to_ax(ax, MW_h96e10, MW_h96e10,
        debug=True, uberdebug=True, print_latex=True)
    fig.show()


    v13_h10_map = match_Harris_VandenBerg_data(MW_v13, MW_h96e10)
    compare_Harris_Vandenberg_data_sets(MW_h96e10, MW_h96e10, MW_v13)
    plot_hist_and_KS_test_Harris_Vandenberg(MW_h96e10, MW_h96e10,
        MW_v13, bins=10)


    MW_mc05 = read_McLaughlin_vanderMarel2005_data()
    c05_h10_map = match_Harris_McLaughlinVandeMarel_data(MW_mc05, MW_h96e10)
    compare_Harris_McLaughlin_vandeMarel2005_data_sets(MW_mc05, MW_h96e10)
