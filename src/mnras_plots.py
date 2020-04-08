import os
import copy
import numpy
import argparse
import matplotlib
from matplotlib import pyplot
pyplot.style.use("tlrh")

from mw_m31_gc_observations import (
    read_MWandM31_data,
    add_age_MWandM31_to_ax,
    add_Rgc_MWandM31_to_ax,
    add_Harris_FeH_Rgc_logM_MW_to_ax,
    add_CaldwellRomanowsky_FeH_Rgc_logM_M31_to_ax,
)
from read_auriga_snapshot import AurigaOverview
from main import (
    l5_sims,
    l4_sims,
    l3_sims,
    master_process,
)
from augcs import (
    save_feabund,
    save_FeH_logM_hist,
    plot_FeH_obs_and_4_and_10_and_21,
    plot_FeH_mean_vs_std,
    plot_FeH_ratios,
    save_rgal,
    save_Rgc_logM_hist,
    plot_rgal_mean_vs_std,
    plot_rgal_ratios,
    save_FeHRgc_logM_hist,
    average_logM_FeHRgc_histogram_for_all_sims,
    average_logM_FeHRgc_histogram_for_all_sims_diverging,
)


def figure_1_top(MW_v13, M31_cr16):
    fname = "figures/MW_M31_Age_Histogram.png"
    fig, ax = pyplot.subplots(figsize=(12, 9))
    with suppress_stdout():
        add_age_MWandM31_to_ax(ax, MW_v13, M31_cr16)
    ax.set_xlim(4, 14)
    ax.set_yscale("log")
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_1_bottom(MW_h96e10, M31_cr16_scaled_to_MW, MW_v13, MW_rvir, M31_rvir):
    fname = "figures/MW_M31_Rgc_Histogram.png"
    fig, ax = pyplot.subplots(figsize=(12, 9))
    with suppress_stdout():
        add_Rgc_MWandM31_to_ax(ax, MW_h96e10, M31_cr16_scaled_to_MW,
            M31_c11=None, MW_v13=MW_v13, show_cr16_age_subset=True,
            density=False, MW_rvir=MW_rvir, M31_rvir=M31_rvir,
        )
    ax.set_xlim(0.15, 260)
    ax.set_xscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=18)
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_2_top(MW_h96e10, plot_values_skip):
    fname = "figures/MW_RgcFeH_HistogramMassWeighted_Harris1996ed2010data.png"
    fig, ax = pyplot.subplots(figsize=(12, 9))
    with suppress_stdout():
        add_Harris_FeH_Rgc_logM_MW_to_ax(ax, MW_h96e10, do_scatter=False,
            plot_values_skip=plot_values_skip, debug=False,
        )
    fig.suptitle("Milky Way GCS")
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_2_bottom(M31_cr16_scaled_to_MW, plot_values_skip):
    fname = "figures/M31_RgcFeH_HistogramMassWeighted_CaldwellRomanowsky2016data.png"
    fig, ax = pyplot.subplots(figsize=(12, 9))
    with suppress_stdout():
        add_CaldwellRomanowsky_FeH_Rgc_logM_M31_to_ax(ax, M31_cr16_scaled_to_MW,
            do_scatter=False, plot_values_skip=plot_values_skip, debug=False,
        )
    fig.suptitle("Andromeda GCS")
    fig.savefig(fname)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_3(auriga, MW_h96e10, M31_cr16, age_min):
    fname = "figures/FeH_obs_Au4-4_Au4-10_Au4-21_{:.1f}.png".format(age_min)
    with suppress_stdout():
        fig = plot_FeH_obs_and_4_and_10_and_21(auriga, MW_h96e10, M31_cr16, age_min=age_min)
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_4(auriga, MW_h96e10, M31_cr16, age_min):
    fname = "figures/FeH_mu_sigma_{:.1f}.png".format(age_min)
    with suppress_stdout():
        fig, mu_insituold_minus_mu_accreted_old = plot_FeH_mean_vs_std(
            auriga, MW_h96e10, M31_cr16, age_min=args.age_min, debug=False)
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_5(auriga, MW_h96e10, M31_cr16, age_min):
    fname = "figures/logMFeH_withRatios_{:.1f}.png".format(age_min)
    with suppress_stdout():
        fig = plot_FeH_ratios(auriga,
            ["istars", "iold", "insitu_old", "accreted_old"],
            MW_h96e10, M31_cr16, age_min=age_min)
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_6(auriga, MWRgc, MW_h96e10, M31Rgc, age_min):
    fname = "figures/LogRgc_mu_sigma_{:.1f}.png".format(age_min)
    with suppress_stdout():
        fig = plot_rgal_mean_vs_std(auriga, numpy.log10(MWRgc), MW_h96e10,
            numpy.log10(M31Rgc), age_min=age_min)
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_7(auriga, MW_h96e10, M31_cr16_scaled_to_MW, age_min):
    fname = "figures/logMRgc_withRatios_{:.1f}.png".format(age_min)
    with suppress_stdout():
        fig = plot_rgal_ratios(auriga,
            ["istars", "iold", "insitu_old", "accreted_old"],
            MW_h96e10, M31_cr16_scaled_to_MW, age_min=age_min, yscale="log")
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_8(auriga, l4_sims, age_min):
    fname = "figures/Au4-median_RgcFeH_HistogramMassWeighted_iold_{:.1f}.png".format(age_min)
    # NB: not using plot_values_skip
    fig = average_logM_FeHRgc_histogram_for_all_sims(auriga, l4_sims,
        verbose=True, debug=False, age_min=age_min)

    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_9_top(auriga, l4_sims, MW_h96e10, plot_values_skip, age_min):
    fname = "figures/Au4-median_RgcFeH_HistogramMassWeighted_MW_iold_diverging_{:.1f}.png".format(age_min)
    fig = average_logM_FeHRgc_histogram_for_all_sims_diverging(auriga, l4_sims,
        MW_h96e10=MW_h96e10, age_min=age_min, verbose=False, debug=False,
        plot_values_skip=plot_values_skip
    )
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def figure_9_bottom(auriga, l4_sims, M31_cr16_scaled_to_MW, plot_values_skip, age_min):
    fname = "figures/Au4-median_RgcFeH_HistogramMassWeighted_M31_iold_diverging_{:.1f}.png".format(age_min)
    fig = average_logM_FeHRgc_histogram_for_all_sims_diverging(auriga, l4_sims,
        M31_cr16=M31_cr16_scaled_to_MW, age_min=age_min, verbose=True, debug=False,
        plot_values_skip=plot_values_skip
    )
    fig.savefig(fname)
    pyplot.close(fig)
    os.system("convert -trim {} {}".format(fname, fname.replace(".png", "-trim.png")))
    os.remove(fname)
    print("Saved {}".format(fname))


def new_argument_parser():
    args = argparse.ArgumentParser(description="Plot figures for Auriga GCS paper")
    args.add_argument("--fig1", dest="fig1", action="store_true", help="Plot Figure 1")
    args.add_argument("--fig2", dest="fig2", action="store_true", help="Plot Figure 2")
    args.add_argument("--fig3", dest="fig3", action="store_true", help="Plot Figure 3")
    args.add_argument("--fig4", dest="fig4", action="store_true", help="Plot Figure 4")
    args.add_argument("--fig5", dest="fig5", action="store_true", help="Plot Figure 5")
    args.add_argument("--fig6", dest="fig6", action="store_true", help="Plot Figure 6")
    args.add_argument("--fig7", dest="fig7", action="store_true", help="Plot Figure 7")
    args.add_argument("--fig8", dest="fig8", action="store_true", help="Plot Figure 8")
    args.add_argument("--fig9", dest="fig9", action="store_true", help="Plot Figure 9")
    args.add_argument("--all", dest="all", action="store_true", help="Plot all")

    args.add_argument("--age_min", dest="age_min", default=10.0, type=float, help="Age cut [Gyr]")
    args.add_argument("--npz", dest="npz", action="store_true", help="(Re)generate npz dumps")
    args.add_argument("-N", dest="N", default=16, type=int, help="Parallel processes")
    args.add_argument("--silence", dest="silence", action="store_true", help="Silence plot output")

    return args


if __name__ == "__main__":
    # Parse arguments
    args, unknown = new_argument_parser().parse_known_args()
    print("Running {0}".format(__file__))
    print("  fig1:    {0}".format(args.fig1))
    print("  fig2:    {0}".format(args.fig2))
    print("  fig3:    {0}".format(args.fig3))
    print("  fig4:    {0}".format(args.fig4))
    print("  fig5:    {0}".format(args.fig5))
    print("  fig6:    {0}".format(args.fig6))
    print("  fig7:    {0}".format(args.fig7))
    print("  fig8:    {0}".format(args.fig8))
    print("  fig9:    {0}".format(args.fig9))
    print("  all:     {0}".format(args.all))
    print("  age_min: {0}".format(args.age_min))
    print("  npz:     {0}".format(args.npz))
    print("  N:       {0}".format(args.N))
    print("  silence: {0}".format(args.silence))


    if args.silence:
        from tlrh_util import suppress_stdout
    else:  # override suppress_stdout to not suppress stdout :+1:
        from contextlib import contextmanager
        @contextmanager
        def suppress_stdout():
            yield

    # Read the data
    MW_h96e10, MW_v13, M31_c11, M31_cr16 = read_MWandM31_data()

    # [Fe/H] for the Milky Way globular cluster system
    MWFeH = MW_h96e10["FeH"][numpy.isfinite(MW_h96e10["FeH"])]  # 152
    # [Fe/H] for the Andromeda globular cluster system
    M31FeH = M31_cr16["[Fe/H]"][numpy.isfinite(M31_cr16["[Fe/H]"])]  # 314

    # Rvir from Patel+ 2017 (2017MNRAS.464.3825P), Table 2.
    # Mvir, Rvir calculated for rho/rho_crit = 357 in WMAP9 cosmology
    # Adopting the values for Mvir,MW = 1e12 MSun; Mvir_M31 = 1.5e12 Msun
    MW_rvir = 261.
    M31_rvir = 299.

    M31_c11_scaled_to_MW = copy.copy(M31_c11)
    M31_c11_scaled_to_MW["Rgc"] = M31_c11_scaled_to_MW["Rgc"] / M31_rvir * MW_rvir
    M31_c11_scaled_to_MW["Rproj"] = M31_c11_scaled_to_MW["Rproj"] / M31_rvir * MW_rvir
    M31_c11_scaled_to_MW["Rdeproj"] = M31_c11_scaled_to_MW["Rdeproj"] / M31_rvir * MW_rvir
    M31_cr16_scaled_to_MW = copy.copy(M31_cr16)
    M31_cr16_scaled_to_MW["Rgc"] = M31_cr16_scaled_to_MW["Rgc"] / M31_rvir * MW_rvir
    M31_cr16_scaled_to_MW["Rproj"] = M31_cr16_scaled_to_MW["Rproj"] / M31_rvir * MW_rvir
    M31_cr16_scaled_to_MW["Rdeproj"] = M31_cr16_scaled_to_MW["Rdeproj"] / M31_rvir * MW_rvir

    # Rgc for the Milky Way globular cluster system
    MWRgc = MW_h96e10["R_gc"][numpy.isfinite(MW_h96e10["R_gc"])]  # 157
    # Rgc for the Andromeda globular cluster system
    M31Rgc = M31_cr16_scaled_to_MW["Rgc"][numpy.isfinite(M31_cr16_scaled_to_MW["Rgc"])]  # 441
    M31Rproj = M31_cr16_scaled_to_MW["Rproj"][numpy.isfinite(M31_cr16_scaled_to_MW["Rproj"])]  # 441

    # Read the simulations
    auriga = AurigaOverview(verbose=False)

    # Generate npz files
    if args.npz:
        #  8 threads --> 4m35s
        # 16 threads --> 1m42s
        # 30 threads --> 2m14s
        master_process(auriga, { **l4_sims }, lambda run:
            save_feabund(run, age_min=args.age_min), nthreads=args.N)
        master_process(auriga, { **l4_sims }, lambda run:
            save_FeH_logM_hist(run, age_min=args.age_min), nthreads=args.N)
        master_process(auriga, { **l4_sims }, lambda run:
            save_rgal(run, MW_rvir, age_min=args.age_min), nthreads=args.N)
        master_process(auriga, { **l4_sims }, lambda run:
            save_Rgc_logM_hist(run, MW_rvir, age_min=args.age_min), nthreads=args.N)
        master_process(auriga, { **l4_sims }, lambda run:
            save_FeHRgc_logM_hist(run, MW_rvir, age_min=args.age_min), nthreads=args.N)

    # For Figure 2 and Figure 9
    plot_values_skip = [(4,3), (3,4), (4,4)]

    if args.fig1 or args.all:
        print("\nFigure 1")
        figure_1_top(MW_v13, M31_cr16)
        figure_1_bottom(MW_h96e10, M31_cr16_scaled_to_MW, MW_v13, MW_rvir, M31_rvir)
    if args.fig2 or args.all:
        print("\nFigure 2")
        figure_2_top(MW_h96e10, plot_values_skip)
        figure_2_bottom(M31_cr16_scaled_to_MW, plot_values_skip)
    if args.fig3 or args.all:
        print("\nFigure 3")
        figure_3(auriga, MW_h96e10, M31_cr16, args.age_min)
    if args.fig4 or args.all:
        print("\nFigure 4")
        figure_4(auriga, MW_h96e10, M31_cr16, args.age_min)
    if args.fig5 or args.all:
        print("\nFigure 5")
        figure_5(auriga, MW_h96e10, M31_cr16, args.age_min)
    if args.fig6 or args.all:
        print("\nFigure 6")
        figure_6(auriga, MWRgc, MW_h96e10, M31Rgc, args.age_min)
    if args.fig7 or args.all:
        print("\nFigure 7")
        figure_7(auriga, MW_h96e10, M31_cr16_scaled_to_MW, args.age_min)
    if args.fig8 or args.all:
        print("\nFigure 8")
        figure_8(auriga, l4_sims, args.age_min)
    if args.fig9 or args.all:
        print("\nFigure 9")
        figure_9_top(auriga, l4_sims, MW_h96e10, plot_values_skip, args.age_min)
        figure_9_bottom(auriga, l4_sims, M31_cr16_scaled_to_MW, plot_values_skip, args.age_min)
