import copy
import numpy
import scipy
import colorcet
import matplotlib
from matplotlib import pyplot
from scipy.stats import binned_statistic
from scipy.stats import binned_statistic_2d

from main import COLORS
from main import l4_sims
from cosmology import calculate_rvir
from tlrh_util import suppress_stdout
from mw_m31_gc_observations import bin_milkyway_data
from mw_m31_gc_observations import bin_andromeda_data
from convert_auriga_units import compute_iron_over_hydrogen_abundance


def select_gc_candidates(s, sf, age_min=10.0, verbose=True):
    istars, = numpy.where(
        (s.type == 4) & (s.age > 0.)
        & (s.halo == 0) & (s.subhalo == 0)
        # & ( (s.halo != 0) | ((s.halo == 0) & (s.subhalo != 0)) )
    )

    with suppress_stdout():
        # s.age contains star- and wind (age < 0.) particles. sqrt(a**3) has negative argument --> runtimewarning
        age_Gyr = s.cosmology_get_lookback_time_from_a( s.age, is_flat=True )
        # age_Gyr is nan for wind particles --> runtimewarning in greater than
        iold = numpy.intersect1d( istars, numpy.where(age_Gyr > age_min) )

    insitu = numpy.intersect1d(
        istars,
        numpy.where(numpy.in1d(s.id, s.insitu))
    )
    accreted = numpy.intersect1d(
        istars,
        numpy.where(numpy.in1d(s.id, s.insitu, invert=True))
    )
    insitu_old = numpy.intersect1d(
        iold,
        numpy.where(numpy.in1d(s.id, s.insitu))
    )
    accreted_old = numpy.intersect1d(
        iold,
        numpy.where(numpy.in1d(s.id, s.insitu, invert=True))
    )

    if verbose:
        print("{0} has {1: 10d} stars".format(s.name, len(istars)))
        print("{0} has {1: 10d} insitu stars".format(s.name, len(insitu), age_min))
        print("{0} has {1: 10d} accreted stars".format(s.name, len(accreted), age_min))
        print("{0} has {1: 10d} old (> {2} Gyr) stars".format(s.name, len(iold), age_min))
        print("{0} has {1: 10d} old (> {2} Gyr) insitu stars".format(s.name, len(insitu_old), age_min))
        print("{0} has {1: 10d} old (> {2} Gyr) accreted stars".format(s.name, len(accreted_old), age_min))

    return istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr


###############################################################################
##                                Metallicity                                ##
###############################################################################
def save_feabund(run, age_min=10.0):
    print("  save_feabund for {0} /w age_min = {1}".format(run.name, age_min))

    s, sf = run.load_snapshot(loadonlytype=[0,4], verbose=False)
    if not hasattr(run, "insitu"):
        run.set_insitu_stars()
    s.insitu = run.insitu
    istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
        = select_gc_candidates(s, sf, age_min=age_min)

    for mask, mask_name in zip(
            [istars, iold, insitu, accreted, insitu_old, accreted_old],
            ["istars", "iold", "insitu", "accreted", "insitu_old", "accreted_old"]
        ):
        feabund = compute_iron_over_hydrogen_abundance(s, sf, mask)

        numpy.savez("{0}/{1}-{2}_feabund_{3}_{4:.1f}.npz".format(run.outdir,
            run.name, s.snapnr, mask_name, age_min), feabund=feabund)

    # data = np.load("{0}/{1}-{2}_feabund_{3}_{4:.1f}.npz".format(
    #     run.outdir, run.name, s.snapnr, mask_name, age_min))
    # feabund = data["feabund"]; print(feabund)

    del s, sf


def save_FeH_logM_hist(run, age_min=10.0, bins=39, myrange=(-3, 1.0)):
    print("  save_FeH_logM_hist for {0} /w age_min = {1}".format(
        run.name, age_min))

    s, sf = run.load_snapshot(loadonlytype=[0,4], verbose=False)
    if not hasattr(run, "insitu"):
        run.set_insitu_stars()
    s.insitu = run.insitu
    istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
        = select_gc_candidates(s, sf, age_min=age_min)

    for mask, mask_name in zip(
            [istars, iold, insitu, accreted, insitu_old, accreted_old],
            ["istars", "iold", "insitu", "accreted", "insitu_old", "accreted_old"]
        ):
        feabund = compute_iron_over_hydrogen_abundance(s, sf, mask)
        Msum, Medges, Mcnt = binned_statistic(feabund, s.mass[mask]*1e10,
            bins=bins, statistic="sum", range=myrange)

        numpy.savez("{0}/{1}-{2}_FeH_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
            run.outdir, run.name, s.snapnr, mask_name, age_min, bins, myrange),
            Msum=Msum, Medges=Medges, Mcnt=Mcnt, Ngc=len(mask))

    # data = numpy.load("{0}/{1}-{2}_FeH_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
    #     run.outdir, run.name, s.snapnr, mask_name, age_min, bins, myrange))
    # Msum = data["Msum"]; print(Msum)

    del s, sf


def plot_FeH_obs_and_4_and_10_and_21(auriga, MW_h96e10, M31_cr16, age_min=10.0,
        nbins=39, myrange=(-3, 1.0), yscale="log", use_fit=True, show_gauss_all=False):

    fig, (ax_obs, ax_4, ax_10, ax_21) = pyplot.subplots(
        4, 1, figsize=(12, 18))

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates: all",
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }

    # To fit a Gaussian g(x) where p[0] mu; p[1] sigma
    from scipy.optimize import leastsq
    fitfunc  = lambda p, x: p[2]/(numpy.sqrt(2*numpy.pi)*p[1]) * numpy.exp(-0.5*((x-p[0])/p[1])**2)
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))
    xvals = numpy.linspace(-4.0, 2, 256)

    ### Get the FeH distributions
    # Milky Way
    # Here is dataset2.1: [Fe/H] for the Milky Way globular cluster system
    MWFeH = MW_h96e10["FeH"][numpy.isfinite(MW_h96e10["FeH"])]  # 152

    # Andromeda
    # Here is dataset2.2: [Fe/H] for the Andromeda globular cluster system
    M31FeH = M31_cr16["[Fe/H]"][numpy.isfinite(M31_cr16["[Fe/H]"])]  # 314

    ### Plot the FeH distributions
    # Milky Way
    MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10); & " + \
        r"N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(MWFeH))
    counts, edges = numpy.histogram(MWFeH, range=(-3, 1), bins=39, density=True)
    ax_obs.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c=COLORS["MW"], lw=4, label=MW_GCS_label)

    # Only mu, sigma from literature. Normalization of 0.3 and 0.75 cheated for
    # a visually more pleasing Gaussian ...
    ax_obs.plot(xvals, fitfunc([-0.51, 0.23, 0.3], xvals),  # MW MRP
        c=COLORS["MW"], lw=2, ls="dashed")
    ax_obs.plot(xvals, fitfunc([-1.59, 0.34, 0.75], xvals),  # MW MPP
        c=COLORS["MW"], lw=2, ls="dashed", label="MW - double Gaussian, Harris 2001")

    # Fit Gaussian to MW all
    mu, sigma = numpy.mean(MWFeH), numpy.std(MWFeH, ddof=1)
    counts, edges = numpy.histogram(MWFeH, bins=39, range=(-3, 1), density=True)
    c, success = leastsq( errfunc, [mu, sigma, 1],
        args=( (edges[1:]+edges[:-1])/2, counts)
    )
    trans = matplotlib.transforms.blended_transform_factory(
        ax_obs.transData, ax_obs.transAxes)
    ax_obs.text(c[0], 0, u"\\textbf{\u2193}", color=COLORS["MW"], size=30,
        va="bottom", ha="center", transform=trans)

    print("\nFitting g(x) to MW [Fe/H] observations")
    print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}\n".format(
        c[0], c[1], c[2]))
    if show_gauss_all:
        ax_obs.plot(xvals, fitfunc(c, xvals),  # MW all
            c=COLORS["MW"], lw=2, ls="dashed")

    # Indicate M31
    M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(M31FeH))
    counts, edges = numpy.histogram(M31FeH, range=(-3, 1), bins=39, density=True)
    ax_obs.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
        c=COLORS["M31"], lw=4, label=M31_GCS_label)

    # Fit Gaussian to M31 all
    mu, sigma = numpy.mean(M31FeH), numpy.std(M31FeH, ddof=1)
    counts, edges = numpy.histogram(M31FeH, bins=39, range=(-3, 1), density=True)
    c, success = leastsq( errfunc, [mu, sigma, 1],
        args=( (edges[1:]+edges[:-1])/2, counts)
    )
    trans = matplotlib.transforms.blended_transform_factory(
        ax_obs.transData, ax_obs.transAxes)
    ax_obs.text(c[0], 0, u"\\textbf{\u2193}", color=COLORS["M31"], size=30,
        va="bottom", ha="center", transform=trans)
    print("\nFitting g(x) to M31 [Fe/H] observations")
    print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}\n".format(
        c[0], c[1], c[2]))
    if show_gauss_all:
        ax_obs.plot(xvals, fitfunc([c[0], c[1], c[2]], xvals),  # M31 all
            c=COLORS["M31"], lw=2, ls="dashed")

    ax_obs.legend(frameon=False, fontsize=14)


    mu_insituold_minus_mu_accreted_old = []
    for level in [4]:
        for halo, this_ax in zip([4, 10, 21], [ax_4, ax_10, ax_21]):
            run = auriga.getrun(level, halo)
            # Need to load the data to have s.mass available ...
            s, sf = run.load_snapshot(loadonlytype=[4], verbose=False)
            if not hasattr(run, "insitu"):
                run.set_insitu_stars()
            s.insitu = run.insitu
            istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
                = select_gc_candidates(s, sf, age_min=age_min)

            # Figures to plot the FeH and Mass-Weighted FeH distributions
            xvals = numpy.linspace(-4.0, 2, 256)

            for mask, mask_name in zip(
                    [istars, iold, insitu_old, accreted_old],
                    ["istars", "iold", "insitu_old", "accreted_old"]
                ):
                data = numpy.load("{0}/{1}-{2}_feabund_{3}_{4:.1f}.npz".format(
                    run.outdir, run.name, run.nsnaps-1, mask_name, age_min))
                feabund = data["feabund"];
                icorrect, = numpy.where( (feabund > -6) & (feabund < 4) )
                feabund = feabund[icorrect]

                # Start with initial guess wat numpy gives for mu and sigma
                mu, sigma = numpy.mean(feabund), numpy.std(feabund, ddof=1)
                counts, edges = numpy.histogram(feabund, bins=78, range=(-6, 2), density=True)
                c, success = leastsq( errfunc, [mu, sigma, 1],
                    args=( (edges[1:]+edges[:-1])/2, counts)
                )

                # Numpy calculated mu, sigma
                print("{0} -> mu = {1:.2f} sigma={2:.2f}".format(run.name, mu, sigma))
                x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                # ax.plot(x, stats.norm.pdf(x, mu, sigma))

                # Plot the fit
                print("Fitting g(x) to [Fe/H]")
                print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}".format(
                    c[0], c[1], c[2]))

                this_ax.axvline(c[0], c=COLORS[mask_name],
                    ls="solid" if mask_name != "istars" else "dotted", lw=1)

                # Plot the distribution of the data
                fit_info_label = r"$\hspace*{{-1cm}}$ {0} $\vspace*{{0.25cm}}$ \\ $\mu$ = {1:.2f}, $\sigma$ = {2:.2f} $\vspace*{{0.25cm}}$".format(
                    LABELS[mask_name], c[0], c[1])
                this_ax.plot((edges[1:]+edges[:-1])/2, counts,  # drawstyle="steps-mid",
                    c=COLORS[mask_name], lw=4 if mask_name != "istars" else 2,
                    ls="solid" if mask_name != "istars" else "dotted",
                    label=fit_info_label)

                # Add the legend
                handles, labels = this_ax.get_legend_handles_labels()
                legend1 = this_ax.legend(
                    [h for h in handles[0:4]], labels[0:4],
                    loc="upper left", frameon=False,
                    fontsize=14
                )
                # Push the text down (to get the marker up)
                for txt in legend1.get_texts():
                    txt.set_y(-20)

            # Indicate in upper right corner simulation name
            this_ax.text(0.98, 0.98, run.name, ha="right", va="top", transform=this_ax.transAxes)
            # this_ax.set_ylabel("Normalized Count")

        # Axes limits for all
        for this_ax in fig.axes:
            this_ax.set_xlim(-3, 1)
            this_ax.set_ylim(0, 1.1)

        fig.text(0.06, 0.5, "Normalized Count",
            rotation="vertical", ha="center", va="center")
        ax_21.set_xlabel("[Fe/H]")

        fig.subplots_adjust(wspace=0, hspace=0)
        return fig


def plot_FeH_mean_vs_std(auriga, MW_h96e10, M31_cr16, use_fit=True,
        plot_nr=False, show_gauss_all=False, debug=False,
        age_min=10.0, nbins=39, myrange=(-3, 1.0), yscale="log",
        Mv_Sun=4.83, mass_to_light=1.7):

    fig, ax = pyplot.subplots(figsize=(12, 9))
    LABELS = {
        "istars": "All star particles", "iold": "GC candidates: all",
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }

    ### Get the FeH distributions
    # Milky Way
    # Here is dataset2.1: [Fe/H] for the Milky Way globular cluster system
    MWFeH = MW_h96e10["FeH"][numpy.isfinite(MW_h96e10["FeH"])]  # 152

    # Andromeda
    # Here is dataset2.2: [Fe/H] for the Andromeda globular cluster system
    M31FeH = M31_cr16["[Fe/H]"][numpy.isfinite(M31_cr16["[Fe/H]"])]  # 314

    # Indicate the Milky Way MRP and MPP, and MW all
    ax.errorbar(numpy.mean(MWFeH), numpy.std(MWFeH, ddof=1),
        xerr=scipy.stats.sem(MWFeH), marker="x", c=COLORS["MW"],
        ms=20, mew=6, elinewidth=6, linestyle="none", label="MW", zorder=100)
    ax.plot(-0.51, 0.23, "o", ms=20, c=COLORS["MW"],
        label="MW (`red' / metal-rich)")
    ax.axvline(-1, c=COLORS["MW"], lw=2, ls=":")
    ax.plot(-1.59, 0.34, "o", ms=16, mfc="none", mew=6, c=COLORS["MW"],
        label="MW (`blue' / metal-poor)")

    # Indicate Andromeda mean + std (for the entire population)
    ax.errorbar(numpy.mean(M31FeH), numpy.std(M31FeH, ddof=1),
        xerr=scipy.stats.sem(M31FeH), marker="x", c=COLORS["M31"],
        ms=20, mew=6, elinewidth=6, linestyle="none", label="M31", zorder=100)

    ### Get the FeH distributions weighted by mass
    # Milky Way
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["M_Vt"])
    )
    MW_GCS_label_mass = r"\begin{tabular}{p{4cm}l}MW (H96e10); & " + \
        r"N$_{{\text{{GC}}}} = {0}\end{{tabular}}".format(len(mwgcs))
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_h96e10["M_Vt"][mwgcs])
    ) * mass_to_light
    Msum_MW, Medges_MW, Mcnt_MW = binned_statistic(
        MW_h96e10["FeH"][mwgcs], MW_GCS_mass,
        bins=nbins, statistic="sum", range=myrange
    )

    # Andromeda
    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(M31_cr16["[Fe/H]"])
            & (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    M31_GCS_label_mass = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
        r" N$_{{\text{{GC}}}}$ = {0}\end{{tabular}}".format(len(i_has_FeH_and_logM))
    Msum_M31, Medges_M31, Mcnt_M31 = binned_statistic(
        M31_cr16["[Fe/H]"][i_has_FeH_and_logM],
        10**M31_cr16["LogM"][i_has_FeH_and_logM],
        bins=nbins, statistic="sum", range=myrange
    )

    # To fit a Gaussian g(x) where p[0] mu; p[1] sigma
    from scipy.optimize import leastsq
    fitfunc  = lambda p, x: p[2]/(numpy.sqrt(2*numpy.pi)*p[1]) * numpy.exp(-0.5*((x-p[0])/p[1])**2)
    errfunc  = lambda p, x, y: (y - fitfunc(p, x))


    mu_insituold_minus_mu_accreted_old = []
    for level in [4]:
        for halo in l4_sims[level]:
            run = auriga.getrun(level, halo)
            # Need to load the data to have s.mass available ...
            s, sf = run.load_snapshot(loadonlytype=[4], verbose=False)
            if not hasattr(run, "insitu"):
                run.set_insitu_stars()
            s.insitu = run.insitu
            istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
                = select_gc_candidates(s, sf, age_min=age_min)

            if debug:
                # Figures to plot the FeH and Mass-Weighted FeH distributions
                xvals = numpy.linspace(-4.0, 2, 256)
                fig_debug, (ax_debug, ax_debug2) = pyplot.subplots(
                    2, 1, figsize=(12, 9))
                fig_debug_mass, ax_debug_mass = pyplot.subplots(
                    1, 1, figsize=(12, 9))

                ### Plot the FeH distributions
                # Milky Way
                MW_GCS_label = r"\begin{tabular}{p{4cm}l}MW (H96e10); & " + \
                    r"Ngc = {0}\end{{tabular}}".format(len(MWFeH))
                counts, edges = numpy.histogram(MWFeH, range=(-3, 1), bins=39, density=True)
                ax_debug.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
                    c=COLORS["MW"], lw=4, label=MW_GCS_label)

                # And Milky Way Mass-Weighted
                ax_debug_mass.plot(
                    (Medges_MW[1:]+Medges_MW[:-1])/2, Msum_MW,
                    lw=4, c=COLORS["MW"], label=MW_GCS_label_mass,
                    drawstyle="steps-mid"
                )

                # Only mu, sigma from literature. 0.3 and 0.75 cheated for
                # a visually more pleasing Gaussian ...
                ax_debug.plot(xvals, fitfunc([-0.51, 0.23, 0.3], xvals),  # MW MRP
                    c=COLORS["MW"], lw=2, ls="dashed")
                ax_debug.plot(xvals, fitfunc([-1.59, 0.34, 0.75], xvals),  # MW MPP
                    c=COLORS["MW"], lw=2, ls="dashed")

                # Fit Gaussian to MW all
                mu, sigma = numpy.mean(MWFeH), numpy.std(MWFeH, ddof=1)
                counts, edges = numpy.histogram(MWFeH, bins=39, range=(-3, 1), density=True)
                c, success = leastsq( errfunc, [mu, sigma, 1],
                    args=( (edges[1:]+edges[:-1])/2, counts)
                )

                print("\nFitting g(x) to MW [Fe/H] observations")
                print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}\n".format(
                    c[0], c[1], c[2]))
                if show_gauss_all:
                    ax_debug.plot(xvals, fitfunc(c, xvals),  # MW all
                        c=COLORS["MW"], lw=2, ls="dashed")


                # Indicate M31
                M31_GCS_label = r"\begin{tabular}{p{4cm}l}M31 (CR16); & " + \
                    r" Ngc = {0}\end{{tabular}}".format(len(M31FeH))
                counts, edges = numpy.histogram(M31FeH, range=(-3, 1), bins=39, density=True)
                ax_debug.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid",
                    c=COLORS["M31"], lw=4, label=M31_GCS_label)

                # And M31 Mass-Weighted
                ax_debug_mass.plot(
                    (Medges_M31[1:]+Medges_M31[:-1])/2, Msum_M31,
                    lw=4, c=COLORS["M31"], label=M31_GCS_label_mass,
                    drawstyle="steps-mid"
                )

                # Fit Gaussian to M31 all
                mu, sigma = numpy.mean(M31FeH), numpy.std(M31FeH, ddof=1)
                counts, edges = numpy.histogram(M31FeH, bins=39, range=(-3, 1), density=True)
                c, success = leastsq( errfunc, [mu, sigma, 1],
                    args=( (edges[1:]+edges[:-1])/2, counts)
                )
                print("\nFitting g(x) to M31 [Fe/H] observations")
                print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}\n".format(
                    c[0], c[1], c[2]))
                if show_gauss_all:
                    ax_debug.plot(xvals, fitfunc([c[0], c[1], c[2]], xvals),  # M31 all
                        c=COLORS["M31"], lw=2, ls="dashed")

                ax_debug.legend(frameon=False, fontsize=16)
                ax_debug_mass.legend(frameon=False, fontsize=16)


            for mask, mask_name in zip(
                    [istars, iold, insitu_old, accreted_old],
                    ["istars", "iold", "insitu_old", "accreted_old"]
                ):
                data = numpy.load("{0}/{1}-{2}_feabund_{3}_{4:.1f}.npz".format(
                    run.outdir, run.name, run.nsnaps-1, mask_name, age_min))
                feabund = data["feabund"];
                icorrect, = numpy.where( (feabund > -6) & (feabund < 4) )
                feabund = feabund[icorrect]

                # Start with initial guess wat numpy gives for mu and sigma
                mu, sigma = numpy.mean(feabund), numpy.std(feabund, ddof=1)
                counts, edges = numpy.histogram(feabund, bins=78, range=(-6, 2), density=True)
                c, success = leastsq( errfunc, [mu, sigma, 1],
                    args=( (edges[1:]+edges[:-1])/2, counts)
                )

                if mask_name == "insitu_old":
                    mu_insitu_old = c[0]
                if mask_name == "accreted_old":
                    mu_accreted_old = c[0]
                if debug:
                    # Numpy calculated mu, sigma
                    print("{0} -> mu = {1:.2f} sigma={2:.2f}".format(run.name, mu, sigma))
                    x = numpy.linspace(mu - 3*sigma, mu + 3*sigma, 100)
                    # ax_debug.plot(x, stats.norm.pdf(x, mu, sigma))

                    # Plot the fit
                    print("Fitting g(x) to [Fe/H]")
                    print("  mu = {:.3f}\n  sigma = {:.3f}\n  a = {:.3f}".format(
                        c[0], c[1], c[2]))

                    # Could also just plot a Gaussian w/ numpy.mean and numpy.std
                    # ax_debug.plot(xvals, fitfunc((mu, sigma), xvals),
                    #     c=COLORS[mask_name], ls="dashed", lw=2,
                    #     label="numpy.mean = {0:.2f}\nnumpy.std = {1:.2f}".format(mu, sigma))
                    # Plot the fit results
                    fit_info_label = "{0} ({1})\nmu = {2:.2f}\nsigma = {3:.2f}".format(
                        run.name, LABELS[mask_name], c[0], c[1])
                    # ax_debug2.plot(xvals, fitfunc(c, xvals), c=COLORS[mask_name],
                    #     ls="dashed" if mask_name != "istars" else "dotted", lw=2,
                    #     label=fit_info_label)
                    ax_debug2.axvline(c[0], c=COLORS[mask_name],
                        ls="solid" if mask_name != "istars" else "dotted", lw=1)


                    # Plot the distribution of the data
                    fit_info_label = r"$\hspace*{{-1cm}}$ {0} ({1}) \\ \begin{{tabular}}{{p{{1cm}}ll}}".format(
                        run.name, LABELS[mask_name]) + \
                        r"mu & = & {0:.2f} \\ sigma & = & {1:.2f} \end{{tabular}}".format(c[0], c[1])
                    ax_debug2.plot((edges[1:]+edges[:-1])/2, counts,  # drawstyle="steps-mid",
                        c=COLORS[mask_name], lw=4 if mask_name != "istars" else 2,
                        ls="solid" if mask_name != "istars" else "dotted",
                        label=fit_info_label)
                    # Mass-Weighted
                    Msum, Medges, Mcnt = binned_statistic(feabund, s.mass[mask][icorrect]*1e10,
                        bins=nbins, statistic="sum", range=myrange)
                    ax_debug_mass.plot((Medges[1:]+Medges[:-1])/2, Msum,  # drawstyle="steps-mid",
                        c=COLORS[mask_name], lw=4 if mask_name != "istars" else 2,
                        ls="solid" if mask_name != "istars" else "dotted",
                        label="{0} ({1})".format(run.name, LABELS[mask_name])
                    )

                    ax_debug2.set_xlabel("[Fe/H]")
                    # ax_debug2.set_ylabel("Normalized Count")
                    # Common ylabel
                    fig_debug.text(0.06, 0.5, "Normalized Count",
                        rotation="vertical", ha="center", va="center")
                    # fig_debug.suptitle("{0}".format(run.name))
                    ax_debug_mass.set_xlabel("[Fe/H]")
                    ax_debug_mass.set_ylabel("Mass-Weighted Count")
                    # fig_debug_mass.suptitle("{0}".format(run.name))

                    # Legend, but w/o istars
                    handles, labels = ax_debug2.get_legend_handles_labels()
                    legend1 = ax_debug2.legend(
                        [h for h in handles[1:4]], labels[1:4],
                        bbox_to_anchor=(0., 1.07, 1., .102),
                        loc="upper left", frameon=False,
                        fontsize=16 if run.name != "Au4-1" else 12
                    )
                    ax_debug_mass.legend(loc="upper left", frameon=False, fontsize=16)

                    # Also, push the marker up
                    for txt in legend1.get_texts():
                        # txt.set_ha("center") # horizontal alignment of text item
                        # txt.set_x(-5) # x-position
                        txt.set_y(-39) # y-position

                # TODO: alternatively to stats.sem(feabund) we could use
                # curve_fit to to retrieve perr from pcov?
                mu_to_plot = c[0] if use_fit else mu
                sigma_to_plot = c[1] if use_fit else sigma
                ax.errorbar(mu_to_plot, sigma_to_plot, # xerr=scipy.stats.sem(feabund),
                    marker="x" if mask_name != "istars" else "^",
                    ms=15 if mask_name != "istars" else 10,
                    mew=4 if mask_name != "istars" else 2,
                    alpha=1 if mask_name != "istars" else 0.5,
                    c=COLORS[mask_name], linestyle="none", elinewidth=0,
                    label=LABELS[mask_name] if halo==1 else None)
                if plot_nr:
                    ax.text(mu_to_plot, sigma_to_plot, run.name.replace("Au4-", ""),
                        ha="center", va="center", transform=ax.transData, fontsize=14)
            if debug:
                # Meh does not work?
                # ax_debug.tick_params(axis="both", labelbottom=False, bottom=False)
                # ax_debug2.tick_params(axis="both", labeltop=False, top=False)
                for this_ax in [ax_debug, ax_debug2, ax_debug_mass]:
                    this_ax.set_xlim(-3, 1)
                for this_ax in [ax_debug, ax_debug2]:
                    this_ax.set_ylim(0, 1.1)

                fig_debug.subplots_adjust(wspace=0, hspace=0)
                fig_debug.savefig("{0}/{1}_FeH.png".format(run.outdir, run.name))
                fig_debug.savefig("{0}/{1}_FeH.pdf".format(run.outdir, run.name))
                display(fig_debug)
                pyplot.close(fig_debug)

                ax_debug_mass.set_yscale("log")
                ax_debug_mass.set_ylim(1e5, 1e10)
                fig_debug_mass.subplots_adjust(wspace=0, hspace=0)
                fig_debug_mass.savefig("{0}/{1}_FeH_mass.png".format(run.outdir, run.name))
                fig_debug_mass.savefig("{0}/{1}_FeH_mass.pdf".format(run.outdir, run.name))
                display(fig_debug_mass)
                pyplot.close(fig_debug_mass)

            mu_insituold_minus_mu_accreted_old.append(
                mu_insitu_old - mu_accreted_old)
            # break
        # break

    handles, labels = ax.get_legend_handles_labels()
    legend1 = ax.legend([h for h in handles[0:4]], labels[0:4],
        loc="upper right", frameon=False, fontsize=18)
    ax.legend([h for h in handles[4:]], labels[4:],
        loc="lower left", frameon=False, fontsize=18)
    ax.add_artist(legend1)

    ax.set_xlabel(r"Mean: $\langle$ [Fe/H] $\rangle$")
    ax.set_ylabel(r"Standard Deviation: $\sigma$( [Fe/H] )")
    # ax.set_xlim(-1.5, -0.25)
    # ax.set_ylim(0.4, 1.6)
    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

    return fig, mu_insituold_minus_mu_accreted_old


def bin_FeH_all_sims(auriga, sims, mask_name, nbins, myrange, ax, age_min=10.0,
        plot_all=False, hide_label=True, do_fit=False, verbose=False):

    LABELS = {
        "istars": "all stars", "iold": "old ({:.1f} Gyr)".format(age_min),
        "accreted": "accreted", "insitu": "insitu",
        "accreted_old": "old accreted", "insitu_old": "old insitu"
    }

    data_per_bin = [list() for b in range(nbins)]
    correlate_mass_normalisation = []
    for level in sims.keys():
        for halo in sims[level]:
            run = auriga.getrun(level=level, halo=halo)

            Au_data = numpy.load("{0}/{1}-{2}_FeH_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
                run.outdir, run.name, run.nsnaps-1, mask_name, age_min, nbins, myrange))
            Msum = Au_data["Msum"]
            Medges = Au_data["Medges"]
            Mcnt = Au_data["Mcnt"]
            Ngc = Au_data["Ngc"]
            Au_label  = r"\begin{{tabular}}{{p{{4cm}}l}}{0} ({1}); & Ngc = {2}\end{{tabular}}"\
                .format(run.name, LABELS[mask_name], Ngc)

            if hide_label: Au_label = None

            for b in range(nbins):
                data_per_bin[b].append(Msum[b])

            # The part on the left seems liner. What does the normalization depend on
            # / correlate with? We fit ax+b to [Fe/H] in range -3.0 to -1.0 to get b
            if do_fit:
                from scipy.optimize import leastsq

                fitfunc  = lambda p, x: p[0]*(x+3) + p[1]
                errfunc  = lambda p, x, y: (y - fitfunc(p, x))

                xvals = (Medges[1:]+Medges[:-1])/2
                inner = numpy.where(xvals < -1.0)

                c, success = leastsq( errfunc, [1, 6.5],
                    args=(xvals[inner], numpy.log10(Msum[inner])))
                if verbose:
                    print("Fitting ax+b to [Fe/H]<-1.0")
                    print("  a = {:.3f}\n  b = {:.3f}".format(c[0], c[1]))
                ax.plot(xvals[inner], Msum[inner], c="b", lw=2)
                xvals = numpy.linspace(-3.0, -0.5, 64)
                ax.plot(xvals, 10**fitfunc((1, 6.5), xvals), c="y", lw=2)
                ax.plot(xvals, 10**fitfunc(c, xvals), c="r", lw=2)

                s, sf = run.load_header(verbose=False)
                M200 = sf.data["fmc2"][0]  # 1e10 Msun
                correlate_mass_normalisation.append( (run.name, c[0], c[1], M200) )

            if plot_all:
                ax.plot((Medges[1:]+Medges[:-1])/2, Msum, c="grey" if hide_label else None,
                    drawstyle="steps-mid", alpha=0.3 if hide_label else None,
                    linewidth=None if hide_label else 4, label=Au_label)

    return Medges, data_per_bin, correlate_mass_normalisation


def plot_FeH_ratios(auriga, mask_names, MW_h96e10, M31_cr16,
        age_min=10.0, nbins=39, myrange=(-3, 1.0), yscale="log",
        Mv_Sun=4.83, mass_to_light=1.7, debug=False):

    fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1,figsize=(12, 16),
        sharex=True, gridspec_kw={"height_ratios": [9, 3, 3]})
    # cax.axis("off")

    # Milky Way
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["M_Vt"])
    )
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_h96e10["M_Vt"][mwgcs])
    ) * mass_to_light
    Msum_MW, Medges_MW, Mcnt_MW = binned_statistic(
        MW_h96e10["FeH"][mwgcs], MW_GCS_mass,
        bins=nbins, range=myrange, statistic="sum")

    MW_GCS_label = r"\begin{tabular}{p{3.45cm}l}MW (H96e10); & " + \
        r"Ngc = {0};\end{{tabular}}M/L = 1.7".format(len(mwgcs))
    ax1.plot(
        (Medges_MW[1:]+Medges_MW[:-1])/2, Msum_MW,
        lw=4, c="#542788", label=MW_GCS_label,
        drawstyle="steps-mid"
    )

    # Andromeda
    with suppress_stdout():  # invalid value encountered in < and/or >
        i_has_FeH_and_logM, = numpy.where(
            numpy.isfinite(M31_cr16["[Fe/H]"])
            & (M31_cr16["[Fe/H]"] > -6) & (M31_cr16["[Fe/H]"] < 4)
            & numpy.isfinite(M31_cr16["LogM"])
        )
    Msum_M31, Medges_M31, Mcnt_M31 = binned_statistic(
        M31_cr16["[Fe/H]"][i_has_FeH_and_logM],
        10**M31_cr16["LogM"][i_has_FeH_and_logM],
        bins=nbins, statistic="sum", range=myrange
    )


    M31_GCS_label = r"\begin{tabular}{p{3.45cm}l}M31 (CR16); & " + \
        r" Ngc = {0};\end{{tabular}}M/L = 2.0".format(len(i_has_FeH_and_logM))
    ax1.plot(
        (Medges_M31[1:]+Medges_M31[:-1])/2, Msum_M31,
        lw=4, c="#c51b7d", label=M31_GCS_label,
        drawstyle="steps-mid"
    )

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates: all",
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }
    for mask_name in mask_names:
        style = {
            "c": COLORS[mask_name],
            "ms": 8, "alpha": 1, "linewidth": 4,
            "linestyle": "solid" if mask_name != "istars" else "dotted",
            "label": "{0}".format(LABELS[mask_name])
        }

        # Auriga
        print("\n\nUSING MASK: {0}".format(mask_name))
        Medges, data_per_bin, correlate_mass_normalisation = bin_FeH_all_sims(
            auriga, l4_sims, mask_name, nbins, myrange, ax1, age_min=age_min,
            plot_all=False, do_fit=False, hide_label=True
        )

        percentiles = numpy.array(
            [numpy.percentile(data_per_bin[b], [25, 50, 75])
            for b in range(nbins)]
        )
        ax1.plot(
            (Medges[1:]+Medges[:-1])/2, percentiles[::,1],
            **style
        )
        ax1.fill_between(
            (Medges[1:]+Medges[:-1])/2, percentiles[::,0], percentiles[::,2],
            color=COLORS[mask_name], alpha=0.3
        )

        for ax_ratio, ratiomass in zip([ax2, ax3], [Msum_MW, Msum_M31]):

            # Ratios
            percentiles = numpy.array(
                [numpy.percentile(data_per_bin[b], [25, 50, 75])
                for b in range(nbins)]
            )
            style1 = copy.copy(style); style1["label"] = None
            # style1["drawstyle"] = "steps-mid"
            ax_ratio.plot(
                (Medges[1:]+Medges[:-1])/2, percentiles[::,1]/ratiomass,
                **style1
            )
            ax_ratio.fill_between(
                (Medges[1:]+Medges[:-1])/2,
                percentiles[::,0]/ratiomass, percentiles[::,2]/ratiomass,
                color=COLORS[mask_name], alpha=0.3
            )


    # Plot Settings
    ax1.set_ylabel("Total Mass [Msun]")
    ax1.set_yscale(yscale)
    ax1.set_ylim(5e3, 1.1e10)
    ax1.legend(loc="upper left", frameon=False, fontsize=16)

    for ax in [ax1, ax2, ax3]:
        ax.set_xticks(numpy.arange(-4, 1.5, 0.5))
        ax.set_xticks(numpy.arange(-4, 1.25, 0.25), minor=True)

        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

        ax.set_xlim(myrange[0], myrange[1])
        ax.set_yscale(yscale)

    for ax in [ax2, ax3]:
        ax.set_ylim(1e0, 3e5)
        ax.set_yticks([1e0, 1e2, 1e4])
        ax.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$"])
        ax.set_ylabel("Mass Ratio")
    # ax2.set_ylabel(r"$\Sigma_{M_{\rm Au}} / \Sigma_{M_{\rm MW}} $")
    ax2.text(0.05, 0.95, "Milky Way", ha="left", va="top", transform=ax2.transAxes)
    # ax3.set_ylabel(r"$\Sigma_{M_{\rm Au}} / \Sigma_{M_{\rm M31}} $")
    ax3.text(0.05, 0.95, "Andromeda", ha="left", va="top", transform=ax3.transAxes)
    ax3.set_xlabel("[Fe/H]")

    pyplot.tight_layout()
    fig.subplots_adjust(hspace=0)
    return fig


###############################################################################
##                           Galactocentric radius                           ##
###############################################################################
def save_rgal(run, MW_rvir, age_min=10.0):
    print("  save_rgal for {0} /w MW_rvir = {1}, age_min = {2}".format(
        run.name, MW_rvir, age_min))

    s, sf = run.load_snapshot(loadonlytype=range(6), verbose=False)
    if not hasattr(run, "insitu"):
        run.set_insitu_stars()
    s.insitu = run.insitu
    istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
        = select_gc_candidates(s, sf, age_min=age_min)

    if MW_rvir > 10:
        print("\nScaling to MW size")
        Mvirs, rvirs = calculate_rvir(s, sf, cosmo="WMAP9", verbose=True, debug=False)
        Au_rvir = rvirs[2]  # tophat
        rvir_factor = MW_rvir/Au_rvir
        print("  SUBFIND\n    mean   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmm2"][0], 1e3 * sf.data["frm2"][0]))
        print("    crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmc2"][0], 1e3 * sf.data["frc2"][0]))
        print("    tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]\n".format(
            1e10 * sf.data["fmt2"][0], 1e3 * sf.data["frt2"][0]))
        print("  MW_rvir = {0:.2f} kpc [WMAP9]".format(MW_rvir))
        print("  Au_rvir = {0:.2f} kpc [WMAP9]".format(Au_rvir))
        print("  rvir_factor = {0:.2f}".format(rvir_factor))
    else:
        rvir_factor = 1.0

    for mask, mask_name in zip(
            [istars, iold, insitu, accreted, insitu_old, accreted_old],
            ["istars", "iold", "insitu", "accreted", "insitu_old", "accreted_old"]
        ):
        rgc = 1000 * s.r()[mask]*rvir_factor

        numpy.savez("{0}/{1}-{2}_rgal_{3}_{4:.1f}.npz".format(
            run.outdir, run.name, s.snapnr, mask_name, age_min), rgc=rgc)

    # data = numpy.load("{0}/{1}-{2}_rgal_{3}_{4:.1f}.npz".format(
    #     run.outdir, run.name, s.snapnr, mask_name, age_min))
    # feabund = data["rgc"]; print(feabund)

    del s, sf


def save_Rgc_logM_hist(run, MW_rvir, age_min=10.0, nbins=32, myrange=(0.1, 500)):
    print("  save_logM_Rgc_hist for {0} /w MW_rvir = {1}, age_min = {2:.2f}".format(
        run.name, MW_rvir, age_min))

    s, sf = run.load_snapshot(loadonlytype=range(6), verbose=False)
    if not hasattr(run, "insitu"):
        run.set_insitu_stars()
    s.insitu = run.insitu
    istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
        = select_gc_candidates(s, sf, age_min=age_min)

    # Logarithmic bins in x-direction
    bins = numpy.power(10, numpy.linspace(numpy.log10(myrange[0]),
        numpy.log10(myrange[1]), nbins))

    if MW_rvir > 10:
        print("\nScaling to MW size")
        Mvirs, rvirs = calculate_rvir(s, sf, cosmo="WMAP9", verbose=True, debug=False)
        Au_rvir = rvirs[2]  # tophat
        rvir_factor = MW_rvir/Au_rvir
        print("  SUBFIND\n    mean   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmm2"][0], 1e3 * sf.data["frm2"][0]))
        print("    crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmc2"][0], 1e3 * sf.data["frc2"][0]))
        print("    tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]\n".format(
            1e10 * sf.data["fmt2"][0], 1e3 * sf.data["frt2"][0]))
        print("  MW_rvir = {0:.2f} kpc [WMAP9]".format(MW_rvir))
        print("  Au_rvir = {0:.2f} kpc [WMAP9]".format(Au_rvir))
        print("  rvir_factor = {0:.2f}".format(rvir_factor))
    else:
        rvir_factor = 1.0

    for mask, mask_name in zip(
            [istars, iold, insitu, accreted, insitu_old, accreted_old],
            ["istars", "iold", "insitu", "accreted", "insitu_old", "accreted_old"]
        ):

        Msum, Medges, Mcnt = binned_statistic(1000*s.r()[mask]*rvir_factor,
            s.mass[mask]*1e10, bins=bins, statistic="sum")

        numpy.savez("{0}/{1}-{2}_Rgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
            run.outdir, run.name, s.snapnr, mask_name, age_min, nbins, myrange),
            Msum=Msum, Medges=Medges, Mcnt=Mcnt, Ngc=len(mask))

    # data = np.load("{0}/{1}-{2}_Rgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
    #     run.outdir, run.name, s.snapnr, mask_name, age_min, nbins, myrange))
    # Msum = data["Msum"]; print(Msum)

    del s, sf


def plot_rgal_mean_vs_std(auriga, MWRgc, MW_h96e10, M31Rgc,
        age_min=10.0, plot_nr=False):

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates: all ({:.1f})".format(age_min),
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }

    fig, ax = pyplot.subplots(figsize=(12, 9))

    for level in [4]:
        for halo in l4_sims[level]:
            run = auriga.getrun(level, halo)
            for mask_name in ["istars", "iold", "insitu_old", "accreted_old"]:
                data = numpy.load("{0}/{1}-{2}_rgal_{3}_{4:.1f}.npz".format(
                    run.outdir, run.name, run.nsnaps-1, mask_name, age_min))
                ifinite, = numpy.where(numpy.isfinite(numpy.log10(data["rgc"])))
                rgc = numpy.log10(data["rgc"][ifinite])

                # print(run.name, mask_name)
                # print(data["rgc"].shape)
                # print(len(numpy.where(numpy.isfinite(data["rgc"]))[0]))
                # print(numpy.log10(data["rgc"]).shape)
                # print(len(numpy.where(numpy.isfinite(numpy.log10(data["rgc"])))[0]))
                # print(numpy.mean(rgc), numpy.std(rgc), scipy.stats.sem(rgc))

                ax.errorbar(numpy.mean(rgc), numpy.std(rgc),
                    # xerr=scipy.stats.sem(rgc),
                    marker="x" if mask_name != "istars" else "^",
                    alpha=1 if mask_name != "istars" else 0.5,
                    ms=15 if mask_name != "istars" else 10,
                    mew=4 if mask_name != "istars" else 2,
                    linestyle="none", c=COLORS[mask_name],
                    label=LABELS[mask_name] if halo==1 else None)
                if plot_nr:
                    ax.text(numpy.mean(rgc), numpy.std(rgc),
                        run.name.replace("Au4-", ""), ha="center", va="center",
                        transform=ax.transData, fontsize=14)

    # Add the observations
    i_has_FeH_and_Rgc, = numpy.where(
        numpy.isfinite(MW_h96e10["FeH"])
        & numpy.isfinite(MW_h96e10["R_gc"])
    )
    i_metal_rich, = numpy.where(MW_h96e10["FeH"][i_has_FeH_and_Rgc] > -1)
    i_metal_poor, = numpy.where(MW_h96e10["FeH"][i_has_FeH_and_Rgc] <= -1)

    MWRgc_rich = numpy.log10(MW_h96e10["R_gc"][i_has_FeH_and_Rgc][i_metal_rich])
    MWRgc_poor = numpy.log10(MW_h96e10["R_gc"][i_has_FeH_and_Rgc][i_metal_poor])
    ax.errorbar(numpy.mean(MWRgc_rich), numpy.std(MWRgc_rich), marker="o", ms=20,
        c=COLORS["MW"], linestyle="none", label="MW (`red' / metal-rich)")
    ax.errorbar(numpy.mean(MWRgc_poor), numpy.std(MWRgc_poor), marker="o", ms=16, mfc="none",
        mew=6, c=COLORS["MW"], linestyle="none", label="MW (`blue' / metal-poor)")

    # Indicate the Milky Way MRP and MPP, and MW all
    ax.errorbar(numpy.mean(MWRgc), numpy.std(MWRgc),
        xerr=scipy.stats.sem(MWRgc), marker="x", c=COLORS["MW"],
        ms=20, mew=6, elinewidth=6, linestyle="none", label="MW", zorder=100)

    # Indicate Andromeda mean + std (for the entire population)
    ax.errorbar(numpy.mean(M31Rgc), numpy.std(M31Rgc),
        xerr=scipy.stats.sem(M31Rgc), marker="x", c=COLORS["M31"],
        ms=20, mew=6, elinewidth=6, linestyle="none", label="M31", zorder=100)

    ax.set_xlim(0, 1.4)
    ax.set_ylim(0.3, 0.85)
    ax.set_xlabel(r"Mean: $<\log_{10}(r_{\rm gal} {\rm [kpc]})>$")
    ax.set_ylabel(r"Standard Deviation: $\sigma$( $\log_{10}(r_{\rm gal} {\rm [kpc]})$ )")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
    ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")
    ax.legend(loc="upper left", frameon=False, fontsize=18)

    return fig


def bin_rgal_all_sims(auriga, sims, mask_name, nbins, myrange, ax, age_min=10.0,
        plot_all=False, hide_label=True, do_fit=False, verbose=False):

    LABELS = {
        "istars": "all stars", "iold": "old ({:.1f})".format(age_min),
        "insitu": "insitu", "accreted": "accreted",
        "accreted_old": "old accreted", "insitu_old": "old insitu"
    }

    data_per_bin = [list() for b in range(nbins)]
    correlate_mass_normalisation = []
    for level in sims.keys():
        for halo in sims[level]:
            run = auriga.getrun(level=level, halo=halo)

            Au_data = numpy.load("{0}/{1}-{2}_Rgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
            run.outdir, run.name, run.nsnaps-1, mask_name, age_min, nbins, myrange))
            Msum = Au_data["Msum"]
            Medges = Au_data["Medges"]
            Mcnt = Au_data["Mcnt"]
            Ngc = Au_data["Ngc"]
            Au_label  = r"\begin{{tabular}}{{p{{4cm}}l}}{0} ({1}); & Ngc = {2}\end{{tabular}}"\
                .format(run.name, LABELS[mask_name], Ngc)
            if hide_label: Au_label = None

            for b in range(nbins-1):
                data_per_bin[b].append(Msum[b])

            # The part on the left seems liner. What does the normalization depend on
            # / correlate with? We fit ax+b to [Fe/H] in range -3.0 to -1.0 to get b
            if do_fit:
                from scipy.optimize import leastsq

                fitfunc  = lambda p, x: 0*x + p[0]
                errfunc  = lambda p, x, y: (y - fitfunc(p, x))

                xvals = (Medges[1:]+Medges[:-1])/2
                inner = numpy.where((xvals > 1) & (xvals < 100))

                c, success = leastsq( errfunc, [8.5],
                    args=(numpy.log10(xvals[inner]), numpy.log10(Msum[inner])))
                if verbose:
                    print("Fitting ax+b to 1 < Rgc < 100")
                    print("  a = {:.3f}\n  b = {:.3f}".format(c[0], c[1]))
                ax.plot(xvals[inner], Msum[inner], c="b", lw=2)
                xvals = numpy.log10(numpy.linspace(1, 100, 64))
                ax.plot(10**xvals, 10**fitfunc((8.5,), 10**xvals), c="y", lw=2)
                ax.plot(10**xvals, 10**fitfunc(c, xvals), c="r", lw=2)

                s, sf = run.load_header(verbose=False)
                M200 = sf.data["fmc2"][0]  # 1e10 Msun
                correlate_mass_normalisation.append( (run.name, c[0], M200) )

            if plot_all:
                ax.plot((Medges[1:]+Medges[:-1])/2, Msum, c="grey" if hide_label else None,
                    drawstyle="steps-mid", alpha=0.3 if hide_label else None,
                    linewidth=None if hide_label else 4, label=Au_label)

    return Medges, data_per_bin, correlate_mass_normalisation


def plot_rgal_ratios(auriga, mask_names, MW_h96e10, M31_cr16, age_min=10.0,
        nbins=32, myrange=(0.1, 500), yscale="log",
        Mv_Sun=4.83, mass_to_light=1.7, debug=False):

    fig, (ax1, ax2, ax3) = pyplot.subplots(3, 1,figsize=(12, 16),
        sharex=True, gridspec_kw={"height_ratios": [9, 3, 3]})

    # Logarithmic bins in x-direction
    bins = numpy.power(10, numpy.linspace(numpy.log10(myrange[0]),
        numpy.log10(myrange[1]), nbins))

    # Milky Way
    mwgcs, = numpy.where(
        numpy.isfinite(MW_h96e10["M_Vt"])
    )
    MW_GCS_mass = numpy.power(10,
        0.4*(Mv_Sun - MW_h96e10["M_Vt"][mwgcs])
    ) * mass_to_light
    Msum_MW, Medges_MW, Mcnt_MW = binned_statistic(
        MW_h96e10["R_gc"][mwgcs], MW_GCS_mass,
        bins=bins, range=myrange, statistic="sum")

    # Andromeda
    i_has_logM, = numpy.where(
        numpy.isfinite(M31_cr16["LogM"])
    )
    Msum_M31, Medges_M31, Mcnt_M31 = binned_statistic(
        M31_cr16["Rgc"][i_has_logM],
        10**M31_cr16["LogM"][i_has_logM],
        bins=bins, range=myrange, statistic="sum",
    )

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates: all",
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }
    for mask_name in mask_names:
        style = {
            "c": COLORS[mask_name],
            "ms": 8, "alpha": 1, "linewidth": 4,
            "linestyle": "solid" if mask_name != "istars" else "dotted",
            "label": "{0}".format(LABELS[mask_name]),
        }

        # Auriga
        print("\n\nUSING MASK: {0}".format(mask_name))
        Medges, data_per_bin, correlate_mass_normalisation = bin_rgal_all_sims(
            auriga, l4_sims, mask_name, nbins, myrange, ax1, age_min=age_min,
            plot_all=False, do_fit=False, hide_label=True
        )

        percentiles = numpy.array(
            [numpy.percentile(data_per_bin[b], [25, 50, 75])
            for b in range(nbins-1)]
        )
        ax1.plot(
            (Medges[1:]+Medges[:-1])/2, percentiles[::,1],
            **style
        )
        ax1.fill_between(
            (Medges[1:]+Medges[:-1])/2, percentiles[::,0], percentiles[::,2],
            color=COLORS[mask_name], alpha=0.3
        )

        for ax_ratio, ratiomass in zip([ax2, ax3], [Msum_MW, Msum_M31]):
            # Ratios
            percentiles = numpy.array(
                [numpy.percentile(data_per_bin[b], [25, 50, 75])
                for b in range(nbins-1)]
            )
            style1 = copy.copy(style); style1["label"] = None
            ax_ratio.plot(
                (Medges[1:]+Medges[:-1])/2, percentiles[::,1]/ratiomass,
                **style1
            )
            ax_ratio.fill_between(
                (Medges[1:]+Medges[:-1])/2,
                percentiles[::,0]/ratiomass, percentiles[::,2]/ratiomass,
                color=COLORS[mask_name], alpha=0.3
            )

    # Plot Milky Way
    MW_GCS_label = r"\begin{tabular}{p{3.5cm}l}MW (H96e10); & " + \
        r"Ngc = {0};\end{{tabular}}M/L = 1.7".format(len(mwgcs))
    ax1.plot(
        (Medges_MW[1:]+Medges_MW[:-1])/2, Msum_MW,
        lw=4, c=COLORS["MW"], label=MW_GCS_label,
        drawstyle="steps-mid"
    )

    # Plot Andromeda
    M31_GCS_label = r"\begin{tabular}{p{3.5cm}l}M31 (CR16); & " + \
        r" Ngc = {0};\end{{tabular}}M/L = 2.0".format(len(i_has_logM))
    ax1.plot(
        (Medges_M31[1:]+Medges_M31[:-1])/2, Msum_M31,
        lw=4, c=COLORS["M31"], label=M31_GCS_label,
        drawstyle="steps-mid"
    )

    # Plot Settings
    ax1.set_ylabel("Total Mass [Msun]")
    ax1.set_yscale(yscale)
    ax1.set_ylim(4e3, 1e10)
    ax1.legend(loc="lower center", frameon=False, fontsize=16)

    for ax in [ax1, ax2, ax3]:
        # ax.set_xticks(numpy.arange(-4, 1.5, 0.5))
        # ax.set_xticks(numpy.arange(-4, 1.25, 0.25), minor=True)

        ax.grid(color='grey', ls=":", lw=1, alpha=0.2, which="minor")
        ax.grid(color='grey', ls=":", lw=1, alpha=0.5, which="major")

        ax.set_xlim(0.75*myrange[0], myrange[1])
        ax.set_xscale("log")
        ax.set_yscale(yscale)

    for ax in [ax2, ax3]:
        ax.set_ylim(1e0, 3e5)
        ax.set_yticks([1e0, 1e2, 1e4])
        ax.set_yticklabels([r"$10^0$", r"$10^2$", r"$10^4$"])
        ax.set_ylabel("Mass Ratio")
    # ax2.set_ylabel(r"$\Sigma_{M_{\rm Au}} / \Sigma_{M_{\rm MW}} $")
    ax2.text(0.05, 0.95, "Milky Way", ha="left", va="top", transform=ax2.transAxes)
    # ax3.set_ylabel(r"$\Sigma_{M_{\rm Au}} / \Sigma_{M_{\rm M31}} $")
    ax3.text(0.05, 0.95, "Andromeda", ha="left", va="top", transform=ax3.transAxes)
    ax3.set_xlabel(r"$r_{\rm gal}$ [kpc]")

    pyplot.tight_layout()
    fig.subplots_adjust(hspace=0)

    return fig


###############################################################################
##                   Metallicity and Galactocentric radius                   ##
###############################################################################
def save_FeHRgc_logM_hist(run, MW_rvir, age_min=10.0,
        xbins=[1, 3, 8, 15, 30, 125], xbins_name=None,
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0], ybins_name=None,
        verbose=True, debug=False):
    print("  save_FeHRgc_logM_hist for {0} /w MW_rvir = {1}, age_min = {2}".format(
        run.name, MW_rvir, age_min))

    if xbins_name is None:
        xbins_name = str(xbins)
    if ybins_name is None:
        ybins_name = str(ybins)

    s, sf = run.load_snapshot(loadonlytype=range(6), verbose=False)
    if not hasattr(run, "insitu"):
        run.set_insitu_stars()
    s.insitu = run.insitu
    istars, iold, insitu, accreted, insitu_old, accreted_old, age_Gyr\
        = select_gc_candidates(s, sf, age_min=age_min)

    if MW_rvir > 10:
        print("\nScaling to MW size")
        Mvirs, rvirs = calculate_rvir(s, sf, cosmo="WMAP9", verbose=True, debug=False)
        Au_rvir = rvirs[2]  # tophat
        rvir_factor = MW_rvir/Au_rvir
        print("  SUBFIND\n    mean   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmm2"][0], 1e3 * sf.data["frm2"][0]))
        print("    crit   M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]".format(
            1e10 * sf.data["fmc2"][0], 1e3 * sf.data["frc2"][0]))
        print("    tophat M200, r200 = {0:.2e} MSun, {1:.2f} kpc [Planck XVI]\n".format(
            1e10 * sf.data["fmt2"][0], 1e3 * sf.data["frt2"][0]))
        print("  MW_rvir = {0:.2f} kpc [WMAP9]".format(MW_rvir))
        print("  Au_rvir = {0:.2f} kpc [WMAP9]".format(Au_rvir))
        print("  rvir_factor = {0:.2f}".format(rvir_factor))
    else:
        rvir_factor = 1.0

    for mask, mask_name in zip(
            [istars, iold, insitu, accreted, insitu_old, accreted_old],
            ["istars", "iold", "insitu", "accreted", "insitu_old", "accreted_old"]
        ):

        feabund = compute_iron_over_hydrogen_abundance(s, sf, mask)
        Mcount, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            1000*s.r()[mask]*rvir_factor, feabund, s.mass[mask]*1e10,
            bins=[xbins, ybins], statistic="count")
        Msum, Rgc_edges, FeH_edges, Mcnt = binned_statistic_2d(
            1000*s.r()[mask]*rvir_factor, feabund, s.mass[mask]*1e10,
            bins=[xbins, ybins], statistic="sum")

        if verbose:
            print("\nMask '{0}' has {1: 10d} particles".format(mask_name, len(mask)))

            # Sanity check: are we taking all data into account for the chosen binning?
            ismaller_than_xmin, = numpy.where(
                1e3*s.r()[mask]*rvir_factor < xbins[0] )
            print("  Rgc < {0: 8.2f} kpc has {1: 10d} particles.".format(
                xbins[0], len(ismaller_than_xmin) ))

            ibigger_than_xmax, = numpy.where(
                1e3*s.r()[mask]*rvir_factor > xbins[-1] )
            print("  Rgc > {0: 8.2f} kpc has {1: 10d} particles.".format(
                xbins[-1], len(ibigger_than_xmax) ))

            ismaller_than_ymin, = numpy.where(
                feabund < ybins[0] )
            print("  FeH < {0: 8.2f}     has {1: 10d} particles.".format(
                ybins[0], len(ismaller_than_ymin) ))

            ibigger_than_ymax, = numpy.where(
                feabund > ybins[-1] )
            print("  FeH > {0: 8.2f}     has {1: 10d} particles.".format(
                ybins[-1], len(ibigger_than_ymax) ))

            istars_in_range, = numpy.where(
                ( 1e3*s.r()[mask]*rvir_factor > xbins[0] )
                & ( 1e3*s.r()[mask]*rvir_factor < xbins[-1] )
                & ( feabund > ybins[0] )
                & ( feabund < ybins[-1] )
            )
            istars_out_of_range, = numpy.where(
                ( 1e3*s.r()[mask]*rvir_factor < xbins[0] )
                | ( 1e3*s.r()[mask]*rvir_factor > xbins[-1] )
                | ( feabund < ybins[0] )
                | ( feabund > ybins[-1] )
            )
            print("  " + "-"*34 + " +")
            print("  # stars out of range:  {0: 10d}".format(len(istars_out_of_range)))
            print("  # stars in     range:  {0: 10d}".format(len(istars_in_range)))
            print("  " + "-"*34 + " +")
            print("  # stars in      mask:  {0: 10d} [as a reminder ;-)]".format(len(mask)))
            print("{0: 6.2f} % of (number of) stars in mask taken into account\n".format(
                 100 * len(istars_in_range) / len(mask) ))

            print("  # stars in our 2D bins {0: 10d} [{1}]".format(len(Mcnt),
                "fed into 2d hist function - some may be out of range!"))
            print("  # stars in our 2D bins {0: 10d} [{1}]".format(
                int(Mcount.sum()), "sanity check - this is the sum of Mcount"))
            print("    Mass in mask in range{0:10.2e}".format( Msum.flatten().sum() ))
            print("    Total mass in mask   {0:10.2e}".format( 1e10 * s.mass[mask].sum() ))
            print("    Total mass in stars  {0:10.2e}".format( 1e10 * s.mass[istars].sum() ))
            print("{0: 6.2f} % of mass in  mask taken into account".format(
                 100 * Msum.flatten().sum() / (1e10*s.mass[mask].sum())) )
            print("{0: 6.2f} % of mass in stars taken into account\n".format(
                 100 * Msum.flatten().sum() / (1e10*s.mass[istars].sum())) )

        if debug:
            # Mcnt: this assigns to each element of `sample` an integer that
            #       represents the bin in which this observation falls.
            print("\nMsum\n", Msum)
            print("\nMcount\n", Mcount)
            print("\nRgc_edges\n", Rgc_edges)
            print("\nFeH_edges\n", FeH_edges)
            print("\nMcnt\n", Mcnt)

        numpy.savez("{0}/{1}-{2}_FeHRgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
            run.outdir, run.name, s.snapnr, mask_name, age_min, xbins_name, ybins_name),
            Mcount=Mcount, Msum=Msum, Rgc_edges=Rgc_edges, FeH_edges=FeH_edges,
            Mcnt=Mcnt, Ngc=len(mask))

    # data = np.load("{0}/{1}-{2}_FeHRgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
    #     run.outdir, run.name, s.snapnr, mask_name, age_min, xbins_name, ybins_name))
    # Msum = data["Msum"]; print(Msum)

    del s, sf



def show_bins(ax, xbins, ybins, c=None, ls=None):
    transx = matplotlib.transforms.blended_transform_factory(
        ax.transData, ax.transAxes)
    for x in xbins:
        ax.axvline(x, c="k" if c is None else c, ls="-" if ls is None else ls)
        # ax.text(x, 1.01, str(x), fontsize=16, ha="center", transform=transx)

    transy = matplotlib.transforms.blended_transform_factory(
        ax.transAxes, ax.transData)
    for y in ybins[:-1]:
        ax.axhline(y, c="k" if c is None else c, ls="-" if ls is None else ls)
        # ax.text(1.01, y, str(y), fontsize=16, va="center", transform=transy)


def plot_values(ax, Rgc_edges, FeH_edges, Msum, Mcount, verbose=True, skip=[]):
    xmid = (Rgc_edges[1:] + Rgc_edges[:-1]) / 2
    xmidlog = ( numpy.log10(Rgc_edges[1:]) + numpy.log10(Rgc_edges[:-1]) ) / 2
    ymid = (FeH_edges[1:] + FeH_edges[:-1]) / 2
    if verbose:
        print("{0:<10s}{1:<10s}{2:<10s}{3:<10s}{4:<10s}".format(
            "logx", "y", "value/1e6", "logvalue", "Ngcs"))
    for i, (logx, rows) in enumerate(zip(xmidlog, Msum)):
        for j, (y, value) in enumerate(zip(ymid, rows)):
            Ngcs = int(Mcount[i][j])
            with suppress_stdout():  # divide by zero encountered in log10
                logvalue = numpy.log10(value)
                logvalue = logvalue if numpy.isfinite(logvalue) else 0
            if verbose:
                print("{0:<10.1f}{1:<10.1f}{2:<10.1f}{3:<10.2f}{4:<10d}".format(
                    logx, y, value/1e6, logvalue, Ngcs))
            if (i,j) in skip:
                continue
            ax.text(10**logx, y, "{0:.2f} ({1})".format(logvalue, Ngcs),
                ha="center", va="center", fontsize=18)


def average_logM_FeHRgc_histogram_for_all_sims(auriga, sims, age_min=10.0,
        xbins=[1, 3, 8, 15, 30, 125], xbins_name=None,
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0], ybins_name=None,
        do_show_bins=True, do_plot_values=True, do_show_25bins=False, norm=None,
        MW_h96e10=None, M31_cr16=None, do_fit=False, verbose=True, debug=False,
        plot_values_skip=[]):

    if xbins_name is None:
        xbins_name = str(xbins)
    if ybins_name is None:
        ybins_name = str(ybins)

    all_masks = [
        "istars", "insitu", "accreted", "iold", "insitu_old", "accreted_old"
    ]

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates ({:.1f})".format(age_min),
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }

    # Allocate data - perhaps this should be array not list but w/e works, works
    Msum_per_bin = dict()
    Mcount_per_bin = dict()
    for mask_name in all_masks:
        if mask_name not in Msum_per_bin.keys():
            Msum_per_bin[mask_name] = \
                [[list() for a in range(len(xbins))] for b in range(len(xbins))]
            Mcount_per_bin[mask_name] = \
                [[list() for a in range(len(xbins))] for b in range(len(xbins))]

    # Gather data
    for level in sims.keys():
        for halo in sims[level]:
            run = auriga.getrun(level=level, halo=halo)
            for mask_name in all_masks:

                Au_data = numpy.load("{0}/{1}-{2}_FeHRgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
                run.outdir, run.name, run.nsnaps-1, mask_name, age_min, xbins_name, ybins_name))
                Mcount = Au_data["Mcount"]
                Msum = Au_data["Msum"]
                Rgc_edges = Au_data["Rgc_edges"]
                FeH_edges = Au_data["FeH_edges"]
                Mcnt = Au_data["Mcnt"]
                Ngc = Au_data["Ngc"]
                Au_label  = r"\begin{{tabular}}{{p{{6cm}}l}}{0}; & Ngc = {1}\end{{tabular}}"\
                    .format(run.name, Ngc)

                for col in range(len(xbins)-1):
                    for row in range(len(ybins)-1):
                        # print(col, row, numpy.log10(Msum[col][row]))
                        (Msum_per_bin[mask_name][col][row]).append(Msum[col][row])
                        (Mcount_per_bin[mask_name][col][row]).append(Mcount[col][row])

            if debug: break
        if debug: break

    for mask_name in ["iold", "insitu_old", "accreted_old"]:
        mworm31 = ""
        # Allocate 2D hist to pcolormesh
        Msum_mean = numpy.zeros(Msum.shape)
        Msum_std = numpy.zeros(Msum.shape)
        Msum_percentiles_16 = numpy.zeros(Msum.shape)
        Msum_percentiles_50 = numpy.zeros(Msum.shape)
        Msum_percentiles_84 = numpy.zeros(Msum.shape)
        Msum_sum = numpy.zeros(Msum.shape)

        Mcount_mean = numpy.zeros(Msum.shape)
        Mcount_std = numpy.zeros(Msum.shape)
        Mcount_percentiles_16 = numpy.zeros(Msum.shape)
        Mcount_percentiles_50 = numpy.zeros(Msum.shape)
        Mcount_percentiles_84 = numpy.zeros(Msum.shape)
        Mcount_sum = numpy.zeros(Msum.shape)

        if MW_h96e10 is not None:
            if verbose: print("Okidoki, Harris data given. Now we shall plot Delta")
            mworm31 = "MW"
            MW_GCS_mass, Mcount_h96e10, Msum_h96e10, Rgc_edges_h96e10, \
                FeH_edges_h96e10, Mcnt_h96e10 = bin_milkyway_data(MW_h96e10, xbins, ybins)
            norm = matplotlib.colors.Normalize(vmin=0.8, vmax=5.8)
            cmap = colorcet.cm["bgyw_r"]
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:,-1] = 0.7  # hack in an alpha=0.7
            my_cmap = matplotlib.colors.ListedColormap(my_cmap)

        if M31_cr16 is not None:
            if verbose: print("Okidoki, CR16 data given. Now we shall plot Delta")
            mworm31 = "M31"
            M31_GCS_mass, Mcount_cr16, Msum_cr16, Rgc_edges_cr16, \
                FeH_edges_cr16, Mcnt_cr16 = bin_andromeda_data(M31_cr16, xbins, ybins)
            norm = matplotlib.colors.Normalize(vmin=0.8, vmax=5.8)
            cmap = colorcet.cm["bgyw_r"]
            my_cmap = cmap(np.arange(cmap.N))
            my_cmap[:,-1] = 0.7  # hack in an alpha=0.7
            my_cmap = matplotlib.colors.ListedColormap(my_cmap)

        for col in range(len(xbins)-1):
            for row in range(len(ybins)-1):
                Msum_mean[col][row] = numpy.mean(Msum_per_bin[mask_name][col][row])
                Msum_std[col][row] = numpy.std(Msum_per_bin[mask_name][col][row])
                Msum_sum[col][row] = numpy.sum(Msum_per_bin[mask_name][col][row])
                Msum_percentile = numpy.percentile(Msum_per_bin[mask_name][col][row], [16, 50, 84])
                Msum_percentiles_16[col][row] = Msum_percentile[0]
                Msum_percentiles_50[col][row] = Msum_percentile[1]
                Msum_percentiles_84[col][row] = Msum_percentile[2]
                print("{0:<5d}{1:<5d}{2:<10.2e}{3:<10.2e}".format(
                    col, row, Msum_mean[col][row], Msum_percentiles_50[col][row]))

                Mcount_mean[col][row] = numpy.mean(Mcount_per_bin[mask_name][col][row])
                Mcount_std[col][row] = numpy.std(Mcount_per_bin[mask_name][col][row])
                Mcount_sum[col][row] = numpy.sum(Mcount_per_bin[mask_name][col][row])
                Mcount_percentile = numpy.percentile(Mcount_per_bin[mask_name][col][row], [16, 50, 84])
                Mcount_percentiles_16[col][row] = Mcount_percentile[0]
                Mcount_percentiles_50[col][row] = Mcount_percentile[1]
                Mcount_percentiles_84[col][row] = Mcount_percentile[2]
                if debug:
                    print("{0:<5d}{1:<5d}{2:<10.2e}{3:<10.2e}".format(
                        col, row, Mcount_mean[col][row], Mcount_percentiles_50[col][row]))

        numpy.savez("../out/Au-all_RgcFeH_HistogramMassWeighted_median-savez_{0}_{1}bins.npz".format(
            mask_name.replace("_", "-"),
            "25" if not xbins_name else str(len(xbins)*len(ybins)) ),
            Msum_mean=Msum_mean, Msum_std=Msum_std, Msum_sum=Msum_sum,
            Mcount_mean=Mcount_mean, Mcount_std=Mcount_std, Mcount_sum=Mcount_sum,
            Mcount_percentiles_16=Mcount_percentiles_16,
            Mcount_percentiles_50=Mcount_percentiles_50,
            Mcount_percentiles_84=Mcount_percentiles_84,
        )

        if MW_h96e10 is not None:
            if debug:
                print(Msum_percentiles_50)
                print(Msum_h96e10)
            fraction = numpy.log10(Msum_percentiles_50 / Msum_h96e10)

            # In case we want to cheat the three bins in upper right corner
            # fraction[Msum_h96e10 < 1] = numpy.log10(Msum_percentiles_50[Msum_h96e10 < 1])
            if debug: print("MW fraction:\n{0}\n".format(fraction))

        if M31_cr16 is not None:
            print("Msum_cr16\n", Msum_cr16)
            fraction = numpy.log10(Msum_percentiles_50 / Msum_cr16)
            # In case we want to cheat the three bins in upper right corner
            # fraction[Msum_cr16 < 1] = numpy.log10(Msum_percentiles_50[Msum_cr16 < 1])
            print("M31 fraction:\n{0}\n".format(fraction))

        # Median
        fig1, ax1 = pyplot.subplots(figsize=(12, 9))
        # with suppress_stdout():  # divide by zero encountered in log10
        if MW_h96e10 is None and M31_cr16 is None:
            if debug: print("Hank\n", numpy.log10(Msum_percentiles_50.T))
            cbar = ax1.pcolormesh(Rgc_edges, FeH_edges, numpy.log10(Msum_percentiles_50.T),
                cmap=colorcet.cm["bgy"], alpha=0.7) # , vmin=3, vmax=9.1)
            boundaries = None
            axcb = pyplot.colorbar(cbar, norm=norm, boundaries=boundaries,
                orientation="horizontal")
        else:
            print("PLOTTING FRACTION")
            cbar = ax1.pcolormesh(Rgc_edges, FeH_edges, fraction.T,
                norm=norm, cmap=my_cmap)
            cbar = pyplot.cm.ScalarMappable(cmap=my_cmap)  # cmap=cmap --> alpha = 1; cmap=my_cmap --> alpha 0.7
            cbar._A = []
            boundaries = numpy.linspace(0.8, 5.8, 256)
            axcb = pyplot.colorbar(cbar, norm=norm, boundaries=boundaries,
                orientation="horizontal")  #  ticks=[1, 2, 3, 4, 5]
        axcb.ax.set_xlabel(r"log$_{10}$( Median Auriga Mass [Msun] )")
        if MW_h96e10 is not None:
            axcb.ax.set_xlabel(r"log$_{10}(\Sigma$M$_{\rm Au} / \Sigma$M$_{\rm MW}$)")
            # axcb.ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
            # axcb.ax.set_xticklabels([r"${0:.1f}$".format(i) for i in range (0, 7)])
            axcb.set_clim(vmin=0, vmax=8)
            # axcb.ax.set_xticks(range(0, 7))
            # axcb.ax.set_xticklabels([r"${0:.1f}$".format(i) for i in range (0, 7)])
        if M31_cr16 is not None:
            axcb.ax.set_xlabel(r"log$_{10}(\Sigma$M$_{\rm Au} / \Sigma$M$_{\rm M31}$)")
            # axcb.ax.set_xticklabels([r"${0:.1f}$".format(i) for i in range (0, 7)])
        if do_plot_values:
            # NB, here we hack in the simulated values of the uppermost three bins
            # We do not colorcode to make the point that the values are not ratios,
            # but we do print the simulated value (but the observations have zero mass
            # in the three uppermost right bins)
            if MW_h96e10 is not None:
                fraction[Msum_h96e10 < 1] = numpy.log10(Msum_percentiles_50[Msum_h96e10 < 1])
            if M31_cr16 is not None:
                fraction[Msum_cr16 < 1] = numpy.log10(Msum_percentiles_50[Msum_cr16 < 1])
            plot_values(ax1, Rgc_edges, FeH_edges,
                Msum_percentiles_50 if MW_h96e10 is None and M31_cr16 is None else 10**fraction,
                Mcount_percentiles_50, verbose=verbose, skip=plot_values_skip)

        if do_fit:
            from scipy.optimize import curve_fit

            fitfunc = lambda x, a, b: a*x + b
            x = numpy.log10( (Rgc_edges[1:]+Rgc_edges[:-1])/2)
            y = (FeH_edges[1:]+FeH_edges[:-1])/2
            w = Msum_percentiles_50
            nz = w.nonzero()
            popt, pcov = curve_fit(fitfunc,
                x[nz[0]], y[nz[1]], sigma=1/w[nz])
            perr = numpy.sqrt(numpy.diag(pcov))

            print("Fitting ax+b [Fe/H]-Rgc, weighted by mass")
            fitlabel = "  a = {0:+6.2f} +/- {1:6.2f}\n  b = {2:+6.2f} +/- {3:6.2f}".format(
                popt[0], perr[0], popt[1], perr[1] )
            print(fitlabel)
            xvals = numpy.log10(numpy.linspace(1e-2, 1e4, 64))
            ax1.plot(10**xvals, fitfunc(xvals, *popt), c="r", lw=4, label=fitlabel)
            ax1.legend(loc="lower left", framealpha=1)

        # Std
        # fig2, ax2 = pyplot.subplots(figsize=(12, 9))
        # if debug: print("Debug is on, so std will be zero everywhere b/c only one simulation used")
        # with suppress_stdout():  # divide by zero encountered in log10
        #     cbar = ax2.pcolormesh(Rgc_edges, FeH_edges, numpy.log10(Msum_std.T),
        #         vmin=6.5, vmax=9.2)
        # axcb = pyplot.colorbar(cbar, orientation="horizontal")
        # axcb.ax.set_xlabel(r"log$_{10}$( Std Mass [Msun] )")
        # if do_plot_values:
        #     plot_values(ax2, Rgc_edges, FeH_edges, Msum_std,
        #         Mcount_std, verbose=verbose)

        # Both
        for ax in [ax1]:
            ax.set_xlim(xbins[0], xbins[-1])
            ax.set_ylim(ybins[0], ybins[-1])
            ax.set_xlabel(r"$r_{\rm gal}$ [kpc]")
            ax.set_ylabel("[Fe/H]")
            ax.set_xscale("log")

            if do_show_bins: show_bins(ax, xbins, ybins)
            if do_show_25bins:
                show_bins(ax, [1, 3, 8, 15, 30, 125],
                    [-2.5, -1.5, -1.0, -0.5, -0.3, 0])

        # Median
        fig1.suptitle("{0} {1} {2}".format(
            "Auriga", LABELS[mask_name],
            "vs {0}".format(mworm31) if mworm31 else ""
        ))
        # fig1.savefig("../out/Au-all_{0}_{1}_{2}bins_median.png".format(
        #     "RgcFeH_HistogramMassWeighted{0}".format("_"+mworm31 if mworm31 else ""),
        #     mask_name.replace("_", "-"),
        #      "25" if not xbins_name else str(len(xbins)*len(ybins))
        # ))

        # Std
        # fig2.suptitle("{0}: {1} {2}".format(
        #     "Auriga Std", labels[mask_name],
        #     "vs {0}".format(mworm31) if mworm31 else ""
        # ))
        # fig2.savefig("../out/Au-all_{0}_{1}_{2}bins_std.png".format(
        #     "RgcFeH_HistogramMassWeighted{0}".format("_"+mworm31 if mworm31 else ""),
        #     mask_name.replace("_", "-"),
        #     "25" if not xbins_name else str(len(xbins)*len(ybins))
        # ))

        # if debug: break
        return fig1


def average_logM_FeHRgc_histogram_for_all_sims_diverging(auriga, sims, age_min=10.0,
        xbins=[1, 3, 8, 15, 30, 125], xbins_name=None,
        ybins=[-2.5, -1.5, -1.0, -0.5, -0.3, 0], ybins_name=None,
        MW_h96e10=None, M31_cr16=None, cmap=colorcet.cm["coolwarm"],
        verbose=True, debug=False, plot_values_skip=[]):

    if xbins_name is None:
        xbins_name = str(xbins)
    if ybins_name is None:
        ybins_name = str(ybins)

    all_masks = [
        "istars", "insitu", "accreted", "iold", "insitu_old", "accreted_old"
    ]

    LABELS = {
        "istars": "All star particles", "iold": "GC candidates ({:.1f})".format(age_min),
        "accreted_old": "GC candidates: accreted", "insitu_old": "GC candidates: {\it in situ}"
    }

    # Allocate data - perhaps this should be array not list but w/e works, works
    Msum_per_bin = dict()
    Mcount_per_bin = dict()
    for mask_name in all_masks:
        if mask_name not in Msum_per_bin.keys():
            Msum_per_bin[mask_name] = \
                [[list() for a in range(len(xbins))] for b in range(len(xbins))]
            Mcount_per_bin[mask_name] = \
                [[list() for a in range(len(xbins))] for b in range(len(xbins))]

    # Gather data
    for level in sims.keys():
        for halo in sims[level]:
            run = auriga.getrun(level=level, halo=halo)
            for mask_name in all_masks:

                Au_data = numpy.load("{0}/{1}-{2}_FeHRgc_logM_{3}_{4:.1f}_hist_{5}_{6}.npz".format(
                run.outdir, run.name, run.nsnaps-1, mask_name, age_min, xbins_name, ybins_name))
                Mcount = Au_data["Mcount"]
                Msum = Au_data["Msum"]
                Rgc_edges = Au_data["Rgc_edges"]
                FeH_edges = Au_data["FeH_edges"]
                Mcnt = Au_data["Mcnt"]
                Ngc = Au_data["Ngc"]
                Au_label  = r"\begin{{tabular}}{{p{{6cm}}l}}{0}; & Ngc = {1}\end{{tabular}}"\
                    .format(run.name, Ngc)

                for col in range(len(xbins)-1):
                    for row in range(len(ybins)-1):
                        # print(col, row, numpy.log10(Msum[col][row]))
                        (Msum_per_bin[mask_name][col][row]).append(Msum[col][row])
                        (Mcount_per_bin[mask_name][col][row]).append(Mcount[col][row])

            if debug: break
        if debug: break

    for mask_name in ["iold"]:
        mworm31 = ""
        # Allocate 2D hist to pcolormesh
        Msum_mean = numpy.zeros(Msum.shape)
        Msum_std = numpy.zeros(Msum.shape)
        Msum_percentiles_16 = numpy.zeros(Msum.shape)
        Msum_percentiles_50 = numpy.zeros(Msum.shape)
        Msum_percentiles_84 = numpy.zeros(Msum.shape)
        Msum_sum = numpy.zeros(Msum.shape)

        Mcount_mean = numpy.zeros(Msum.shape)
        Mcount_std = numpy.zeros(Msum.shape)
        Mcount_percentiles_16 = numpy.zeros(Msum.shape)
        Mcount_percentiles_50 = numpy.zeros(Msum.shape)
        Mcount_percentiles_84 = numpy.zeros(Msum.shape)
        Mcount_sum = numpy.zeros(Msum.shape)

        if MW_h96e10 is not None:
            if verbose: print("Okidoki, Harris data given. Now we shall plot Delta")
            mworm31 = "MW"
            MW_GCS_mass, Mcount_h96e10, Msum_h96e10, Rgc_edges_h96e10, \
                FeH_edges_h96e10, Mcnt_h96e10 = bin_milkyway_data(MW_h96e10, xbins, ybins)

        if M31_cr16 is not None:
            if verbose: print("Okidoki, CR16 data given. Now we shall plot Delta")
            mworm31 = "M31"
            M31_GCS_mass, Mcount_cr16, Msum_cr16, Rgc_edges_cr16, \
                FeH_edges_cr16, Mcnt_cr16 = bin_andromeda_data(M31_cr16, xbins, ybins)

        for col in range(len(xbins)-1):
            for row in range(len(ybins)-1):
                Msum_mean[col][row] = numpy.mean(Msum_per_bin[mask_name][col][row])
                Msum_std[col][row] = numpy.std(Msum_per_bin[mask_name][col][row])
                Msum_sum[col][row] = numpy.sum(Msum_per_bin[mask_name][col][row])
                Msum_percentile = numpy.percentile(Msum_per_bin[mask_name][col][row], [16, 50, 84])
                Msum_percentiles_16[col][row] = Msum_percentile[0]
                Msum_percentiles_50[col][row] = Msum_percentile[1]
                Msum_percentiles_84[col][row] = Msum_percentile[2]
                if debug:
                    print("{0:<5d}{1:<5d}{2:<10.2e}{3:<10.2e}".format(
                        col, row, Msum_mean[col][row], Msum_percentiles_50[col][row]))

                Mcount_mean[col][row] = numpy.mean(Mcount_per_bin[mask_name][col][row])
                Mcount_std[col][row] = numpy.std(Mcount_per_bin[mask_name][col][row])
                Mcount_sum[col][row] = numpy.sum(Mcount_per_bin[mask_name][col][row])
                Mcount_percentile = numpy.percentile(Mcount_per_bin[mask_name][col][row], [16, 50, 84])
                Mcount_percentiles_16[col][row] = Mcount_percentile[0]
                Mcount_percentiles_50[col][row] = Mcount_percentile[1]
                Mcount_percentiles_84[col][row] = Mcount_percentile[2]
                if debug:
                    print("{0:<5d}{1:<5d}{2:<10.2e}{3:<10.2e}".format(
                        col, row, Mcount_mean[col][row], Mcount_percentiles_50[col][row]))

        numpy.savez("../out/Au-all_RgcFeH_HistogramMassWeighted_median-savez_{0}_{1}bins.npz".format(
            mask_name.replace("_", "-"),
            "25" if not xbins_name else str(len(xbins)*len(ybins)) ),
            Msum_mean=Msum_mean, Msum_std=Msum_std, Msum_sum=Msum_sum,
            Mcount_mean=Mcount_mean, Mcount_std=Mcount_std, Mcount_sum=Mcount_sum,
            Mcount_percentiles_16=Mcount_percentiles_16,
            Mcount_percentiles_50=Mcount_percentiles_50,
            Mcount_percentiles_84=Mcount_percentiles_84,
        )

        if MW_h96e10 is not None:
            if debug:
                print(Msum_percentiles_50)
                print(Msum_h96e10)
            fraction = numpy.log10(Msum_percentiles_50 / Msum_h96e10)

            # In case we want to cheat the three bins in upper right corner
            # fraction[Msum_h96e10 < 1] = numpy.log10(Msum_percentiles_50[Msum_h96e10 < 1])
            if debug: print("MW fraction:\n{0}\n".format(fraction))

        if M31_cr16 is not None:
            fraction = numpy.log10(Msum_percentiles_50 / Msum_cr16)
            # In case we want to cheat the three bins in upper right corner
            # fraction[Msum_cr16 < 1] = numpy.log10(Msum_percentiles_50[Msum_cr16 < 1])
            if debug: print("M31 fraction:\n{0}\n".format(fraction))


        # mean_fraction_of_finite = (numpy.mean(fraction[numpy.isfinite(fraction)]))
        median_fraction_of_finite = (numpy.median(fraction[numpy.isfinite(fraction)]))
        # print("Diverging cmap around mean: {0}".format(mean_fraction_of_finite))
        print("Diverging cmap around median: {0}".format(median_fraction_of_finite))
        print(fraction-median_fraction_of_finite)
        the_min = numpy.min(fraction[numpy.isfinite(fraction)] - median_fraction_of_finite)
        print(the_min)
        the_max = numpy.max(fraction[numpy.isfinite(fraction)] - median_fraction_of_finite)
        print(the_max)
        vlim = max(numpy.abs(the_min), numpy.abs(the_max))
        print(vlim)

        fig1, ax1 = pyplot.subplots(figsize=(12, 9))

        cbar = ax1.pcolormesh(
            Rgc_edges, FeH_edges, (fraction-median_fraction_of_finite).T,
            cmap=cmap, vmin=-vlim*1.1, vmax=vlim*1.1
        )
        axcb = pyplot.colorbar(cbar, orientation="horizontal")

        axcb.ax.set_xlabel(r"log$_{10}$( Median Auriga Mass [Msun] )")
        if MW_h96e10 is not None:
            axcb.ax.set_xlabel(r"log$_{10}(\Sigma$M$_{\rm Au} / \Sigma$M$_{\rm MW})$"+
                r" $-$ median thereof")
        if M31_cr16 is not None:
            axcb.ax.set_xlabel(r"log$_{10}(\Sigma$M$_{\rm Au} / \Sigma$M$_{\rm M31})$"+
                r" $-$ median thereof")


        # Show values
        # NB, here we hack in the simulated values of the uppermost three bins
        # We do not colorcode to make the point that the values are not ratios,
        # but we do print the simulated value (but the observations have zero mass
        # in the three uppermost right bins)
        if MW_h96e10 is not None:
            fraction[Msum_h96e10 < 1] = numpy.log10(Msum_percentiles_50[Msum_h96e10 < 1])
        if M31_cr16 is not None:
            fraction[Msum_cr16 < 1] = numpy.log10(Msum_percentiles_50[Msum_cr16 < 1])
        plot_values(ax1, Rgc_edges, FeH_edges,
            10**(fraction-median_fraction_of_finite),
            Mcount_percentiles_50, verbose=verbose, skip=plot_values_skip)

        ax1.set_xlim(xbins[0], xbins[-1])
        ax1.set_ylim(ybins[0], ybins[-1])
        ax1.set_xlabel(r"$r_{\rm gal}$ [kpc]")
        ax1.set_ylabel("[Fe/H]")
        ax1.set_xscale("log")

        show_bins(ax1, [1, 3, 8, 15, 30, 125],
            [-2.5, -1.5, -1.0, -0.5, -0.3, 0])

        # Median
        fig1.suptitle("{0} {1} {2}".format(
            "Auriga", LABELS[mask_name],
            "vs {0}".format(mworm31) if mworm31 else ""
        ))
        # fig1.savefig("../out/Au-all_{0}_{1}_{2}bins_median.png".format(
        #     "RgcFeH_HistogramMassWeighted{0}".format("_"+mworm31 if mworm31 else ""),
        #     mask_name.replace("_", "-"),
        #     "25" if not xbins_name else str(len(xbins)*len(ybins))
        # ))

        return fig1
