import numpy
import scipy
from matplotlib import pyplot


def fit_to_data(ax, x, y, yerr, fitfunc, x0, xmid=0, ymid=0, x_ana=None):
    # Inspired by https://stackoverflow.com/a/39729036
    from scipy.optimize import curve_fit

    # curve_fit eats slightly different input than leastsq.
    # fitfunc is leastsq style; func is curve_fit style
    func = lambda x, a, b: fitfunc([a, b], (x-xmid))

    # ax.axvline(xmid, ls=":", c="r")
    # ax.axhline(ymid, ls=":", c="r")

    popt, pcov = curve_fit(func, x, y, sigma=yerr, absolute_sigma=True)
    perr = numpy.sqrt(numpy.diagonal(pcov))

    # Initial guess
    if x_ana is None: x_ana = numpy.linspace(numpy.min(x), numpy.max(x), 128)
    label = "\n".join([r"$p_i$[{}] = {: .3f}".format(i, p)
        for i, p in enumerate(x0)])
    ax.text(0.02, 0.98, label, ha="left", va="top", transform=ax.transAxes)
    pyplot.plot(x_ana, func(x_ana, *x0), c="k", lw=2, ls=":")

    # Best-fit model
    label = "\n".join([r"$p_f$[{}] = {: .3f} +/- {: .3f}".format(i, p, pe)
        for i, (p, pe) in enumerate(zip(popt, perr))])
    ax.text(0.98, 0.98, label, ha="right", va="top", transform=ax.transAxes)
    pyplot.plot(x_ana, func(x_ana, *popt), c="k", lw=4)

    # Confidence interval
    bound_upper = func(x_ana, *(popt + perr))
    bound_lower = func(x_ana, *(popt - perr))
    ax.fill_between(x_ana, bound_lower, bound_upper, color="yellow", alpha=0.35)

    return popt, perr
