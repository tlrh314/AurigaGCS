import os
import time
import numpy
import warnings
import argparse
import matplotlib
from matplotlib import pyplot
from matplotlib import gridspec
from tlrh_util import suppress_stdout
from tlrh_util import print_progressbar

pyplot.style.use("tlrh")
from read_auriga_snapshot import AurigaOverview


def parallel_helper(args):
    return get_insitu_for_snapnr(*args)


def get_insitu_for_snapnr(run, snapnr, insitu_def, verbose=False, debug=False):
    previous_redshift = run.redshifts[snapnr-1]
    previous_time = run.times[snapnr-1]
    previous_a = 1/(previous_redshift+1)
    current_redshift = run.redshifts[snapnr]
    current_time = run.times[snapnr]
    current_a = 1/(current_redshift+1)
    if snapnr is 0:
        previous_redshift = 1000  # should be infinity
        # perhaps calculate lookbacktime for a=0 and see what happens?
        previous_time = current_time
        previous_a = 0

    if verbose:
        print("\nEating snapshot: {0}/{1}".format(snapnr, run.nsnaps-1))
        print("  previous redshift: {0}".format(previous_redshift))
        print("  current  redshift: {0}".format(current_redshift))
        print("  previous time: {0}".format(previous_time))
        print("  current  time: {0}".format(current_time))
        print("  previous a: {0}".format(previous_a))
        print("  current  a: {0}".format(current_a))

    with suppress_stdout():
        s, sf = run.load_snapshot(snapnr=snapnr, loadonlytype=[4], verbose=True)

    required = ["id", "pos", "age", "halo", "subhalo"]
    if not all(x in s.data.keys() for x in required):
        missing = [x for x in required if x not in s.data.keys()]
        if verbose:
            print("WARNING: s does not have all required info. Saving empty file.")
            print("required: {0}".format(required))
            print("missing:  {0}\n".format(missing))
        if not os.path.exists("../out/{0}/insitu".format(s.name)):
            os.mkdir("../out/{0}/insitu".format(s.name))
        numpy.savetxt("../out/{0}/insitu/{0}_insitu_{1}_{2}.txt".format(
            s.name, snapnr, insitu_def), numpy.array([], dtype=numpy.uint64),
            fmt='%d',)
        numpy.savetxt("../out/{0}/insitu/{0}_satellite_{1}.txt".format(
            s.name, snapnr), numpy.array([], dtype=numpy.uint64),
            fmt='%d',)
        # optional
        istars = numpy.array([], dtype=numpy.uint64)
        insitu = numpy.array([], dtype=numpy.uint64)
        isatellite = numpy.array([], dtype=numpy.uint64)
        igalaxy = numpy.array([], dtype=numpy.uint64)
        igalaxy_10percentr200 = numpy.array([], dtype=numpy.uint64)
    else:
        # s.age is actually scalefactor, so redshift z = (1/a -1)
        # born is the snapshot in which a particle first appeared
        s_redshifts =  (1/s.age - 1)

        istars, = numpy.where(
            (s.type == 4) & (s.age > 0.)
        )
        igalaxy, = numpy.where(
            (s.type == 4) & (s.age > 0.) & (s.halo == 0) & (s.subhalo == 0)
        )
        igalaxy_10percentr200, = numpy.where(
            (s.type == 4) & (s.age > 0.) & (s.halo == 0) & (s.subhalo == 0)
            & (s.r() < s.galrad)
        )

        # To find accreted stars we look for stars born outside of halo 0 subhalo 0
        # Stars in this subset that live within r200 at z=0 can then be classified
        # as accreted.
        isatellite, = numpy.where(
            (s.type == 4) & (s.age > 0.)
            & ( (s.halo != 0) |  ((s.halo == 0) & (s.subhalo != 0)) )
        )


        # TODO: perhaps not fixed def, but function of galradfrac?
        if insitu_def == "galrad":
            if verbose: print("\nInsitu requires r < galrad (=10% of virial radius)")
            insitu, = numpy.where(
                # Stars in the main halo + main subhalo,
                # within 10% of the virial radius (--> disk)
                (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)
                & (s.r() < s.galrad)
                # only look at stars that have formed in the timespan from last
                # snapshot until now the condition s.age < current_a should not
                # make a difference because no star can be older of course
                & (s.age > 0.) & (s.age > previous_a) & (s.age < current_a)
            )
        elif insitu_def == "virial":
            if verbose: print("\nInsitu requires r < 10*galrad (= virial radius)")
            insitu, = numpy.where(
                # Stars in the main halo + main subhalo,
                # within 10% of the virial radius (--> disk)
                (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)
                & (s.r() < 10*s.galrad)  # galrad is 10% of virial radius
                # only look at stars that have formed in the timespan from last
                # snapshot until now the condition s.age < current_a should not
                # make a difference because no star can be older of course
                & (s.age > 0.) & (s.age > previous_a) & (s.age < current_a)
            )
        elif insitu_def == "30kpc":
            if verbose: print("\nInsitu requires r < 30 kpc")
            insitu, = numpy.where(
                # Stars in the main halo + main subhalo,
                # within 10% of the virial radius (--> disk)
                (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)
                & (s.r() < float(30)/1000)  # galrad is 10% of virial radius
                # only look at stars that have formed in the timespan from last
                # snapshot until now the condition s.age < current_a should not
                # make a difference because no star can be older of course
                & (s.age > 0.) & (s.age > previous_a) & (s.age < current_a)
            )

        if not os.path.exists("../out/{0}/insitu".format(s.name)):
            os.mkdir("../out/{0}/insitu".format(s.name))
        numpy.savetxt("../out/{0}/insitu/{0}_insitu_{1}_{2}.txt".format(
            s.name, snapnr, insitu_def), s.id[insitu], fmt='%d')
        numpy.savetxt("../out/{0}/insitu/{0}_satellite_{1}.txt".format(
            s.name, snapnr), s.id[isatellite], fmt='%d')

    if verbose:
        print("  len(istars):     {0}".format(len(istars)))
        print("  len(insitu):     {0}".format(len(insitu)))
        print("  len(isatellite): {0}".format(len(isatellite)))
        print("  len(igalaxy):    {0}".format(len(igalaxy)))
        print("  len(igalaxy_10percentr200):  {0}".format(len(igalaxy_10percentr200)))
        print("")

    # At end of function we delete s and sf
    del s, sf


def get_insitu_vs_accreted_for_run(run, insitu_def, Nthreads=8, verbose=False):
    if not hasattr(run, "times"):
        run.get_snapshot_spacing()

    print("Generating insitu lists")
    for snapnr in range(run.nsnaps):
        print_progressbar(snapnr, run.nsnaps)
        get_insitu_for_snapnr(run, snapnr, insitu_def, verbose=verbose)
    print("Done generating insitu lists")

    return

    from multiprocessing import Pool
    pool = Pool(processes=Nthreads)
    job_args = [(snapnr,) for snapnr in range(run.nsnaps)]
    pool_result = pool.map(parallel_helper, job_args)


def plot_all_insitu_stars(run, snapnr, verbose=False, debug=False):
    if verbose:
        print("Eating snapshot: {0}/{1}".format(snapnr, run.nsnaps-1))

    if not hasattr(run, "insitu"):
        print("{0}/{1} does not have insitu attr".format(snapnr, run.nsnaps-1))
        return
    if not hasattr(run, "born_in_satellite"):
        print("{0}/{1} does not have born_in_satellite attr".format(snapnr, run.nsnaps-1))
        return

    s, sf = run.load_snapshot(snapnr=snapnr, loadonlytype=[4], verbose=True)

    required = ["id", "pos", "age", "halo", "subhalo"]
    if not all(x in s.data.keys() for x in required):
        missing = [x for x in required if x not in s.data.keys()]
        print("WARNING: s does not have all required info.")
        print("required: {0}".format(required))
        print("missing:  {0}\n".format(missing))
        return

    start = time.time()
    insitu, = numpy.where(s.id in run.insitu)
    print("Selected insitu stars in {0:.2f} s".format(time.time()-start))

    start = time.time()
    isatellite, = numpy.where(s.id in run.born_in_satellite)
    print("Selected born_in_satellite stars in {0:.2f} s".format(time.time()-start))


    istars, = numpy.where(
        (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)  # only main galaxy, not satellites
        & (s.age > 0.) & (s.r() > 0.)
    )
    insitu, = numpy.where(
        (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)  # only main galaxy, not satellites
        & (s.age > 0.) & (s.r() > 0.) & numpy.in1d(s.id, run.insitu)
    )
    accreted, = numpy.where(
        (s.type == 4) & (s.halo == 0) & (s.subhalo == 0)  # only main galaxy, not satellites in **current** snapshot
        & (s.age > 0.) & (s.r() > 0.) & numpy.in1d(s.id, run.born_in_satellite)
    )

    print("Insitu:   {0}".format(len(insitu)))
    print("Accreted: {0}".format(len(accreted)))
    print("Insitu/total: {0}".format( s.mass[insitu].sum() /
        (s.mass[accreted].sum() + s.mass[insitu].sum()) ) )
    print("istars: {0}".format(len(istars)))

    fig = pyplot.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 3)
    axxz = fig.add_subplot(gs.new_subplotspec((0, 0), colspan=2))
    axzy = fig.add_subplot(gs.new_subplotspec((1, 2), rowspan=2))
    axxy = fig.add_subplot(gs.new_subplotspec((1, 0), colspan=2, rowspan=2),
                          sharex=axxz, sharey=axzy)
    axt = fig.add_subplot(gs.new_subplotspec((0, 2)))
    axt.axis("off")
    gs.update(wspace=0, hspace=0)

    # Accreted
    (x, y, z) = s.pos[accreted,2], s.pos[accreted,1], s.pos[accreted,0]
    axxz.plot(1000*x, 1000*z, "rD", mec="red", ms=0.1, rasterized=True, alpha=0.7)
    axzy.plot(1000*z, 1000*y, "rD", mec="red", ms=0.1, rasterized=True, alpha=0.7)
    axxy.plot(1000*x, 1000*y, "rD", mec="red", ms=0.1, rasterized=True, alpha=0.7)

    # Insitu
    (x, y, z) = s.pos[insitu,2], s.pos[insitu,1], s.pos[insitu,0]
    axxz.plot(1000*x, 1000*z, "gD", mec="green", ms=1, rasterized=True, alpha=0.7)
    axzy.plot(1000*z, 1000*y, "gD", mec="green", ms=1, rasterized=True, alpha=0.7)
    axxy.plot(1000*x, 1000*y, "gD", mec="green", ms=1, rasterized=True, alpha=0.7)

    phi = numpy.arange(0, 2*numpy.pi, 0.01)
    radfrac = 1
    axxz.plot(radfrac*s.galrad*1e3*numpy.cos(phi), radfrac*s.galrad*1e3*numpy.sin(phi), c="k", lw=4)
    axzy.plot(radfrac*s.galrad*1e3*numpy.cos(phi), radfrac*s.galrad*1e3*numpy.sin(phi), c="k", lw=4)
    axxy.plot(radfrac*s.galrad*1e3*numpy.cos(phi), radfrac*s.galrad*1e3*numpy.sin(phi), c="k", lw=4)

    xlim = ylim = 3*1000*radfrac*s.galrad
    axxz.set_xlim([-xlim, xlim])
    axxz.set_ylim([-ylim/2, ylim/2])
    axzy.set_xlim([-xlim/2, ylim/2])
    axzy.set_ylim([-ylim, ylim])
    axxy.set_xlim([-xlim, xlim])
    axxy.set_ylim([-ylim, ylim])

    axxz.set_ylabel("z [kpc]")
    axzy.set_xlabel("z [kpc]")
    axxy.set_xlabel("x [kpc]")
    axxy.set_ylabel("y [kpc]")

    labels = [s.name, "snapnr = {0}".format(snapnr)]
    for i, label in enumerate(labels):
        axt.text(0.5, 0.9-0.1*i, label, fontsize=16, ha="center", va="center", transform=axt.transAxes)

    pyplot.savefig("../out/{0}/insitu/{0}_ins_{1:03d}.png".format(run.name, snapnr))
    pyplot.close()

    # At end of function we delete s and sf
    del s, sf


def set_insitu_for_run(run, insitu_def, verbose=False, regenerate=False):
    start = time.time()
    binary_file = "../out/{0}/insitu/{0}_insitu_{1}".format(run.name, insitu_def)
    binary_file_satellite = "../out/{0}/insitu/{0}_satellite".format(run.name)
    if os.path.exists(binary_file) and os.path.exists(binary_file_satellite) \
            and not regenerate:
        run.insitu = numpy.fromfile(binary_file, dtype="uint64")
        if verbose:
            print("\nRead insitu stars (binary) in {0:.2f} s".format(time.time()-start))
            print("Found {0} insitu stars".format(len(run.insitu)))

        run.born_in_satellite= numpy.fromfile(binary_file_satellite, dtype="uint64")
        if verbose:
            print("\nRead born_in_satellite stars (binary) in {0:.2f} s".format(time.time()-start))
            print("Found {0} born_in_satellite stars".format(len(run.born_in_satellite)))
    else:
        # Save insitu list for each snapshot
        if verbose:
            print("Strap in, generating insitu file. This will take a while...")
        get_insitu_vs_accreted_for_run(run, insitu_def, verbose=verbose)
        if verbose:
            print("Generated insitu lists in {0:.2f} s".format(time.time()-start))

        # Read the insitu lists and combine
        insitu = []
        isatellite = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for snapnr in range(run.nsnaps):
                insitu.extend(
                    numpy.loadtxt("../out/{0}/insitu/{0}_insitu_{1}_{2}.txt".format(
                        run.name, snapnr, insitu_def),
                        dtype="uint64", ndmin=1)
                )
                isatellite.extend(
                    numpy.loadtxt("../out/{0}/insitu/{0}_satellite_{1}.txt".format(
                        run.name, snapnr),
                        dtype="uint64", ndmin=1)
                )

        # print(len(insitu))
        # print(len(numpy.unique(numpy.array(insitu))))
        run.insitu = numpy.array(insitu).astype('uint64')
        run.insitu.tofile(binary_file)
        run.born_in_satellite = numpy.array(isatellite).astype('uint64')
        run.born_in_satellite.tofile(binary_file_satellite)
        if verbose: print("Read insitu stars in {0:.2f} s".format(time.time()-start))

        if verbose: print("Cleaning up...")
        for snapnr in range(run.nsnaps):
            fname = "../out/{0}/insitu/{0}_insitu_{1}_{2}.txt".format(
                run.name, snapnr, insitu_def)
            if os.path.exists(fname):
                os.remove(fname)
                if verbose: print("  removed: {0}".format(fname))
            fname = "../out/{0}/insitu/{0}_satellite_{1}.txt".format(
                run.name, snapnr)
            if os.path.exists(fname):
                os.remove(fname)
                if verbose: print("  removed: {0}".format(fname))

        if verbose: print("... done cleaning up")


def new_argument_parser():
    args = argparse.ArgumentParser(
        description="Generate InSitu lists for Auriga Simulations")
    args.add_argument("-i", "--halo", dest="haloes",
        help="Halo Number", nargs="+", type=int,
        default=[24], choices=range(1, 31))
    args.add_argument("-l", "--level", dest="level",
        help="Simulation Level", nargs="+", type=int,
        default=[4], choices=[2, 3, 4, 5, 6])
    args.add_argument("-d", "--def", dest="insitu_def",
        help="Definition of insitu", type=str,
        default="virial", choices=["galrad", "virial", "30kpc"])
    args.add_argument("-r", "--regenerate", dest="regenerate", action="store_true",
        help="Toggle flag to force-regenerate insitu file", default=False)
    args.add_argument("-v", "--verbose", dest="verbose", action="store_true",
        help="Toggle verbose flag", default=False)
    args.add_argument("-p", "--plot", dest="plot", action="store_true",
        help="Toggle plot flag", default=False)
    args.add_argument("-a", "--all", dest="regenerate_all", action="store_true",
        help="Toggle flag to regenerate all levels/haloes/definitions", default=False)

    return args


if __name__ == "__main__":
    print(">>> Parsing Arguments")
    args, unknown = new_argument_parser().parse_known_args()
    for k, v in args._get_kwargs():
        print("  {0:<15}: {1}".format(k, v))

    auriga = AurigaOverview()

    if args.regenerate_all:
        allsims = {
            5: [6, 9, 16, 24],
            4: range(1, 31),
            3: [6, 16, 21, 23, 24, 27],
        }
        for insitu_def in ["galrad", "virial", "30kpc"]:
            for level in allsims.keys():
                for halo in allsims[level]:
                    run = auriga.getrun(level=level, halo=halo)
                    if not hasattr(run, "times"):
                        run.get_snapshot_spacing()
                    set_insitu_for_run(run, insitu_def,
                        verbose=args.verbose, regenerate=True)
        sys.exit(0)

    for level in args.level:
        for halo in args.haloes:
            run = auriga.getrun(level, halo)
            if not hasattr(run, "times"):
                run.get_snapshot_spacing()

            # Inspect the method
            # get_insitu_for_snapnr(run, 100, angs.insitu_def, verbose=True, debug=True)
            set_insitu_for_run(run, args.insitu_def, verbose=args.verbose,
                regenerate=args.regenerate)

            if args.plot:
                run.set_insitu_stars()
                start = time.time()
                with warnings.catch_warnings():
                    # warnings.simplefilter("ignore")
                    for snapnr in range(run.nsnaps):
                        plot_all_insitu_stars(run, snapnr)
