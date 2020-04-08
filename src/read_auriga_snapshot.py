import re
import os
import sys
import glob
import time
import numpy
import warnings
from scipy import interpolate
from matplotlib import pyplot
from tlrh_util import suppress_stdout
from collections import OrderedDict

# from areposnap.cosmological_factors import CosmologicalFactors
from convert_auriga_units import compute_iron_over_hydrogen_abundance


class AurigaRun(object):
    def __init__(self, rundir, level, halo, verbose=False):
        self.rundir = rundir
        self.level = level
        self.halo = halo
        self.name = "Au{0}-{1}".format(level, halo)
        self.snapshots = list()
        self.find_all_snapshots(verbose=verbose)
        self.highfreqstars = list()
        self.find_all_highfreqstars(verbose=verbose)
        self.insitu_def = "virial"

        self.outdir = "../out/{0}".format(self.name)
        if not os.path.isdir(self.outdir):
            os.mkdir(self.outdir)
            print("Created: {0}".format(self.outdir))

        try:
            haloint = int(halo)
            # self.set_insitu_stars(verbose=verbose)
            if verbose: print("{0}: run.insitu is available".format(self.name))
        except ValueError:
            if verbose: print("{0}: run.insitu is not available".format(self.name))
        except Exception as e:
            if "Neither" in str(e) and "nor" in str(e) and "exists" in str(e):
                if verbose: print("{0}: run.insitu is not available".format(self.name))
            else:
                raise

    def __str__(self):
        return "{0:<20}{1:<60s}{2}".format("Au{0}-{1}".format(self.level, self.halo),
            self.rundir, self.nsnaps)

    def find_all_snapshots(self, verbose=False):
        self.snapshots = sorted(
            [snap for snap in glob.glob(self.rundir + "/output/snap*")
                    if "robs_failure" not in snap and "highfreqstars" not in snap],
            key=lambda s: [ int(c) for c in re.split('(\d+)', s) if c.isdigit()]
        )
        if verbose:
            print("INFO in find_all_snapshots")
            [print("  {0}".format(snapshot)) for snapshot in self.snapshots]
        self.nsnaps = len(self.snapshots)
        if self.nsnaps == 65:
            print("WARNING in find_all_snapshots: {0} has nsnaps = 65, force-setting it to 64 ...".format(
                self.name))
            self.nsnaps = 64  # because we don't believe that, aye

    def find_all_highfreqstars(self, verbose=False):
        self.highfreqstars = sorted(
            [snap for snap in glob.glob(self.rundir + "/output/snap*")
                    if "highfreqstars" in snap],
            key=lambda s: [ int(c) for c in re.split('(\d+)', s) if c.isdigit()]
        )
        self.nhighfreqstars = len(self.highfreqstars)
        if verbose:
            print("Found {0} highfreqstars".format(self.nhighfreqstars))

    def load_header(self, snapnr=None, redshift=None, tlookback=None, verbose=True):
        # Give snapnr to load a certain snapshot. If not given, use z=0
        if snapnr is None: snapnr = self.nsnaps-1

        # If redshift is given, find the snapshot closest to this redshift
        if redshift is not None and type(redshift) is float:
            if not hasattr(self, "redshifts"): self.get_snapshot_spacing(verbose=verbose)
            snapnr = (numpy.abs(self.redshifts - redshift)).argmin()
            print("Loading snapshot closest to z = {0}".format(redshift))
            print("The closest snapshot is {0} (z = {1:.2f})".format(snapnr, self.redshifts[snapnr]))

        # If tlookback given, find the snapshot closest ot this tlookback
        if tlookback is not None and type(tlookback) is float or type(tlookback) is int:
            if not hasattr(self, "times"): self.get_snapshot_spacing(verbose=verbose)
            snapnr = (numpy.abs(self.times - tlookback)).argmin()
            print("Loading snapshot closest to tlookback = {0}".format(tlookback))
            print("The closest snapshot is {0} (tlookback = {1:.2f})".format(snapnr, self.times[snapnr]))

        from areposnap.gadget_subfind import load_subfind
        from areposnap.gadget import gadget_readsnap
        sf = load_subfind(snapnr, dir=self.rundir+"/output/")
        s = gadget_readsnap(snapnr, snappath=self.rundir+"/output/",
            subfind=sf, onlyHeader=True)

        # Sneak some more info into the s instance
        s.halo_number = self.halo
        s.level = self.level
        s.snapnr = snapnr
        s.name = self.name

        if verbose:
            print("\n{0}".format(s.name))
            print("redshift: {0}".format(s.redshift))
            print("time    : {0}".format(s.time))
            print("center  : {0}\n".format(s.center))

        return s, sf

    def load_snapshot(self, snapnr=None, redshift=None, tlookback=None,
            loadonlytype=[4], rotate=True, verbose=True):

        # Give snapnr to load a certain snapshot. If not given, use z=0
        if snapnr is None: snapnr = self.nsnaps-1

        # If redshift is given, find the snapshot closest to this redshift
        if redshift is not None and type(redshift) is float:
            if not hasattr(self, "redshifts"): self.get_snapshot_spacing(verbose=verbose)
            snapnr = (numpy.abs(self.redshifts - redshift)).argmin()
            print("Loading snapshot closest to z = {0}".format(redshift))
            print("The closest snapshot is {0} (z = {1:.2f})".format(snapnr, self.redshifts[snapnr]))

        # If tlookback given, find the snapshot closest ot this tlookback
        if tlookback is not None and type(tlookback) is float or type(tlookback) is int:
            if not hasattr(self, "times"): self.get_snapshot_spacing(verbose=verbose)
            snapnr = (numpy.abs(self.times - tlookback)).argmin()
            print("Loading snapshot closest to tlookback = {0}".format(tlookback))
            print("The closest snapshot is {0} (tlookback = {1:.2f})".format(snapnr, self.times[snapnr]))

        if verbose:
            return eat_snap_and_fof(self.level, self.halo, snapnr, verbose=verbose, run=self,
                snappath=self.rundir+"/output/", loadonlytype=loadonlytype, rotate=rotate)
        else:
            with suppress_stdout():
                return eat_snap_and_fof(self.level, self.halo, snapnr, verbose=verbose, run=self,
                    snappath=self.rundir+"/output/", loadonlytype=loadonlytype)

    def get_total_stellar_mass_versus_time(self, verbose=False):
        if verbose: print("\nRunning get_total_stellar_mass_versus_time")

        if not hasattr(self, "nsnaps"):
            self.find_all_snapshots(verbose=verbose)

        binary_file = "../out/{0}/{0}_total_stellar_mass".format(self.name)
        if os.path.exists(binary_file):
            self.total_stellar_mass = numpy.fromfile(binary_file)
        else:
            self.total_stellar_mass = numpy.zeros(self.nsnaps)
            # TODO move insitu_fraction from RenaudOgertGieles2017 to here?
            # self.insitu_fraction = numpy.zeros(self.nsnaps)
            for snapnr in range(self.nsnaps):
                if verbose: print("  Eating snapshot: {0}/{1}".format(
                    snapnr, self.nsnaps-1))
                s, sf = self.load_snapshot(snapnr=snapnr,
                    loadonlytype=[4], verbose=True)

                required = ["pos", "halo", "subhalo", "mass"]
                if not all(x in s.data.keys() for x in required):
                    missing = [x for x in required if x not in s.data.keys()]
                    if not verbose: print("  Eating snapshot: {0}/{1}".format(
                        snapnr, self.nsnaps-1))
                    print("    WARNING: not all required data is available!")
                    print("    required: {0}".format(required))
                    print("    missing:  {0}\n".format(missing))
                    continue

                with suppress_stdout():
                    sr = s.r()
                stars_in_main_galaxy, = numpy.where(
                    (s.halo == 0) & (s.subhalo == 0) & (s.type == 4) & (s.age > 0.)
                )

                self.total_stellar_mass[snapnr] = s.mass[stars_in_main_galaxy].sum()*1e10
                if verbose: print("    Success!\n")

            self.total_stellar_mass.tofile(binary_file)
            del s, sf

    def get_snapshot_spacing(self, verbose=False):
        if not hasattr(self, "nsnaps"):
            self.find_all_snapshots(verbose=verbose)
        from areposnap.gadget import gadget_readsnap
        times = numpy.zeros(self.nsnaps)
        redshifts = numpy.zeros(self.nsnaps)
        for i, snapnr in enumerate(range(self.nsnaps)):
            # CAUTION, reruns seem to complain "couldnt find units in file" for each snapshot :o
            with suppress_stdout():
                s = gadget_readsnap(snapnr, snappath=self.rundir+"/output/", onlyHeader=True)
            s.cosmology_init()
            redshifts[i] = s.redshift
            times[i] = s.cosmology_get_lookback_time_from_a( s.time.astype('float64'), is_flat=True )

            if verbose:
                print("snapnr = {0:03d},\tz = {1:.5f},\ta = {2:.5f},\ttime = {3:.2f} Gyr".format(
                    snapnr, s.redshift, s.time, times[i]))

        self.cosmo = s.cosmo
        del s
        # s.time is actually scalefactor a which can be computed as a = 1/(1+z)
        self.times, self.redshifts = times, redshifts

    def get_snapshot_spacing_highfreqstars(self, verbose=False, from_snapshots=False):
        if not hasattr(self, "highfreqstars"):
            self.find_all_highfreqstars(verbose=verbose)
        times = numpy.zeros(self.nhighfreqstars)
        redshifts = numpy.zeros(self.nhighfreqstars)

        output_list = self.rundir + "/output_list_high_freq_stars.txt"
        if os.path.exists(output_list) and os.path.isfile(output_list) and not from_snapshots:
            a = numpy.zeros(self.nhighfreqstars)
            with open(output_list) as w:
                for i, line in enumerate(w.readlines()):
                    a[i] = line  # a = 1/(1+z)
                    redshifts[i] = 1/a[i] - 1
                    times[i] = self.cosmology_get_lookback_time_from_a( a[i].astype('float64'), is_flat=True )
            self.times_highfreqstars, self.redshifts_highfreqstars = times, redshifts
            return

        from tlrh_util import print_progressbar
        from areposnap.gadget import gadget_readsnap
        print("Setting redshifts for highfreqstars")
        for i, snapnr in enumerate(range(self.nhighfreqstars)):
            print_progressbar(snapnr, self.nhighfreqstars, whitespace="  ")
            # CAUTION, reruns seem to complain "couldnt find units in file" for each snapshot :o
            with suppress_stdout():
                # What's up with 2763?? O_o ... 000, 001, ..., 2761, 2762, 6764
                s = gadget_readsnap(snapnr if snapnr != 2763 else 2764,
                    snappath=self.rundir+"/output/",
                    snapbase="snapshot_highfreqstars_",
                    snapdirbase="snapdir_highfreqstars_",
                    onlyHeader=True
                )
            s.cosmology_init()
            redshifts[i] = s.redshift
            times[i] = s.cosmology_get_lookback_time_from_a( s.time.astype('float64'), is_flat=True )

            if verbose:
                print("snapnr = {0:03d},\tz = {1:.5f},\ta = {2:.5f},\ttime = {3:.2f} Gyr".format(
                    snapnr, s.redshift, s.time, times[i]))

        self.cosmo = s.cosmo
        del s
        # s.time is actually scalefactor a which can be computed as a = 1/(1+z)
        self.times_highfreqstars, self.redshifts_highfreqstars = times, redshifts

    def cosmology_get_lookback_time_from_a( self, a, is_flat=True):
        if not hasattr(self, "cosmo"):
            from areposnap.gadget import gadget_readsnap
            s = gadget_readsnap(127 if self.level == 4 else 63,
                snappath=self.rundir+"/output/", onlyHeader=True)
            s.cosmology_init()
            self.cosmo = s.cosmo
        return self.cosmo.LookbackTime_a_in_Gyr( a, is_flat=is_flat )

    def get_redshift_from_lookback(self, time, debug=False):
        zmin, zmax, epsilon, i = 0, 1337, 0.0001, 1
        while numpy.abs(zmin-zmax) > epsilon:
            i += 1
            zi = (zmin + zmax) / 2
            t = self.cosmology_get_lookback_time_from_a( 1/(1+zi), is_flat=True )
            if debug: print("{0:04d} {1:10.5f} {2:10.5f} {3:10.5f} {4:10.5f}".format(
                i, numpy.abs(zmin-zmax), zi, time, t))

            if time > t:
                zmin = zi
            else:
                zmax = zi
            if i > 42: break  # failsafe
        return zi

    def plot_snapshot_spacing_time(self):
        if not hasattr(self, "times"):
            self.get_snapshot_spacing()
        fig, axt = pyplot.subplots(figsize=(12, 3))
        for t in self.times: axt.axvline(t, c="k", ls="solid")
        axt.set_xlim(0, 14)
        axt.set_ylim(0, 1)
        axt.set_xticks(numpy.arange(0, 16, 2))
        axt.set_yticks([], [])
        axt.set_xlabel("Time [Gyr]")

        redshift_labels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 6.0, 13.0]
        times_at_redshifts = [self.times[(numpy.abs(self.redshifts - i)).argmin()] for i in redshift_labels]
        axb = axt.twiny()
        axb.set_xlabel("Redshift")
        axb.set_xticks(times_at_redshifts)
        axb.set_xticklabels(["{0:.1f}".format(i) for i in redshift_labels])
        axb.set_xlim(0, 14)
        axb.set_yticks([], [])

        pyplot.tight_layout()
        pyplot.savefig("{0}/{1}_snapshot_spacing_time.pdf".format(self.outdir, self.name))
        pyplot.show()

    def plot_snapshot_spacing_redshift(self):
        if not hasattr(self, "times"):
            self.get_snapshot_spacing()

        fig, axb = pyplot.subplots(figsize=(12, 3))
        for z in self.redshifts: axb.axvline(z, c="k", ls="solid")
        axb.set_xscale("log")
        axb.set_xlim(50, 0.01)
        axb.set_ylim(0, 1)
        redshift_labels = [50, 30, 10, 3, 1, 0.5, 0.3, 0.1, 0.03, 0.01]
        axb.set_xticks(redshift_labels)
        axb.set_xticklabels(["{0:.0f}".format(i) if i > 0.99 else "{0:.1f}".format(i)
            if i > 0.099 else "{0:.2f}".format(i) for i in redshift_labels])
        axb.set_yticks([], [])
        axb.set_xlabel("Redshift")

        time_labels = [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0.5]
        redshift_at_times = [self.redshifts[(numpy.abs(self.times - i)).argmin()] for i in time_labels]
        axt = axb.twiny()
        axt.set_xscale("log")
        axt.set_xlabel("Time [Gyr]")
        axt.set_xticks(redshift_at_times)
        axt.set_xticklabels(["{0:.1f}".format(i) for i in time_labels])
        axt.set_xlim(50, 0.01)

        pyplot.tight_layout()
        pyplot.savefig("{0}/{1}_snapshot_spacing_redshift.pdf".format(self.outdir, self.name))
        pyplot.show()

    def plot_time_between_snapshots(self, nbins=32, myrange=None, verbose=False, highfreqstars=False):
        if not highfreqstars:
            if not hasattr(self, "times"):
                self.get_snapshot_spacing()
            times = self.times
            snapshots = self.snapshots
        else:
            if not hasattr(self, "times_highfreqstars"):
                self.get_snapshot_spacing_highfreqstars()
            times = self.times_highfreqstars
            snapshots = self.highfreqstars

        fig, ax = pyplot.subplots(figsize=(12, 9))

        counts, edges = numpy.histogram(
            1000*(times[:-1] - times[1:]), bins=nbins, range=myrange if myrange else None,
        )
        ax.plot((edges[1:]+edges[:-1])/2, counts, drawstyle="steps-mid", c="k", lw=2)

        mean = 1000*numpy.mean(times[:-1] - times[1:])
        ax.axvline(mean, c="k", lw=2, ls=":")
        import matplotlib
        trans = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.text(0.99*mean, 0.03, r"$\mu$ = {0:.2f}".format(mean), ha="right", va="bottom", transform=trans)
        ax.text(1.01*mean, 0.03, "Myr", ha="left", va="bottom", transform=trans)

        ax.text(0.5, 1.01, "Time between snapshots", ha="center", va="bottom", transform=ax.transAxes)
        ax.text(0.03, 0.97, "{0}".format(self.name), ha="left", va="top", transform=ax.transAxes)
        ax.text(0.97, 0.97, "Nsnaps = {0}".format(len(snapshots)), ha="right", va="top", transform=ax.transAxes)
        ax.set_xlabel(r"$\Delta$T [Myr]")
        ax.set_ylabel("Nr. of Snapshots")

        pyplot.tight_layout()
        pyplot.savefig("{0}/{1}_time_between_snapshots{2}.pdf".format(
            self.outdir, self.name, "" if not highfreqstars else "_highfreqstars"))
        pyplot.show()

        if verbose:
            print("{0:<5s}{1:<12s}{2:<12s}{3:<12s}".format(
                "i", "start (Gyr)", "end (Gyr)", "delta T (Myr)"))
            for i, t in enumerate(self.times[:-1]):
                print("{0:<5d}{1:<12.2f}{2:<12.2f}{3:<12.2f}".format(
                    i, times[i], times[i+1], 1000*(times[i] - times[i+1])
                ))

    def set_insitu_stars(self, verbose=False, regenerate=False):
        from insitu import set_insitu_for_run
        set_insitu_for_run(self, self.insitu_def, verbose=verbose,
            regenerate=regenerate)


    def calculate_masses(self, snapnr=None, gc_max_age=10, gc_max_radius=250, print_header=True):
        with suppress_stdout():
            # snapnr=None --> redshift zero
            s, sf = self.load_snapshot(snapnr=snapnr, loadonlytype=range(6))

        M200 = sf.data["fmc2"][0]        # 1e10 Msun
        r200 = sf.data["frc2"][0] * 1e3  # kpc
        istars, = numpy.where(  # stars within 10% of R200
            (s.type == 4) & (s.age > 0.) & (s.r() > 0.) & (s.r() < 0.1*r200/1e3)
            & (s.halo == 0) & (s.subhalo == 0)
        )
        Mstars_tenpercent_r200 = s.mass[istars].sum()

        # Gas fraction within 0.1*R200
        igas, = numpy.where(  # gas within 10% of R200
            (s.type == 0) & (s.r() > 0.) & (s.r() < 0.1*r200/1e3)
            & (s.halo == 0) & (s.subhalo == 0)
        )
        Mgas_tenpercent_r200 = s.mass[igas].sum()
        fgas = Mgas_tenpercent_r200 / (Mgas_tenpercent_r200+Mstars_tenpercent_r200)

        from investigate_auriga_gcs import GlobularClusterSystem
        with suppress_stdout():
            GCS = GlobularClusterSystem(None, self.level, self.halo,
                radius=gc_max_radius, age=gc_max_age)

            GCS.set_FeH_allstars(s, sf)
            GCS.set_age_allstars(s, sf)
            GCS.set_FeH_oldstars(s, sf)

        Mgcs_red = s.mass[GCS.oldstars][GCS.old_red].sum()
        Mgcs_blue = s.mass[GCS.oldstars][GCS.old_blue].sum()
        Mgc = Mgcs_red + Mgcs_blue

        istellarhalo, = numpy.where(  # stars within
            (s.type == 4) & (s.age > 0.) & (s.r() > 0.) & (s.r() < GCS.radius/1000.)
            & (s.halo == 0) & (s.subhalo == 0)
        )
        Mstellarhalo = s.mass[istellarhalo].sum()

        if print_header:
            header = ["Run", "M200", "R200", "Mstar", "M*halo", "Mgc", "fgas"]
            print("{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}{:<10s}".format(*header))
        info = [self.name, M200, r200, Mstars_tenpercent_r200, Mstellarhalo, Mgc, fgas]
        print("{:<10s}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}{:<10.2f}".format(*info))

        del s, sf, GCS
        return info if not print_header else [header, info]


class AurigaOverview(object):
    def __init__(self, basedir="/hits/universe/GigaGalaxy/", verbose=False):
        if os.uname().nodename == 'ChezTimo':
            basedir="/Volumes/Cygnus" + basedir
        if os.uname().nodename == 'ZoltaN':
            basedir="/media/tlrh/Cygnus" + basedir
        if "freya" in os.uname().nodename:
            basedir="/virgo/data/Auriga/"
        self.basedir = basedir
        self.runs = OrderedDict()
        self.find_all_runs()
        if verbose: print("\nOverview of all available Auriga Simulations Runs:\n{0}".format(self))

    def __str__(self):
        s = "  {0:<20}{1:<60s}{2}\n".format("simulation", "rundir", "nsnaps")
        runs = [self.runs[level].get(halo, None) for level in self.runs for halo in self.runs[level].keys()]

        for r in runs:
            s += "  {0}\n".format(r)
        return s

    def getrun(self, level, halo):
        l = self.runs.get(str(level), None)
        if l is None:
            print("ERROR: Au{0}-{1} does not exist".format(level, halo)); return
        h = l.get(str(halo), None)
        if h is None:
            print("ERROR: Au{0}-{1} does not exist".format(level, halo)); return
        return h

    def find_all_runs(self, verbose=False):
        if verbose: print("find_all_runs")
        for rundir in sorted(glob.glob(self.basedir + "level*/" + "halo*"),
                key=lambda s: [ "{:02d}".format(int(c))
                    for c in re.split('(\d+)', s) if c.isdigit()]):
            if verbose: print("rundir: {0}".format(rundir))
            # Only interested in MHD runs, not DMO/Durham runs
            to_skip = [
                "DM", "Durham", "hestia", "variants", "template", "new",
                ".tar", ".toc", ".sh"  # ignore some junk
            ]
            if any(skip in rundir for skip in to_skip) \
                    or ("freya" in os.uname().nodename and "MHD" not in rundir):
                if verbose: print("skipping {0}\n".format(rundir))
                continue
            for s in rundir.split("/"):
                if "level" in s:
                    level = "".join(s.replace("level", "").replace("MHD", "").replace("_", ""))
                if "halo" in s:
                    halo = "".join(s.replace("halo", "").replace("_", ""))
            l = self.runs.get(level, None)
            if l is None: self.runs[level] = dict()
            self.runs[level][halo] = AurigaRun(rundir=rundir, level=level, halo=halo)

    def plot_GCS_level345(self, halo, gc_max_age=10, gc_max_radius=250):
        print("plot_GCS_level345")
        print("  halo          = {0}".format(halo))
        print("  gc_max_age    = {0} Gyr".format(gc_max_age))
        print("  gc_max_radius = {0} kpc\n".format(gc_max_radius))

        from investigate_auriga_gcs import GlobularClusterSystem
        with suppress_stdout():
            GCS = GlobularClusterSystem(self, None, halo,
                radius=gc_max_radius, age=gc_max_age)

        GCS.plot_diagnostics_level345(self, halo)


def default_setup(level=4, halo=24, snapnr=127, loadonlytype=[0, 1, 2, 3, 4, 5]):
    if os.uname().nodename == 'ChezTimo' or os.uname().nodename == 'ZoltaN':
        level, halo, snapnr = 5, 24, 63
    auriga = AurigaOverview(verbose=True)
    run = auriga.getrun(level, halo)
    print("level   : {0}".format(level))
    print("halo    : {0}".format(halo))
    print("snapnr  : {0}".format(snapnr))
    s, sf = run.load_snapshot(snapnr=snapnr, loadonlytype=loadonlytype)

    istars, = numpy.where(
        (s.type == 4) & (s.halo == 0)
        & (s.r() > 0.) & (s.r() < s.galrad)
        & (s.age > 0.)
    )

    idm, = numpy.where(
        (s.type == 1) & (s.halo == 0)
        & (s.r() > 0.) & (s.r() < s.galrad)
    )

    print("The center sits at: "+str(s.center))
    print("Found {0} stars.".format(len(istars)))

    print("Found {0} DMs.".format(len(idm)))


    return s, sf, istars, idm


def eat_snap_and_fof(level, halo_number, snapnr, snappath, run=None, loadonlytype=[4],
                     haloid=0, galradfac=0.1, remove_bulk_vel=True, rotate=True, verbose=True):
    """ Method to eat an Auriga snapshot, given a level/halo_number/snapnr.
        Subfind has been executed 'on-the-fly', during the simulation run.

        @param level: level of the Auriga simulation (3=high, 4='normal' or 5=low).
            Level 3/5 only for halo 6, 16 and 24. See Grand+ 2017 for details.
            Careful when level != 4 because directories may have different names.
        @param halo_number: which Auriga galaxy? See Grand+ 2017 for details.
            Should be an integer in range(1, 31)
        @param snapnr: which snapshot number? This is an integer, in most cases
            in range(1, 128) depending on the number of timesteps of the run.
            The last snapshot would then be 127. Snapshots are written at a
            certain time, but careful because the variable called time is actually
            the cosmological expansion factor a = 1/(1+z). For example, snapnr=127
            has s.time = 1, which corresponds to a redshift of ~0. This makes sense
            because this is the last snapshot and the last snapshot is written at
            redshift zero
        @param snappath: full path to the level/halo directory that contains
            all of the simulation snapshots
        @param loadonlytype: which particle types should be loaded? This should
            be a list of integers. If I'm not mistaken, the options are:
            0 (gas), 1 (dark matter), 2 (unused), 3 (tracers), 4 (stars & wind;
            age > 0. --> stars; age < 0. --> wind), 5 (black holes).
        @param haloid: the ID of the SubFind halo. In case you are interested
            in the main galaxy in the simulation run: set haloid to zero.
            This was a bit confusing to me at first because a zoom-simulation run
            of one Auriga galaxy is also referred to as 'halo', see halo_number.
        @param galradfac: the radius of the galaxy is often used to make cuts in
            the (star) particles. It seems that in general galrad is set to 10%
            of the virial radius R200 of the DM halo that the galaxy sits in. The
            disk does seem to 'end' at 0.1R200.
        @param remove_bulk_vel: boolean to subtract bulk velocity [default True]
        @param rotate: boolean to toggle between rotate / not rotate [default True]
        @param verbose: boolean to print some information

        @return: two-tuple (s, sf) where s is an instance of the gadget_snapshot
            class, and sf is an instance of the subfind class. See Arepo-snap-util,
            gadget_snap.py respectively gadget_subfind.py """

    # Eat the subfind friend of friends output
    if "highfreqstars_" in str(snapnr):
        snapnr = int(snapnr.split("highfreqstars_")[1])
        snapbase = "snapshot_highfreqstars_"
        snapdirbase = "snapdir_highfreqstars_"
        loadonlytype = [4]

        if not hasattr(run, "redshifts"):
            run.get_snapshot_spacing(verbose=verbose)
        if not hasattr(run, "redshifts_highfreqstars"):
            run.get_snapshot_spacing_highfreqstars(verbose=verbose)

        # Well, since we do not have sf for highfreqstars, we cheat. Find the
        # regular snapshot that does have subfind outputs, and then glean the
        # relevant information from there and sneak it in here.
        redshift = run.redshifts_highfreqstars[snapnr if snapnr != 2764 else 2763]
        snapnr_with_sf_closest_to_snapnr_highfreqstars = (numpy.abs(run.redshifts - redshift)).argmin()
        print("{0} / {1} --> {2:.6f}".format(
            snapnr, run.nhighfreqstars-1, redshift))
        s, sf = run.load_header(verbose=verbose, redshift=float(redshift))
        sf.snapnr = snapnr_with_sf_closest_to_snapnr_highfreqstars
        print("  --> {0} / {1} --> {2:.6f}".format(s.snapnr, run.nsnaps-1, s.redshift))
    else:
        from areposnap.gadget_subfind import load_subfind
        sf = load_subfind(snapnr, dir=snappath)
        sf.snapnr = snapnr
        snapbase = "snapshot_"
        snapdirbase = "snapdir_"

    # Eat the Gadget snapshot
    from areposnap.gadget import gadget_readsnap
    s = gadget_readsnap(snapnr, snappath=snappath,
        snapbase=snapbase, snapdirbase=snapdirbase,
        lazy_load=True, subfind=sf, loadonlytype=loadonlytype)
    if "highfreqstars_" not in str(snapnr): sf.redshift = s.redshift
    s.subfind = sf

    if sf is None:  # for highfreqstars
        has_sf_indizes = False
    else:
        try:
            # Sets s.(sub)halo. This allows selecting the halo, e.g. 0 (main 'Galaxy')
            s.calc_sf_indizes(s.subfind)
            has_sf_indizes = True
        except KeyError as e:
            # for example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
            print("WARNING: KeyError encountered in s.calc_sf_indizes")
            if str(e) == "'flty'":
                print("WARNING: KeyError arised because 'flty' was not found in sf.data")
                # print("  sf.data.keys(): {0}".format(sf.data.keys()))
                # print("   s.data.keys(): {0}".format(s.data.keys()))
                has_sf_indizes = False
            else:
                raise

    # for example, Au5-24: snapnr 7-14 breaks due to lack of stars

    s.galrad = None
    if has_sf_indizes:
        # Note that selecting the halo now rotates the disk using the principal axis.
        # rotate_disk is a general switch which has to be set to True to rotate.
        # To then actually do the rotation, do_rotation has to be True as well.
        # Within rotate_disk there are three methods to handle the rotation. Choose
        # one of them, but see the select_halo method for details.
        try:
            matrix = s.select_halo( s.subfind, haloid=haloid, galradfac=galradfac,
                rotate_disk=True, use_principal_axis=True, euler_rotation=False,
                use_cold_gas_spin=False, do_rotation=rotate,
                remove_bulk_vel=remove_bulk_vel, verbose=verbose )
            s.rotmat = matrix.transpose()
            # To use: value_rotated = numpy.dot(value, s.rotmat)
        except KeyError as e:
            # for example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
            print("WARNING: KeyError encountered in s.select_halo")
            print(str(e))
            if str(e) == "'svel'":
                print("WARNING: KeyError arised because 'svel' was not found in s.data")
            elif str(e) == "'pos'":
                print("WARNING: this particular snapshot has no positions.")
            else:
                raise
        except IndexError as e:
            # for example, Au5-24: snapnr 0-3 has empty sf.data; snapnr 4-6 has no stars and no cold-gas spin
            print("WARNING: IndexError encountered in s.select_halo")
            if str(e) == "index 0 is out of bounds for axis 0 with size 0":
                print("WARNING: IndexError arised possibly in get_principal_axis because there are no stars (yet)")
            else:
                raise

        # This means that galrad is 10 % of R200 (200*rho_crit definition)
        # frc2 = Group_R_Crit200
        s.galrad = numpy.maximum(galradfac * sf.data['frc2'][haloid], 0.005)

    # I find it somewhat annoying that age has different shape than other properties...
    # So let's just force age to be of same shape.. by shoving a bunch of zeros in-between
    # I mean, memory is cheap anyway right?
    if "age" in s.data.keys():
        age = numpy.zeros(s.npartall)
        st = s.nparticlesall[:4].sum()
        en = st + s.nparticlesall[4]
        age[st:en] = s.age
        s.data['age'] = age
        del age

    if "gmet" in s.data.keys():
        # Clean negative and zero values of gmet to avoid RuntimeErrors
        s.gmet = numpy.maximum( s.gmet, 1e-40 )

    # Sneak some more info into the s instance
    s.halo_number = halo_number
    s.level = level
    s.snapnr = snapnr
    s.haloid = haloid
    s.name = "Au{0}-{1}".format(s.level, s.halo_number)

    if verbose:
        print("\n{0}".format(s.name))
        print("galrad  : {0}".format(s.galrad))
        print("redshift: {0}".format(s.redshift))
        print("time    : {0}".format(s.time))
        print("center  : {0}\n".format(s.center))

    return s, sf


if __name__ == "__main__":
    # s, sf, istars, idm = default_setup()

    level, halo = 5, 24
    auriga = AurigaOverview(verbose=True)

    run = auriga.getrun(level, halo)
    # run.plot_snapshot_spacing_time()
    # run.plot_snapshot_spacing_redshift()
    # run.get_total_stellar_mass_versus_time()
    run.set_insitu_stars(verbose=True)
