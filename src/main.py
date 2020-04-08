import time
import scipy
import numpy
import numpy as np
import multiprocessing
from scipy.stats import binned_statistic

import colorcet
import colorbrewer
from matplotlib import pyplot
from matplotlib import gridspec
from matplotlib import pyplot as plt
pyplot.style.use("tlrh")


# Here we group available simulations together in combinations of interest
l5_sims = {
    5: [6, 9, 16, 24]
}
l4_sims = {
    4: range(1, 31),
}
l3_sims = {
    3: [6, 16, 21, 23, 24, 27],
}
l45_sims = { **l5_sims, **l4_sims }
all_sims = { **l5_sims, **l4_sims , **l3_sims }
au6_sims = {
    5: [6],
    4: [6],
    3: [6]
}
au16_sims = {
    5: [16],
    4: [16],
    3: [16]
}
au24_sims = {
    5: [24],
    4: [24],
    3: [24]
}

COLORS = {
    "istars": "#dfc27d",
    "iold": "#dfc27d",
    "insitu": "#a6611a",
    "accreted": "#018571",
    "insitu_old": "#a6611a",
    "accreted_old": "#018571",
    "milky_way": "#542788",
    "MW": "#542788",
    "andromeda": "#c51b7d",
    "M31": "#c51b7d"
}

def get_nsims(sims):
    sims_to_use = []
    for level in sims.keys():
        for halo in sims[level]:
            sims_to_use.append( (level, halo) )
    return len(sims_to_use)


def worker_process(auriga, sims_for_this_thread, analysismethod):
    time.sleep(1)
    name = multiprocessing.current_process().name
    print("Worker {0:<02s} started".format(name))
    time.sleep(1)

    for (level, halo) in sims_for_this_thread:
        print("  Worker {0:<02s} now processes: Au{1}-{2}"\
            .format(name, level, halo))

        run = auriga.getrun(level=level, halo=halo)
        tstart = time.time()
        analysismethod(run)
        print("  Running analysismethod operating on {0} took {1:.2f} s".format(run.name, time.time()-tstart))
        del run

    print("Worker {0:<02s} finished".format(name))


def master_process(auriga, sims, analysismethod, workers=[], nthreads=4):
    sims_to_use = []
    for level in sims.keys():
        for halo in sims[level]:
            sims_to_use.append( (level, halo) )

    nsims = len(sims_to_use)
    if nsims < nthreads: nthreads = nsims
    print("Requested to process {0} simulations using {1} threads\n"\
        .format(nsims, nthreads))

    for i in range(nthreads):
        sims_start = int( nsims*i/float(nthreads) )
        sims_stop = int( nsims*(i+1)/float(nthreads) )
        print("Thread {0} gets simulations {1:02d}:{2:02d}"\
            .format(i, sims_start, sims_stop))

        worker = multiprocessing.Process(target=worker_process,
            args=(auriga, sims_to_use[sims_start:sims_stop], analysismethod))
        workers.append(worker)
        worker.start()
    print("")

    for w in workers:  # wait for workers to finish
        w.join()


def worker_process_for_run(run, snap_start, snap_stop,
        analysismethod, highfreqstars):
    time.sleep(1)
    name = multiprocessing.current_process().name
    print("Worker {0:<02s} started".format(name))
    time.sleep(1)

    for snapnr in range(snap_start, snap_stop):
        print("  Worker {0:<02s} now processes: {1} / {2}"\
            .format(name, run.name, snapnr))

        tstart = time.time()
        analysismethod(run, snapnr, highfreqstars)
        print("  Running analysismethod operating on {0} / {1} took {2:.2f} s".format(
            run.name, snapnr, time.time()-tstart))

    print("Worker {0:<02s} finished".format(name))


def master_process_for_run(run, analysismethod,
        highfreqstars, workers=[], nthreads=4):
    if highfreqstars:
        nsnaps = run.nhighfreqstars-1  # b/c 2763 misses
    else:
        nsnaps = run.nsnaps

    if nsnaps < nthreads: nthreads = nsnaps

    print("Requested to process {0} snapshots using {1} threads\n"\
        .format(nsnaps, nthreads))

    for i in range(nthreads):
        snap_start = int( nsnaps*i/float(nthreads) )
        snap_stop = int( nsnaps*(i+1)/float(nthreads) )
        print("Thread {0} gets snapshots {1:02d}:{2:02d}"\
            .format(i, snap_start, snap_stop))

        worker = multiprocessing.Process(target=worker_process_for_run,
            args=(run, snap_start, snap_stop, analysismethod, highfreqstars))
        workers.append(worker)
        worker.start()
    print("")

    for w in workers:  # wait for workers to finish
        w.join()


if __name__ == "__main__":
    from read_auriga_snapshot import AurigaOverview
    auriga = AurigaOverview(verbose=False)
    def tmp(run):
        print("  Hello, I am the analysismethod operating on {0}".format(run.name))
    for sims in [ allsims, au6_sims ]:
        master_process(auriga, sims, tmp)
