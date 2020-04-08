import os
import sys
import numpy
from astropy import units as u
import astropy.coordinates as coord
from matplotlib import pyplot
from contextlib import contextmanager
from astropy.coordinates import ICRS, Distance, Angle


def get_angular_momentum(a, mask):
    # L = r x p = r x mv = (x,y,z) x m*(vx,vy,vz)
    L = numpy.cross( s.pos[mask].astype("f8"),
        (s.mass[mask][:,None].astype("f8") * s.vel[mask].astype("f8")) )
    Ltot = L.sum( axis=0 )
    Ldir = Ltot / numpy.sqrt( (Ltot**2).sum() )


def compute_specific_energy(s, mask, debug=True):
    Epot = s.pot[mask]
    Ekin = 0.5 * numpy.sum(p2(s.vel[mask]), axis=1)

    if debug:
        pyplot.figure(figsize=(12, 9))

        bins, edges = numpy.histogram(-Epot, bins=128, normed=True)
        pyplot.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid", label=r"-Epot")

        bins, edges = numpy.histogram(Ekin, bins=128, normed=True)
        pyplot.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid", label=r"Ekin")

        pyplot.xlabel(r"Specific Energy [erg]")
        pyplot.legend()

        mean = numpy.mean(Ekin)
        median = numpy.median(Ekin)
        std = numpy.std(Ekin)
        print("Kinetic Energy   --> mean, median, std = {0:>10.2g}{1:>10.2g}{2:>10.2g}"
            .format(mean, median, std))

        mean = numpy.mean(Epot)
        median = numpy.median(Epot)
        std = numpy.std(Epot)
        print("Potential Energy --> mean, median, std = {0:>10.2g}{1:>10.2g}{2:>10.2g}"
            .format(mean, median, std))

        pyplot.show()

    return Ekin, Epot


def compute_specific_angular_momentum(s, mask, debug=True):
    L = numpy.cross( s.pos[mask].astype("f8"), s.vel[mask].astype("f8") )

    if debug:
        Ltot = L.sum( axis=0 )
        Ldir = Ltot / numpy.sqrt( (Ltot**2).sum() )
        Lznorm = L[:,0] / numpy.linalg.norm(L, axis=1)
        Lzangle =  numpy.pi/2-numpy.arccos(Lznorm)
        print("Ldir = {0}".format(Ldir))

        fig, (ax1, ax2, ax3) = pyplot.subplots(1, 3, figsize=(16, 5))

        for ax, dim, i in zip([ax1, ax2, ax3], ["x", "y", "x"], [2, 1, 0]):
            mean = numpy.mean(L[:,i])
            median = numpy.median(L[:,i])
            std = numpy.std(L[:,i])
            print("Angular Momentum {0} --> mean, median, std = ")
            print("{1:>10.2g}{2:>10.2g}{3:>10.2g}".format(i, mean, median, std))
            bins, edges = numpy.histogram(L[:,i], bins=128, normed=True)
            ax.plot((edges[1:]+edges[:-1])/2, bins, drawstyle="steps-mid")
            ax.set_xlabel(r"$L_{0}$".format(dim))

        pyplot.show()

    return L


def get_center_of_mass(s, mask, debug=True):
    com = ( (s.pos[mask] * (s.mass[mask][...,None])).sum(axis=1) /
        (s.mass[mask].sum(axis=1)[:,None]) )

    if debug:
        # TODO: this will break due to circular dep
        from auriga_plots import plot_hist2d_of_particle_subset
        labels = [r"Au24-4. $z \, = \, {0:2.2f}$".format(0), r"Stars" + r" : $0 < r < 0.5 r_{200}$"]
        fig = plot_hist2d_of_particle_subset(s, sf, labels, iSurroundings)
        axxz, axzy, axyx, axt = fig.axes

        for e1, e2, c in zip(stream_com[:-1], stream_com[1:],
                matplotlib.cm.viridis(numpy.linspace(0, 1, len(stream_com)-1))):
            axyx.plot((1000*e1[1], 1000*e2[1]), (1000*e1[2], 1000*e2[2]), c=c)
            axxz.plot((1000*e1[2], 1000*e2[2]), (1000*e1[0], 1000*e2[0]), c=c)
            axzy.plot((1000*e1[0], 1000*e2[0]), (1000*e1[1], 1000*e2[1]), c=c)

        l = axyx.scatter(1000*stream_com[:,1], 1000*stream_com[:,2],
            c=matplotlib.cm.viridis(numpy.linspace(0, 1, len(stream_com))))
        l = axxz.scatter(1000*stream_com[:,2], 1000*stream_com[:,0],
            c=matplotlib.cm.viridis(numpy.linspace(0, 1, len(stream_com))))
        l = axzy.scatter(1000*stream_com[:,0], 1000*stream_com[:,1],
            c=matplotlib.cm.viridis(numpy.linspace(0, 1, len(stream_com))))

        for ax in [axxz, axzy, axyx]:
            ax.set_xlim(-5000*s.galrad, 5000*s.galrad)
            ax.set_ylim(-5000*s.galrad, 5000*s.galrad)

        # cs = matplotlib.cm.viridis(numpy.linspace(0,1,50))
        # for e,c in zip(l,cs):
        #     e.set_color(c)
        # pyplot.xlim(-0.001,0.0005)
        # pyplot.ylim(-0.0015,0.001)

        axxz.plot([0], [0], 'rX', markersize=10)
        axyx.plot([0], [0], 'rX', markersize=10)
        axzy.plot([0], [0], 'rX', markersize=10)

        pyplot.show()


# Wang, Ma & Liu (2019, sec 4.1). arXiv 1901.11229v1
def get_Wang2019_X(A, B, PA=(38*numpy.pi)/180):
    """ M31 position angle PA from Kent (1989) """
    return A * numpy.sin(PA) + B * numpy.cos(PA)

def get_Wang2019_Y(A, B, PA=(38*numpy.pi)/180):
    """ M31 position angle PA from Kent (1989) """
    return -A * numpy.cos(PA) + B * numpy.sin(PA)

def get_Wang2019_A(alpha, alpha0, delta):
    return numpy.sin(alpha - alpha0) * numpy.cos(delta)

def get_Wang2019_B(alpha, alpha0, delta, delta0):
    return numpy.sin(delta) * numpy.cos(delta0) - \
        numpy.cos(alpha - alpha0)*numpy.cos(delta)*numpy.sin(delta0)

def apply_Wang2019_deprojection(X, Y, IA=(77.5*numpy.pi)/180):
    """ M31 inclination angle IA from Kent (1989) """
    return numpy.sqrt(X**2 + (Y/numpy.cos(IA))**2)

def calculate_M31_Rgc_Wang2019(coordinates, deproject=False, debug=False,
        M31_coord=coord.SkyCoord("0 42 44.3503", "41 16 08.634",
                frame="icrs", equinox="J2000", unit=(u.hourangle, u.deg)),
        M31_distance=Distance(780, unit=u.kpc)):
    # Preferred position is given by the NASA Extragalactic Database as
    # 00h42m44.3503s +41d16m08.634s Equatorial J2000.0  (2010ApJS..189...37E)
    #   M31 RA and Dec are also published by Kent (1989)
    # Distance McConnachie+ (2005), Conn+ (2012): 785 +/- 25 kpc; 779 +19/-18 kpc
    #   Alternatively, Freedman & Madore (1990) 773 +/- 36 kpc
    #   NED found 409 distance measurements /w mean 784 kpc, median 776 kpc.
    #   but in that table no homogenization or corrections have been applied.


    # Given coordinates
    test = coordinates[0]
    if not isinstance(coordinates, coord.SkyCoord):
        ra, dec = [], []
        for c in coordinates:
            ra.append(c.ra); dec.append(c.dec)
        coordinates = coord.SkyCoord(ra, dec)
    assert test.ra == coordinates[0].ra, "RA broke"
    assert test.dec == coordinates[0].dec, "DEC broke"

    alpha = coordinates.ra.radian
    delta = coordinates.dec.radian

    # M31
    alpha0 = M31_coord.ra.radian
    delta0 = M31_coord.dec.radian

    A = get_Wang2019_A(alpha, alpha0, delta)
    B = get_Wang2019_B(alpha, alpha0, delta, delta0)
    # Have to convert to arcmin b/c can't do sqrt(X^2 + Y^2) when x and y are angles
    X = Angle(get_Wang2019_X(A, B), unit=u.radian).arcmin
    Y = Angle(get_Wang2019_Y(A, B), unit=u.radian).arcmin

    if debug:
        print("A = {0}".format(A))
        print("B = {0}".format(B))
        print("X = {0}".format(X))
        print("Y = {0}\n".format(Y))

    if deproject:
        # De-projected two-dimensional galactocentric radius.
        Rproj = apply_Wang2019_deprojection(X, Y)
    else:
        # Projected two-dimensional galactocentric radius.
        Rproj = numpy.sqrt(X**2 + Y**2)  # in arcmin

    X = Distance(
       Angle(X, unit=u.arcmin).radian * M31_distance,
       unit=M31_distance.unit, allow_negative=True,
    ).value
    Y = Distance(
       Angle(Y, unit=u.arcmin).radian * M31_distance,
       unit=M31_distance.unit, allow_negative=True,
    ).value
    Rproj = Distance(
       Angle(Rproj, unit=u.arcmin).radian * M31_distance,
       unit=M31_distance.unit
    ).value

    return X, Y, Rproj


# https://gist.github.com/jonathansick/9399842
def correct_rgc(coord, glx_ctr=ICRS('00h42m44.33s', '+41d16m07.5s'),
        glx_PA=Angle('37d42m54s'),
        glx_incl=Angle('77.5d'),
        glx_dist=Distance(783, unit=u.kpc)):
    # TODO: reference for the 783 kpc distance
    """Computes deprojected galactocentric distance.
    Inspired by: http://idl-moustakas.googlecode.com/svn-history/
        r560/trunk/impro/hiiregions/im_hiiregion_deproject.pro
    Parameters
    ----------
    coord : :class:`astropy.coordinates.ICRS`
        Coordinate of points to compute galactocentric distance for.
        Can be either a single coordinate, or array of coordinates.
    glx_ctr : :class:`astropy.coordinates.ICRS`
        Galaxy center.
    glx_PA : :class:`astropy.coordinates.Angle`
        Position angle of galaxy disk.
    glx_incl : :class:`astropy.coordinates.Angle`
        Inclination angle of the galaxy disk.
    glx_dist : :class:`astropy.coordinates.Distance`
        Distance to galaxy.
    Returns
    -------
    obj_dist : class:`astropy.coordinates.Distance`
        Galactocentric distance(s) for coordinate point(s).
    """
    # distance from coord to glx centre
    sky_radius = glx_ctr.separation(coord)
    # TODO: what on earth is this avg_dec doing here?
    avg_dec = 0.5 * (glx_ctr.dec + coord.dec).radian
    x = (glx_ctr.ra - coord.ra) * numpy.cos(avg_dec)
    y = glx_ctr.dec - coord.dec
    # azimuthal angle from coord to glx  -- not completely happy with this
    phi = glx_PA - Angle('90d') \
            + Angle(numpy.arctan(y.arcsec / x.arcsec), unit=u.rad)

    # TODO: this does not even look remotely like the original
    # https://github.com/moustakas/moustakas-projects/blob/master/
    # hiiregions/im_hiiregion_deproject.pro
    # These two lines below looks like cartesian coordinates are calculated
    # from polar coordinates, where the radius sky_radius is obtained by
    # calculating the angular difference, and the angle is obtained by taking
    # the position angle into account as well as polar coordinate angle
    # phi=arctan(y/x). O_o what's going on here?

    # convert to coordinates in rotated frame, where y-axis is galaxy major
    # ax; have to convert to arcmin b/c can't do sqrt(x^2+y^2) when x and y
    # are angles
    xp = (sky_radius * numpy.cos(phi.radian)).arcmin
    yp = (sky_radius * numpy.sin(phi.radian)).arcmin

    # de-project
    ypp = yp / numpy.cos(glx_incl.radian)
    obj_radius = numpy.sqrt(xp ** 2 + ypp ** 2)  # in arcmin
    obj_dist = Distance(Angle(obj_radius, unit=u.arcmin).radian * glx_dist,
            unit=glx_dist.unit)

    # Computing PA in disk (unused)
    obj_phi = Angle(numpy.arctan(ypp / xp), unit=u.rad)
    # TODO Zero out very small angles, i.e.
    # if numpy.abs(Angle(xp, unit=u.arcmin)) < Angle(1e-5, unit=u.rad):
    #     obj_phi = Angle(0.0)

    return obj_dist


@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_settings = numpy.seterr(all="ignore")
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            numpy.seterr(**old_settings)


def print_progressbar(i, N, whitespace=""):
    pbwidth = 42

    progress = float(i)/(N-1)
    block = int(round(pbwidth*progress))
    text = "\r{0}Progress: [{1}] {2:.1f}%".format(whitespace,
        "#"*block + "-"*(pbwidth-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()

    if i == (N-1):
        print(" .. done")
