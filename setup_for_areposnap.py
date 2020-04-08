#! /usr/bin/env python3
from numpy.distutils.core import setup, Extension
import os
import numpy as np

#VIP:
# the first path can probably be removed because of the automatic detection below
#incl_dirs = ['/usr/local/lib/python2.6/site-packages/numpy/core/include/numpy','/usr/local/include', '/u/system/Power6/libs/gsl/1.14/include']
#libs_dirs = ['/usr/local/lib']

#OSX:
# the first path can probably be removed because of the automatic detection below
incl_dirs = [
    # '/mpcdf/soft/SLE_15/packages/skylake/gsl/gcc_9/2.4/include'
    '/mpcdf/soft/SLE_15/packages/skylake/gsl/intel_19_0_5/2.4/include',
    '/mpcdf/soft/SLE_15/packages/x86_64/intel_parallel_studio/2019.5/mkl/include'
]
libs_dirs = [
    # '/mpcdf/soft/SLE_15/packages/skylake/gsl/gcc_9/2.4/lib'
    '/mpcdf/soft/SLE_15/packages/skylake/gsl/intel_19_0_5/2.4/lib',
    '/mpcdf/soft/SLE_15/packages/x86_64/intel_parallel_studio/2019.5/mkl/include'
]

# find numpy include path automatically
incl_dirs += [os.path.join(np.get_include(),'numpy')]

libs = ['gsl','gslcblas','m']

defines = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]

grid = Extension(   'areposnap.grid',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/grid.c'])

pyeos  = Extension( 'areposnap.pyeos',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/pyeos.c','src/libs/eos.c'])

pyhelm_eos  = Extension( 'areposnap.pyhelm_eos',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/pyhelm_eos.c','src/libs/helm_eos.c'])

pyopal_eos  = Extension( 'areposnap.pyopal_eos',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
            sources = ['src/libs/pyopal_eos.c','src/libs/opal_eos.c'])

pysph  = Extension( 'areposnap.pysph',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
            extra_compile_args = ['-qopenmp'],
            extra_link_args = ['-qopenmp'],
                    sources = ['src/libs/pysph.c','src/libs/sph.c'])

calcGrid = Extension(   'areposnap.calcGrid',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
            extra_compile_args = ['-qopenmp'],
            extra_link_args = ['-qopenmp'],
                    sources = ['src/libs/calcGrid.c','src/libs/sph.c', 'src/libs/treef.c'])

ic  = Extension(    'areposnap.ic',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/ic.c','src/libs/pyhelm_eos.c','src/libs/helm_eos.c'])

rgadget  = Extension(   'areposnap.rgadget',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/rgadget.c','src/libs/gadgetSnap.c'])

boundmass  = Extension( 'areposnap.boundmass',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/boundmass.c','src/libs/gadgetSnap.c'])

createICs  = Extension( 'areposnap.createICs',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/createICs.c','src/libs/gadgetSnap.c','src/libs/pyhelm_eos.c','src/libs/helm_eos.c','src/libs/rgadget.c'])

opalopacities  = Extension( 'areposnap.opalopacities',
            define_macros = defines,
            include_dirs = incl_dirs,
            libraries    = libs,
            library_dirs = libs_dirs,
                    sources = ['src/libs/opacities/opacities.pyf', 'src/libs/opacities/xztrin21.f'])

setup(  name = 'Python script lib',
        version = '2.0',
        description = 'Scripts to work with GADGET/LEAFS/Arepo input & output + EOS + opacities',
        author = 'Ruediger',
        author_email = 'ruediger.pakmor@h-its.org',
        package_dir = {
            'areposnap': 'src',
            'areposnap.stellar_ics': 'src',
            'areposnap.stellar_ics': 'src',
        },
        py_modules = [
            'areposnap.arepo_ics',  'areposnap.arepo_run',  'areposnap.const',
            'areposnap.cosmological_factors', 'areposnap.data_plot',
            'areposnap.eos',  'areposnap.fortran',  'areposnap.gadget',
            'areposnap.gadget_run', 'areposnap.gadget_snap', 'areposnap.setup',
            'areposnap.gadget_subfind', 'areposnap.gadget_tree', 'areposnap.masses',
            'areposnap.yag', 'areposnap.artis', 'areposnap.decays',
            'areposnap.gadget_utils', 'areposnap.get_path_hack', 'areposnap.sfigure',
            'areposnap.ionization_hhe', 'areposnap.loaders', 'areposnap.loadmodules',
            'areposnap.mesa', 'areposnap.network_data',  'areposnap.object',
            'areposnap.pp_tracer',  'areposnap.ppgrid',  'areposnap.ppgrid_yann_util',
            'areposnap.startup',  'areposnap.tracer',  'areposnap.utilities',
            'areposnap.powerspec_matter', 'areposnap.powerspec', 'areposnap.crenergy',
            'areposnap.leafs_snap',  'areposnap.leafs_tracer', 'areposnap.leafs',
            'areposnap.leafs2avizo', 'areposnap.leafs_input', 'areposnap.leafs_analysis',
            'areposnap.opacities', 'areposnap.yann_tracer', 'areposnap.decays',
            'areposnap.ellipse', 'areposnap.parallel_decorators',
            'areposnap.parse_particledata', 'areposnap.mock_read',
            ],
        packages = ['areposnap.stellar_ics', 'areposnap.publications'],
        ext_modules = [grid, pyeos, pyhelm_eos, pyopal_eos, pysph, calcGrid, ic, rgadget, boundmass, createICs, opalopacities],
        data_files=[('areposnap/eostable', [
            'src/eostable/EOS5_data',  'src/eostable/GN93hz',
            'src/eostable/helm_table.dat',  'src/eostable/species_star.txt',
            'src/eostable/species05.txt', 'src/eostable/species_wd_co.txt',
            'src/eostable/decaydata', 'src/eostable/species384.txt',
            'src/eostable/data_logR', 'src/eostable/data_logT',
            'src/eostable/data_logkappa', 'src/eostable/leveldata']),
            ('areposnap/publications', ['src/publications/colors-tol.mplstyle',
            'src/publications/colors-wong.mplstyle',
            'src/publications/hide_top_right_spines.mplstyle',
            'src/publications/publications.mplstyle']),
            # ('areposnap/stellar_ics/templates', ['src/stellar_ics/templates/template-params',
            # 'src/stellar_ics/templates/template-run.cmd.bridge',
            # 'src/stellar_ics/templates/template-run.cmd.sandy',
            # 'src/stellar_ics/templates/template-run.cmd.haswell',
            # 'src/stellar_ics/templates/template-run.cmd.forhlr'])
        ]
)
