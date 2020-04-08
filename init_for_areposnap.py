import os
import sys
import glob

dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(dir + "/areposnap")
# print("\n\nsys.path\n{0}\n\n".format(sys.path))

modules = glob.glob(os.path.dirname(__file__)+"/*.py")
__all__ = [ os.path.basename(f)[:-3]
    for f in modules if os.path.isfile(f) and not f.endswith('__init__.py')
]

__all__.append("gadget_tree")

# print("\n\n__all__\n{0}\n\n".format(__all__))
