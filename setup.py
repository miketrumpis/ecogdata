import os
from glob import glob
from setuptools import setup, Extension, find_packages
from numpy.distutils.command import build_src
# import Cython.Compiler.Main
# build_src.Pyrex = Cython
# build_src.have_pyrex = True
# from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs)


# slepian_projection = Extension(
#     'ecogdata.filt.time._slepian_projection',
#     ['src/ecogdata/filt/time/_slepian_projection.pyx'],
#     include_dirs = dirs,
#     libraries=(['m'] if os.name != 'nt' else []),
#     extra_compile_args=['-O3']
#     )


if __name__=='__main__':
    setup(
        name='ecogdata',
        version='0.1',
        package_dir={'': 'src'},
        packages=find_packages(where='src'),
        scripts=glob('scripts/*.py'),
        # ext_modules=cythonize([slepian_projection]),
        # cmdclass={'build_ext': build_ext},
        package_data={'ecogdata.expconfig': ['*.txt']}
    )
