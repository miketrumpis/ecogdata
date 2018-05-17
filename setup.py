from setuptools import setup, Extension, find_packages
from numpy.distutils.command import build_src
import Cython
import Cython.Compiler.Main
build_src.Pyrex = Cython
build_src.have_pyrex = True
from Cython.Distutils import build_ext
import Cython
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs
    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()


dirs = list(numpy_include_dirs)

slepian_projection = Extension(
    'ecogdata.filt.time._slepian_projection',
    ['ecogdata/filt/time/_slepian_projection.pyx'],
    include_dirs = dirs, 
    libraries=['m'],
    extra_compile_args=['-O3']
    )

with open('requirements.txt') as f:
    reqs = f.readlines()
    reqs = map(str.strip, reqs)
    reqs = filter(None, reqs)

if __name__=='__main__':
    setup(
        name = 'ecogdata',
        version = '0.1',
        packages = find_packages(),
        ext_modules = [slepian_projection],
        cmdclass = {'build_ext': build_ext},
        install_requires=reqs
    )
