#!/usr/bin/env python
import os
from glob import glob
import setuptools
import numpy

try:
    from numpy.distutils.misc_util import get_numpy_include_dirs

    numpy_include_dirs = get_numpy_include_dirs()
except AttributeError:
    numpy_include_dirs = numpy.get_include()

header_dirs = list(numpy_include_dirs)

if __name__ == "__main__":
    filter_extension = setuptools.Extension(
        'ecogdata.filt.time._slepian_projection',
        ['ecogdata/filt/time/_slepian_projection.pyx'],
        include_dirs = header_dirs,
        libraries=(['m'] if os.name != 'nt' else []),
        extra_compile_args=['-O3']
    )
    setuptools.setup(ext_modules=[filter_extension], scripts=glob('scripts/*.py'))
