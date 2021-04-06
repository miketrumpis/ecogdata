import numpy as np
from scipy.ndimage import convolve1d as convolve1d_serial
from scipy.signal import lfilter as lfilter_serial
from nitime.algorithms import multi_taper_psd as multi_taper_psd_serial

from ecogdata.filt.time.blocked_filter import bfilter as bfilter_serial
from ecogdata.filt.time.blocked_filter import overlap_add as overlap_add_serial
from ecogdata.parallel.array_split import split_at
import ecogdata.parallel.sharedmem as shm


# Parallelized re-definitions

# redefine bfilter to be a void function (put output into y)
def bfilter_void(b, a, x, y, **kwargs):
    return bfilter_serial(b, a, x, out=y, **kwargs)

# parallelize bfilter_void
bfilter_para = split_at(split_arg=(2, 3))(bfilter_void)

# Redefine bfilter to operate inplace or not using parallel driver
def bfilter(b, a, x, out=None, **kwargs):
    if out is None:
        out = x
    return bfilter_para(b, a, x, out, **kwargs)

# WIP
# if hasattr(bfilter_para, 'uses_parallel'):
#     bfilter.uses_parallel = bfilter_para.uses_parallel

overlap_add = split_at()(overlap_add_serial)

# multi taper spectral estimation
multi_taper_psd = split_at(splice_at=(1, 2))(multi_taper_psd_serial)

# convolution
convolve1d = split_at(split_arg=0)(convolve1d_serial)

# linear filtering wrapper -- converts lfilter to a "void" method rather than returning an array
# @split_at(split_arg=(2, 3, 4), splice_at=(0,))
def lfilter_void(b, a, x, y, zi, **kwargs):
    kwargs['zi'] = zi
    y[:], zi = lfilter_serial(b, a, x, **kwargs)
    return zi

lfilter_para = split_at(split_arg=(2, 3, 4), splice_at=(0,))(lfilter_void)

def lfilter(b, a, x, out=None, **kwargs):
    if out is None:
        # Can check parallel with the first split array
        if lfilter_para(b, a, x, None, None, check_parallel=True):
            out = shm.shared_ndarray(x.shape, x.dtype.char)
        else:
            out = np.empty_like(x)
    zi = kwargs.pop('zi', None)
    if zi is None:
        zi = np.zeros((len(x), len(b) - 1))
    zi = lfilter_para(b, a, x, out, zi, **kwargs)
    return out, zi

# if hasattr(lfilter_para, 'uses_parallel'):
#     lfilter.uses_parallel = lfilter_para.uses_parallel

# Convenience wrappers
def filtfilt(arr, b, a, bsize=10000):
    """
    Docstring
    """
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1, filtfilt=True)

# if hasattr(bfilter, 'uses_parallel'):
#     filtfilt.uses_parallel = bfilter.uses_parallel
