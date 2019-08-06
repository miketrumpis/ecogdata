import numpy as np
from scipy.ndimage import convolve1d
from scipy.signal import lfilter as lfilter_scipy
from nitime.algorithms import multi_taper_psd

from ecogdata.filt.time.blocked_filter import bfilter as bfilter_serial
from ecogdata.filt.time.blocked_filter import overlap_add
from ecogdata.parallel.array_split import split_at, shared_ndarray


# Parallelized re-definitions

# Turn bfilter's output optional parameter into a split parameter.
@split_at(split_arg=(2, 3))
def bfilter_inner(b, a, x, y, **kwargs):
    bfilter_serial(b, a, x, out=y, **kwargs)


# Redefine bfilter to operate inplace or not
def bfilter(b, a, x, out=None, **kwargs):
    if out is None:
        out = x
    bfilter_inner(b, a, x, out, **kwargs)


overlap_add = split_at()(overlap_add)

# multi taper spectral estimation
multi_taper_psd = split_at(splice_at=(1,2))(multi_taper_psd)

# convolution
convolve1d = split_at(split_arg=0)(convolve1d)

# linear filtering wrapper -- converts lfilter to a "void" method rather than returning an array
@split_at(split_arg=(2, 3, 4), splice_at=(0,))
def lfilter_inner(b, a, x, y, zi, **kwargs):
    kwargs['zi'] = zi
    y[:], zi = lfilter_scipy(b, a, x, **kwargs)
    return zi

def lfilter(b, a, x, out=None, **kwargs):
    if out is None:
        out = shared_ndarray(x.shape, x.dtype.char)
    zi = kwargs.pop('zi', None)
    if zi is None:
        zi = np.zeros((len(x), len(b) - 1))
    zi = lfilter_inner(b, a, x, out, zi, **kwargs)
    return out, zi

# Convenience wrappers
def filtfilt(arr, b, a, bsize=10000):
    """
    Docstring
    """
    # needs to be axis=-1 for contiguity
    bfilter(b, a, arr, bsize=bsize, axis=-1, filtfilt=True)
