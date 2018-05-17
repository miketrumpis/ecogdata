# brief numerical utility functions
from __future__ import division
from __future__ import print_function
from builtins import map
from builtins import zip
from builtins import range
import numpy as np
from scipy.integrate import simps
import scipy.ndimage as ndimage

from ecogdata.util import *
from ecogdata.parallel.array_split import split_at

def ndim_prctile(x, p, axis=0):
    xs = np.sort(x, axis=axis)
    dim = xs.shape[axis]
    idx = np.round( float(dim) * np.asarray(p) / 100 ).astype('i')
    slicer = [slice(None)] * x.ndim
    slicer[axis] = idx
    return xs[slicer]

def nanpercentile(*args, **kwargs):
    """
    Light wrap of nanpercentile from numpy. This version presents output
    consistent with the percentile function from versions 1.9.x to 1.10.x.
    The behavior from version 1.11.x is already consistent.
    """
    axis = kwargs.get('axis', None)
    res = np.nanpercentile(*args, **kwargs)

    import distutils.version as vs
    if vs.LooseVersion(np.__version__) < vs.LooseVersion('1.11.0'):
        if axis is not None:
            print(axis)
            while axis < 0:
                axis += args[0].ndim
            print(axis)
            return np.rollaxis(res, axis)
    return res

def unity_normalize(x, axis=None):
    if axis is None:
        mn = x.min(axis=axis)
        mx = x.max(axis=axis)
        return (x-mn)/(mx-mn)

    x = np.rollaxis(x, axis)
    mn = x.min(axis=-1)
    mx = x.max(axis=-1)
    while mn.ndim > 1:
        mn = mn.min(axis=-1)
        mx = mx.max(axis=-1)
    
    slicer = [slice(None)] + [np.newaxis]*(x.ndim-1)
    x = x - mn[slicer]
    x = x / (mx-mn)[slicer]
    return np.rollaxis(x, 0, axis+1)

def density_normalize(x, raxis=None):
    if raxis is None:
        return x / np.nansum(x)

    if x.ndim > 2:
        shape = x.shape
        if raxis not in (0, -1, x.ndim-1):
            raise ValueError('can only normalized in contiguous dimensions')

        if raxis == 0:
            x = x.reshape(x.shape[0], -1)
        else:
            raxis = -1
            x = x.reshape(-1, x.shape[-1])
        xn = density_normalize(x, raxis=raxis)
        return xn.reshape(shape)

    # roll repeat axis to last axis
    x = np.rollaxis(x, raxis, start=2)
    x = x / np.nansum(x, 0)
    return np.rollaxis(x, 1, start=raxis)

def center_samples(x, axis=-1):
    # normalize samples with a "Normal" transformation
    mu = x.mean(axis=axis)
    sig = x.std(axis=axis)
    if x.ndim > 1:
        slices = [slice(None)] * x.ndim
        slices[axis] = None
        mu = mu[slices]
        sig = sig[slices]
    y = x - mu
    y /= sig
    return y

def sphere_samples(x, axis=-1):
    # normalize samples by projecting to a hypersphere
    norm = np.linalg.norm(axis=axis)
    slices = [slice(None)] * x.ndim
    slices[axis] = None
    y = x / norm
    return y


def roc(null_samp, sig_samp):
    # Create an empirical ROC from samples of a target distribution
    # and samples of a no-target distribution. For threshold values,
    # use only the union of sample values.
    
    mn = min( null_samp.min(), sig_samp.min() )
    mx = max( null_samp.max(), sig_samp.max() )
    
    #thresh = np.linspace(mn, mx, n)
    thresh = np.union1d(null_samp, sig_samp)

    # false pos is proportion of samps in null samp that are > thresh
    false_hits = (null_samp >= thresh[:,None]).astype('d')
    false_pos = np.sum(false_hits, axis=1) / len(null_samp)

    # true pos is proportion of samps in sig samp that are > thresh
    hits = (sig_samp >= thresh[:,None]).astype('d')
    true_pos = np.sum(hits, axis=1) / len(sig_samp)

    return np.row_stack( (false_pos[::-1], true_pos[::-1]) )


def integrate_roc(roc_pts):
    x, y = roc_pts
    same_x = np.r_[False, np.diff(x) == 0]
    # only keep pts on the curve where x increases
    x = x[~same_x]
    y = y[~same_x]
    # append end points for some kind of interpolation
    x = np.r_[0, x, 1]
    y = np.r_[0, y, 1]
    cp = simps(y, x=x, even='avg')
    return cp


def savitzky_golay(y, window_size, order, deriv=0, rate=1, axis=-1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.

    Parameters
    ----------

    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute
        (default = 0 means only smoothing)

    Returns
    -------

    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).

    Notes
    -----

    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.

    Examples
    --------

    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()

    References
    ----------

    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = list(range(order+1))
    half_window = (window_size -1) // 2
    # precompute coefficients
    ## b = np.mat(
    ##     [[k**i for i in order_range]
    ##      for k in range(-half_window, half_window+1)]
    ##     )
    ## m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    ix = np.arange(-half_window, half_window+1, dtype='d')
    bt = np.array( [np.power(ix, k) for k in order_range] )
    if np.iterable(deriv):
        scl = [ rate**d * factorial(d) for d in deriv ]
        scl = np.array(scl).reshape( len(deriv), 1 )
    else:
        scl = rate**deriv * factorial(deriv)
    m = np.linalg.pinv(bt.T)[deriv] * scl

    if m.ndim == 2:
        ys = [ndimage.convolve1d(y, mr[::-1], mode='constant', axis=axis)
              for mr in m]
        return np.array(ys)
    
    return ndimage.convolve1d(y, m[::-1], mode='constant', axis=axis)


def _autovectorize(fn, exclude_after=-1):
    def vfn(*args, **kwargs):
        kwargs.pop('axis', None)
        va = list(map(np.atleast_2d, args))
        used = list(range(len(va)))
        res = [ fn(*a_, **kwargs) for a_ in zip(*va) ]
        return np.array(res).squeeze()
    return vfn


def bootstrap_stat(*arrays, **kwargs):
    """
    This method parallelizes simple bootstrap resampling over the 
    1st axis in the arrays. This can be used only with methods that 
    are vectorized over one dimension (e.g. have an "axis" keyword 
    argument).

    kwargs must include the method keyed by "func"

    func : the method to reduce the sample

    n_boot : number of resampling steps

    assoc_args : arguments are associated? if so, resample the 
                 entire argument tuple at once (default True)

    autovec : naively vectorize a function of 1D args (default False)

    rand_seed : seed for random state

    args : method arguments to concatenate to the array list

    extra : any further arguments are passed directly to func

    """
    # If axis is given as positive it will have to be increased
    # by one slot
    axis = kwargs.setdefault('axis', -1)
    if axis >= 0:
        kwargs['axis'] = axis + 1

    func = kwargs.pop('func', None)
    n_boot = kwargs.pop('n_boot', 1000)
    rand_seed = kwargs.pop('rand_seed', None)
    args = kwargs.pop('args', [])
    splice_args = kwargs.pop('splice_args', None)
    assoc_args = kwargs.pop('assoc_args', True)
    autovec = kwargs.pop('autovec', False)
    if func is None:
        raise ValueError('func must be set')
    if autovec:
        func = _autovectorize(func)
    
    np.random.RandomState(rand_seed)

    b_arrays = list()
    if assoc_args:
        r = len(arrays[0])
        resamp = np.random.randint(0, r, r*n_boot)
    for arr in arrays:
        if not assoc_args:
            r = len(arr)
            resamp = np.random.randint(0, r, r*n_boot)
        b_arr = np.take(arr, resamp, axis=0)
        b_arr.shape = (n_boot, r) + arr.shape[1:]
        b_arrays.append(b_arr)

    if not splice_args:
        # try to determine automatically
        
        test_input = [b[0] for b in b_arrays] + list(args)

        test_output = func(*test_input, **kwargs)
        if isinstance(test_output, tuple):
            outputs = list(range(len(test_output)))
        else:
            outputs = (0,)
    else:
        outputs = splice_args
        print(outputs)

    p_func = split_at(
        split_arg=list(range(len(b_arrays))), 
        splice_at=outputs
        )(func)

    inputs = b_arrays + list(args)
    b_output = p_func(*inputs, **kwargs)
    return b_output
