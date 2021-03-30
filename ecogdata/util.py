# ye olde utilities module
import os
import copy
import errno
from warnings import warn
from glob import glob
from decorator import decorator
import inspect
from contextlib import contextmanager
import numpy as np
from scipy.ndimage import convolve1d
from scipy.integrate import simps
from scipy.signal.windows import dpss


# ye olde Bunch object
class Bunch(dict):
    def __init__(self, *args, **kw):
        dict.__init__(self, *args, **kw)
        self.__dict__ = self

    def __repr__(self):
        k_rep = list(self.keys())
        if not len(k_rep):
            return 'an empty Bunch'
        v_rep = [str(type(self[k])) for k in k_rep]
        mx_c1 = max([len(s) for s in k_rep])
        mx_c2 = max([len(s) for s in v_rep])
        table = ['{0:<{col1}} : {1:<{col2}}'.format(k, v, col1=mx_c1, col2=mx_c2)
                 for (k, v) in zip(k_rep, v_rep)]
        table = '\n'.join(table)
        return table.strip()

    def __copy__(self):
        d = dict([(k, copy.copy(v)) for k, v in list(self.items())])
        return Bunch(**d)

    def copy(self):
        return copy.copy(self)

    def __deepcopy__(self, memo):
        d = dict([(k, copy.deepcopy(v)) for k, v in list(self.items())])
        return Bunch(**d)

    def deepcopy(self):
        return copy.deepcopy(self)


# Matrix-indexing manipulations
def flat_to_mat(mn, idx, col_major=True):
    idx = np.asarray(idx, dtype='l')
    # convert a flat matrix index into (i,j) style
    (m, n) = mn if col_major else mn[::-1]
    if (idx < 0).any() or (idx >= m * n).any():
        raise ValueError(
            'The flat index does not lie inside the matrix: ' + str(mn)
        )
    j = idx // m
    i = (idx - j * m)
    return (i, j) if col_major else (j, i)


def mat_to_flat(mn, i, j, col_major=True):
    i = np.asarray(i, dtype='l')
    j = np.asarray(j, dtype='l')
    if (i < 0).any() or (i >= mn[0]).any() \
            or (j < 0).any() or (j >= mn[1]).any():
        raise ValueError('The matrix index does not fit the geometry: ' + str(mn))
    # covert matrix indexing to a flat (linear) indexing
    (fast, slow) = (i, j) if col_major else (j, i)
    block = mn[0] if col_major else mn[1]
    idx = slow * block + fast
    return idx


def flat_to_flat(mn, idx, col_major=True):
    # convert flat indexing from one convention to another
    i, j = flat_to_mat(mn, idx, col_major=col_major)
    return mat_to_flat(mn, i, j, col_major=not col_major)


# Introspection: use inspect.signature to address numpy issue #gh-12225
def get_default_args(func):
    """
    returns a dictionary of arg_name:default_values for the input function
    """
    params = inspect.signature(func).parameters
    kw_params = [p for p in params if params[p].default is not inspect.Parameter.empty]
    if not kw_params:
        return ()
    defaults = [params[p].default for p in kw_params]
    return dict(zip(kw_params, defaults))


# Path trick from SO:
# http://stackoverflow.com/questions/600268/mkdir-p-functionality-in-python
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def nextpow2(n):
    l2 = np.log2(n)
    return int(2 ** (int(l2) + 1 if l2 != int(l2) else l2))


# Reshape into categories with possible padding
def equalize_groups(x, group_sizes, axis=0, fill=np.nan, reshape=True):

    mx_size = max(group_sizes)
    n_groups = len(group_sizes)
    steps = np.r_[0, np.cumsum(group_sizes)]
    new_shape = list(x.shape)
    new_shape[axis] = n_groups * mx_size
    if np.prod(x.shape) == np.prod(new_shape):
        # already has consistent size for equalized groups
        if reshape:
            new_shape[axis] = n_groups
            new_shape.insert(axis + 1, mx_size)
            return x.reshape(new_shape)
        return x
    if x.shape[axis] != steps[-1]:
        raise ValueError('axis {0} in x has wrong size'.format(axis))
    if all([g == mx_size for g in group_sizes]):
        if reshape:
            new_shape[axis] = n_groups
            new_shape.insert(axis + 1, mx_size)
            x = x.reshape(new_shape)
        return x
    y = np.empty(new_shape, dtype=x.dtype)
    new_shape[axis] = n_groups
    new_shape.insert(axis + 1, mx_size)
    y = y.reshape(new_shape)
    y.fill(fill)
    y_slice = [slice(None)] * len(new_shape)
    x_slice = [slice(None)] * len(x.shape)
    for g in range(n_groups):
        y_slice[axis] = g
        y_slice[axis + 1] = slice(0, group_sizes[g])
        x_slice[axis] = slice(steps[g], steps[g + 1])
        y[tuple(y_slice)] = x[tuple(x_slice)]
    if not reshape:
        new_shape[axis] *= mx_size
        new_shape.pop(axis + 1)
        y = y.reshape(new_shape)
    return y


def fenced_out(samps, quantiles=(25, 75), thresh=3.0, axis=None, fences='both'):
    """
    Threshold input sampled based on Tukey's box-plot heuristic. An
    outlier is a value that lies beyond some multiple of of an
    inter-percentile range (3 multiples of the inter-quartile range
    is default). If the sample has an inter-percentile range of zero,
    then the sample median is substituted.


    Parameters
    ----------
    samps: ndarray
        Samples in n dimensions.
    quantiles: tuple
        Quantiles to define the nominal range: usually (25, 75) for inter-quartile
    thresh: float
        This multiples the normal sample range to define the limits of the outer fences.
    axis: int or None
        By default (None), use the entire sample (in all dimensions) to find outliers. If
        an axis is specified, then repeat the outlier detection separately for all 1D samples
        that slice along that dimension.
    fences: str
        By default, reject samples outside of the upper and lower fences (fences='both'). If
        fences='lower' or fences='upper', then only detect outliers that are too low or too high.

    Returns
    -------
    mask: ndarray
        A binary mask of the same size as "samps", where True indicates an inlier and False an outlier.

    """
    samps = np.asanyarray(samps)
    thresh = float(thresh)
    if isinstance(samps, np.ma.MaskedArray):
        samps = samps.filled(np.nan)

    oshape = samps.shape

    if axis is None:
        # do pooled distribution
        samps = samps.ravel()
    else:
        # roll axis of interest to the end
        samps = np.rollaxis(samps, axis, samps.ndim)

    quantiles = list(map(float, quantiles))
    q_lo, q_hi = np.nanpercentile(samps, quantiles, axis=-1)
    extended_range = thresh * (q_hi - q_lo)
    if (extended_range == 0).any():
        warn('Equal percentiles: estimating outlier range from median value', RuntimeWarning)
        m = (extended_range > 0).astype('d')
        md_range = thresh * q_hi
        extended_range = extended_range * m + md_range * (1 - m)

    high_cutoff = q_hi + extended_range / 2
    low_cutoff = q_lo - extended_range / 2
    if fences.lower() not in ('lower', 'upper', 'both'):
        warn('"fences" value not understood: marking all outliers', RuntimeWarning)
        fences = 'both'
    check_lo = fences.lower() in ('lower', 'both')
    check_hi = fences.lower() in ('upper', 'both')
    # don't care about warnings about comparing nans to reals
    with np.errstate(invalid='ignore'):
        out_mask = np.ones(samps.shape, dtype='?')
        if check_hi:
            out_mask &= samps < high_cutoff[..., None]
        if check_lo:
            out_mask &= samps > low_cutoff[..., None]
    # be sure to reject nans as well
    out_mask &= ~np.isnan(samps)

    if axis is None:
        out_mask.shape = oshape
    else:
        out_mask = np.rollaxis(out_mask, samps.ndim - 1, axis)
    return out_mask


def search_results(path, filter=''):
    from ecogdata.datastore.h5utils import load_bunch
    filter = filter + '*.h5'
    existing = sorted(glob(os.path.join(path, filter)))
    if existing:
        print('Precomputed results exist:')
        for n, path in enumerate(existing):
            print('\t(%d)\t%s' % (n, path))
        mode = input(
            'Enter a choice to load existing work,'
            'or (c) to compute new results: '
        )
        try:
            return load_bunch(existing[int(mode)], '/')
        except ValueError:
            return Bunch()


def input_as_2d(in_arr=0, out_arr=0):
    """
    A decorator to reshape input to be 2D and then bring output back
    to original size (possibly with loss of last dimension).
    Vectors will also be reshaped to (1, N) on input.

    Parameters
    ----------
    in_arr : int (sequence)
        position of argument(s) (input) to be reshaped
    out_arr : int (sequence)
        Non-negative position(s) of output to be reshaped.
        If None, then no output is reshaped.
        If "all", then all outputs are reshaped.
        If the method has a single return value, it is reshaped.

    """

    if not np.iterable(in_arr):
        in_arr = (in_arr,)
    if out_arr is not None and not isinstance(out_arr, str) and not np.iterable(out_arr):
        out_arr = (out_arr,)

    @decorator
    def _wrap(fn, *args, **kwargs):
        args = list(args)
        shapes = list()
        for p in in_arr:
            x = args[p]
            # if it's not an ndarray, then hope for the best
            if isinstance(x, np.ndarray):
                shp = x.shape
                x = x.reshape(-1, shp[-1])
            else:
                return fn(*args, **kwargs)
            args[p] = x
            shapes.append(shp)
        r = fn(*args, **kwargs)
        if out_arr is None or r is None:
            return r
        # Need to be avoid re-assigning the variable from another scope,
        # so using _out_arr locally
        if isinstance(out_arr, str) and out_arr.lower() == 'all':
            _out_arr = tuple(range(len(r)))
        else:
            _out_arr = out_arr
        if not isinstance(r, tuple):
            _out_array = (0,)
            r = (r,)
        returns = list()
        for i in range(len(r)):
            if i not in _out_arr:
                returns.append(r[i])
                continue
            x = r[i]
            n_out = len(x.shape)
            # relying on the 1st encountered shape to represent the output shape
            shp = shapes[0]
            # check to see if the function ate the last dimension
            if n_out < 2:
                shp = shp[:-1]
            elif shp[-1] != x.shape[-1]:
                shp = shp[:-1] + (x.shape[-1],)
            x = x.reshape(shp)
            returns.append(x)
        if len(returns) < 2:
            return returns[0]
        return tuple(returns)
    return _wrap


class ToggleState:
    """
    A callable that flips an internal state within the scope of a with-statement context (subject to a possible hard
    over-ride)
    """

    def __init__(self, init_state=True, name='', permanent_state=None):
        self.__permstate = permanent_state
        if self.__permstate is not None:
            self.state = self.__permstate
        else:
            self.state = init_state
        self.name = name

    @contextmanager
    def __call__(self, status=None):
        prev_status = self.state
        if self.__permstate is not None:
            warn('Object {} state is permanently {}. '
                 'The present context has not changed the state.'.format(self.name, self.__permstate),
                 RuntimeWarning)
            self.state = self.__permstate
        elif status is not None:
            self.state = status
        else:
            self.state = not prev_status
        try:
            yield
        finally:
            self.state = prev_status

    def __bool__(self):
        return self.state


def ndim_prctile(x, p, axis=0):
    xs = np.sort(x, axis=axis)
    dim = xs.shape[axis]
    idx = np.round(float(dim) * np.asarray(p) / 100).astype('i')
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
        return (x - mn) / (mx - mn)

    x = np.rollaxis(x, axis)
    mn = x.min(axis=-1)
    mx = x.max(axis=-1)
    while mn.ndim > 1:
        mn = mn.min(axis=-1)
        mx = mx.max(axis=-1)

    slicer = [slice(None)] + [np.newaxis] * (x.ndim - 1)
    x = x - mn[slicer]
    x = x / (mx - mn)[slicer]
    return np.rollaxis(x, 0, axis + 1)


def density_normalize(x, raxis=None):
    if raxis is None:
        return x / np.nansum(x)

    if x.ndim > 2:
        shape = x.shape
        if raxis not in (0, -1, x.ndim - 1):
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

    mn = min(null_samp.min(), sig_samp.min())
    mx = max(null_samp.max(), sig_samp.max())

    # thresh = np.linspace(mn, mx, n)
    thresh = np.union1d(null_samp, sig_samp)

    # false pos is proportion of samps in null samp that are > thresh
    false_hits = (null_samp >= thresh[:, None]).astype('d')
    false_pos = np.sum(false_hits, axis=1) / len(null_samp)

    # true pos is proportion of samps in sig samp that are > thresh
    hits = (sig_samp >= thresh[:, None]).astype('d')
    true_pos = np.sum(hits, axis=1) / len(sig_samp)

    return np.row_stack((false_pos[::-1], true_pos[::-1]))


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


def dpss_windows(N, NW, K):
    """
    Convenience wrapper of scipy's DPSS window method that always returns K orthonormal tapers and eigenvalues.

    Parameters
    ----------
    N: int
        Sequence length
    NW: float
        Sequence time-frequency product (half integers).
    K: int
        Number of DPSS to calculate (typically <= 2 * NW)

    Returns
    -------
    dpss: ndarray
        (K, N) orthonormal tapers
    eigs: ndarray
        K eigenvalues (bandpass concentration ratios)

    """
    vecs, eigs = dpss(N, NW, Kmax=int(K), sym=False, norm=2, return_ratios=True)
    return vecs, eigs


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
    order_range = list(range(order + 1))
    half_window = (window_size - 1) // 2
    # precompute coefficients
    # b = np.mat(
    # [[k**i for i in order_range]
    # for k in range(-half_window, half_window+1)]
    # )
    ## m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    ix = np.arange(-half_window, half_window + 1, dtype='d')
    bt = np.array([np.power(ix, k) for k in order_range])
    if np.iterable(deriv):
        scl = [rate ** d * factorial(d) for d in deriv]
        scl = np.array(scl).reshape(len(deriv), 1)
    else:
        scl = rate ** deriv * factorial(deriv)
    m = np.linalg.pinv(bt.T)[deriv] * scl

    if m.ndim == 2:
        ys = [convolve1d(y, mr[::-1], mode='constant', axis=axis) for mr in m]
        return np.array(ys)

    return convolve1d(y, m[::-1], mode='constant', axis=axis)


def _autovectorize(fn, exclude_after=-1):
    def vfn(*args, **kwargs):
        kwargs.pop('axis', None)
        va = list(map(np.atleast_2d, args))
        used = list(range(len(va)))
        res = [fn(*a_, **kwargs) for a_ in zip(*va)]
        return np.array(res).squeeze()

    return vfn


# def bootstrap_stat(*arrays, **kwargs):
#     """
#     This method parallelizes simple bootstrap resampling over the
#     1st axis in the arrays. This can be used only with methods that
#     are vectorized over one dimension (e.g. have an "axis" keyword
#     argument).
#
#     kwargs must include the method keyed by "func"
#
#     func : the method to reduce the sample
#
#     n_boot : number of resampling steps
#
#     assoc_args : arguments are associated? if so, resample the
#                  entire argument tuple at once (default True)
#
#     autovec : naively vectorize a function of 1D args (default False)
#
#     rand_seed : seed for random state
#
#     args : method arguments to concatenate to the array list
#
#     extra : any further arguments are passed directly to func
#
#     """
#     # If axis is given as positive it will have to be increased
#     # by one slot
#     axis = kwargs.setdefault('axis', -1)
#     if axis >= 0:
#         kwargs['axis'] = axis + 1
#
#     func = kwargs.pop('func', None)
#     n_boot = kwargs.pop('n_boot', 1000)
#     rand_seed = kwargs.pop('rand_seed', None)
#     args = kwargs.pop('args', [])
#     splice_args = kwargs.pop('splice_args', None)
#     assoc_args = kwargs.pop('assoc_args', True)
#     autovec = kwargs.pop('autovec', False)
#     if func is None:
#         raise ValueError('func must be set')
#     if autovec:
#         func = _autovectorize(func)
#
#     np.random.RandomState(rand_seed)
#
#     b_arrays = list()
#     if assoc_args:
#         r = len(arrays[0])
#         resamp = np.random.randint(0, r, r * n_boot)
#     for arr in arrays:
#         if not assoc_args:
#             r = len(arr)
#             resamp = np.random.randint(0, r, r * n_boot)
#         b_arr = np.take(arr, resamp, axis=0)
#         b_arr.shape = (n_boot, r) + arr.shape[1:]
#         b_arrays.append(b_arr)
#
#     if not splice_args:
#         # try to determine automatically
#
#         test_input = [b[0] for b in b_arrays] + list(args)
#
#         test_output = func(*test_input, **kwargs)
#         if isinstance(test_output, tuple):
#             outputs = list(range(len(test_output)))
#         else:
#             outputs = (0,)
#     else:
#         outputs = splice_args
#         print(outputs)
#
#     p_func = split_at(
#         split_arg=list(range(len(b_arrays))),
#         splice_at=outputs
#     )(func)
#
#     inputs = b_arrays + list(args)
#     b_output = p_func(*inputs, **kwargs)
#     return b_output
