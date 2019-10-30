"""
One-stop shopping for digital filtering of arrays
"""
import numpy as np
from .design import butter_bp, cheby1_bp, cheby2_bp, notch
from ecogdata.util import get_default_args, input_as_2d
from ecogdata.parallel.array_split import shared_ndarray, shared_copy
import scipy.signal as signal
from nitime.algorithms.autoregressive import AR_est_YW

__all__ = [ 'filter_array', 'notch_all', 'downsample', 'ma_highpass',
            'common_average_regression', 'ar_whiten_blocks',
            'harmonic_projection' ]

def _get_poles_zeros(destype, **filt_args):
    if destype.lower().startswith('butter'):
        return butter_bp(**filt_args)

    des_lookup = dict(cheby1=cheby1_bp, cheby2=cheby2_bp, notch=notch)
    desfun = des_lookup[destype]    
    def_args = get_default_args(desfun)
    extra_arg = [k for k in list(filt_args.keys()) if k not in list(def_args.keys())]
    # should only be one extra key
    if len(extra_arg) > 1:
        raise ValueError('too many arguments for filter type '+destype)
    extra_arg = filt_args.pop( extra_arg.pop() )

    return desfun(extra_arg, **filt_args)

@input_as_2d()
def filter_array(
        arr, ftype='butterworth', inplace=True, out=None, block_filter='parallel',
        design_kwargs=dict(), filt_kwargs=dict()
        ):
    """
    Filter an ND array timeseries on the last dimension. For computational
    efficiency, the timeseries are blocked into partitions (10000 points
    by default) and split over multiple threads (not supported on Windoze).

    Parameters
    ----------
    arr: ndarray
        Timeseries in the last dimension (can be 1D).
    ftype: str
        Filter type to design.
    inplace: bool
        If True, then arr must be a shared memory array. Otherwise a
        shared copy will be made from the input. This is a shortcut for using "out=arr".
    out: ndarray
        If not None, place filter output here (if inplace is specified, any output array is ignored).
    block_filter: str or callable
        Specify the run-time block filter to apply. Can be "parallel" or
        "serial", or can be a callable that follows the basic signature of
        `ecogdata.filt.time.block_filter.bfilter`.
    design_kwargs: dict
        Design parameters for the filter (e.g. lo, hi, Fs, ord)
    filt_kwargs: dict
        Processing parameters (e.g. filtfilt, bsize)

    Returns
    -------
    arr_f: ndarray
        Filtered timeseries, same shape as input.
    
    """
    b, a = _get_poles_zeros(ftype, **design_kwargs)
    if isinstance(block_filter, str):
        if block_filter.lower() == 'parallel':
            from ecogdata.parallel.split_methods import bfilter
            block_filter = bfilter
        elif block_filter.lower() == 'serial':
            from .blocked_filter import bfilter
            block_filter = bfilter
        else:
            raise ValueError('Block filter type not known: {}'.format(block_filter))
    if not callable(block_filter):
        raise ValueError('Provided block filter is not callable: {}'.format(block_filter))
    def_args = get_default_args(block_filter)
    # Set some defaults and then update with filt_kwargs
    def_args['bsize'] = 10000
    def_args['filtfilt'] = True
    def_args.update(filt_kwargs)
    def_args['out'] = out
    if inplace:
        # enforce that def_args['out'] is arr?
        def_args['out'] = arr
        block_filter(b, a, arr, **def_args)
        return arr
    else:
        # still use bfilter for memory efficiency
        if def_args['out'] is None:
            arr_f = shared_ndarray(arr.shape, arr.dtype.char)
            def_args['out'] = arr_f
        block_filter(b, a, arr, **def_args)
        # raise
        return def_args['out']

def notch_all(
        arr, Fs, lines=60.0, nzo=3,
        nwid=3.0, inplace=True, nmax=None, **filt_kwargs
        ):
    """Apply notch filtering to a array timeseries.

    Parameters
    ----------
    arr : ndarray
        timeseries
    Fs : float
        sampling frequency
    lines : [list of] float(s)
        One or more lines to notch.
    nzo : int (default 3)
        Number of zeros for the notch filter (more zeros --> deeper notch).
    nwid : float (default 3)
        Affects distance of notch poles from zeros (smaller --> closer).
        Zeros occur on the unit disk in the z-plane. Note that the 
        stability of a digital filter depends on poles being within
        the unit disk.
    nmax : float (optional)
        If set, notch all multiples of (scalar-valued) lines up to nmax.

    Returns
    -------
    notched : ndarray
    
    """

    if inplace:
        # If filtering inplace, then set the output array as such
        filt_kwargs['out'] = arr
    elif filt_kwargs.get('out', None) is None:
        # If inplace is False and no output array is set,
        # then make the array copy here and do inplace filtering on the copy
        arr_f = shared_copy(arr)
        arr = arr_f
        filt_kwargs['out'] = arr
    # otherwise an output array is set

    if isinstance(lines, (float, int)):
        # repeat lines until nmax
        nf = lines
        if nmax is None:
            nmax = nf
        nmax = min(nmax, Fs / 2.0)
        lines = [nf * i for i in range(1, int(nmax // nf) + 1)]
    else:
        lines = [x for x in lines if x < Fs/2]

    notch_defs = get_default_args(notch)
    notch_defs['nwid'] = nwid
    notch_defs['nzo'] = nzo
    notch_defs['Fs'] = Fs
    for nf in lines:
        notch_defs['fcut'] = nf
        arr_f = filter_array(arr, 'notch', inplace=False, design_kwargs=notch_defs, **filt_kwargs)
    return arr_f

def downsample(x, fs, appx_fs=None, r=None, filter_inplace=False):
    """Integer downsampling with antialiasing.

    One (and only one) of the parameters 'appx_fs' and 'r' must be
    specified to determine the downsample factor.

    The anti-aliasing filter is a type-1 Chebyshev filter with small
    passband ripple and monotonic decreasing stopband attenuation:

    * pass-band corner: 0.4 * new_fs
    * stop-band corner: 0.5 * new_fs
    * passband ripple: 0.5 dB
    * Nyquist attenuation: 20 dB

    Parameters
    ----------
    x: ndarray
        timeseries
    fs: float
        Original sampling frequency
    appx_fs: float
        Approximate resampling frequency. The timeseries will be
        downsampled by an integer amount to meet or exceed this
        sampling rate.
    r: int
        The integer downsampling rate
    filter_inplace: bool
        If True, attempt to filter in-place. This can be done if x is shared memory. This will modify x.

    Returns
    -------
    y : ndarray
        Downsampled timeseries
    new_fs : float
        New sampling rate

    """
    if appx_fs is None and r is None:
        return x
    if appx_fs is not None and r is not None:
        raise ValueError('only specify new fs or resample factor, not both')

    if appx_fs is not None:
        # new sampling interval must be a multiple of old sample interval,
        # so find the closest match that is >= appx_fs
        r = int( np.ceil(fs / appx_fs) )

    
    num_pts = x.shape[-1] // r
    num_pts += int( ( x.shape[-1] - num_pts*r ) > 0 )

    new_fs = fs / r

    # design a cheby-1 lowpass filter 
    # wp: 0.4 * new_fs
    # ws: 0.5 * new_fs
    # design specs with halved numbers, since filtfilt will be used
    wp = 2 * 0.4 * new_fs / fs
    ws = 2 * 0.5 * new_fs / fs
    ord, wc = signal.cheb1ord(wp, ws, 0.25, 10)
    fdesign = dict(ripple=0.25, hi=0.5 * wc * fs, Fs=fs, ord=ord)
    x_lp = filter_array(
        x, ftype='cheby1', inplace=filter_inplace, design_kwargs=fdesign
        )
    
    x_ds = x_lp[..., ::r].copy()
    return x_ds, new_fs

def upsample(x, fs, appx_fs=None, r=None, interp='sinc'):
    if appx_fs is None and r is None:
        return x
    if appx_fs is not None and r is not None:
        raise ValueError('only specify new fs or resample factor, not both')

    if appx_fs is not None:
        # new sampling interval must be a multiple of old sample interval,
        # so find the closest match that is <= appx_fs
        r = int( np.floor(appx_fs / fs) )

    x_up = shared_ndarray(x.shape[:-1] + (r * x.shape[-1],), x.dtype.char)
    x_up[..., ::r] = x

    new_fs = fs * r

    from ecogdata.filt.blocks import BlockedSignal
    from scipy.fftpack import fft, ifft, fftfreq
    from scipy.interpolate import interp1d

    if interp == 'sinc':
        x_up *= r
        op_size = 2**16
        xb = BlockedSignal(x_up, op_size, partial_block=True)
        for block in xb.fwd():
            Xf = fft(block, axis=-1)
            freq = fftfreq(Xf.shape[-1]) * new_fs
            Xf[..., np.abs(freq) > fs/2.0] = 0
            block[:] = ifft(Xf).real
        return x_up, new_fs
    elif interp == 'linear':
        # always pick up the first and last sample when skipping by r
        op_size = r * 10000 + 1
        t = np.arange(op_size)
        xb = BlockedSignal(x_up, op_size, overlap=1, partial_block=True)
        for block in xb.fwd():
            N = block.shape[-1]
            fn = interp1d(t[:N][::r], block[..., ::r], axis=-1,
                          bounds_error=False, fill_value=0,
                          assume_sorted=True)
            block[:] = fn(t[:N])
        return x_up, new_fs
    
    # design a cheby-1 lowpass filter 
    # wp: 0.4 * new_fs
    # ws: 0.5 * new_fs
    # design specs with halved numbers, since filtfilt will be used
    wp = 2 * 0.4 * fs / new_fs
    ws = 2 * 0.5 * fs / new_fs
    ord, wc = signal.cheb1ord(wp, ws, 0.25, 10)
    fdesign = dict(ripple=0.25, hi=0.5 * wc * new_fs, Fs=new_fs, ord=ord)
    filter_array(x_up, ftype='cheby1', inplace=True, design_kwargs=fdesign)
    x_up *= r
    return x_up, new_fs
    
def ma_highpass(x, fc, progress=False, fir_filt=False):
    """
    Implement a stable FIR highpass filter using a moving average.
    """

    from ecogdata.parallel.split_methods import convolve1d
    n = int(round(fc ** -1.0))
    if not n%2:
        n += 1
    h = np.empty(n)
    h.fill( -1.0 / n )
    h[n//2] += 1
    return convolve1d(x, h)
    if fir_filt:
        return h

@input_as_2d()
def common_average_regression(data, mu=(), inplace=True):
    """
    Return the residual of each channel after regressing a 
    common signal (by default the channel-average).
    """
    if not len(mu):
        mu = data.mean(0)
    beta = data.dot(mu) / np.sum( mu**2 )
    data_r = data if inplace else data.copy()
    for chan, b in zip(data_r, beta):
        chan -= b * mu
    return data_r


@input_as_2d()
def ar_whiten_blocks(blocks, p=50):
    """AR(p) Autoregressive whitening of timeseries blocks.
    """
    bw = np.empty_like(blocks)
    for n in range(len(blocks)):
        b, _ = AR_est_YW(blocks[n], p)
        bw[n] = signal.lfilter(np.r_[1, -b], [1], blocks[n])
    return bw

@input_as_2d()
def harmonic_projection(data, f0, stdevs=2):
    """Harmonic artifact cancellation through direct sinusoid projections.

    This method attempts a robust projection of a signal's line noise
    onto a single-frequency ("atomic") complex exponential. To avoid fitting
    signal to the line atom, high amplitude samples are masked.

    Parameters
    ----------
    data : ndarray
        Timeseries
    f0 : float
        Line frequency (in normalized frequency).
    stdevs : float
        Threshold for amplitude masking in multiples of the standard deviation.

    Returns
    -------
    y : ndarray

    Note
    ----
    This method is best applied to short-ish intervals.

    """
    
    n = data.shape[-1]
    sigma = data.std(1)
    m_2d = np.abs(data) > stdevs*sigma[:,None]
    data = np.ma.masked_array(data, m_2d)
    cs = np.cos(2*np.pi*f0*np.arange(n))
    sn = np.sin(2*np.pi*f0*np.arange(n))
    alpha = data.dot(cs) / (0.5 * n)
    beta = data.dot(sn) / (0.5 * n)
    h = alpha[:,None] * cs + beta[:,None] * sn
    return data.data - h.data
