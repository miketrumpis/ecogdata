"""
Simple filter design wrappings
"""
import numpy as np
import scipy.signal as signal
from numpy import poly

__all__ = [ 'butter_bp', 
            'cheby1_bp', 
            'cheby2_bp', 
            'notch',
            'ellip_bp',
            'savgol',
            'plot_filt',
            'continuous_amplitude_linphase' ]

def _bandpass_params(lo, hi):
    (lo, hi) = list(map(float, (lo, hi)))
    if not (lo > 0 or hi > 0):
        raise ValueError('no cutoff frequencies set')
    if lo and not hi > 0:
        return lo, 'highpass'
        ## return sig.filter_design.butter(
        ##     ord, 2 * lo / Fs, btype='highpass'
        ##     )
    if hi and not lo > 0:
        return hi, 'lowpass'
        ## return sig.filter_design.butter(
        ##     ord, 2 * hi / Fs, btype='lowpass'
        ##     )
    return np.array([lo, hi]), 'bandpass'

def butter_bp(lo=0, hi=0, Fs=2.0, ord=3):

    # note: "lo" corresponds to highpass cutoff
    #       "hi" corresponds to lowpass cutoff
    freqs, btype = _bandpass_params(lo, hi)
    return signal.butter(ord, 2*freqs/Fs, btype=btype)
    
def cheby1_bp(ripple, lo=0, hi=0, Fs=2.0, ord=3):
    freqs, btype = _bandpass_params(lo, hi)
    return signal.cheby1(ord, ripple, 2*freqs/Fs, btype=btype)
    
def cheby2_bp(rstop, lo=0, hi=0, Fs=2.0, ord=3):
    freqs, btype = _bandpass_params(lo, hi)
    return signal.cheby2(ord, rstop, 2*freqs/Fs, btype=btype)

def ellip_bp(atten, ripple, lo=0, hi=0, hp_width=0, lp_width=0, Fs=2.0):
    if hp_width == 0 and lo > 0:
        hp_width = 0.1 * (hi - lo)
        if lo - hp_width <= 0:
            # set hp_width to halfway between 0 and lo
            hp_width = 0.5 * lo
            print('bad HP stopband, adjusting to {0:.1f}'.format(hp_width))
    if lp_width == 0 and hi > 0:
        lp_width = 0.1 * (hi - lo)
        if hi + lp_width >= Fs/2:
            # make lp_width to halfway between hi and Nyquist
            lp_width = 0.5 * (Fs/2 - hi)
            print('bad LP stopband, adjusting to {0:.1f}'.format(lp_width))
    if lo > 0 and hi > 0:
        # bandpass design
        wp = np.array([lo, hi]) * 2 / Fs
        ws = np.array([lo - hp_width, hi + lp_width]) * 2 / Fs
        btype = 'bandpass'
    elif lo > 0:
        # highpass design
        wp = 2 * lo / Fs
        ws = 2 * (lo-hp_width) / Fs
        btype = 'highpass'
    elif hi > 0:
        # lowpass design
        wp = 2 * hi / Fs
        ws = 2 * (hi + lp_width) / Fs
        btype = 'lowpass'

    order, wn = signal.ellipord(wp, ws, ripple, atten)
    return signal.ellip(order, ripple, atten, wn, btype=btype)

def savgol(T, ord=3, Fs=1.0, smoothing=True):
    """Return a Savitzky-Golay FIR filter for smoothing or residuals.

    Parameters
    ----------
    T : float
        Window length (in seconds if sampling rate is given).
    ord : int
        Order of local polynomial.
    Fs : float
        Sampling rate
    smoothing : bool
        Filter coefficients yield local polynomial fits by default.
        Alternatively, the local polynomial can be subtracted from
        the signal if smoothing==False

    Returns
    -------
    b : array
        FIR filter
    a : constant (1)
        denominator of filter polynomial
    
    """

    N = int(T * Fs)
    N += 1 - N % 2
    b = signal.savgol_coeffs(N, ord)
    if not smoothing:
        b *= -1
        # midpoint add one?
        b[ N//2 ] += 1
        #b[0] += 1
    return b, 1

def notch(fcut, Fs=2.0, nwid=3.0, npo=None, nzo=3):

    f0 = fcut * 2 / Fs
    fw = nwid * 2 / Fs

    z = [np.exp(1j*np.pi*f0), np.exp(-1j*np.pi*f0)]
    
    # find the polynomial with the specified (multiplicity of) zeros
    b = poly( np.array( z * int(nzo) ) )
    # the polynomial with the specified (multiplicity of) poles
    if npo is None:
        npo = nzo
    a = poly( (1-fw) * np.array( z * int(npo) ) )
    return (b, a)

def plot_filt(
        b, a, Fs=2.0, n=2048, log=True, logx=False, db=False, 
        filtfilt=False, phase=False, ax=None, minx=1e-5, maxx=None, **plot_kws
        ):
    import matplotlib.pyplot as plt
    if maxx is None:
        maxx = Fs / 2
    if logx:
        hi = np.log10(maxx)
        lo = np.log10(minx)
        w = np.logspace(lo, hi, n)
    else:
        w = np.linspace(minx, maxx, n)
    _, f = signal.freqz(b, a, worN=w * (2*np.pi/Fs))
    if ax:
        plt.sca(ax)
        fig = ax.figure
    else:
        fig = plt.figure()
    if filtfilt:
        m = np.abs(f)**2
    else:
        m = np.abs(f)

    if log and db:
        # assume dB actually preferred
        log = False

    if db:
        m = 20*np.log(m)
    if logx and log:
        plt.loglog( w, m, **plot_kws )
        plt.ylabel('Magnitude')
    elif log:
        plt.semilogy( w, m, **plot_kws )
        plt.ylabel('Magnitude')
    elif logx:
        plt.semilogx( w, m, **plot_kws )
        plt.ylabel('Magnitude (dB)' if db else 'Magnitude')
    else:
        plt.plot( w, m, **plot_kws )
        plt.ylabel('Magnitude (dB)' if db else 'Magnitude')
    plt.xlabel('Frequency (Hz)')
    plt.title('Frequency response' + (' (filtfilt)' if filtfilt else ''))
    if not filtfilt and phase:
        plot_kws['ls'] = '--'
        ax2 = plt.gca().twinx()
        ax2.plot( w, np.angle(f), **plot_kws )
        ax2.set_ylabel('radians')
    return fig


def continuous_amplitude_linphase(ft_samps):
    """Given Fourier Transform samples of a linear phase system,
    return functions of amplitude and phase such that the amplitude
    function is continuous (ie, not a magnitude function), and that
    f(e) = a(e)exp(j*p(e))

    Parameters
    ----------

    ft_samps: ndarray
      N complex samples of the fourier transform

    Returns
    -------

    (a, p): ndarray
      (continuous) amplitude and phase functions
    """
    npts = len(ft_samps)
    p_jumps = np.unwrap(np.angle(ft_samps))
    p_diff = np.diff(p_jumps)
    # assume there is not a filter zero at point 0 or 1
    p_slope = p_diff[0]
    zeros = np.where(np.pi - (p_diff-p_slope) <= (np.pi-1e-5))[0] + 1
    zeros = np.where(np.abs(p_diff-p_slope) >= 1)[0] + 1
                     
    zeros = np.r_[zeros, npts]

    # now get magnitude from ft_samps
    # assumption: amplitude from 0 to first filter zero is positive 
    a = np.abs(ft_samps)
    psign = np.sign(p_slope)
    k=1
    for lower, upper in zip(zeros[:-1], zeros[1:]):
        a[lower:upper] *= np.power(-1, k)
        p_jumps[lower:upper] += k*psign*np.pi
        k += 1

    return a, p_jumps
