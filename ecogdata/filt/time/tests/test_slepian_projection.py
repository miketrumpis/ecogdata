from numpy.testing import assert_almost_equal
import numpy as np

from ecogdata.filt.time.projection_filters import \
    slepian_projection, moving_projection, _moving_projection_preserve


def gen_sig(am=False, w0=80, bw=30, nfreqs=10):
    """Generate test signal with given bandwidth. The band is
    shifted to w0 either through amplitude modulation or by
    adding w0 to the zero-centered frequencies.

    The sampling frequency is 1000 Hz.
    """
    freqs = (np.random.rand(10) - 0.5) * bw
    if w0 and not am:
        freqs += w0
    amps = np.random.randn(10)
    phs = np.random.rand(10) * 2 * np.pi
    tx = np.arange(2000) * 2 * np.pi / 1000.
    narrowband = amps[:, None] * np.cos(freqs[:, None]*tx + phs[:, None])
    narrowband = narrowband.sum(0)
    if am:
        modulated = narrowband * np.cos(tx * w0)
        return modulated, narrowband
    return narrowband


def test_baseband_recon():
    """Does the projection filter preserve a lowpass baseband signal?"""
    modulated, baseband = gen_sig(am=True)
    recon = slepian_projection(modulated, 50, 1000.0, w0=80, baseband=True)
    # I guess < 0.1% error is good -- sometimes boundary effects, so
    # check within interior of signal
    err = recon[150:-150] - baseband[150:-150]
    rel_error = np.sum(err ** 2) / np.sum(baseband[150:-150] ** 2)
    assert rel_error < 1e-3


def test_narrowband_recon():
    """Does the projection filter preserve a narrowband signal?"""
    nb = gen_sig()
    nb_est = slepian_projection(nb, 50, 1000.0, w0=80)
    # I guess < 0.1% error is good -- sometimes boundary effects, so
    # check within interior of signal
    err = nb_est[150:-150] - nb[150:-150]
    rel_error = np.sum(err ** 2) / np.sum(nb[150:-150] ** 2)
    assert rel_error < 1e-3


def test_narrowband_rejection():
    """Does the projection filter miss an out-of-band narrowband signal?"""
    nb = gen_sig()
    # Narrow-band signal extends from 65-95 Hz
    # This slepian projector extends from 100 - 200 Hz and
    # should reject the narrow-band signal reasonably well
    nb_est = slepian_projection(nb, 50, 1000.0, w0=150)
    rel_leak = np.sum(nb_est[150:-150] ** 2) / np.sum(nb[150:-150] ** 2)
    assert rel_leak < 1e-3


def test_bandpass_power():
    """Does the projection remove the correct amount of broadband power?"""
    sg = np.random.randn(2000)
    # This bandpass window is from 150 to 250 (and -150 to -250),
    # Given the hypothetical bandwidth of 500 Hz, it should
    # take up ~ 20% of the signal power (which is pretty much unit-valued)
    sg_bp = slepian_projection(sg, 50, 1000, w0=200)
    assert_almost_equal(sg_bp.var(), 0.2, decimal=1)
    # This bandpass window is from -50 to 50,
    # Given the hypothetical bandwidth of 500 Hz, it should
    # take up ~ 10% of the signal power (which is pretty much unit-valued)
    sg_bp = slepian_projection(sg, 50, 1000, w0=0)
    assert_almost_equal(sg_bp.var(), 0.1, decimal=1)


def test_shapes():
    sg_1d = np.random.randn(200)
    sg_2d = np.random.randn(3, 200)
    sg_3d = np.random.randn(2, 2, 200)

    assert slepian_projection(sg_1d, 20, 1000.0).shape == sg_1d.shape
    assert slepian_projection(sg_2d, 20, 1000.0).shape == sg_2d.shape
    assert slepian_projection(sg_3d, 20, 1000.0).shape == sg_3d.shape


def test_projection_return_types():

    x = np.random.randn(2000)

    # basic lowpass
    y1 = slepian_projection(x, 100, 1e3)
    assert y1.dtype not in np.sctypes['complex']

    # basic bandpass twosided
    y1 = slepian_projection(x, 100, 1e3, w0=150)
    assert y1.dtype not in np.sctypes['complex']

    # bandpass onesided
    y1 = slepian_projection(x, 100, 1e3, w0=150, onesided=True)
    assert y1.dtype in np.sctypes['complex']

    # bandpass demodulated
    y1 = slepian_projection(x, 100, 1e3, w0=150, baseband=True)
    assert y1.dtype not in np.sctypes['complex']

    # bandpass onesided
    y1 = slepian_projection(x, 100, 1e3, w0=150, baseband=True, onesided=True)
    assert y1.dtype in np.sctypes['complex']


def test_moving_projection_recon():
    """Does the projection filter preserve a lowpass signal?"""
    nb = gen_sig(w0=0, bw=50, nfreqs=20)
    N = 100
    nb_est = moving_projection(nb, N, 75, Fs=1000.0)
    # I guess < 0.1% error is good -- sometimes boundary effects, so
    # check within interior of signal
    err = nb_est[N:-N] - nb[N:-N]
    rel_error = np.sum(err ** 2) / np.sum(nb[N:-N] ** 2)
    assert rel_error < 1e-3


def test_moving_projection_sizes():
    for M in (200, 1000, 10000):
        x = np.random.randn(M)
        for N in (100, 500, 800):
            if N > M:
                continue
            try:
                y = moving_projection(x, N, 0.1)
                assert len(y) == len(x)
            except:
                assert False, 'shapes failed: M={0}, N={1}'.format(M, N)


def test_twodim_moving_projection():
    """Check consistency in 2-dimensional calculation"""

    x = np.random.randn(2000)
    x = np.row_stack((x,) * 10)
    y = moving_projection(x, 200, 5/200.)

    err = y - y[0]
    assert np.sum(err ** 2) < 1e-8


def test_multidim_moving_projection():
    """Check consistency in multidimensional calculation"""

    x_ = np.random.randn(2000)

    x = np.empty((5, 10, 2000))
    x[:] = x_
    y = moving_projection(x, 200, 5/200.)

    err = y - y[0, 0]
    assert np.sum(err ** 2) < 1e-8


def test_consistency():
    """Check consistency between optimized and regular methods"""

    x = np.random.randn(10000)

    y1 = moving_projection(x, 200, 5/200.)
    y2 = _moving_projection_preserve(x, 200, 5/200.)
    # return y1, y2
    err = y1 - y2
    assert np.sum(err ** 2) < 1e-8


def test_return_types():

    x = np.random.randn(2000)

    y1 = moving_projection(x, 200, 5/200., f0=6/200., baseband=False)
    assert y1.dtype not in np.sctypes['complex']

    y2 = moving_projection(x, 200, 5/200., f0=6/200., baseband=True)
    assert y2.dtype in np.sctypes['complex']
