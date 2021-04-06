import os.path as pt
import itertools
import numpy as np
import scipy.signal as signal

from ecogdata.util import Bunch
from ecogdata.channel_map import ChannelMap
from ecogdata.datastore.h5utils import load_bunch
import ecogdata.parallel.sharedmem as shm
from .units import convert_scale


_transform_lookup = {
    # '2014-08-21_ddc' : 'transforms_2014-08-21_0.5HP.h5',
    # '2014-08-21' : 'transforms_2014-08-21_0.5HP.h5',
    '2014-08-21': 'transforms_2014-08-21.h5',
    '2014-08-21_ddc': 'transforms_2014-08-21.h5',
    '2014-09-23_ddc': 'transforms_2014-09-23.h5',
    '2014-09-23': 'transforms_2014-09-23.h5'
}


def convert_volts_to_amps(v_rec, to_unit='nA', **conv_kws):
    return _convert(v_rec, v_rec.units, to_unit, inverted=True, **conv_kws)


def convert_amps_to_volts(c_rec, to_unit='uV', **conv_kws):
    return _convert(c_rec, c_rec.units, to_unit, **conv_kws)


def tf_to_circuit(b, a):
    b, a = signal.normalize(b, a)
    Rs = b[0]
    Rct = (b[1] / a[1]) - Rs
    Cdl = (a[1] * Rct) ** -1
    return (Rs, Rct, Cdl)


def circuit_to_tf(Rs, Rct, Cdl):
    tau = Rct * Cdl
    b = np.array([Rs, (Rs + Rct) / tau])
    a = np.array([1.0, 1.0 / tau])
    return b, a


def _convert(
        rec, from_unit, to_unit, inverted=False, tfs=(),
        Rct=None, Cdl=None, Rs=None, prewarp=False, **sm_kwargs
):
    from_type = from_unit[-1]
    to_type = to_unit[-1]

    def _override_tf(b, a):
        b, a = signal.normalize(b, a)
        Rs_, Rct_, Cdl_ = tf_to_circuit(b, a)
        b, a = circuit_to_tf(
            (Rs if Rs else Rs_),
            (Rct if Rct else Rct_),
            (Cdl if Cdl else Cdl_)
        )
        return b, a

    if not len(tfs):
        from ecogdata.expconfig import params
        session = rec.name.split('.')[0]
        # backwards compatibility
        session = session.split('/')[-1]
        if session in _transform_lookup:
            # XXX: ideally should find the expected unit scale of the
            # transforms in the transform Bunch -- for now assume pico-scale
            to_scale = convert_scale(1, from_unit, 'p' + from_type)
            from_scale = convert_scale(1, 'p' + to_type, to_unit)
            stash_path = pt.sep.join([params.stash_path,
                                      'devices',
                                      'impedance_transfer'])
            transforms = pt.join(stash_path, _transform_lookup[session])
            tfs = load_bunch(transforms, '/')
        else:
            # do an analog integrator --
            # will be converted to 1 + z / (1 - z) later
            # do this on pico scale also (?)
            to_scale = convert_scale(1, from_unit, 'p' + from_type)
            from_scale = convert_scale(1, 'p' + to_type, to_unit)
            tfs = Bunch(aa=np.array([1, 0]), bb=np.array([0, 1]))
    else:
        to_scale = convert_scale(1, from_unit, 'p' + from_type)
        from_scale = convert_scale(1, 'p' + to_type, to_unit)

    if tfs.aa.ndim > 1:
        from ecogdata.filt.time import bfilter
        conv = rec.deepcopy()
        cmap = conv.chan_map
        bb, aa = smooth_transfer_functions(tfs.bb.T, tfs.aa.T, **sm_kwargs)
        for n, ij in enumerate(zip(*cmap.to_mat())):
            b = bb[ij]
            a = aa[ij]
            b, a = _override_tf(b, a)
            if prewarp:
                T = conv.Fs ** -1
                z, p, k = signal.tf2zpk(b, a)
                z = 2 / T * np.tan(z * T / 2)
                p = 2 / T * np.tan(p * T / 2)
                b, a = signal.zpk2tf(z, p, k)
            if inverted:
                zb, za = signal.bilinear(a, b, fs=conv.Fs)
            else:
                zb, za = signal.bilinear(b, a, fs=conv.Fs)
            bfilter(zb, za, conv.data[n], bsize=10000)
    else:
        from ecogdata.parallel.split_methods import bfilter
        # avoid needless copy of data array
        rec_data = rec.pop('data')
        conv = rec.deepcopy()
        rec.data = rec_data
        conv_data = shm.shared_ndarray(rec_data.shape, rec_data.dtype.char)
        conv_data[:] = rec_data
        b = tfs.bb
        a = tfs.aa
        if prewarp:
            T = conv.Fs ** -1
            z, p, k = signal.tf2zpk(b, a)
            z = 2 / T * np.tan(z * T / 2)
            p = 2 / T * np.tan(p * T / 2)
            b, a = signal.zpk2tf(z, p, k)
        if inverted:
            zb, za = signal.bilinear(a, b, fs=conv.Fs)
        else:
            zb, za = signal.bilinear(b, a, fs=conv.Fs)
        bfilter(zb, za, conv_data, bsize=10000)
        conv.data = conv_data

    conv.data *= (from_scale * to_scale)
    conv.units = to_unit
    return conv


def smooth_transfer_functions(bb, aa, mask_poles=-1, mask_zeros=-1):
    g = aa.shape[:2]
    cm = ChannelMap(np.arange(g[0] * g[1]), g, col_major=False)

    # going to punch out the corners of all maps, as well as ...
    # * poles that are < 1 Hz
    # * zeros that are < 1 Hz

    nzr = bb.shape[-1] - 1
    npl = aa.shape[-1] - 1

    aa = aa.reshape(len(cm), -1)
    bb = bb.reshape(len(cm), -1)

    k = np.zeros(len(aa))
    z = np.zeros((len(aa), nzr))
    p = np.zeros((len(aa), npl))

    m = np.ones(len(aa), dtype='?')
    for n in range(len(aa)):
        try:
            z[n], p[n], k[n] = signal.tf2zpk(bb[n], aa[n])
        except BaseException:
            # if there is a dimension error, because xfer fun is null
            m[n] = False
            pass

    # corners
    m[[0, 7, 64 - 8, 64 - 1]] = False
    if mask_poles > 0:
        m[p.max(1) > -2 * np.pi * mask_poles] = False
    if mask_zeros > 0:
        m[z.min(1) < -2 * np.pi * mask_zeros] = False

    z, p, k = [x[m] for x in (z, p, k)]

    cm = cm.subset(m.nonzero()[0])

    # then the order-of-magnitude maps will be smoothed with a median filter
    z_ = cm.embed(np.log10(-z), axis=0, fill='median')
    p_ = cm.embed(np.log10(-p), axis=0, fill='median')
    k_ = cm.embed(np.log10(k), axis=0, fill='median')

    z = -np.power(10, z_)
    p = -np.power(10, p_)
    k = np.power(10, k_)

    aa_sm = np.zeros(g + (npl + 1,))
    bb_sm = np.zeros(g + (nzr + 1,))
    for ij in itertools.product(range(g[0]), range(g[1])):
        bb_sm[ij], aa_sm[ij] = signal.zpk2tf(z[ij], p[ij], k[ij])

    return bb_sm, aa_sm


def plot_smooth_transforms(bb, aa, w_lo=1e-1, w_hi=1e4, **kwargs):
    import matplotlib.pyplot as pp
    import seaborn as sns
    # Fix until MPL or seaborn gets straightened out
    import warnings
    with warnings.catch_warnings():
        import matplotlib as mpl
        warnings.simplefilter('ignore', mpl.cbook.MatplotlibDeprecationWarning)
        sns.reset_orig()
    bb_s, aa_s = smooth_transfer_functions(bb, aa, **kwargs)

    g = bb.shape[:2]
    fr = np.logspace(np.log10(w_lo), np.log10(w_hi), 200)
    om = fr * 2 * np.pi
    with sns.plotting_context('notebook'), sns.axes_style('whitegrid'):
        f, axs = pp.subplots(*g, sharex=True, sharey=True)
        for i, j in itertools.product(range(g[0]), range(g[1])):
            _, h1 = signal.freqs(bb[i, j], aa[i, j], worN=om)
            _, h2 = signal.freqs(bb_s[i, j], aa_s[i, j], worN=om)
            axs[i, j].loglog(fr, np.c_[np.abs(h1), np.abs(h2)])
    pp.show()
