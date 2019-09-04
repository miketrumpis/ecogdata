from ecogdata.parallel.split_methods import multi_taper_psd
from ecogdata.filt.time import *
from ecogdata.util import nextpow2, fenced_out, get_default_args

from .plot_util import filled_interval, light_boxplot
from .colormaps import nancmap
from ecogdata.devices.units import nice_unit_text

import seaborn as sns

sns.reset_orig()

psd_colors = ["#348ABD", "#A60628"]


# complementary colors are: #BD6734, #06A684

########################################################################
## -------- Utility functions: computation, plotting, arranging --------

def band_power(f, pf, fc=None, root_hz=True):
    # xxx: freq axis must be last axis
    # and assuming real and one-sided
    f_mask = f < fc if fc else slice(None)
    p_slice = pf[..., f_mask]
    if root_hz:
        igrl = np.nansum(p_slice ** 2, axis=-1)
    else:
        igrl = np.nansum(p_slice, axis=-1)

    # apply df (assuming regular freq grid)
    df = f[1] - f[0]
    return igrl * df


def bad_channel_mask(pwr_est, iqr=4.0, **kwargs):
    # estimated power should be log-transformed
    if np.all(pwr_est >= 0):
        pwr_est = np.log(pwr_est)

    ## psds = np.exp(np.log(psds).mean(1))
    ## pwr_est = np.log( psds[..., fx<200].sum(-1) )

    ### Channel selection

    # first off, throw out anything pushing numerical precision
    m = pwr_est > np.log(1e-8)
    # automatically reject anything 2 orders of magnitude
    # lower than the median broadband power
    m[pwr_est < (np.median(pwr_est[m]) - np.log(1e2))] = False
    # now also apply a fairly wide outlier rejection
    kwargs.setdefault('quantiles', (25, 75))
    kwargs.setdefault('thresh', iqr)
    kwargs.setdefault('low', True)
    msub = fenced_out(pwr_est[m], **kwargs)
    m[m] = msub
    return m


def plot_psds(
        f, gf, df, fc, title, ylims=(), root_hz=True, units='V',
        iqr_thresh=None
):
    """Plot spectral power density estimate for each array channel
    (and possibly ground channels). Compute RMS power for the bandpass
    determined by "fc".

    Parameters
    ----------
    f : sequence
        frequency vector
    gf : ndarray
        psd matrix for grounded input channels
    df : ndarray
        psd matrix for array signal channels
    fc : float
        cutoff frequency for RMS power calculation
    title : str
        plot title
    ylims : pair (optional)
        plot y-limits
    root_hz : (boolean)
        units normalized by 1/sqrt(Hz) (true) or 1/Hz (false)

    Returns
    -------
    figure

    """

    # compute outliers based on sum power
    if not iqr_thresh:
        iqr_thresh = get_default_args(fenced_out)['thresh']

    import matplotlib.pyplot as pp
    fig = pp.figure()
    fx = (f > 1) & (f < fc)
    # apply a wide-tolerance mask -- want to avoid plotting any
    # channels with zero (or negligable) power
    s_pwr = band_power(f, df, fc=fc, root_hz=root_hz)
    m = bad_channel_mask(np.log(s_pwr), iqr=iqr_thresh)
    df = df[m]
    pp.semilogy(
        f[fx], df[0, fx], color=psd_colors[0], label='sig channels'
    )
    pp.semilogy(
        f[fx], df[1:, fx].T, color=psd_colors[0], label='_nolegend_'
    )
    df_band_pwr = (df[:, fx] ** 2).mean()
    avg_d = np.sqrt(df_band_pwr * f[-1])
    pp.axhline(
        y=np.sqrt(df_band_pwr), color='chartreuse', linestyle='--',
        linewidth=4, label='sig avg RMS/$\sqrt{Hz}$'
    )

    if gf is not None and len(gf):
        pp.semilogy(f[fx], gf[0, fx], color=psd_colors[1], label='ground channels')
        if len(gf):
            pp.semilogy(f[fx], gf[1:, fx].T, color=psd_colors[1], label='_nolegend_')
        gf_band_pwr = (gf[:, fx] ** 2).mean()
        avg_g = np.sqrt(gf_band_pwr * f[-1])
        pp.axhline(
            y=np.sqrt(gf_band_pwr), color='k', linestyle='--', linewidth=4,
            label='gnd avg RMS/$\sqrt{Hz}$'
        )

    pp.legend(loc='upper right')
    units = nice_unit_text(units).strip('$')
    if root_hz:
        units_label = '$' + units + '/\sqrt{Hz}$'
    else:
        units_label = '$%s^{2}/Hz$' % units
    pp.ylabel(units_label);
    pp.xlabel('Hz (half-BW %d Hz)' % int(f[-1]))
    title = title + '\nSig RMS %1.2e' % avg_d
    if gf is not None:
        title = title + '; Gnd RMS %1.2e' % avg_g
    pp.title(title)
    pp.grid(which='both')
    if ylims:
        pp.ylim(ylims)
        offscreen = df[:, fx].mean(axis=1) < ylims[0]
        if np.any(offscreen):
            pp.gca().annotate(
                '%d chans off-screen' % offscreen.sum(),
                (200, ylims[0]), xycoords='data',
                xytext=(50, 3 * ylims[0]), textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05)
            )
    return fig


def plot_mean_psd(
        f, gf, df, fc, title, ylims=(), root_hz=True, units='V',
        iqr_thresh=None
):
    """Plot the mean spectral power density estimate for array
    channels (and possibly ground channels). Compute RMS power for the
    bandpass determined by "fc". Plot outlier PSDs individually.

    Parameters
    ----------
    f : sequence
        frequency vector
    gf : ndarray
        psd matrix for grounded input channels
    df : ndarray
        psd matrix for array signal channels
    fc : float
        cutoff frequency for RMS power calculation
    title : str
        plot title
    ylims : pair (optional)
        plot y-limits
    root_hz : (boolean)
        units normalized by 1/sqrt(Hz) (true) or 1/Hz (false)
    iqr_thresh : float (optional)
        set the outlier threshold (as a multiple of the interquartile
        range)

    Returns
    -------
    figure

    """

    import matplotlib.pyplot as pp
    # compute outliers based on sum power
    if not iqr_thresh:
        iqr_thresh = get_default_args(fenced_out)['thresh']

    s_pwr = band_power(f, df, fc=fc, root_hz=root_hz)
    s_pwr_mask = bad_channel_mask(np.log(s_pwr), iqr=iqr_thresh)
    ## s_pwr_mask = nut.fenced_out(np.log(s_pwr), thresh=iqr_thresh)
    ## s_pwr_mask = s_pwr_mask & (s_pwr > 0)
    s_pwr_mean = np.mean(s_pwr[s_pwr_mask])

    df = np.log(df)
    s_psd_mn = np.mean(df[s_pwr_mask], axis=0)
    s_psd_stdev = np.std(df[s_pwr_mask], axis=0)
    s_psd_lo = s_psd_mn - s_psd_stdev
    s_psd_hi = s_psd_mn + s_psd_stdev
    s_psd_mn, s_psd_lo, s_psd_hi = map(
        np.exp, (s_psd_mn, s_psd_lo, s_psd_hi)
    )
    avg_d = np.sqrt(s_pwr[s_pwr_mask].mean())

    fig, ln = filled_interval(
        pp.semilogy, f, s_psd_mn, (s_psd_lo, s_psd_hi), psd_colors[0]
    )

    sig_baseline = s_psd_mn[f > f.max() / 2].mean()
    legends = [r'mean signal PSD $\pm \sigma$']
    df_o = None
    if np.any(~s_pwr_mask):
        df_o = np.exp(df[~s_pwr_mask])
        o_lines = pp.semilogy(f, df_o.T, '#BD6734', lw=0.5)
        ln.append(o_lines[0])
        legends.append('outlier signal PSDs')
        # let's label these lines
        chan_txt = 'outlier sig chans: ' + \
                   ', '.join([str(c) for c in (~s_pwr_mask).nonzero()[0]])
        y = 0.5 * (np.ceil(np.log(s_psd_mn.max())) + np.log(sig_baseline))
        pp.text(200, np.exp(y), chan_txt, fontsize=10, va='baseline')

    if gf is not None and len(gf):
        g_pwr = band_power(f, gf, fc=fc, root_hz=root_hz)
        if len(g_pwr) > 1:
            g_pwr_mask = fenced_out(np.log(g_pwr), thresh=iqr_thresh)
        else:
            g_pwr_mask = np.array([True])
        g_pwr_mean = np.mean(g_pwr[g_pwr_mask])

        gf = np.log(gf)
        g_psd_mn = np.mean(gf[g_pwr_mask], axis=0)
        g_psd_stdev = np.std(gf[g_pwr_mask], axis=0)
        g_psd_lo = g_psd_mn - g_psd_stdev
        g_psd_hi = g_psd_mn + g_psd_stdev
        g_psd_mn, g_psd_lo, g_psd_hi = map(
            np.exp, (g_psd_mn, g_psd_lo, g_psd_hi)
        )
        avg_g = np.sqrt(g_pwr[g_pwr_mask].mean())

        fig, g_ln = filled_interval(
            pp.semilogy, f, g_psd_mn, (g_psd_lo, g_psd_hi), psd_colors[1],
            ax=fig.axes[0]
        )
        ln.extend(g_ln)
        legends.append(r'mean grounded input $\pm \sigma$')
        if np.any(~g_pwr_mask):
            o_lines = pp.semilogy(
                f, np.exp(gf[~g_pwr_mask]).T, '#06A684', lw=0.5
            )
            ln.append(o_lines[0])
            legends.append('outlier grounded PSDs')
            chan_txt = 'outlier gnd chans: ' + \
                       ', '.join([str(c) for c in (~g_pwr_mask).nonzero()[0]])
            y = sig_baseline ** 0.33 * g_psd_mn.mean() ** 0.67

            pp.text(200, y, chan_txt, fontsize=10, va='baseline')

    pp.legend(ln, legends, loc='upper right', fontsize=11)
    units = nice_unit_text(units).strip('$')
    if root_hz:
        units_label = '$' + units + '/\sqrt{Hz}$'
    else:
        units_label = '$%s^{2}/Hz$' % units
    pp.ylabel(units_label);
    pp.xlabel('Hz (half-BW %d Hz)' % int(f[-1]))
    if gf is not None and len(gf):
        title = title + \
                '\nGnd RMS %1.2e; Sig RMS %1.2e (to %d Hz)' % (avg_g, avg_d, fc)
    else:
        title = title + \
                '\nSig RMS %1.2e (to %d Hz)' % (avg_d, fc)
    pp.title(title)
    pp.grid(which='both')
    if ylims:
        pp.ylim(ylims)
        if df_o is not None:
            offscreen = df_o.mean(axis=1) < ylims[0]
            if np.any(offscreen):
                pp.gca().annotate(
                    '%d chans off-screen' % offscreen.sum(),
                    (200, ylims[0]), xycoords='data',
                    xytext=(50, 3 * ylims[0]), textcoords='data',
                    arrowprops=dict(facecolor='black', shrink=0.05)
                )

    return fig


def lite_mux_load(fname, bandpass=(2, -1)):
    raise NotImplementedError('removed this function')


def _old_block_psds(data, btime, Fs, max_blocks=-1, **mtm_kw):
    """
    Parameters
    ----------
    data : 2D or 3D array
        Array with timeseries in the last axis. If 3D, then it
        has already been cut into blocks.
    """
    if 'jackknife' not in mtm_kw:
        mtm_kw['jackknife'] = False
    if 'adaptive' not in mtm_kw:
        mtm_kw['adaptive'] = False
    if 'NW' not in mtm_kw:
        mtm_kw['NW'] = 2.5

    if data.ndim > 2:
        nblock, bsize = data.shape[1:]
        blk_data = data
    else:
        bsize = int(round(Fs * btime))
        npt = data.shape[-1]
        nblock = npt // bsize
        blk_data = data[..., :bsize * nblock].reshape(-1, nblock, bsize)
    if max_blocks > 0:
        nblock = min(max_blocks, nblock)

    nfft = nextpow2(bsize)
    # try to keep computation chunks to modest sizes.. 6 GB
    # so (2*NW) * nfft * nchan * sizeof(complex128) * comp_blocks < 3 GB
    n_tapers = 2.0 * mtm_kw['NW']
    from ecogdata.expconfig import params as global_params
    mem_limit = float(global_params.memory_limit) / 2.0
    comp_blocks = mem_limit / n_tapers / nfft / len(data) / (2 * data.dtype.itemsize)
    comp_blocks = max(1, int(comp_blocks))
    n_comp_blocks = nblock // comp_blocks + int(nblock % comp_blocks > 0)
    psds = list()
    for blocks in np.array_split(blk_data[:, :nblock], n_comp_blocks, axis=1):
        freqs, psds_, _ = multi_taper_psd(
            blocks.copy(), NFFT=nfft, Fs=Fs, **mtm_kw
        )
        psds.append(psds_)

    psds = np.concatenate(psds, axis=1)
    return freqs, psds


def _old_logged_estimators(psds, sem=True):
    # *** used with block_psds above ***
    # returns (blockwise mean, channel mean, channel mean +/- stderr)
    lg_psds = np.mean(np.log(psds), axis=1)
    mn = np.mean(lg_psds, axis=0)
    err = np.std(lg_psds, axis=0)
    if sem:
        err = err / np.sqrt(lg_psds.shape[0])
    return map(np.exp, (lg_psds, mn, mn - err, mn + err))


def block_psds(data, btime, Fs, max_blocks=-1, old_blocks=False, **mtm_kw):
    """
    Parameters
    ----------
    data : 2D or 3D array
        Array with timeseries in the last axis. If 3D, then it
        has already been cut into blocks, indexed in the first dimension
        (or second dimension, if old_blocks==True).
    btime : float
        Length (in seconds) of the blocking.
    Fs : float
        Sampling frequency
    max_blocks : int
        Limits the number of blocks computed (-1 for no limit).
    old_blocks : {True | False}
        (for old code) revert to prior block geometry (in 2nd axis).

    Returns
    -------
    freqs : array
        Frequency bins
    psds : ndarray
        Power spectral density of blocks.
    """

    if old_blocks:
        return _old_block_psds(data, btime, Fs, max_blocks=max_blocks, **mtm_kw)

    if 'jackknife' not in mtm_kw:
        mtm_kw['jackknife'] = False
    if 'adaptive' not in mtm_kw:
        mtm_kw['adaptive'] = False
    if 'NW' not in mtm_kw:
        mtm_kw['NW'] = 2.5

    if data.ndim > 2:
        nblock, nchan, bsize = data.shape
        blk_data = data.reshape(nblock * nchan, bsize)
    else:
        bsize = int(round(Fs * btime))
        nchan, npt = data.shape
        nblock = npt // bsize
        blk_data = data[..., :bsize * nblock].reshape(-1, nblock, bsize)
        blk_data = blk_data.transpose(1, 0, 2).copy()
        blk_data = blk_data.reshape(nblock * nchan, bsize)
    if max_blocks > 0:
        nblock = min(max_blocks, nblock)

    nfft = nextpow2(bsize)
    # try to keep computation chunks to modest sizes.. 6 GB
    # so (2*NW) * nfft * nchan * sizeof(complex128) * comp_blocks < 3 GB
    n_tapers = 2.0 * mtm_kw['NW']
    from ecogdata.expconfig import params as global_params
    mem_limit = float(global_params.memory_limit) / 2.0
    comp_blocks = mem_limit / n_tapers / nfft / (2 * data.dtype.itemsize)
    comp_blocks = max(1, int(comp_blocks))
    n_comp_blocks = nchan * nblock // comp_blocks + \
                    int((nchan * nblock) % comp_blocks > 0)
    psds = list()

    for blocks in np.array_split(blk_data[:nblock * nchan], n_comp_blocks, axis=0):
        ## freqs, psds_, _ = multi_taper_psd(
        ##     blocks.copy(), NFFT=nfft, Fs=Fs, **mtm_kw
        ##     )
        freqs, psds_, _ = multi_taper_psd(
            blocks, NFFT=nfft, Fs=Fs, **mtm_kw
        )
        # unwrap into blocks x chans x pts
        psds.append(psds_)

    psds = np.concatenate(psds, axis=0)
    return freqs, psds.reshape(-1, nchan, psds.shape[1])


def logged_estimators(psds, sem=True, old_blocks=False):
    if old_blocks:
        return _old_logged_estimators(psds, sem=sem)

    # *** used with block_psds above ***
    # returns (blockwise mean, channel mean, channel mean +/- stderr)
    lg_psds = np.mean(np.log(psds), axis=0)
    mn = np.mean(lg_psds, axis=0)
    err = np.std(lg_psds, axis=0)
    if sem:
        err = err / np.sqrt(lg_psds.shape[0])
    return map(np.exp, (lg_psds, mn, mn - err, mn + err))


def safe_avg_power(
        data, bsize=2000, iqr_thresh=3.0, mean=True, mask_per_chan=False
):
    if data.ndim < 3:
        b_data = BlockedSignal(data, bsize, axis=-1)
        iterator = b_data.fwd()
        nblock = b_data.nblock
        nchan = data.shape[0]
    else:
        nblock, nchan = data.shape[:2]
        iterator = (b for b in data)
    rms_vals = np.zeros((nblock, nchan))
    for n, blk in enumerate(iterator):
        rms_vals[n] = blk.std(axis=-1)
    # If mask_per_chan is True, then evaluate outliers relative to
    # each channel's samples. Otherwise eval outliers relative to the
    # whole sample.
    axis = 0 if mask_per_chan else None
    if not mean:
        return rms_vals
    omask = fenced_out(rms_vals, thresh=iqr_thresh, axis=0)
    rms_vals[~omask] = np.nan
    return np.nanmean(rms_vals, axis=0)


def semivariogram(data, normed=True):
    # zero mean for temporal expectation
    dzm = data - data.mean(-1)[:, None]
    # dzm = data - data.mean(0)
    # I think this adjustment is ok -- this enforces the assumption
    # that every site has similar marginal statistics
    var = dzm.var(-1)
    dzm = dzm / np.sqrt(var)[:, None]
    # dzm = dzm - dzm.std(-1)[:,None]
    # err = dzm[:,None,:] - dzm[None,:,:]
    # sv1 = np.mean( err**2, axis=-1 ) * 0.5
    cxx = np.cov(dzm, bias=1)
    if normed:
        return 1 - cxx
    else:
        return (1 - cxx) * np.median(var)


def _covar_lite(X, normed=False):
    X = X - np.mean(X, axis=1, keepdims=1)
    C = np.einsum('ij,kj->ik', X, X)
    dof = X.shape[1] - 1
    if normed:
        np.power(X, 2, out=X)
        V = np.sum(X, axis=1)
        np.sqrt(V, out=V)
        C /= V
        C /= V[:, None]
        return C
    return C / dof


# def safe_corrcoef(
#         data, bsize=2000, iqr_thresh=3.0,
#         mean=True, normed=True, semivar=False, ar_whiten=False
# ):
#     if data.ndim < 3:
#         bdata = BlockedSignal(data, bsize, partial_block=False)
#         nchan = data.shape[0]
#         nblock = bdata.nblock
#         iterator = bdata.fwd()
#     else:
#         nblock, nchan = data.shape[:2]
#         iterator = (b for b in data)
#     cxx = np.zeros((nblock, nchan, nchan))
#     results = []
#     if not semivar:
#         # fn = partial(_covar_lite, normed=normed)
#         fn = np.corrcoef if normed else np.cov
#     else:
#         fn = partial(ergodic_semivariogram, normed=normed, mask_outliers=False)
#         pool = mp.Pool(min(8, mp.cpu_count()))
#     for n, blk in enumerate(iterator):
#         if ar_whiten:
#             blk = ar_whiten_blocks(blk)
#         if semivar:
#             results.append(pool.apply_async(fn, (blk,)))
#         else:
#             cxx[n] = fn(blk)
#     if semivar:
#         pool.close()
#         pool.join()
#         for n, r in enumerate(results):
#             cxx[n] = r.get()
#
#     cxx_norm = np.nansum(np.nansum(cxx ** 2, axis=-1), axis=-1)
#     mask = nut.fenced_out(cxx_norm, thresh=iqr_thresh)
#     if not mean:
#         return cxx
#     return np.nanmean(cxx[mask], axis=0)


def centered_pair_maps(pair_table, idx1, idx2):
    # Pair table is an (npair,) + (x,y,z) shape array with information
    # regarding each pair of electrodes. E.g. the upper-triangular
    # portion of a correlation matrix
    #
    # idx1, idx2 are the respective array coordinates of each
    # electrode in the pair
    mi, mj = np.row_stack((idx1, idx2)).max(0)
    offset = np.array([mi, mj])
    mx_at_dist = len(set([tuple(i) for i in idx1]))
    cmap = np.zeros((mx_at_dist, 2 * mi + 1, 2 * mj + 1) + pair_table.shape[1:])
    cmap.fill(np.nan)
    cnt = np.zeros((2 * mi + 1, 2 * mj + 1), 'i')
    for coh, ij1, ij2 in zip(pair_table, idx1, idx2):
        # put the same value down for both symmetrical site-to-site offsets
        rel_i, rel_j = ij1 - ij2 + offset
        n = cnt[rel_i, rel_j]
        cmap[n, rel_i, rel_j] = coh
        cnt[rel_i, rel_j] += 1
        rel_i, rel_j = ij2 - ij1 + offset
        cmap[n, rel_i, rel_j] = coh
        cnt[rel_i, rel_j] += 1
    cnt[cnt == 0] = 1
    # cmap /= cnt[ (slice(None), slice(None)) + (None,)*(pair_coh.ndim-1) ]
    return cmap


########################################################################
## -------- "High-level" functions: give data and get a figure ---------

## XXX: needs hella refactor

def plot_avg_psds(
        data_or_path, d_chans, g_chans, title, bsize_sec=2,
        notches=(), Fs=1, iqr_thresh=None, units='V', **mtm_kw
):
    # make two plots with
    # 1) all spectra
    # 2) average +/- sigma, and outliers
    if not isinstance(data_or_path, np.ndarray):
        data, Fs = lite_mux_load(data_or_path, notches=notches)
        data = data[:, 10000:].copy()
    else:
        data = data_or_path

    freqs, psds = block_psds(data + 1e-10, bsize_sec, Fs, **mtm_kw)

    # psds = np.mean(psds, axis=1)
    psds, p_mn, p_err_lo, p_err_hi = logged_estimators(psds, sem=False)
    g_psds = np.sqrt(psds[g_chans]) if len(g_chans) else None
    d_psds = np.sqrt(psds[d_chans])
    ## g_psds = psds[g_chans] if len(g_chans) else None
    ## d_psds = psds[d_chans]

    ttl_str = '%s Fs=%d' % (title, round(Fs))
    ymax = 10 ** np.ceil(np.log10(d_psds.max()) + 1)
    ymin = ymax * 1e-6
    fig = plot_psds(
        freqs, g_psds, d_psds, Fs / 2, ttl_str,
        ylims=(ymin, ymax),
        iqr_thresh=iqr_thresh, units=units, root_hz=True
    )
    fig_avg = plot_mean_psd(
        freqs, g_psds, d_psds, Fs / 2, ttl_str,
        ylims=(ymin, ymax),
        iqr_thresh=iqr_thresh, units=units, root_hz=True
    )

    return fig, fig_avg


# def plot_centered_rxx(
#         data_or_path, d_chans, chan_map, label,
#         notches=(), bandpass=(2, -1), Fs=1, pitch=1.0, cmap='bwr', normed=True, clim=None
# ):
#     import matplotlib.pyplot as pp
#     from seaborn import JointGrid, jointplot
#     if not isinstance(data_or_path, np.ndarray):
#         data, Fs = lite_mux_load(
#             data_or_path, bandpass=bandpass, notches=notches
#         )
#         data = data[:, 10000:].copy()
#     else:
#         data = data_or_path
#
#     cxx = safe_corrcoef(data[d_chans], 2000, normed=normed)
#     n = cxx.shape[0]
#
#     cxx_pairs = cxx[np.triu_indices(n, k=1)]
#
#     if np.iterable(pitch):
#         pitch_x, pitch_y = pitch
#     else:
#         pitch_x = pitch_y = pitch
#     chan_combs = ut.channel_combinations(chan_map, scale=(pitch_x, pitch_y))
#
#     # make sure pairs are sorted like upper triangular indices
#     pairs = zip(chan_combs.p1, chan_combs.p2)
#     ix = [x for x, y in sorted(enumerate(pairs), key=lambda x: x[1])]
#     idx1 = chan_combs.idx1[ix];
#     idx2 = chan_combs.idx2[ix]
#     dists = chan_combs.dist[ix]
#
#     centered_rxx = centered_pair_maps(cxx_pairs, idx1, idx2)
#     y, x = centered_rxx.shape[-2:]
#     midx = int(x / 2);
#     xx = (np.arange(x) - midx) * pitch_x
#     midy = int(y / 2);
#     yy = (np.arange(y) - midy) * pitch_y
#     # centered_rxx[:,midy,midx] = 1
#     with sns.axes_style('ticks'):
#
#         jgrid = JointGrid(
#             np.random.rand(50), np.random.rand(50), ratio=4,
#             xlim=(xx[0], xx[-1]), ylim=(yy[0], yy[-1]), size=8
#         )
#
#         cm = nancmap(cmap, nanc=(.5, .5, .5, .5))
#
#         ## Joint plot
#         ax = jgrid.ax_joint
#         if clim is None:
#             clim = (-1, 1) if normed else np.percentile(centered_rxx, [2, 98])
#         ax.imshow(
#             np.nanmean(centered_rxx, axis=0), clim=clim, cmap=cm,
#             extent=[xx[0], xx[-1], yy[0], yy[-1]]
#         )
#         ax.set_xlabel('Site-site distance (mm)')
#         ax.set_ylabel('Site-site distance (mm)')
#
#         ## Marginal-X
#         ax = jgrid.ax_marg_x
#         ax.spines['left'].set_visible(True)
#         ax.yaxis.tick_left()
#         pp.setp(ax.yaxis.get_majorticklines(), visible=True)
#         pp.setp(ax.get_yticklabels(), visible=True)
#         # arrange as samples over all x-distances
#         rxx_mx = np.reshape(centered_rxx, (-1, x))
#
#         vals = list()
#         for c in rxx_mx.T:
#             valid = ~np.isnan(c)
#             if valid.any():
#                 vals.append(np.percentile(c[valid], [25, 50, 75]))
#             else:
#                 vals.append([np.nan] * 3)
#
#         mx_lo, mx_md, mx_hi = map(np.array, zip(*vals))
#         filled_interval(
#             ax.plot, xx, mx_md, (mx_lo, mx_hi), cm(0.6), ax=ax, lw=2, alpha=.6
#         )
#         ax.set_yticks(np.linspace(-1, 1, 6))
#         ax.set_ylim(clim)
#
#         ## Marginal-Y
#         ax = jgrid.ax_marg_y
#         ax.spines['top'].set_visible(True)
#         ax.xaxis.tick_top()
#         pp.setp(ax.xaxis.get_majorticklines(), visible=True)
#         pp.setp(ax.get_xticklabels(), visible=True)
#         rxx_my = np.reshape(np.rollaxis(centered_rxx, 2).copy(), (-1, y))
#         vals = list()
#         for c in rxx_my.T:
#             valid = ~np.isnan(c)
#             if valid.any():
#                 vals.append(np.percentile(c[valid], [25, 50, 75]))
#             else:
#                 vals.append([np.nan] * 3)
#
#         my_lo, my_md, my_hi = map(np.array, zip(*vals))
#         filled_interval(
#             ax.plot, yy, my_md, (my_lo, my_hi), cm(0.6),
#             ax=ax, lw=2, alpha=.6, fillx=True
#         )
#         ax.set_xticks(np.linspace(-1, 1, 6))
#         pp.setp(ax.xaxis.get_ticklabels(), rotation=-90)
#         ax.set_xlim(clim)
#
#         jgrid.fig.subplots_adjust(left=0.1, bottom=.1)
#
#     jgrid.ax_marg_x.set_title(
#         'Average centered correlation map: ' + label, fontsize=12
#     )
#     return jgrid.fig

#
# def spatial_variance(data, chan_map, label, normed=False):
#     import matplotlib.pyplot as pp
#     from seaborn import despine, xkcd_rgb
#
#     cxx = safe_corrcoef(data, 2000, normed=normed, semivar=True)
#     n = cxx.shape[0]
#     cxx_pairs = cxx[np.triu_indices(n, k=1)]
#     rms = safe_avg_power(data)
#     var_mu = (rms ** 2).mean()
#     var_se = (rms ** 2).std() / np.sqrt(len(rms))
#
#     chan_combs = chan_map.site_combinations
#     dist = chan_combs.dist
#     if np.iterable(chan_map.pitch):
#         pitch_x, pitch_y = chan_map.pitch
#     else:
#         pitch_x = pitch_y = chan_map.pitch
#     binsize = np.ceil(10 * (pitch_x ** 2 + pitch_y ** 2) ** 0.5) / 10.0
#     clrs = pp.rcParams['axes.prop_cycle'].by_key()['color']
#     pts, lines = covar_to_iqr_lines(dist, cxx_pairs, binsize=binsize, linewidths=1, colors=clrs[0])
#     xb, yb = pts
#     # set a fairly wide range for nugget and sill
#     bounds = {'nugget': (0, yb[0]), 'sill': (np.mean(yb), var_mu + 5 * var_se),
#               'nu': (0.4, 10), 'theta': (0.5, None)}
#     p = matern_semivariogram(
#         dist, y=cxx_pairs, theta=1, nu=1, sill=var_mu, nugget=yb[0] / 5.0,
#         free=('theta', 'nu', 'nugget', 'sill'), dist_limit=0.67,
#         wls_mode='irls', fit_mean=True, fraction_nugget=False, bounds=bounds)
#
#     f, ax = pp.subplots(figsize=(8, 5))
#     ax.scatter(dist, cxx_pairs, s=5, color='gray', alpha=0.2, rasterized=True, label='Pairwise semivariance')
#     ax.plot(*pts, color=clrs[0], ls='--', marker='o', ms=8, label='Binned semivariance')
#     ax.add_collection(lines)
#     ax.axhline(var_mu, lw=1, color=xkcd_rgb['reddish orange'], label='Avg signal variance', alpha=0.5)
#     ax.axhline(var_mu + var_se, lw=0.5, color=xkcd_rgb['reddish orange'], linestyle='--', alpha=0.5)
#     ax.axhline(var_mu - var_se, lw=0.5, color=xkcd_rgb['reddish orange'], linestyle='--', alpha=0.5)
#     ax.axhline(p['nugget'], lw=2, color=xkcd_rgb['dark lavender'], alpha=0.5, label='Noise "nugget" (uV^2)')
#     ax.axhline(p['sill'], lw=2, color=xkcd_rgb['teal green'], alpha=0.5, label='Spatial var. "sill" (uV^2)')
#     xm = np.linspace(dist.min(), dist.max(), 100)
#     model_label = 'Model: ' + make_matern_label(theta=p['theta'], nu=p['nu'])
#     ax.plot(xm, matern_semivariogram(xm, **p), color=clrs[1], label=model_label)
#     ax.set_xlabel('Site-site distance (mm)')
#     if normed:
#         units = '(normalized)'
#     else:
#         units = '(uV^2)'
#     ax.set_ylabel('Semivariance ' + units)
#     despine(fig=f)
#     leg = ax.legend(loc='upper left', ncol=3, frameon=True)
#     for h in leg.legendHandles:
#         h.set_alpha(1)
#         try:
#             h.set_sizes([15] * len(h.get_sizes()))
#         except:
#             pass
#     ax.set_title(label + ' spatial variogram')
#     f.tight_layout(pad=0.2)
#     return f
#
#
# def scatter_correlations(
#         data_or_path, d_chans, chan_map, mask, title,
#         notches=(), bandpass=(2, -1), Fs=1, highlight='rows', pitch=1.0
# ):
#     # plot the pairwise correlation values against distance of the pair
#     # Highlight channels that
#     # 1) share a row (highlight='rows')
#     # 2) share a column (highlight='cols')
#     # 3) either of the above (highlight='rows+cols')
#     # 4) are neighbors on a row (highlight='rownabes')
#     # 5) are neighbors on a column (highlight='colnabes')
#     # 6) any neighbor (4-5) (highlight='allnabes')
#     import matplotlib.pyplot as pp
#     if not isinstance(data_or_path, np.ndarray):
#         data, Fs = lite_mux_load(
#             data_or_path, bandpass=bandpass, notches=notches
#         )
#         data = data[:, 10000:].copy()
#     else:
#         data = data_or_path
#
#     # data[g_chans] = np.nan
#     cxx = safe_corrcoef(data[d_chans[mask]], 2000)
#     n = cxx.shape[0]
#
#     cxx_pairs = cxx[np.triu_indices(n, k=1)]
#     if np.iterable(pitch):
#         pitch_x, pitch_y = pitch
#     else:
#         pitch_x = pitch_y = pitch
#     chan_combs = ut.channel_combinations(chan_map.subset(mask),
#                                          scale=(pitch_x, pitch_y))
#
#     # make sure pairs are sorted like upper triangular indices
#     # xxx: should already be sorted as a result of itertools.combinations()
#     idx1 = chan_combs.idx1;
#     idx2 = chan_combs.idx2
#     dists = chan_combs.dist
#
#     fig = pp.figure()
#
#     panels = highlight.split(',')
#     if panels[0] == highlight:
#         pp.subplot(111)
#         pp.scatter(
#             dists, cxx_pairs, 9, label='_nolegend_', edgecolors='none', alpha=0.25, rasterized=True
#         )
#         fig.tight_layout()
#         fig.subplots_adjust(top=0.95)
#         fig.text(0.5, .96, title, fontsize=16, va='baseline', ha='center')
#         return fig
#
#     # xxx: hardwired for 16 channel muxing with grounded input on 1st chan
#     mux_index = np.arange(len(d_chans)).reshape(-1, 15).transpose()[1:]
#     nrow, ncol = mux_index.shape
#     mux_index -= np.arange(1, ncol + 1)
#
#     labels = []
#     colors = dict(rows='#E480DA', cols='#80E48A')
#     cxx = safe_corrcoef(data[d_chans], 2000)
#     cxx_pairs = cxx[np.triu_indices(len(cxx), k=1)]
#     chan_combs = ut.channel_combinations(chan_map, scale=(pitch_x, pitch_y))
#
#     # make sure pairs are sorted like upper triangular indices
#     # xxx: should already be sorted as a result of itertools.combinations()
#     idx1 = chan_combs.idx1;
#     idx2 = chan_combs.idx2
#     dists = chan_combs.dist
#
#     for n, highlight in enumerate(panels):
#         pp.subplot(len(panels), 1, n + 1)
#         pp.scatter(
#             dists, cxx_pairs, 9, edgecolor='none', label='_nolegend_', alpha=0.25, rasterized=True
#         )
#         if highlight in ('rows', 'rows+cols'):
#             for row in mux_index:
#                 row = [r for r in row if mask[r]]
#                 if len(row) < 2:
#                     continue
#                 subset = chan_map.subset(row)
#                 subcxx = cxx[row][:, row][np.triu_indices(len(row), k=1)]
#                 subdist = ut.channel_combinations(
#                     subset, scale=(pitch_x, pitch_y)
#                 ).dist
#                 c = pp.scatter(
#                     subdist, subcxx, 20, colors['rows'],
#                     edgecolor='white', label='_nolegend_'
#                 )
#             # set label on last one
#             c.set_label('row combo')
#         if highlight in ('cols', 'rows+cols'):
#             for col in mux_index.T:
#                 col = [c for c in col if mask[c]]
#                 if len(col) < 2:
#                     continue
#                 subset = chan_map.subset(col)
#                 subcxx = cxx[col][:, col][np.triu_indices(len(col), k=1)]
#                 subdist = ut.channel_combinations(
#                     subset, scale=(pitch_x, pitch_y)
#                 ).dist
#                 c = pp.scatter(
#                     subdist, subcxx, 20, colors['cols'],
#                     edgecolor='white', label='_nolegend_'
#                 )
#             # set label on last one
#             c.set_label('col combo')
#
#         if highlight in ('rownabes', 'allnabes'):
#             row_cxx = list()
#             row_dist = list()
#             for row in mux_index:
#                 row = [r for r in row if mask[r]]
#                 if len(row) < 2:
#                     continue
#                 for i1, i2 in zip(row[:-1], row[1:]):
#                     ii = np.where(
#                         (chan_combs.p1 == min(i1, i2)) & \
#                         (chan_combs.p2 == max(i1, i2))
#                     )[0][0]
#                     row_cxx.append(cxx_pairs[ii])
#                     row_dist.append(dists[ii])
#             c = pp.scatter(
#                 row_dist, row_cxx, 20, colors['rows'],
#                 edgecolor='white', label='row neighbors'
#             )
#         if highlight in ('colnabes', 'allnabes'):
#             col_cxx = list()
#             col_dist = list()
#             for col in mux_index.T:
#                 col = [c for c in col if mask[c]]
#                 if len(col) < 2:
#                     continue
#                 for i1, i2 in zip(col[:-1], col[1:]):
#                     ii = np.where(
#                         (chan_combs.p1 == min(i1, i2)) & \
#                         (chan_combs.p2 == max(i1, i2))
#                     )[0][0]
#                     col_cxx.append(cxx_pairs[ii])
#                     col_dist.append(dists[ii])
#             c = pp.scatter(
#                 col_dist, col_cxx, 20, colors['cols'],
#                 edgecolor='white', label='col neighbors'
#             )
#         pp.legend(loc='best')
#
#     ax = pp.gca()
#     ax.set_xlabel('Distance (mm)')
#     ax.set_ylabel('Correlation coef.')
#     fig.tight_layout()
#     fig.subplots_adjust(top=0.95)
#     fig.text(0.5, .96, title, fontsize=16, va='baseline', ha='center')
#
#     return fig


def plot_mux_columns(
        data_or_path, d_chans, g_chans, title,
        bandpass=(2, -1), notches=(), Fs=1, color_lims=True,
        units='uV'
):
    import matplotlib.pyplot as pp
    if not isinstance(data_or_path, np.ndarray):
        data, Fs = lite_mux_load(
            data_or_path, bandpass=bandpass, notches=notches
        )
        data = data[:, 10000:].copy()
    else:
        data = data_or_path

    # data[g_chans] = np.nan
    rms = safe_avg_power(data, 2000)
    rms[g_chans] = np.nan
    if color_lims:
        vals = rms[np.isfinite(rms)]
        # basically try to clip out anything small
        vals = vals[vals > 1e-2 * np.median(vals)]
        quantiles = np.percentile(vals, [5., 95.])
        clim = tuple(quantiles)
    else:
        clim = (np.nanmin(rms), np.nanmax(rms))

    d_rms = rms[d_chans].reshape(-1, 15)
    if len(g_chans):
        rms = np.column_stack((rms[g_chans], d_rms))
    else:
        rms = d_rms
    # rms.shape = (-1, 16)
    fig = pp.figure()
    cm = nancmap('hot', nanc='dodgerblue')
    pp.imshow(rms.T, origin='upper', cmap=cm, clim=clim)
    cbar = pp.colorbar()
    cbar.set_label(nice_unit_text(units) + ' RMS')
    pp.title(title)
    pp.xlabel('data column')
    ax = fig.axes[0]
    ax.set_aspect('auto')
    ax.set_xticks(range(rms.shape[0]))
    return fig


def plot_rms_array(
        data_or_path, d_chans, chan_map, title,
        notches=(), bandpass=(2, -1), Fs=1, color_lims=True,
        units='uV'
):
    import matplotlib.pyplot as pp
    if not isinstance(data_or_path, np.ndarray):
        data, Fs = lite_mux_load(
            data_or_path, bandpass=bandpass, notches=notches
        )
        data = data[:, 10000:].copy()
    else:
        data = data_or_path

    # data[g_chans] = np.nan
    rms = safe_avg_power(data, 2000)
    rms = rms[d_chans]
    if color_lims:
        vals = rms[np.isfinite(rms)]
        # basically try to clip out anything small
        vals = vals[vals > 1e-2 * np.median(vals)]
        quantiles = np.percentile(vals, [5., 95.])
        clim = tuple(quantiles)
    else:
        clim = (np.nanmin(rms), np.nanmax(rms))
    rms_arr = chan_map.embed(rms)
    # rms_arr = np.ones(chan_map.geometry)*np.nan
    # np.put(rms_arr, chan_map, rms)
    cm = nancmap('hot', nanc='dodgerblue')

    f = pp.figure()
    pp.imshow(rms_arr, origin='upper', cmap=cm, clim=clim)
    cbar = pp.colorbar()
    cbar.set_label(nice_unit_text(units) + ' RMS')

    pp.title(title)
    return f


# def plot_site_corr(
#         data_or_path, d_chans, title,
#         notches=(), bandpass=(2, -1), Fs=1
# ):
#     import matplotlib.pyplot as pp
#     if not isinstance(data_or_path, np.ndarray):
#         data, Fs = lite_mux_load(
#             data_or_path, bandpass=bandpass, notches=notches
#         )
#         data = data[:, 10000:].copy()
#     else:
#         data = data_or_path
#
#     # data[g_chans] = np.nan
#     cxx = safe_corrcoef(data[d_chans], 2000)
#     n = cxx.shape[0]
#     cxx.flat[0:n * n:n + 1] = np.nan
#
#     cm = pp.cm.jet
#
#     f = pp.figure()
#     pp.imshow(cxx, cmap=cm)
#     cbar = pp.colorbar()
#     cbar.set_label('avg corr coef')
#
#     pp.title(title)
#     return f
#
#
# def plot_site_corr_new(
#         data, chan_map, title,
#         notches=(), bandpass=(2, -1), bsize=2000, cmap=None, normed=True,
#         stagger_x=False, stagger_y=False
# ):
#     import matplotlib.pyplot as pp
#     # data[g_chans] = np.nan
#     cxx = safe_corrcoef(data, bsize, normed=normed)
#     n = cxx.shape[0]
#     cxx.flat[0:n * n:n + 1] = np.nan
#
#     clim = (-1, 1) if normed else np.percentile(cxx, [2, 98])
#     if cmap is None:
#         from .colormaps as diverging_cm
#         cmap = diverging_cm(clim[0], clim[1], ((0, 0, 0), (1, 0, 0)))
#
#     f, axs = pp.subplots(1, 2, figsize=(12, 5))
#
#     corr_ax = axs[0]
#     graph_ax = axs[1]
#
#     im = corr_ax.imshow(cxx, cmap=cmap, norm=pp.Normalize(*clim))
#     cbar = pp.colorbar(im, ax=corr_ax, use_gridspec=True)
#     cbar.set_label('avg corr coef')
#     corr_ax.axis('image')
#
#     plot_electrode_graph(cxx, chan_map, ax=graph_ax, stagger_y=stagger_y, stagger_x=stagger_x)
#
#     f.subplots_adjust(top=0.9, left=0.05, right=0.95, wspace=0.1)
#     f.text(0.5, 0.92, title, ha='center', va='baseline', fontsize=20)
#     return f
#

def plot_channel_mask(
        data, chan_map, title, units='V', bsize=2000,
        quantiles=(50, 80), iqr=3
):
    import matplotlib.pyplot as pp
    from seaborn import violinplot
    rms = safe_avg_power(data, bsize=bsize, iqr_thresh=7)
    rms = np.log(rms)
    mask = bad_channel_mask(rms, quantiles=quantiles, iqr=iqr)
    f = pp.figure(figsize=(7, 4))
    ax = f.add_subplot(121)
    with sns.axes_style('whitegrid'):
        violinplot(
            np.ma.masked_invalid(rms).compressed(),
            alpha=0.5, widths=0.5, names=[' '],
            color=sns.xkcd_rgb['amber'], orient='v'
        )
        sns.despine(ax=ax, left=True)
        ax.plot(np.ones(mask.sum()) * 1.3, rms[mask], 'k+')
        if np.sum(~mask):
            ax.plot(np.ones(np.sum(~mask)) * 1.3, rms[~mask], 'r+')
        ax.set_yticklabels(['%.1f' % s for s in np.exp(ax.get_yticks())])
        ax.set_ylabel(nice_unit_text(units) + ' RMS')
    ax.set_title('Distribution of log-power')
    ax = f.add_subplot(122)
    site_mask = np.ones(chan_map.geometry) * np.nan
    site_mask.flat[chan_map.subset(mask.nonzero()[0])] = 1
    site_mask.flat[chan_map.subset((~mask).nonzero()[0])] = 0
    N = pp.cm.binary.N
    im = ax.imshow(
        site_mask,
        cmap=pp.cm.winter, norm=pp.cm.colors.BoundaryNorm([0, .5, 1], N),
        alpha=0.5, origin='upper'
    )
    cbar = pp.colorbar(im)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(('rejected', 'accepted'))
    ax.axis('image')
    ax.set_title('Inlier electrodes')
    f.text(0.5, 0.02, title, ha='center', va='baseline', fontsize=18)
    return f, mask


def sinusoid_gain(data, ref, chan_map, log=True, **im_kws):
    import matplotlib.pyplot as pp
    ## d_rms = data.std(1)
    ## r_rms = ref.std()
    ## gain = d_rms / r_rms
    data = data - data.mean(axis=-1, keepdims=1)
    ref = ref - ref.mean()
    gain = np.dot(data, ref) / np.dot(ref, ref)

    f = pp.figure(figsize=(7.5, 4))
    ax = pp.subplot2grid((1, 100), (0, 0), colspan=25)

    light_boxplot(
        np.log10(gain) if log else gain, names=[''],
        mark_mean=True, box_ls='solid', ax=ax
    )
    ax.set_ylabel('log10 gain' if log else 'gain')

    ax = pp.subplot2grid((1, 100), (0, 25), colspan=75)
    _, cbar = chan_map.image(gain, ax=ax, **im_kws)
    cbar.set_label('array gain')
    return f

