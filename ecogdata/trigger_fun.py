import numpy as np
import scipy.signal as signal

import ecogdata.util as ut
from ecogdata.expconfig.exp_descr import StimulatedExperiment
import ecogdata.parallel.array_split as array_split


def _auto_level(ttl, verbose=False):
    """Iteratively refine an estimate of the high-level cluster
    of points in a TTL signal.
    """

    n = ttl.size
    mn = ttl.mean()
    # refine until the current subset is < 1% of the signal
    # while float(ttl.size) / n > 1e-3:
    # refine until the current subset is less than 1000 pts (good heuristic??)
    while float(ttl.size) > 1000:
        ttl = ttl[ttl > mn]
        if verbose:
            mn = ttl.mean()
            sz = len(ttl)
            pct = 100 * float(sz) / n
            print('new level: {0:.2f}; subset size: {1} ({2:.1f} %)'.format(mn, sz, pct))
        if len(ttl):
            mn = ttl.mean()
            # it's possible there are > 1000 values all clipped
            # to the same high level, in which case lower the
            # current level by a hair and break
            if np.abs(mn - ttl.max()) < 1e-8:
                mn *= 0.99
                break
        else:
            break
    return mn


def process_trigger(trig_chan, thresh=0.5, uniform=True, clean=False):
    """Pull event timing from one or many logical-level channels.

    Parameters
    ----------
    trig_chan : ndarray
        Vector(s) of event timing square waves.
    thresh : float (0.5)
        Relative threshold for detecting a rising edge.
    uniform : bool
        If (quasi-) periodic triggers, then use a uniform distribution
        heuristic to debounce triggers
    clean : bool
        Check rising edge times for spurious edges (e.g. from noisy trigger)

    Returns
    -------
    pos_edge : ndarray
        Sequence of event times (indices)
    digital_trigger : ndarray
        Binarized trigger vector
    """
    trig_chan = np.atleast_2d(trig_chan)
    if trig_chan.dtype.char != '?':
        trig_chan = trig_chan - trig_chan.mean()
        thresh = thresh * _auto_level(trig_chan)
        trig_chan = trig_chan > thresh
    digital_trigger = np.any(trig_chan, axis=0).astype('i')
    pos_edge = np.where(np.diff(digital_trigger) > 0)[0] + 1

    if uniform:
        # Mask out any edges with *very* small ISI.
        # Assume uniformly distributetd ISIs, and that 5-95 percentile
        # represents 90% of the mass and up to 10% of the mass is evenly
        # concentrated below p(5). Reject any ISI lower than
        # p(95) - (p(95) - p(5)) * 10 / 9
        isi_raw = np.diff(pos_edge)
        p5, p95 = np.percentile(isi_raw, [5, 95])
        min_credible_isi = p95 - (p95 - p5) / 0.9
        pos_edge_ = clean_dirty_trigger(pos_edge,
                                        isi_guess=min_credible_isi)
        sdiff = np.setdiff1d(pos_edge, pos_edge_)
        if len(sdiff):
            print('Warning.. spurious triggers auto-detected.')
            rej = pos_edge.searchsorted(sdiff) - 1
            print('Rejected ISIs were', isi_raw[rej])
        pos_edge = pos_edge_
    if clean:
        pos_edge = clean_dirty_trigger(pos_edge)
    return pos_edge, digital_trigger


def clean_dirty_trigger(pos_edges, isi_guess=None):
    """Clean spurious event times (with suspect inter-stimulus intervals).

    Parameters
    ----------
    pos_edges : array-like
        Sequence of timestamps
    isi_guess : int (optional)
        Prior for ISI. Otherwise guess ISI based on 90th percentile.

    Returns
    -------
    array
        The pruned timestamps.
    """
    if len(pos_edges) < 3:
        return pos_edges
    df = np.diff(pos_edges)
    if isi_guess is None:
        isi_guess = np.percentile(df, 90)

    # lose any edges that are < half of the isi_guess
    edge_mask = np.ones(len(pos_edges), '?')

    for i in range(len(pos_edges)):
        if not edge_mask[i]:
            continue

        # look ahead and kill any edges that are
        # too close to the current edge
        pt = pos_edges[i]
        kill_mask = (pos_edges > pt) & (pos_edges < pt + isi_guess / 2)
        edge_mask[kill_mask] = False

    return pos_edges[edge_mask]

# define some trigger-locked aggregating utilities


def trigs_and_conds(trig_code):
    if isinstance(trig_code, np.ndarray) or \
            isinstance(trig_code, tuple) or \
            isinstance(trig_code, list):
        trigs, conds = trig_code
    elif isinstance(trig_code, StimulatedExperiment):
        try:
            trigs = trig_code.time_stamps
        except AttributeError:
            trigs = trig_code.trig_times
        conds, _ = trig_code.enumerate_conditions()
    return trigs, conds


def _ep_trigger_avg(x, trig_code, pre=0, post=0, iqr_thresh=-1, envelope=False):
    """
    Average response to 1 or more experimental conditions

    Arguments
    ---------
    x: data (nchan, npts)
    trig_code : sequence-type (2, stim) or StimulatedExperiment
        First row is the trigger indices, second row is a condition
        ID (integer). Condition ID -1 codes for a flagged trial to
        be skipped. If a StimulatedExperiment, then triggers and
        conditions are available from this object.
    pre, post : ints
        Number of pre- and post-stim samples in interval. post + pre > 0
        default: 0 and stim-to-stim interval
    sum_limit : int
        Do partial sum up to this many terms
    iqr_thresh : float
        If set, do simple outlier detection on all groups of repeated
        conditions based on RMS power in the epoch interval. The iqr_thresh
        multiplies the width of the inter-quartile range to determine the
        "inlier" range of RMS power.

    Returns
    -------
    avg
        (nchan, ncond, epoch_length)
    n_avg
        number of triggers found for each condition
    skipped
        (nskip, nchan, epoch_length) epochs that were not averaged

    """
    x.shape = (1,) + x.shape if x.ndim == 1 else x.shape
    #pos_edge = trig_code[0]; conds = trig_code[1]
    pos_edge, conds = trigs_and_conds(trig_code)
    epoch_len = int(np.round(np.median(np.diff(pos_edge))))

    n_cond = len(np.unique(conds))
    n_pt = x.shape[1]

    if not (post or pre):
        post = epoch_len

    # this formula should provide consistent epoch lengths,
    # no matter the offset
    epoch_len = int(round(post + pre))
    pre = int(round(pre))
    post = epoch_len - pre

    # edit trigger list to exclude out-of-bounds epochs
    while pos_edge[0] - pre < 0:
        pos_edge = pos_edge[1:]
        conds = conds[1:]
    while pos_edge[-1] + post >= n_pt:
        pos_edge = pos_edge[:-1]
        conds = conds[:-1]

    avg = np.zeros((x.shape[0], n_cond, epoch_len), x.dtype)
    n_avg = np.zeros((x.shape[0], n_cond), 'i')

    for n, c in enumerate(np.unique(conds)):
        trials = np.where(conds == c)[0]
        if not len(trials):
            continue
        epochs = extract_epochs(x, pos_edge, trials, pre, post)
        if iqr_thresh > 0:
            pwr = np.sqrt(np.sum(epochs**2, axis=-1))
            # analyze outlier trials per channel
            out_mask = ut.fenced_out(
                pwr, thresh=iqr_thresh, axis=1, low=False
            )
            epochs = epochs * out_mask[:, :, None]
            n_avg[:, n] = np.sum(out_mask, axis=1)
        else:
            n_avg[:, n] = len(trials)

        if envelope:
            epochs = signal.hilbert(
                epochs, N=ut.nextpow2(epoch_len), axis=-1
            )
            epochs = np.abs(epochs[..., :epoch_len])**2

        avg[:, c - 1, :] = np.sum(epochs, axis=1) / n_avg[:, c - 1][:, None]

    x.shape = [x for x in x.shape if x > 1]
    if envelope:
        np.sqrt(avg, avg)
    return avg, n_avg


# create a parallelized version (but only split big jobs)
ep_trigger_avg = array_split.split_at(splice_at=(0, 1), split_over=100)(_ep_trigger_avg)


def iter_epochs(x, pivots, selected=(), pre=0, post=0, fill=np.nan):
    """
    Generator that yields epochs pivoted at the specified triggers.

    Parameters
    ----------
    x : data (n_chan, n_pt)
    pivots : array-like or StimulatedExperiment
        A sequence of literal pivot samples, or an experiment wrapper
        containing the timestamps.
    selected : sequence
        Indices into trig_code for a subset of stims. If empty, return *ALL*
        epochs (*a potentially very large array*)
    pre, post : ints
        Number of pre- and post-stim samples in interval. post + pre > 0
        default: 0 and stim-to-stim interval

    """

    x = np.atleast_2d(x) if x.ndim == 1 else x
    if isinstance(pivots, StimulatedExperiment):
        pivots, _ = trigs_and_conds(pivots)
    if not np.iterable(pivots):
        pivots = [pivots]

    if not (post or pre):
        if len(pivots) > 1:
            print('Default epoch length based on median inter-trial time')
            post = int(np.median(np.diff(pivots)))
        else:
            print('Default epoch length 200 pts')
            post = 200

    epoch_len = int(round(post + pre))
    pre = int(round(pre))
    post = epoch_len - pre

    if len(selected):
        if hasattr(selected, 'dtype') and selected.dtype.char == '?':
            selected = np.where(selected)[0]
        pivots = np.take(pivots, selected)

    epoch = np.empty((x.shape[0], epoch_len), x.dtype)

    for k in pivots:
        if k - pre < 0:
            start_put = pre - k
            pre = k
        else:
            start_put = 0
        if k + post >= x.shape[1]:
            stop_put = x.shape[1] - k + pre
            post = x.shape[1] - k
        else:
            stop_put = pre + post

        grab_idx = (slice(None), slice(k - pre, k + post))
        put_idx = (slice(None), slice(start_put, stop_put))
        if start_put > 0 or stop_put < pre + post:
            epoch.fill(fill)
        epoch[put_idx] = x[grab_idx]
        yield epoch.copy()
    return


def extract_epochs(x, pivots, selected=(), pre=0, post=0, fill=np.nan):
    """
    Extract an array of epochs pivoted at the specified triggers. Note
    that this method follows a garbage-in, garbage-out policy
    regarding time stamps and epoch intervals. A ValueError will be
    raised if the requested interval goes out of bounds in the recording.

    Parameters
    ----------
    x : data (n_chan, n_pt)
    pivots : array-like or StimulatedExperiment
        A sequence of literal pivot samples, or an experiment wrapper
        containing the timestamps.
    selected : sequence
        Indices into trig_code for a subset of stims. If empty, return *ALL*
        epochs (*a potentially very large array*)
    pre, post : ints
        Number of pre- and post-stim samples in interval. post + pre > 0
        default: 0 and stim-to-stim interval

    Returns
    -------
    epochs : array (n_chan, n_epoch, epoch_len)

    """
    x = np.atleast_2d(x) if x.ndim == 1 else x
    if isinstance(pivots, StimulatedExperiment):
        pivots, _ = trigs_and_conds(pivots)
    if not np.iterable(pivots):
        pivots = [pivots]

    if not (post or pre):
        if len(pivots) > 1:
            print('Default epoch length based on median inter-trial time')
            post = int(np.median(np.diff(pivots)))
        else:
            print('Default epoch length 200 pts')
            post = 200

    epoch_len = int(round(post + pre))
    pre = int(round(pre))
    post = epoch_len - pre

    if len(selected):
        if hasattr(selected, 'dtype') and selected.dtype.char == '?':
            selected = np.where(selected)[0]
        pivots = np.take(pivots, selected)

    epochs = np.empty((x.shape[0], len(pivots), epoch_len), x.dtype)
    gen_epochs = iter_epochs(x, pivots, pre=pre, post=post, fill=fill)
    for n, e in enumerate(gen_epochs):
        epochs[:, n, :] = e
    return epochs
