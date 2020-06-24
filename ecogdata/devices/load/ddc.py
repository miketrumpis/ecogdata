import os.path as osp
import numpy as np
import scipy.io as sio
from ecogdata.channel_map import ChannelMap
from ecogdata.util import Bunch, mat_to_flat

from ecogdata.parallel.sharedmem import shared_copy
from ecogdata.parallel.split_methods import filtfilt
import ecogdata.filt.time as ft

from ..units import convert_dyn_range, convert_scale
from .open_ephys import load_open_ephys_channels

rows = np.array(    
    [0, 3, 2, 1, 0, 0, 4, 5, 6, 7, 6, 7, 1, 4, 4, -1, 0, 3, 1, 2, 3, 3, 4, 5, 6, 5, 6, 7, 2, 1, 5, 2, -1, 4, 4, 1, 7, 6, 7, 6, 5, 4, 3, 3, 1, 2, 0, 3, -1, 5, 1, 2, 7, 6, 5, 6, 5, 4, 2, 0, 2, 1, 0, 3]
    )

columns = np.array(
    [2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 0, 2, 1, -1, 3, 3, 3, 1, 1, 0, 3, 3, 3, 1, 1, 3, 0, 1, 2, 3, -1, 7, 4, 7, 4, 4, 6, 6, 6, 6, 7, 6, 5, 4, 5, 5, -1, 4, 6, 6, 5, 7, 7, 5, 5, 5, 7, 6, 5, 4, 4, 4]
    )

_dyn_range_lookup = dict(
    [ (3, (-0.6, 150.0)), (7, (-1.4, 350.0)), (0, (-0.04, 12.5)),
      (1, (-0.2, 50)), (2, (-0.4, 100)), (4, (-0.88, 200)), (5, (-0.1, 250)) ]
    )

def _load_cooked_pre_august_2014(pth, test, dyn_range, Fs):
    test_dir = osp.join(pth, test)

    chans_int = sio.loadmat(osp.join(test_dir, 'recs.mat'))['adcreads_sort']
    trigs = sio.loadmat(osp.join(test_dir, 'trigs.mat'))['stim_trig_sort']
    order = sio.loadmat(osp.join(test_dir, 'channels.mat'))['channel_numbers_sort']

    # this tells me how to reorder the row/column vectors above to match
    # the channel order in the file
    order = order[:, -1]

    _columns = columns[order]
    _rows = rows[order]

    electrode_chans = _rows >= 0

    chan_flat = mat_to_flat( 
        (8,8), _rows[electrode_chans], 7 - _columns[electrode_chans], 
        col_major=False 
        )
    chan_map = ChannelMap( chan_flat, (8,8), col_major=False, pitch=0.406 )
    
    chans = np.zeros(chans_int.shape, dtype='d')
    dr_lo, dr_hi = _dyn_range_lookup[dyn_range]
    #chans = (chans_int * ( Fs * (dr_hi - dr_lo) * 2**-20 )) + dr_lo*Fs
    chans = convert_dyn_range(chans_int, 2**20, (dr_lo, dr_hi))

    data = shared_copy(chans[electrode_chans])
    disconnected = chans[~electrode_chans]

    binary_trig = (np.any( trigs==1, axis=0 )).astype('i')
    if binary_trig.any():
        pos_edge = np.where( np.diff(binary_trig) > 0 )[0] + 1
    else:
        pos_edge = ()

    return data, disconnected, trigs, pos_edge, chan_map

def _load_cooked(pth, test, half=False, avg=False):
    # august 21 test -- now using a common test-name prefix
    # with different recording channels appended
    test_pfx = osp.join(pth, test)

    chans = sio.loadmat(test_pfx + '.ndata.mat')['raw_data'].T
    try:
        trigs = sio.loadmat(test_pfx + '.ndatastim.mat')['qw'].T
        trigs = trigs.squeeze()
    except IOError:
        trigs = np.zeros(10)

    _columns = np.roll(columns, 2)
    _rows = np.roll(rows, 2)

    electrode_chans = _rows >= 0

    chan_flat = mat_to_flat( 
        (8,8), _rows[electrode_chans], 7 - _columns[electrode_chans], 
        col_major=False 
        )
    chan_map = ChannelMap( chan_flat, (8,8), col_major=False, pitch=0.406 )
    
    # don't need to convert dynamic range
    #chans = np.zeros(chans_int.shape, dtype='d')
    #dr_lo, dr_hi = _dyn_range_lookup[dyn_range]
    #chans = convert_dyn_range(chans_int, 2**20, (dr_lo, dr_hi))

    if avg:
        chans = 0.5 * (chans[:, 1::2] + chans[:, 0::2])
        trigs = trigs[:, 1::2]
    if half:
        chans = chans[:, 1::2]
        trigs = trigs[:, 1::2]

    data = shared_copy(chans[electrode_chans])
    disconnected = chans[~electrode_chans]

    binary_trig = (np.any( trigs==1, axis=0 )).astype('i')
    if binary_trig.any():
        pos_edge = np.where( np.diff(binary_trig) > 0 )[0] + 1
    else:
        pos_edge = ()

    return data, disconnected, trigs, pos_edge, chan_map

def load_ddc(
        exp_path, test, electrode, drange,
        bandpass=(), notches=(), 
        save=True, snip_transient=True,
        Fs=1000, units='nA', **extra
        ):

    half = extra.get('half', False)
    avg = extra.get('avg', False)
    # returns data in coulombs, i.e. values are < 1e-12
    (data, 
     disconnected, 
     trigs, 
     pos_edge, 
     chan_map) = _load_cooked(exp_path, test, half=half, avg=avg)
    if half or avg:
        Fs = Fs / 2.0
    if 'a' in units.lower():
        data *= Fs
        data = convert_scale(data, 'a', units)
    elif 'c' in units.lower():
        data = convert_scale(data, 'c', units)

    if bandpass:
        (b, a) = ft.butter_bp(lo=bandpass[0], hi=bandpass[1], Fs=Fs)
        filtfilt(data, b, a)
    
    if notches:
        ft.notch_all(
            data, Fs, lines=notches, inplace=True, filtfilt=True
            )
    
    if snip_transient:
        snip_len = min(10000, pos_edge[0]) if len(pos_edge) else 10000
        data = data[..., snip_len:].copy()
        disconnected = disconnected[..., snip_len:].copy()
        if len(pos_edge):
            trigs = trigs[..., snip_len:]
            pos_edge -= snip_len

    dset = Bunch()
    dset.data = data
    dset.pos_edge = pos_edge
    dset.trigs = trigs
    dset.ground_chans = disconnected
    dset.Fs = Fs
    dset.chan_map = chan_map
    dset.bandpass = bandpass
    dset.transient_snipped = snip_transient
    dset.units = units
    dset.notches = notches
    return dset

def load_openephys_ddc(
        exp_path, test, electrode, drange, trigger_idx, rec_num='auto',
        bandpass=(), notches=(),
        save=False, snip_transient=True,
        units='nA', **extra
        ):

    rawload = load_open_ephys_channels(exp_path, test, rec_num=rec_num)
    all_chans = rawload.chdata
    Fs = rawload.Fs

    d_chans = len(rows)
    ch_data = all_chans[:d_chans]
    if np.iterable(trigger_idx):
        trigger = all_chans[int(trigger_idx[0])]
    else:
        trigger = all_chans[int(trigger_idx)]

    electrode_chans = rows >= 0
    chan_flat = mat_to_flat( 
        (8,8), rows[electrode_chans], 7 - columns[electrode_chans], 
        col_major=False 
        )
    chan_map = ChannelMap( chan_flat, (8,8), col_major=False, pitch=0.406 )
    
    dr_lo, dr_hi = _dyn_range_lookup[drange] # drange 0 3 or 7
    ch_data = convert_dyn_range(ch_data, (-2**15, 2**15), (dr_lo, dr_hi))
    
    data = shared_copy(ch_data[electrode_chans])
    disconnected = ch_data[~electrode_chans]

    trigger -= trigger.mean()
    binary_trig = ( trigger > 100 ).astype('i')
    if binary_trig.any():
        pos_edge = np.where( np.diff(binary_trig) > 0 )[0] + 1
    else:
        pos_edge = ()

    # change units if not nA
    if 'a' in units.lower():
        # this puts it as picoamps
        data *= Fs
        data = convert_scale(data, 'pa', units)
    elif 'c' in units.lower():
        data = convert_scale(data, 'pc', units)

        
    if bandpass: # how does this logic work? 
        (b, a) = ft.butter_bp(lo=bandpass[0], hi=bandpass[1], Fs=Fs)
        filtfilt(data, b, a)
    
    if notches:
        ft.notch_all(
            data, Fs, lines=notches, inplace=True, filtfilt=True
            )
    
    if snip_transient:
        snip_len = min(10000, pos_edge[0]) if len(pos_edge) else 10000
        data = data[..., snip_len:].copy()
        if len(disconnected):
            disconnected = disconnected[..., snip_len:].copy()
        if len(pos_edge):
            trigger = trigger[..., snip_len:]
            pos_edge -= snip_len

    dset = Bunch()
    dset.data = data
    dset.pos_edge = pos_edge
    dset.trigs = trigger
    dset.ground_chans = disconnected
    dset.Fs = Fs
    dset.chan_map = chan_map
    dset.bandpass = bandpass
    dset.transient_snipped = snip_transient
    dset.units = units
    dset.notches = notches
    return dset

