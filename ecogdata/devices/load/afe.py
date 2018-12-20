from builtins import range
import tables
import numpy as np
import os
import ecogdata.util as ut
import ecogdata.filt.time as ft
import ecogdata.devices.electrode_pinouts as epins
from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.parallel.split_methods import filtfilt

from ..units import convert_dyn_range, convert_scale

# different ranges code for dynamic range of charge
range_lookup = dict(
    [ (0, 0.13), (1, 0.25), (2, 0.5),
      (3, 1.2), (4, 2.4), (5, 4.8), (6, 7.2), (7, 9.6) ]
      )

# valid units: (micro)Volts, (pico)Coulombs, (nano)Amps
# encoding: 'uv', 'v', 'pc', 'c', 'na', 'a'

def load_afe_aug21(
        exp_pth, test, electrode, n_data, range_code, cycle_rate, units='nA',
        bandpass=(), save=False, notches=(), snip_transient=True, **extra
        ):
    h5 = tables.open_file(os.path.join(exp_pth, test+'.h5'))
    Fs = h5.root.Fs.read()

    n_row = h5.root.numRow.read()
    n_data_col = h5.root.numCol.read()
    n_col = h5.root.numChan.read()
    # data rows are 4:67 -- acquiring AFE chans 31:0 and 63:32 on two columns
    data_rows = slice(4, 67, 2) 
    
    full_data = h5.root.data[:].reshape(n_col, n_row, -1)

    #data_chans = shared_ndarray( (32*n_data_col, full_data.shape[-1]) )
    data_chans = full_data[:n_data_col, data_rows].reshape(-1, full_data.shape[-1])
    trig_chans = full_data[-10:, -1]
    del full_data

    trig = np.any( trig_chans > 1, axis = 0 ).astype('i')
    pos_edge = np.where( np.diff(trig) > 0 )[0] + 1
    

    # convert dynamic range to charge or current
    if 'v' not in units.lower():
        pico_coulombs = range_lookup[range_code]
        convert_dyn_range(
            data_chans, (-1.4, 1.4), pico_coulombs, out=data_chans
            )
        if 'a' in units.lower():
            i_period = 563e-6
            data_chans /= i_period
            convert_scale(data_chans, 'pa', units)
    elif units.lower() != 'v':
        convert_scale(data_chans, 'v', units)

    ## # only use this one electrode (for now)
    chan_map, disconnected = epins.get_electrode_map('psv_61_afe')[:2]
    connected = np.setdiff1d(np.arange(n_data), disconnected)
    disconnected = disconnected[ disconnected < n_data ]
        
    data = shared_ndarray( (len(connected), data_chans.shape[-1]) )
    data[:, :] = data_chans[connected]
    ground_chans = data_chans[disconnected]

    del data_chans
    # do a little extra to kill DC
    data -= data.mean(axis=1)[:,None]
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
        if ground_chans is not None:
            ground_chans = ground_chans[..., snip_len:].copy()
        if len(pos_edge):
            trig = trig[..., snip_len:]
            pos_edge -= snip_len

    dset = ut.Bunch()

    dset.data = data
    dset.ground_chans = ground_chans
    dset.chan_map = chan_map
    dset.Fs = Fs
    dset.pos_edge = pos_edge
    dset.bandpass = bandpass
    dset.trig = trig
    dset.transient_snipped = snip_transient
    dset.units = units
    dset.notches = notches
    return dset

    

def load_afe(
        exp_pth, test, electrode, n_data, range_code, cycle_rate, 
        units='nA', bandpass=(), save=True, notches=(), 
        snip_transient=True, **extra
        ):

    h5 = tables.open_file(os.path.join(exp_pth, test+'.h5'))

    data = h5.root.data[:]
    Fs = h5.root.Fs[0,0]

    if data.shape[1] > n_data:
        trig_chans = data[:,n_data:]
        trig = np.any( trig_chans > 1, axis = 1 ).astype('i')
        pos_edge = np.where( np.diff(trig) > 0 )[0] + 1
    else:
        trig = None
        pos_edge = ()
    
    data_chans = data[:,:n_data].T.copy(order='C')

    # convert dynamic range to charge or current
    if 'v' not in units.lower():
        pico_coulombs = range_lookup[range_code]
        convert_dyn_range(
            data_chans, (-1.4, 1.4), pico_coulombs, out=data_chans
            )
        if 'a' in units.lower():
            # To convert to amps, need to divide coulombs by the 
            # integration period. This is found approximately by
            # finding out how many cycles in a scan period were spent
            # integrating. A scan period is now hard coded to be 500
            # cycles. The cycling rate is given as an argument.
            # The integration period for channel i should be:
            # 500 - 2*(n_data - i)
            # That is, the 1st channel is clocked out over two cycles
            # immediately after the integration period. Meanwhile other 
            # channels still acquire until they are clocked out.
            n_cycles = 500
            #i_cycles = n_cycles - 2*(n_data - np.arange(n_data))
            i_cycles = n_cycles - 2*n_data
            i_period = i_cycles / cycle_rate
            data_chans /= i_period #[:,None]
            convert_scale(data_chans, 'pa', units)
    elif units.lower() != 'v':
        convert_scale(data, 'v', units)

    # only use this one electrode (for now)
    chan_map, disconnected = epins.get_electrode_map('psv_61_afe')[:2]
    connected = np.setdiff1d(np.arange(n_data), disconnected)
    disconnected = disconnected[ disconnected < n_data ]
    
    chan_map = chan_map.subset(list(range(len(connected))))
    
    data = shared_ndarray( (len(connected), data_chans.shape[-1]) )
    data[:,:] = data_chans[connected]
    ground_chans = data_chans[disconnected].copy()
    del data_chans

    if bandpass:
        # do a little extra to kill DC
        data -= data.mean(axis=1)[:,None]
        (b, a) = ft.butter_bp(lo=bandpass[0], hi=bandpass[1], Fs=Fs)
        filtfilt(data, b, a)
    if notches:
        for freq in notches:
            (b, a) = ft.notch(freq, Fs=Fs, ftype='cheby2')
            filtfilt(data, b, a)

    ## detrend_window = int(round(0.750*Fs))
    ## ft.bdetrend(data, bsize=detrend_window, type='linear', axis=-1)
    else:
        data -= data.mean(axis=1)[:,None]

    if snip_transient:
        snip_len = min(10000, pos_edge[0]) if len(pos_edge) else 10000
        data = data[..., snip_len:].copy()
        ground_chans = ground_chans[..., snip_len:].copy()
        if len(pos_edge):
            trig = trig[..., snip_len:]
            pos_edge -= snip_len

    dset = ut.Bunch()

    dset.data = data
    dset.ground_chans = ground_chans
    dset.chan_map = chan_map
    dset.Fs = Fs
    dset.pos_edge = pos_edge
    dset.bandpass = bandpass
    dset.trig = trig
    dset.transient_snipped = snip_transient
    dset.units = units
    dset.notches = notches
    return dset