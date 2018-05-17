from builtins import zip
from builtins import map
from builtins import range
import os
import numpy as np
import ecogdata.filt.time as ft
import ecogdata.util as ut
from ecogdata.datastore import load_bunch, save_bunch
from ecogdata.trigger_fun import process_trigger
from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.parallel.split_methods import filtfilt
import ecogdata.devices.electrode_pinouts as epins

from . import DataPathError
from .util import try_saved, tdms_info
from ..units import convert_scale

mux_gain = dict(
    mux3 = 10,
    mux5 = 10,
    joe_mux3 = 1,
    mux4 = 27.6,
    mux6 = 20,
    mux7 = 12,
    mux7_lg = 3,
    stim_mux1 = 1,
    stim_mux64 = 4,
    stim_v4 = 4
    )

mux_headstages = list(mux_gain.keys())

# different mux/daq combinations have sampled the
# digital out lines in various orders
mux_sampling = dict(
    mux4 = [3, 0, 2, 1],
    mux5 = [1, 0, 2, 3],
    mux6 = [0, 2, 1, 3],
    mux7_1card = [1, 2, 0, 3],
    mux7_2card_flip = [2, 0, 3, 1],
    stim4_1card = [3, 1, 0, 2]
    )

def _permute_mux(channels, rows, daq_variant):
    if daq_variant not in mux_sampling:
        return channels
    p_order = mux_sampling[daq_variant]
    cshape = channels.shape
    if channels.ndim < 3:
        channels = channels.reshape(-1, rows, cshape[-1])
    crange = list(range(channels.shape[0]))

    permuting = p_order + crange[4:]
    channels = channels[ permuting ]
    return channels.reshape(cshape)

def rawload_mux(
        exp_path, test, version, daq_variant='', data_only=False, shm=True
        ):
    """
    Find and load data recorded from the MUX style headstage. Return all
    recording columns by default, otherwise only return the electrode
    data.
    
    """
    raw_data = None
    shm_arr = ('/data',) if shm else ()
    try:
        raw_data = load_bunch(
            os.path.join(exp_path, test+'.h5'), '/',
            shared_arrays=shm_arr
            )
    except IOError:
        raw_data = load_bunch(
            os.path.join(exp_path, test+'.mat'), '/',
            shared_arrays=shm_arr
            )
    try:
        Fs = raw_data.Fs
    except:
        Fs = raw_data.fs
    shape = raw_data.data.shape
    if shape[1] < shape[0]:
        raw_data.data = raw_data.data.transpose().copy()
    nrow, ncol_data = list(map(int, (raw_data.numRow, raw_data.numCol)))
    # scale data channels
    raw_data.data[:ncol_data*nrow] /= mux_gain[version]
    # correct for permuted digital out sampling
    if not daq_variant:
        # if daq info (new style) is not given, try to look up sampling order
        # based on the mux version (old style)
        daq_variant = version
    raw_data.data = _permute_mux(raw_data.data, nrow, daq_variant)
    if data_only:
        raw_data.data = raw_data.data[:ncol_data*nrow]

    try:
        # stim-mux converted h5 files (Virginia's conversion)
        # do not have info
        info = tdms_info(raw_data.info)
    except AttributeError:
        info = None
    return raw_data.data, Fs, (nrow, ncol_data), info
    
def load_mux(
        exp_path, test, electrode, headstage,
        ni_daq_variant='', mux_connectors=(),
        bandpass=(), notches=(), 
        trigger=0, bnc=(),
        mux_notches=(),
        save=False, snip_transient=True,
        units='uV'
        ):

    """
    Load data from the MUX style headstage acquisition. Data is expected 
    to be organized along columns corresponding to the MUX units. The
    columns following sensor data columns are assumed to be a stimulus
    trigger followed by other BNC channels.

    The electrode information must be provided to determine the
    arrangement of recorded and grounded channels within the sensor
    data column.
    
    This preprocessing routine returns a Bunch container with the
    following items
    
    dset.data : nchan x ntime data array
    dset.ground_chans : m x ntime data array of grounded ADC channels
    dset.bnc : un-MUXed readout of the BNC channel(s)
    dset.chan_map : the channel-to-electrode mapping vector
    dset.Fs : sampling frequency
    dset.name : path + expID for the given data set
    dset.bandpass : bandpass filtering applied (if any)
    dset.trig : the logical value of the trigger channel (at MUX'd Fs)

    * If saving, then a table of the Bunch is written.
    * If snip_transient, then advance the timeseries past the bandpass
      filtering onset transient.
    
    """

    try:
        dset = try_saved(exp_path, test, bandpass)
        return dset
    except DataPathError:
        pass
        
    # say no to shared memory since it's created later on in this method
    loaded = rawload_mux(exp_path, test, headstage,
                         daq_variant=ni_daq_variant, shm=False)
    channels, Fs, dshape, info = loaded
    nrow, ncol_data = dshape
    if channels.shape[0] >= nrow * ncol_data:
        ncol = channels.shape[0] / nrow
        channels = channels.reshape(ncol, nrow, -1)
    else:
        ncol = channels.shape[0]
        channels.shape = (ncol, -1, nrow)
        channels = channels.transpose(0, 2, 1)
    
    ## Grab BNC data

    if bnc:
        bnc_chans = [ncol_data + int(b) for b in bnc]
        bnc = np.zeros( (len(bnc), nrow * channels.shape[-1]) )
        for bc, col in zip(bnc, bnc_chans):
            bc[:] = channels[col].transpose().ravel()
        bnc = bnc.squeeze()

    try:
        trig_chans = channels[ncol_data+trigger].copy()
        pos_edge, trig = process_trigger(trig_chans)
    except IndexError:
        pos_edge = ()
        trig = ()

    ## Realize channel mapping

    chan_map, disconnected = epins.get_electrode_map(
        electrode, connectors=mux_connectors
        )

    ## Data channels

    # if any pre-processing of multiplexed channels, do it here first
    if mux_notches:
        
        mux_chans = shared_ndarray( (ncol_data, channels.shape[-1], nrow) )
        mux_chans[:] = channels[:ncol_data].transpose(0, 2, 1)
        mux_chans.shape = (ncol_data, -1)
        ft.notch_all(
            mux_chans, Fs, lines=mux_notches, filtfilt=True
            )
        mux_chans.shape = (ncol_data, channels.shape[-1], nrow)
        channels[:ncol_data] = mux_chans.transpose(0, 2, 1)
        del mux_chans
             
    rec_chans = channels[:ncol_data].reshape(nrow*ncol_data, -1)

    if units.lower() != 'v':
        convert_scale(rec_chans, 'v', units)
    
    g_chans = disconnected
    d_chans = np.setdiff1d(np.arange(ncol_data*nrow), g_chans)

    data_chans = shared_ndarray((len(d_chans), rec_chans.shape[-1]))
    data_chans[:,:] = rec_chans[d_chans]
    gnd_data = rec_chans[g_chans]
    del rec_chans
    del channels
    
    # do highpass filtering for stationarity
    if bandpass:
        # manually remove DC from channels before filtering
        if bandpass[0] > 0:
            data_chans -= data_chans.mean(1)[:,None]
            # do a high order highpass to really crush the crappy 
            # low frequency noise
            b, a = ft.butter_bp(lo=bandpass[0], Fs=Fs, ord=5)
            #b, a = ft.cheby1_bp(0.5, lo=bandpass[0], Fs=Fs, ord=5)
        else:
            b = [1]
            a = [1]
        if bandpass[1] > 0:
            b_lp, a_lp = ft.butter_bp(hi=bandpass[1], Fs=Fs, ord=3)
            b = np.convolve(b, b_lp)
            a = np.convolve(a, a_lp)
        
        ## ft.filter_array(
        ##     data_chans, 
        ##     design_kwargs=dict(lo=bandpass[0], hi=bandpass[1], Fs=Fs),
        ##     filt_kwargs=dict(filtfilt=True)
        ##     )
        filtfilt(data_chans, b, a)

    if notches:
        ft.notch_all(
            data_chans, Fs, lines=notches, inplace=True, filtfilt=True
            )

    if snip_transient:
        if isinstance(snip_transient, bool):
            snip_len = int( Fs * 5 )
        else:
            snip_len = int( Fs * snip_transient )
        if len(pos_edge):
            snip_len = max(0, min(snip_len, pos_edge[0] - int(Fs)))
            pos_edge = pos_edge - snip_len
            trig = trig[...,snip_len:].copy()
        data_chans = data_chans[...,snip_len:].copy()
        gnd_data = gnd_data[...,snip_len:].copy()
        if len(bnc):
            bnc = bnc[...,snip_len*nrow:].copy()

    # do blockwise detrending for stationarity
    ## detrend_window = int(round(0.750*Fs))
    ## ft.bdetrend(data_chans, bsize=detrend_window, type='linear', axis=-1)
    dset = ut.Bunch()
    dset.pos_edge = pos_edge
    dset.data = data_chans
    dset.ground_chans = gnd_data
    dset.bnc = bnc
    dset.chan_map = chan_map
    dset.Fs = Fs
    #dset.name = os.path.join(exp_path, test)
    dset.bandpass = bandpass
    dset.trig = trig
    dset.transient_snipped = snip_transient
    dset.units = units
    dset.notches = notches
    dset.info = info

    if save:
        hf = os.path.join(exp_path, test+'_proc.h5')
        save_bunch(hf, '/', dset, mode='w')
    return dset

        
