import os
import scipy.io as sio

from ecogdata.util import Bunch
import ecogdata.devices.electrode_pinouts as epins

from ..units import convert_scale

def load_cooked(exp_path, test, electrode, **kwargs):
    m = sio.loadmat(os.path.join(exp_path, test+'.mat'))
    data = m.pop('dataf').copy(order='C')
    m = sio.loadmat(os.path.join(exp_path, test+'_trig.mat'))
    trigs = m.pop('indxfilt').squeeze()
    Fs = 500.0

    #chan_map = ChannelMap( range(1,9), (2,5), col_major=False )
    chan_map, _ = epins.get_electrode_map(electrode)

    bandpass = (2,-1)
    return data, trigs, Fs, chan_map, bandpass

def load_wireless(
        exp_path, test, electrode,
        bandpass=(), notches=(), 
        save=True, snip_transient=True,
        units='V'
        ):

    data, trigs, Fs, cmap, bpass = load_cooked(exp_path, test, electrode)
    if units.lower() != 'v':
        convert_scale(data, 'v', units)

    dset = Bunch(
        data=data,
        pos_edge=trigs,
        chan_map=cmap,
        Fs=Fs,
        bandpass=bpass,
        units=units
        )
    return dset
