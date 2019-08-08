"""
This module supports loading Blackrock data acquired via a ZIF 
passthrough circuit. The circuit exposes the ZIF pins in two banks of 
30 and 31 pins. The Blackrock system acquires these in 32 channel blocks
(up to a maximum of 96 channels).
"""

import os
import os.path as p
import sys
import gc
import tables
import numpy as np

import ecogdata.parallel.array_split as array_split
from ecogdata.parallel.split_methods import filtfilt
import ecogdata.devices.electrode_pinouts as epins
from ecogdata.filt.time import *
import ecogdata.util as ut

from ..units import convert_scale, convert_dyn_range

def load_blackrock(
        exp_path, test, electrode, connections=(), 
        downsamp=15, page_size=10, bandpass=(), notches=(), 
        save=True, snip_transient=True, lowpass_ord=12, units='uV',
        **extra
        ):
    """
    Load raw data in an HDF5 table stripped from Blackrock NSx format.
    This data should be 16 bit signed integer sampled at 30 kHz. We
    take the approach of resampling to a lower rate (default 2 kHz)
    before subsequent bandpass filtering. This improves the numerical
    stability of the bandpass filter design.

    """

    dsamp_path = p.join(exp_path, 'downsamp')
    nsx_path = p.join(exp_path, 'blackrock')

    ## Get array-to-channel pinouts
    chan_map, disconnected = epins.get_electrode_map(electrode, connectors=connections)[:2]

    # try preproc path first to see if this run has already been downsampled
    load_nsx = True
    if downsamp > 1:
        dsamp_Fs = 3e4 / downsamp
        try:
            test_file = p.join(dsamp_path, test) + '_Fs%d.h5'%dsamp_Fs
            print('searching for', test_file)
            h5f = tables.open_file(test_file)
            downsamp = 1
            load_nsx = False
        except IOError:
            print('Resampled data not found: downsampling to %d Hz'%dsamp_Fs)

    if load_nsx:
        test_file = p.join(nsx_path, test+'.h5')
        h5f = tables.open_file(test_file)

    if downsamp > 1:
        (b, a) = cheby2_bp(60, hi=1.0/downsamp, Fs=2, ord=lowpass_ord)

        if not p.exists(dsamp_path):
            os.mkdir(dsamp_path)
        save_file = p.join(dsamp_path, test) + '_Fs%d.h5'%dsamp_Fs
        h5_save = tables.open_file(save_file, mode='w')
        h5_save.create_array(h5_save.root, 'Fs', dsamp_Fs)
    else:
        # in this case, either the preprocessed data has been found,
        # or downsampling was not requested, which will probably
        # *NEVER* happen
        if load_nsx:
            dlen, nchan = h5f.root.data.shape
            required_mem = dlen * nchan * np.dtype('d').itemsize
            if required_mem > 8e9:
                raise MemoryError(
                    'This dataset would eat %.2f GBytes RAM'%(required_mem/1e9,)
                    )

    dlen, nchan = h5f.root.data.shape
    if dlen < nchan:
        (dlen, nchan) = (nchan, dlen)
        tdim = 1
    else:
        tdim = 0
    
    sublen = dlen / downsamp
    if dlen - sublen*downsamp > 0:
        sublen += 1

    # set up arrays for loaded data and ground chans
    subdata = array_split.shared_ndarray((len(chan_map), sublen))
    if len(chan_map) < nchan:
        gndchan = np.empty((len(disconnected), sublen), 'd')
    else:
        gndchan = None

    # if saving downsampled results, set up H5 table (in c-major fashion)
    if downsamp > 1:
        atom = tables.Float64Atom()
        #filters = tables.Filters(complevel=5, complib='zlib')
        filters = None
        saved_array = h5_save.create_earray(
            h5_save.root, 'data', atom=atom, shape=(0, sublen),
            filters=filters, expectedrows=nchan
            )
    if page_size < 0:
        page_size = nchan
    peel = array_split.shared_ndarray( (page_size, dlen) )
    n = 0
    dstop = 0
    h5_data = h5f.root.data
    while n < nchan:
        start = n
        stop = min(nchan, n+page_size)
        print('processing BR channels %03d - %03d'%(start, stop-1))
        if tdim == 0:
            peel[0:stop-n] = h5_data[:,start:stop].T.astype('d', order='C')
        else:
            peel[0:stop-n] = h5_data[start:stop,:].astype('d')
        if downsamp > 1:
            convert_dyn_range(peel, (-2**15, 2**15), (-8e-3, 8e-3), out=peel)
            print('parfilt', end=' ')
            sys.stdout.flush()
            filtfilt(peel[0:stop-n], b, a)
            print('done')
            sys.stdout.flush()
            print('saving chans', end=' ') 
            sys.stdout.flush()
            saved_array.append(peel[0:stop-n,::downsamp])
            print('done')
            sys.stdout.flush()

        if units.lower() != 'v':
            convert_scale(peel, 'v', units)
        data_chans = np.setdiff1d(np.arange(start,stop), disconnected)
        if len(data_chans):
            dstart = dstop
            dstop = dstart + len(data_chans)
        if len(data_chans) == (stop-start):
            # if all data channels, peel off in a straightforward way
            #print (dstart, dstop)
            subdata[dstart:dstop,:] = peel[0:stop-n,::downsamp]
        else:
            if len(data_chans):
                # get data channels first
                raw_data = peel[data_chans-n, :]
                #print (dstart, dstop), data_chans-n
                subdata[dstart:dstop, :] = raw_data[:, ::downsamp]
            # Now filter for ground channels within this set of channels:
            gnd_chans = [x for x in zip(disconnected,
                                        range(len(disconnected)))
                            if x[0]>=start and x[0]<stop]
            for g in gnd_chans:
                gndchan[g[1], :] = peel[g[0]-n, ::downsamp]
        n += page_size

    del peel
    try:
        Fs = h5f.root.Fs.read()[0,0] / downsamp
    except TypeError:
        Fs = h5f.root.Fs.read() / downsamp
    trigs = h5f.root.trig_idx.read().squeeze()
    if not trigs.shape:
        trigs = ()
    else:
        trigs = np.round( trigs / downsamp ).astype('i')
    h5f.close()
    if downsamp > 1:
        h5_save.create_array(h5_save.root, 'trig_idx', np.asarray(trigs))
        h5_save.close()

    # seems to be more numerically stable to do highpass and 
    # notch filtering after downsampling
    if bandpass:
        lo, hi = bandpass
        (b, a) = butter_bp(lo=lo, hi=hi, Fs=Fs, ord=4)
        filtfilt(subdata, b, a)
        
    if notches:
        notch_all(subdata, Fs, lines=notches, inplace=True, filtfilt=True)

        
    dset = ut.Bunch()
    dset.data = subdata
    dset.ground_chans = gndchan
    dset.chan_map = chan_map
    dset.Fs = Fs
    #dset.name = os.path.join(exp_path, test)
    dset.bandpass = bandpass
    dset.notches = notches
    dset.trig = trigs
    if len(trigs) == subdata.shape[-1]:
        dset.pos_edge = np.where( np.diff(trigs) > 0 )[0] + 1
    else:
        dset.pos_edge = trigs
    dset.units = units
    gc.collect()
    return dset
    

