from __future__ import division
from __future__ import print_function
from builtins import next
from builtins import str
from builtins import map
from builtins import range
import os.path as osp
import sys
from glob import glob
import gc

try:
  from lxml import etree
except ImportError:
  try:
    # Python 2.5
    import xml.etree.cElementTree as etree
  except ImportError:
      import sys
      print("What's wrong with your distro??")
      sys.exit(1)

import numpy as np
import tables

from ecogdata.trigger_fun import process_trigger
from ecogdata.filt.time import cheby2_bp, butter_bp, notch_all, downsample
from ecogdata.util import Bunch, mkdir_p
from ecogdata.datastore import load_bunch, save_bunch
from ecogdata.parallel.array_split import shared_copy, shared_ndarray, parallel_controller
from ecogdata.parallel.split_methods import filtfilt

from . import _OpenEphys as OE
from ecogdata.devices.units import convert_scale
from ecogdata.devices.electrode_pinouts import get_electrode_map

_srates = (1000, 1250, 1500, 2000, 2500, 3000, 1e4/3, 
           4000, 5000, 6250, 8000, 10000, 12500, 15000,
           20000, 25000, 30000)

def get_robust_samplingrate(rec_path):
    settings = glob( osp.join(rec_path, 'settings*.xml') )
    #xml = osp.join(rec_path, 'settings.xml')
    #if not osp.exists(xml):
    if not len(settings):
        return None
    xml = settings[0]
    root = etree.ElementTree(file=xml).getroot()
    sc = root.findall('SIGNALCHAIN')
    processors = list()
    for sc_ in sc:
        processors.extend( sc_.findall('PROCESSOR') )

    rhythm_proc = None
    for proc in processors:
        if proc.attrib['name'] == 'Sources/Rhythm FPGA':
            rhythm_proc = proc
            break
    if rhythm_proc is None:
        print('Did not find Rhythm FPGA processor!')
        raise RuntimeError('bar')
        return
    editor = rhythm_proc.find('EDITOR')
    sr_code = int( editor.attrib['SampleRate'] )
    return float( _srates[sr_code-1] )

def get_robust_recording(session_path, rec_pattern):
    import os
    rec_pattern = rec_pattern.strip(os.path.sep)
    try:
        subdirs = next(os.walk(session_path))[1]
    except StopIteration:
        return []
    candidates = [x for x in subdirs if rec_pattern in x]
    return [osp.join(session_path, c) for c in candidates]

def _prepare_paths(exp_path, test, rec_num):
    """Normalize some info about funky open-ephys paths"""

    # The channel data will be in separate files in the directory
    # exp_path/test.  This method should return the fully-specified
    # recording from the literal recording name, or from a short-cut 
    # path name (e.g. "002" w/o timestamp)
    hits = get_robust_recording(exp_path, test)
    if not len(hits):
        raise IOError('Recording not found: {0} {1}'.format(exp_path, test))
    rec_path = hits[0]

    # regularize the rec_num input -- it might be 
    # * a single string
    # * a single integer
    # * a sequence of integer/strings
    # make it finally a sequence of strings
    if isinstance(rec_num, (str, str)) or isinstance(rec_num, (int, int)):
        rec_num = (rec_num,)
    if np.iterable(rec_num):
        rec_num = [ str(r) for r in rec_num ]
        
    if rec_num[0].lower() == 'auto':
        # try to find the logical recording prefix, i.e. the first one
        # found that has "ADC" channels
        all_files = glob(osp.join(rec_path, '*.continuous'))
        if not len(all_files):
            raise IOError('No files found')
        prefixes = set( [osp.split(f)[1].split('_')[0] for f in all_files] )
        for pre in sorted(list(prefixes)):
            if len( glob(osp.join(rec_path, pre+'*ADC*.continuous')) ):
                rec_num = (pre,)
                print('found prefix:', rec_num)
                break
        # if still auto, then choose one (works with older formats?)
        if rec_num[0].lower() == 'auto':
            rec_num = (pre,)

    return rec_path, rec_num

from tempfile import NamedTemporaryFile

def memmap_open_ephys_channels(
        exp_path, test, rec_num='auto', quantized=False, data_chans='all'
        ):
    
    """Load memory-mapped arrays of the full band timeseries.

    This option provides a way to load massive multi-channel datasets
    sampled at 20 kS/s. Channels are cached to disk in flat files and then
    loaded as "memmap" arrays. Down-sampling is not supported.

                               

    """

    rec_path, rec_num = _prepare_paths(exp_path, test, rec_num)
    OE_type = np.int16 if quantized else float
    NP_type = 'h' if quantized else 'd'

    chan_names = OE.get_filelist(
        rec_path, source=rec_num[0], ctype='CH', channels=data_chans
        )
    n_chan = len( chan_names )
    if not n_chan:
        raise IOError('no channels found')
    from ecogdata.expconfig import params
    # loading in transpose mode, so channels have to be packed
    # in full one after another.
    bytes_per_channel = OE.get_channel_bytes(chan_names[0])
    if not quantized:
        bytes_per_channel *= 4

    load_chans = int( float(params.memory_limit) // (2 * bytes_per_channel) )
    
    if sys.platform == 'win32':
        chans_ = NamedTemporaryFile(mode='ab', delete=False)
        OE.pack(
            rec_path, filename=chans_.file, transpose=True,
            dtype=OE_type, ctype='CH', channels=data_chans,
            chunk_size=load_chans, source=rec_num[0]
            )
        chans = np.memmap(chans_.name, dtype=NP_type).reshape(n_chan, -1)
    else:
        with NamedTemporaryFile(mode='ab') as chans_:
            OE.pack(
                rec_path, filename=chans_.file, transpose=True,
                dtype=OE_type, ctype='CH', channels=data_chans,
                chunk_size=load_chans, source=rec_num[0]
                )
            chans = np.memmap(chans_.name, dtype=NP_type).reshape(n_chan, -1)

    dset = Bunch(chdata = chans)
    for arr in ('ADC', 'AUX'):

        n_extra = len(OE.get_filelist(rec_path, ctype=arr))
        if n_extra:
            if sys.platform == 'win32':
                tfile = NamedTemporaryFile(mode='ab', delete=False)
                OE.pack(
                    rec_path, filename=tfile.file, transpose=True,
                    dtype=OE_type, ctype=arr, chunk_size=load_chans,
                    source=rec_num[0]
                    )
                mm = np.memmap(tfile.name, dtype=NP_type).reshape(n_extra, -1)
                dset[ arr.lower() ] = mm
            else:
                with NamedTemporaryFile(mode='ab') as tfile:
                    OE.pack(
                        rec_path, filename=tfile.file, transpose=True,
                        dtype=OE_type, ctype=arr, chunk_size=load_chans,
                        source=rec_num[0]
                        )
                    mm = np.memmap(tfile.name, dtype=NP_type).reshape(n_extra, -1)
                    dset[ arr.lower() ] = mm
        else:
            dset[ arr.lower() ] = ()

    header = OE.get_header_from_folder(rec_path)
    trueFs = get_robust_samplingrate(rec_path)
    if trueFs is None:
        trueFs = header['sampleRate']
    dset.header = header
    dset.Fs = trueFs
    return dset

def hdf5_open_ephys_channels(
        exp_path, test, hdf5_name, rec_num='auto',
        quantized=False, data_chans='all', downsamp=1,
        load_chans=None
        ):
    
    """Load HDF5-mapped arrays of the full band timeseries.

    This option provides a way to load massive multi-channel datasets
    sampled at 20 kS/s. Down-sampling is supported.

    """

    if downsamp > 1:
        quantized = False
    
    rec_path, rec_num = _prepare_paths(exp_path, test, rec_num)

    chan_names = OE.get_filelist(rec_path, ctype='CH', channels=data_chans, source=rec_num[0])
    n_chan = len( chan_names )
    if not n_chan:
        raise IOError('no channels found')
    from ecogdata.expconfig import params
    # load channel at a time to be able to downsample
    bytes_per_channel = OE.get_channel_bytes(chan_names[0]) 
    if not quantized:
        bytes_per_channel *= 4

    if load_chans is None:
        load_chans = int(float(params.memory_limit) // (2 * bytes_per_channel))
    # get test channel for shape info
    ch_record = OE.loadContinuous(chan_names[0], dtype=np.int16, verbose=False)
    chan_names = chan_names
    ch_data = ch_record['data']
    header = OE.get_header_from_folder(rec_path, source=rec_num[0])
    trueFs = get_robust_samplingrate(rec_path)
    if trueFs is None:
        trueFs = header['sampleRate']

    if downsamp > 1:
        d_len = ch_data[..., ::downsamp].shape[-1]
    else:
        d_len = ch_data.shape[-1]
    if quantized:
        atom = tables.Atom.from_sctype('h')
    else:
        atom = tables.Atom.from_sctype('d')
    
    def _proc_block(block):
        if not quantized:
            block = block * ch_record['header']['bitVolts']
        if downsamp > 1:
            block, _ = downsample(block, trueFs, r=downsamp)
        return block

    ## with closing(tables.open_file(f, mode)) as f:
    with tables.open_file(hdf5_name, mode='w') as h5:
        h5.create_array('/', 'Fs', trueFs/downsamp)
        chans = h5.create_carray(
            '/', 'chdata', atom=atom, shape=(n_chan, d_len)
            )


        # Pack in channel data
        chans[0] = _proc_block(ch_data)
        start_chan = 1
        while True:
            stop_chan = min(len(chan_names), start_chan + load_chans)
            print('load chan', start_chan, 'to', stop_chan)
            ch_data = OE.loadFolderToTransArray(
                rec_path, dtype=np.int16, verbose=False,
                start_chan = start_chan, stop_chan=stop_chan, ctype='CH',
                channels=data_chans, source=rec_num[0]
                )
            chans[start_chan:stop_chan] = _proc_block(ch_data)
            start_chan += load_chans
            if start_chan >= len(chan_names):
                break

    for arr in ('ADC', 'AUX'):

        n_extra = len(OE.get_filelist(rec_path, ctype=arr, source=rec_num[0]))
        if not n_extra:
            continue
        with tables.open_file(hdf5_name, mode='a') as h5:
            chans = h5.create_carray('/', arr.lower(), atom=atom,
                                     shape=(n_extra, d_len))
            start_chan = 0
            while True:
                stop_chan = min(n_extra, start_chan + load_chans)
                ch_data = OE.loadFolderToTransArray(
                    rec_path, dtype=np.int16, verbose=False, source=rec_num[0],
                    start_chan = start_chan, stop_chan=stop_chan, ctype=arr
                    )
                chans[start_chan:stop_chan] = _proc_block(ch_data)
                start_chan += load_chans
                if start_chan >= n_extra:
                    break
    
def load_open_ephys_channels(
        exp_path, test, rec_num='auto', shared_array=False,
        downsamp=1, target_Fs=-1, lowpass_ord=12, page_size=8,
        save_downsamp=True, use_stored=True, store_path='', 
        quantized=False):

    # first off, check if there is a stored file at target_Fs (if valid)
    if use_stored and target_Fs > 0:
        # Look for a previously downsampled data stash
        fname_part = '*{0}*_Fs{1}.h5'.format(test, int(target_Fs))
        # try store_path (if given) and also exp_path
        for p_ in (store_path, exp_path):
            fname = glob(osp.join(p_, fname_part))
            if len(fname) and osp.exists(fname[0]):
                print('Loading from', fname[0])
                channel_data = load_bunch( fname[0], '/' )
                return channel_data

    rec_path, rec_num = _prepare_paths(exp_path, test, rec_num)
    trueFs = get_robust_samplingrate(rec_path)
    if downsamp == 1 and target_Fs > 0:
        if trueFs is None:
            # do nothing
            print('Sampling frequency not robustly determined, '
                  'downsample not calculated for {0:.1f} Hz'.format(target_Fs))
            raise ValueError
        else:
            # find the correct (integer) downsample rate 
            # to get (approx) target Fs
            # target_fs * downsamp <= Fs
            # downsamp <= Fs / target_fs
            downsamp = int( trueFs // target_Fs )
            print(('downsample rate:', downsamp))

    if downsamp > 1 and quantized:
        print('Cannot return quantized data when downsampling')
        quantized = False

        
    all_files = list()
    for pre in rec_num:
        all_files.extend( glob(osp.join(rec_path, pre+'*.continuous')) )
    if not len(all_files):
        raise IOError('No files found')
    c_nums = list()
    chan_files = list()
    aux_files = list()
    aux_nums = list()
    adc_files = list()
    adc_nums = list()
    for f in all_files:
        f_part = osp.splitext( osp.split(f)[1] )[0]
        # File names can be: Proc#_{ADC/CH/AUX}[_N].continuous
        # (the last _N part is not always present!! disgard for now)
        f_parts = f_part.split('_')
        if len(f_parts[-1]) == 1 and f_parts[-1] in '0123456789':
            f_parts = f_parts[:-1]
        ch = f_parts[-1] # last file part is CHx or AUXx
        if ch[0:2] == 'CH':
            chan_files.append(f)
            c_nums.append( int(ch[2:]) )
        elif ch[0:3] == 'AUX': #separate chan and AUX files
            aux_files.append(f)
            aux_nums.append( int(ch[3:]) )
        elif ch[0:3] == 'ADC':
            adc_files.append(f)
            adc_nums.append( int(ch[3:]) )

    if downsamp > 1:
        (b_lp, a_lp) = cheby2_bp(60, hi=1.0/downsamp, Fs=2, ord=lowpass_ord)


    def _load_array_block(files, shared_array=False, antialias=True):
        Fs = 1
        dtype = 'h' if quantized else 'd'

        # start on 1st index of 0th block
        n = 1
        b_cnt = 0
        b_idx = 1

        ch_record = OE.loadContinuous(files[0], dtype=np.int16, verbose=False)
        d_len = ch_record['data'].shape[-1]
        sub_len = d_len // downsamp
        if sub_len * downsamp < d_len:
            sub_len += 1
        proc_block = shared_ndarray( (page_size, d_len), typecode=dtype )
        proc_block[0] = ch_record['data'].astype('d')
        if shared_array:
            saved_array = shared_ndarray( (len(files), sub_len), 
                                          typecode=dtype )
        else:
            saved_array = np.zeros( (len(files), sub_len), dtype=dtype )
        
        for f in files[1:]:
            ch_record = OE.loadContinuous(
                f, dtype=np.int16, verbose=False
                ) # load data
            Fs = float( ch_record['header']['sampleRate'] )
            proc_block[b_idx] = ch_record['data'].astype(dtype)
            b_idx += 1
            n += 1
            if (b_idx == page_size) or (n == len(files)):
                # do dynamic range conversion and downsampling
                # on a block of data
                if not quantized:
                    proc_block *= ch_record['header']['bitVolts']
                if downsamp > 1 and antialias:
                    filtfilt(proc_block, b_lp, a_lp)
                sl = slice(b_cnt*page_size, n)
                saved_array[sl] = proc_block[:b_idx, ::downsamp]
                # update / reset block counters
                b_idx = 0
                b_cnt += 1

        del proc_block
        while gc.collect():
            pass
        return saved_array, Fs, ch_record['header']
    
    # sort CH, AUX, and ADC by the channel number
    sorted_chans = np.argsort(c_nums)
    # sorts list ed on sorted_chans
    chan_files = [chan_files[n] for n in sorted_chans]
    chdata, Fs, header = _load_array_block(chan_files, 
                                           shared_array=shared_array)
    
    aux_data = list()
    if len(aux_files) > 0:
        sorted_aux = np.argsort(aux_nums)
        aux_files = [aux_files[n] for n in sorted_aux]
        aux_data, _, _ = _load_array_block(aux_files, antialias=False)

    adc_data = list()
    if len(adc_files) > 0:
        sorted_adc = np.argsort(adc_nums)
        adc_files = [adc_files[n] for n in sorted_adc]
        adc_data, _, _ = _load_array_block(adc_files, antialias=False)
        
    if not trueFs:
        print('settings.xml not found, relying on sampling rate from ' \
          'recording header files')
        trueFs = Fs
    if downsamp > 1:
        trueFs /= downsamp
    dset = Bunch(chdata=chdata, aux=aux_data, adc=adc_data,
                 Fs=trueFs, header=header)

    if save_downsamp and downsamp > 1:
        fname = '{0}_Fs{1}.h5'.format(osp.split(rec_path)[-1], int(dset.Fs))
        if not len(store_path):
            store_path = exp_path
        mkdir_p(store_path)
        fname = osp.join(store_path, fname)
        print('saving', fname)
        save_bunch(fname, '/', dset, mode='w')

    return dset
    #### stop function, return ch_data, header

def load_open_ephys(exp_path, test, electrode, 
                    bandpass=(), notches=(), units='uV',
                    snip_transient=True, rec_num='auto', 
                    trigger_idx=(), useFs=-1,
                    save_downsamp=True, use_stored=True, store_path='',
                    downsamp=1, memmap=False, connectors=(), **extra):

    chan_map, gnd_chans = get_electrode_map(electrode, connectors=connectors)
    el_chans = np.setdiff1d(
        np.arange(len(chan_map) + len(gnd_chans)), gnd_chans
        )

    if memmap:
        channel_data = memmap_open_ephys_channels(
            exp_path, test, rec_num=rec_num,
            data_chans=list(el_chans+1), **extra
            )
        ecog_chans = channel_data.chdata
        ground_chans = ()
        snip_transient = False
    else:
        # Load Data/ADC/AUX channels (perhaps pre-computed downsample)
        channel_data = load_open_ephys_channels(
            exp_path, test, rec_num=rec_num, shared_array=False,
            target_Fs=useFs, save_downsamp=save_downsamp, 
            use_stored=use_stored, store_path=store_path, **extra
            )
        ground_chans = channel_data.chdata[gnd_chans]
        ecog_chans = shared_copy( channel_data.chdata[el_chans] )

    Fs = channel_data.Fs

    # Now do a pretty standard set of operations (some day will be
    # "standardized" in a data loading class)
    # * separate electrode / trigger / aux data
    # * process trigger edges
    # * bandpass filtering
    # * advance starting index 
    # * convert units


    
    if not np.iterable(trigger_idx):
        trigger_idx = [trigger_idx]
    if not len(trigger_idx):
        trig_chan = ()
    else:
        try:
            trig_chan = channel_data.adc[trigger_idx]
        except IndexError:
            print("No trig chans found")
            trig_chan = ()
        
    try:
        stim_chan = channel_data.adc[max(trigger_idx)+1];
    except IndexError:
        print("Stim chan not loaded")
        stim_chan = ()
    
    if len(trig_chan):
        pos_edge, _ = process_trigger(trig_chan)
    else:
        pos_edge = ()

    with parallel_controller.context(not memmap):
        ### bandpass filter
        if len(bandpass):
            lo, hi = bandpass
            (b, a) = butter_bp(lo=lo, hi=hi, Fs=Fs, ord=4)
            filtfilt(ecog_chans, b, a)

        ### notch filters
        if len(notches):
            notch_all(
                ecog_chans, Fs, lines=notches, inplace=True, filtfilt=True
                )


    
    ### advance index
    if snip_transient:
        if isinstance(snip_transient, bool):
            snip = int(5 * Fs)
        else:
            snip = int(snip_transient * Fs)

        ecog_chans = ecog_chans[:, snip:].copy()
        if len(ground_chans):
            ground_chans = ground_chans[:, snip:].copy()
        if len(trig_chan):
            trig_chan = trig_chan[snip:].copy()
            pos_edge -= snip
            pos_edge = pos_edge[ pos_edge > 0 ]
        if len(stim_chan):
            stim_chan = stim_chan[snip:].copy()

    ### convert units
    if units.lower() != 'uv':
        convert_scale(ecog_chans, 'uv', units)

    dset = Bunch()
    dset.data = ecog_chans
    dset.adc = channel_data.adc # added in for loading in behavior tables
    dset.ground_chans = ground_chans
    dset.chan_map = chan_map
    dset.pos_edge = pos_edge
    dset.trig_chan = trig_chan
    dset.stim_chan = stim_chan
    dset.Fs = Fs
    dset.bandpass = bandpass
    dset.transient_snipped = snip_transient
    dset.notches = notches
    dset.units = units
    return dset

def load_open_ephys_impedance(
        exp_path, test, electrode, magphs=True,
        electrode_connections=()
        ):

    xml = osp.join(osp.join(exp_path, test), 'impedance_measurement.xml')
    if not osp.exists(xml):
        raise IOError('No impedance XMl found')

    root = etree.ElementTree(file=xml).getroot()
    chans = root.getchildren()
    mag = np.zeros(len(chans), 'd')
    phs = np.zeros(len(chans), 'd')
    for n, ch in enumerate(chans):
        # cannot rely on "channel_number" -- it only counts 0-31
        # n = int(ch.attrib['channel_number'])
        mag[n] = float(ch.attrib['magnitude'])
        phs[n] = float(ch.attrib['phase'])
    
    chan_map, gnd_chans = get_electrode_map(electrode,
                                            connectors=electrode_connections)
    ix = np.arange(len(phs))
    cx = np.setdiff1d(ix, gnd_chans)
    chan_mag = mag[cx]
    chan_phs = phs[cx]
    if magphs:
        return (chan_mag, chan_phs), chan_map
    return chan_mag * np.exp(1j * chan_phs * np.pi / 180.0), chan_map

def debounce_trigger(pos_edges):
    if len(pos_edges) < 3:
        return pos_edges
    df = np.diff(pos_edges)
    isi_guess = np.percentile(df, 90)

    # lose any edges that are < half of the isi_guess
    edge_mask = np.ones(len(pos_edges), '?')

    for i in range(len(pos_edges)):
        if not edge_mask[i]:
            continue

        # look ahead and kill any edges that are 
        # too close to the current edge
        pt = pos_edges[i]
        kill_mask = (pos_edges > pt) & (pos_edges < pt + isi_guess/2)
        edge_mask[kill_mask] = False

    return pos_edges[edge_mask]

def plot_Z(
        path_or_Z, electrode, minZ, maxZ, cmap, 
        phs=False, title='', ax=None, cbar=True, clim=None,
        electrode_connections=()
        ):
    # from ecogana.anacode.colormaps import nancmap

    if isinstance(path_or_Z, (str, str)):
        path = osp.abspath(path_or_Z)
        path, test = osp.split(path)
        if not len(title):
            title = test
        Z, chan_map = load_open_ephys_impedance(
            path, test, electrode,
            electrode_connections=electrode_connections
            )
        Z = Z[1] if phs else Z[0]
    else:
        Z = path_or_Z.copy()
        chan_map, _ = get_electrode_map(electrode)
    
    if not phs:
        Z *= 1e-3
        Z_open = Z > maxZ
        Z_shrt = Z < minZ
        np.log10(Z, Z)
        Z[ Z_open ] = 1e20
        Z[ Z_shrt ] = -1
        lo = Z[ ~(Z_open | Z_shrt) ].min()
        hi = Z[ ~(Z_open | Z_shrt) ].max()
    else:
        lo, hi = Z.min(), Z.max()
    if np.abs( lo - round(lo) ) < np.abs( hi - round(hi) ):
        lo = round(lo)
        hi = np.ceil(hi)
    ## else:
    ##     lo = np.floor(lo)
    ##     hi = round(hi)

    if clim is None:
        clim = (lo, hi)
    else:
        lo, hi = clim

    # cm = nancmap(cmap, overc='gray', underc='lightgray')
    r_ = chan_map.image(Z, cmap=cmap, clim=clim, ax=ax, cbar=cbar)
    if cbar:
        f, cb = r_
    else:
        f = r_
    if ax is None:
        ax = f.axes[-(1 + int(cbar))]
    if not phs:
        bad = Z_open.sum() + Z_shrt.sum()
        yld = 100 * (1 - float(bad) / len(Z_open))
        title = title + ' ({0:.2f}% yield)'.format(yld)
    ax.set_title(title)
    if phs:
        if cbar:
            cb.set_label('Phase (degrees)')
        f.tight_layout()
        return f
    if cbar:
        c_ticks = np.array([0.1, 0.2, 0.5,
                            1, 2, 5, 
                            10, 20, 50, 
                            100, 200, 500, 
                            1000, 2000, 5000])
        c_ticks = c_ticks[ (c_ticks >= 10**lo) & (c_ticks <= 10**hi) ]
        cb.set_ticks( np.log10( c_ticks ) )
        cb.set_ticklabels( list(map(str, c_ticks)) )
        cb.set_label(u'Impedance (k\u03A9)')
    f.tight_layout()
    return f

