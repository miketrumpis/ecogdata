import os
import os.path as osp
from glob import glob
import gc
import warnings

from lxml import etree

import numpy as np
import h5py

from ecogdata.expconfig import load_params
from ecogdata.trigger_fun import process_trigger
from ecogdata.filt.time import cheby2_bp, downsample
from ecogdata.util import Bunch, mkdir_p
from ecogdata.datastore import load_bunch, save_bunch
from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.parallel.split_methods import filtfilt

from . import DataPathError, _OpenEphys as OE

from ecogdata.datasource import PlainArraySource
from ecogdata.datasource.memmap import TempFilePool
from ecogdata.devices.electrode_pinouts import get_electrode_map
from ecogdata.devices.units import convert_scale

from .file2data import FileLoader


_srates = (1000, 1250, 1500, 2000, 2500, 3000, 1e4 / 3,
           4000, 5000, 6250, 8000, 10000, 12500, 15000,
           20000, 25000, 30000)


def get_robust_samplingrate(rec_path):
    settings = glob(osp.join(rec_path, 'settings*.xml'))
    if not len(settings):
        return None
    xml = settings[0]
    root = etree.ElementTree(file=xml).getroot()
    sc = root.findall('SIGNALCHAIN')
    processors = list()
    for sc_ in sc:
        processors.extend(sc_.findall('PROCESSOR'))

    rhythm_proc = None
    for proc in processors:
        if proc.attrib['name'] == 'Sources/Rhythm FPGA':
            rhythm_proc = proc
            break
    if rhythm_proc is None:
        print('Did not find Rhythm FPGA processor!')
        raise RuntimeError('bar')
    editor = rhythm_proc.find('EDITOR')
    sr_code = int(editor.attrib['SampleRate'])
    return float(_srates[sr_code - 1])


def get_robust_recording(session_path, rec_pattern):
    rec_pattern = rec_pattern.strip(osp.sep)
    try:
        subdirs = next(os.walk(session_path))[1]
    except StopIteration:
        return []
    candidates = [x for x in subdirs if rec_pattern in x]
    return [osp.join(session_path, c) for c in candidates]


def prepare_paths(exp_path, test, rec_num):
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
    if isinstance(rec_num, str) or isinstance(rec_num, int):
        rec_num = (rec_num,)
    if np.iterable(rec_num):
        rec_num = [str(r) for r in rec_num]

    if rec_num[0].lower() == 'auto':
        # try to find the logical recording prefix, i.e. the first one
        # found that has "ADC" channels
        all_files = glob(osp.join(rec_path, '*.continuous'))
        if not len(all_files):
            raise IOError('No files found')
        prefixes = set(['_'.join(osp.split(f)[1].split('_')[:-1]) for f in all_files])
        for pre in sorted(list(prefixes)):
            if len(glob(osp.join(rec_path, pre + '*ADC*.continuous'))):
                rec_num = (pre,)
                break
        # if still auto, then choose one (works with older formats?)
        if rec_num[0].lower() == 'auto':
            rec_num = (pre,)

    return rec_path, rec_num


def hdf5_open_ephys_channels(
        exp_path, test, hdf5_name, rec_num='auto',
        quantized=False, data_chans='all', downsamp=1,
        load_chans=None):
    """Load HDF5-mapped arrays of the full band timeseries.

    This option provides a way to load massive multi-channel datasets
    sampled at 20 kS/s. Down-sampling is supported.

    """

    downsamp = int(downsamp)
    if downsamp > 1:
        quantized = False

    rec_path, rec_num = prepare_paths(exp_path, test, rec_num)

    chan_names = OE.get_filelist(rec_path, ctype='CH', channels=data_chans, source=rec_num[0])
    n_chan = len(chan_names)
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
        arr_dtype = 'h'
    else:
        arr_dtype = 'f' if load_params().floating_point == 'single' else 'd'

    def _proc_block(block, antialias=True):
        if not quantized:
            block = block * ch_record['header']['bitVolts']
        if downsamp > 1:
            if antialias:
                block, _ = downsample(block, trueFs, r=downsamp)
            else:
                block = block[:, ::downsamp]
        return block

    with h5py.File(hdf5_name, 'w', libver='latest') as h5:
        h5.create_dataset('Fs', data=trueFs / downsamp)
        chans = h5.create_dataset('chdata', dtype=arr_dtype, chunks=True, shape=(n_chan, d_len))

        # Pack in channel data
        chans[0] = _proc_block(ch_data)
        start_chan = 1
        while True:
            stop_chan = min(len(chan_names), start_chan + load_chans)
            print('load chan', start_chan, 'to', stop_chan)
            ch_data = OE.loadFolderToTransArray(
                rec_path, dtype=np.int16, verbose=False,
                start_chan=start_chan, stop_chan=stop_chan, ctype='CH',
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
        with h5py.File(hdf5_name, 'r+', libver='latest') as h5:
            chans = h5.create_dataset(arr.lower(), dtype=arr_dtype, chunks=True, shape=(n_extra, d_len))
            start_chan = 0
            while True:
                stop_chan = min(n_extra, start_chan + load_chans)
                ch_data = OE.loadFolderToTransArray(
                    rec_path, dtype=np.int16, verbose=False, source=rec_num[0],
                    start_chan=start_chan, stop_chan=stop_chan, ctype=arr
                )
                chans[start_chan:stop_chan] = _proc_block(ch_data, antialias=False)
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
                channel_data = load_bunch(fname[0], '/')
                return channel_data

    rec_path, rec_num = prepare_paths(exp_path, test, rec_num)
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
            downsamp = int(trueFs // target_Fs)
            print(('downsample rate:', downsamp))

    if downsamp > 1 and quantized:
        print('Cannot return quantized data when downsampling')
        quantized = False
    downsamp = int(downsamp)

    all_files = list()
    for pre in rec_num:
        all_files.extend(glob(osp.join(rec_path, pre + '*.continuous')))
    if not len(all_files):
        raise IOError('No files found')
    c_nums = list()
    chan_files = list()
    aux_files = list()
    aux_nums = list()
    adc_files = list()
    adc_nums = list()
    for f in all_files:
        f_part = osp.splitext(osp.split(f)[1])[0]
        # File names can be: Proc#_{ADC/CH/AUX}[_N].continuous
        # (the last _N part is not always present!! disgard for now)
        f_parts = f_part.split('_')
        if len(f_parts[-1]) == 1 and f_parts[-1] in '0123456789':
            f_parts = f_parts[:-1]
        ch = f_parts[-1]  # last file part is CHx or AUXx
        if ch[0:2] == 'CH':
            chan_files.append(f)
            c_nums.append(int(ch[2:]))
        elif ch[0:3] == 'AUX':  # separate chan and AUX files
            aux_files.append(f)
            aux_nums.append(int(ch[3:]))
        elif ch[0:3] == 'ADC':
            adc_files.append(f)
            adc_nums.append(int(ch[3:]))

    if downsamp > 1:
        (b_lp, a_lp) = cheby2_bp(60, hi=1.0 / downsamp, Fs=2, ord=lowpass_ord)

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
        proc_block = shared_ndarray((page_size, d_len), typecode=dtype)
        proc_block[0] = ch_record['data'].astype('d')
        if shared_array:
            saved_array = shared_ndarray((len(files), sub_len), typecode=dtype)
        else:
            saved_array = np.zeros((len(files), sub_len), dtype=dtype)

        for f in files[1:]:
            ch_record = OE.loadContinuous(
                f, dtype=np.int16, verbose=False
            )  # load data
            Fs = float(ch_record['header']['sampleRate'])
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
                sl = slice(b_cnt * page_size, n)
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
        print('settings.xml not found, relying on sampling rate from '
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


class OpenEphysLoader(FileLoader):
    scale_to_uv = 0.195
    data_array = 'chdata'
    trigger_array = 'adc'
    aligned_arrays = ('adc', 'aux')
    transpose_array = False
    permissible_types = ['.h5', '.hdf', '.continuous']

    @property
    def primary_data_file(self):
        try:
            data_path, _ = prepare_paths(self.experiment_path, self.recording, 'auto')
            return data_path
        except OSError as e:
            # check for the plain directory first, or also possibly the
            data_path = [osp.join(self.experiment_path, self.recording + ext) for ext in self.permissible_types]
            exist = [osp.exists(p) for p in data_path]
            if any(exist):
                return data_path[exist.index(True)]
            return osp.splitext(data_path[0])[0]

    def raw_sample_rate(self):
        return get_robust_samplingrate(self.primary_data_file)

    def make_channel_map(self):
        if os.path.isdir(self.primary_data_file):
            data_path, rec_num = prepare_paths(self.experiment_path, self.recording, 'auto')
            channel_files = OE.get_filelist(data_path, source=rec_num[0], ctype='CH')
            n_data_channels = len(channel_files)
        else:
            with h5py.File(self.data_file, 'r') as h5file:
                n_data_channels = h5file[self.data_array].shape[0]
        channel_map, grounded, reference = get_electrode_map(self.electrode)
        electrode_chans = [n for n in range(n_data_channels) if n not in grounded + reference]
        return channel_map, electrode_chans, grounded, reference

    def find_trigger_signals(self, data_file):
        if not os.path.isdir(data_file):
            return super(OpenEphysLoader, self).find_trigger_signals(data_file)
        data_path, rec_num = prepare_paths(self.experiment_path, self.recording, 'auto')
        assert data_path == data_file, \
            'Mismatched data sources: named data file {} and raw data file {}'.format(data_file, data_path)
        trigger_idx = self.trigger_idx
        if not np.iterable(trigger_idx):
            trigger_idx = (trigger_idx,)
        if not len(trigger_idx):
            return (), ()
        try:
            trigger_idx = [t + 1 for t in trigger_idx]
            trigger_signal = OE.loadFolderToTransArray(data_file, dtype=float, ctype='ADC', source=rec_num[0],
                                                       channels=trigger_idx, verbose=False)
            pos_edge = process_trigger(trigger_signal)[0]
            return trigger_signal, pos_edge
        except (IndexError, ValueError) as e:
            tb = e.__traceback__
            msg = 'Trigger channels were specified but do not exist'
            if self.raise_on_glitch:
                raise Exception(msg).with_traceback(tb)
            else:
                warnings.warn(msg, RuntimeWarning)
                return (), ()

    def create_downsample_file(self, data_file, resample_rate, downsamp_file, antialias_aligned=False,
                               aggregate_aligned=True):
        if not downsamp_file:
            with TempFilePool(mode='ab') as downsamp_file:
                ds_filename = downsamp_file.name
        else:
            ds_filename = downsamp_file
        downsamp_ratio = self.raw_sample_rate() // resample_rate
        # here the TempFilePool does not remain open so do not need to register a closing callback
        hdf5_open_ephys_channels(self.experiment_path, self.recording, ds_filename, data_chans='all',
                                 downsamp=downsamp_ratio)
        return ds_filename

    def map_raw_data(self, data_file, open_mode, electrode_chans, ground_chans, ref_chans, downsample_ratio):
        """
        Unlike the parent method, this method will directly load and downsample .continuous files to in-memory arrays
        without creating an intermediary MappedSource.

        Parameters
        ----------
        data_file: str
            Open Ephys data path or HDF5 file
        open_mode:
            File mode (for HDF5 only)
        electrode_chans: sequence
        ground_chans: sequence
        ref_chans: sequence
        downsample_ratio: int

        Returns
        -------
        datasource: ElectrodeArraySource
            Datasource for electrodes.
        ground_chans: ElectrodeArraySource
            Datasource for ground channels. May be an empty list
        ref_chans: ElectrodeArraySource:
            Datasource for reference channels. May be an empty list

        """

        # downsample_ratio is 1 if a pre-computed file exists or if a full resolution HDF5 needs to be created here.
        # In these cases, revert to parent method
        if downsample_ratio == 1:
            if os.path.isdir(data_file):
                # mapped_file = data_file + '.h5'
                with TempFilePool(mode='ab') as mf:
                    mapped_file = mf
                print('Take note!! Creating full resolution map file {}'.format(mapped_file))
                # Get a little tricky and see if this source should be writeable. If no, then leave it quantized with
                # a scaling factor. If yes, then convert the source to microvolts.
                quantized = not (self.ensure_writeable or self.bandpass or self.notches)
                hdf5_open_ephys_channels(self.experiment_path, self.recording, str(mapped_file), data_chans='all',
                                         quantized=quantized)
                if quantized:
                    open_mode = 'r'
                else:
                    open_mode = 'r+'
                    self.units_scale = convert_scale(1, 'uv', self.units)
                # Note: pass file "name" directly as a TempFilePool object so that file-closing can be registered
                # downstream!!
                data_file = mapped_file
            print('Calling super to map/load {} in mode {} scaling units {}'.format(data_file, open_mode,
                                                                                    self.units_scale))
            return super(OpenEphysLoader, self).map_raw_data(data_file, open_mode, electrode_chans,
                                                             ground_chans, ref_chans, downsample_ratio)

        # If downsample_ratio > 1 then we are downsampling straight to memory. Invoke a custom loader that handles
        # continuous files
        loaded = load_open_ephys_channels(self.experiment_path, self.recording, shared_array=False,
                                          downsamp=downsample_ratio, save_downsamp=False, use_stored=False,
                                          quantized=False)
        chdata = loaded.chdata
        electrode_data = shared_ndarray((len(electrode_chans), chdata.shape[1]), chdata.dtype.char)
        np.take(chdata, electrode_chans, axis=0, out=electrode_data)
        datasource = PlainArraySource(electrode_data, use_shared_mem=False, adc=loaded.adc, aux=loaded.aux)
        if ground_chans:
            ground_chans = PlainArraySource(chdata[ground_chans], use_shared_mem=False)
        if ref_chans:
            ref_chans = PlainArraySource(chdata[ref_chans], use_shared_mem=False)
        return datasource, ground_chans, ref_chans


def load_open_ephys(exp_path, test, electrode, rec_num='auto', downsamp=1, useFs=-1, memmap=False, **loader_kwargs):
    """
    Load open ephys data from continuous or from pre-computed HDF5. This method more or less preserves the original
    signature.

    Parameters
    ----------
    exp_path: str
        Path to recordings
    test: str
        Recording to load
    electrode: str
        Electrode tag
    rec_num: str
        'auto' or '001', '002', etc.. (still needs to be plugged into OpenEphysLoader)
    downsamp: int
        Downsample ratio (use resample_rate now to specify new sample rate)
    useFs: float
        New sample rate (use resample_rate now to specify new sample rate)
    memmap: bool
        Meaningless argument, historical
    loader_kwargs: dict
        Other FileLoader arguments, mostly similar to previous signature

    Returns
    -------
    dataset: Bunch
        Bunch containing ".data" (a DataSource), ".chan_map" (a ChannelMap), and many other metadata attributes.

    """

    # just silently drop this one
    loader_kwargs.pop('snip_transient', None)
    if loader_kwargs.get('resample_rate', None) is None:
        # If possible, rename useFs to resample_rate
        if useFs > 0:
            loader_kwargs['resample_rate'] = useFs
        elif downsamp > 1:
            # If downsample factor is given, try to find the original sample rate (needs access to primary data)
            try:
                loader_info = OpenEphysLoader(exp_path, test, electrode)
                loader_kwargs['resample_rate'] = loader_info.raw_sample_rate() / downsamp
            except DataPathError:
                raise RuntimeError('Could not find the original sampling rate needed to compute a '
                                   'resample rate given the downsample factor of {}. Try loading again '
                                   'specifying "resample_rate" in Hz to see if a pre-computed downsample '
                                   'can be loaded.'.format(downsamp))
    loader = OpenEphysLoader(exp_path, test, electrode, **loader_kwargs)
    return loader.create_dataset()


def load_open_ephys_impedance(exp_path, test, electrode, magphs=True, electrode_connections=()):

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

    chan_map, gnd_chans, ref_chans = get_electrode_map(electrode, connectors=electrode_connections)
    ix = np.arange(len(phs))
    cx = np.setdiff1d(ix, np.union1d(gnd_chans, ref_chans))
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
        kill_mask = (pos_edges > pt) & (pos_edges < pt + isi_guess / 2)
        edge_mask[kill_mask] = False

    return pos_edges[edge_mask]


def plot_Z(
        path_or_Z, electrode, minZ, maxZ, cmap,
        phs=False, title='', ax=None, cbar=True, clim=None,
        electrode_connections=()):
    # from ecogana.anacode.colormaps import nancmap

    if isinstance(path_or_Z, str):
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
        chan_map = get_electrode_map(electrode)[0]

    if not phs:
        Z *= 1e-3
        Z_open = Z > maxZ
        Z_shrt = Z < minZ
        np.log10(Z, Z)
        Z[Z_open] = 1e20
        Z[Z_shrt] = -1
        lo = Z[~(Z_open | Z_shrt)].min()
        hi = Z[~(Z_open | Z_shrt)].max()
    else:
        lo, hi = Z.min(), Z.max()
    if np.abs(lo - round(lo)) < np.abs(hi - round(hi)):
        lo = round(lo)
        hi = np.ceil(hi)
    # else:
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
        c_ticks = c_ticks[(c_ticks >= 10**lo) & (c_ticks <= 10**hi)]
        cb.set_ticks(np.log10(c_ticks))
        cb.set_ticklabels(list(map(str, c_ticks)))
        cb.set_label(u'Impedance (k\u03A9)')
    f.tight_layout()
    return f
