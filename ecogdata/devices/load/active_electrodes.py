import warnings
from collections import namedtuple
import numpy as np
import h5py

from ecogdata.channel_map import ChannelMap
from ecogdata.trigger_fun import process_trigger

from .file2data import FileLoader

gain = { 
    '2t-as daq v1' : 10,
    '2t-as daq v2' : 10
    }
pitch_lookup = {
    'actv_64' : 0.4,
    'active_1008ch_sp_v2' : (0.3214, 0.25) # pitch is dx, dy
    }

DAQunmix = namedtuple('DAQunmix', ['col', 'row', 'extra_col', 'extra_row'])
active_headstages = ('zif26 to 2x uhdmi', 
                     'zif26 to 2x 20 pins harwin to 2x uhdmi',
                     'zif to 50mil',
                     'zif51_p4-50_demux-14c20r',)


def load_active(exp_path, name, electrode, daq, headstage, bnc=(), trigger_idx=0, **load_kwargs):
    """

    Parameters
    ----------
    exp_path: str
        Path for experiment recordings
    name: str
        Name of the recording to load
    electrode:
        Electrode tag
    daq:
        DAQ equipment tag
    headstage:
        Headstage equipment tag
    bnc: int or sequence
        Columns in the acquired data corresponding to BNC inputs
    trigger_idx: int
        If there are BNC columns, then this one corresponds to a timestamp trigger.
    **load_kwargs: dict
        Other arguments for the FileLoader type

    Returns
    -------
    dataset: Bunch
        Bunch containing ".data" (a DataSource), ".chan_map" (a ChannelMap), and many other metadata attributes.

    """

    loader = ActiveLoader(exp_path, name, electrode, daq, headstage, bnc=bnc, **load_kwargs)
    return loader.create_dataset()


def get_daq_unmix(daq, headstage, electrode, row_order=()):
    daq = daq.lower()
    headstage = headstage.lower()
    electrode = electrode.lower()
    row_order = list(map(int, row_order))
    # e.g. penn data 4/28/2016
    if (daq == '2t-as daq v2') and (headstage == 'zif26 to 2x uhdmi') and \
      (electrode == 'actv_64'):
        col_order = [2, 1, 5, 8, 7, 6, 9, 0, 4, 3]
        if not len(row_order):
            row_order = [0, 1, 2, 3, 7, 4, 6, 5]
        col = [col_order.index(i) for i in range(len(col_order))]
        row = [row_order.index(i) for i in range(len(row_order))]
        # diagnostic channels are last 2 columns
        extra_col = col[-2:]
        col = col[:-2]
        unmix = DAQunmix(np.array(col[::-1]), np.array(row), extra_col, ())

    # e.g. duke data winter/spring 2016
    elif (daq == '2t-as daq v1') and \
      (headstage == 'zif26 to 2x 20 pins harwin to 2x uhdmi')  and \
      (electrode == 'actv_64'):
        col_order = [7, 9, 8, 2, 4, 5, 1, 0, 3, 6]
        col = [col_order.index(i) for i in range(len(col_order))]
        extra_col = [1, 4]
        for c in extra_col:
            col.remove(c)
        col = np.array(col)
        # This is Ken's original order
        if not len(row_order):
            row_order = [6, 5, 1, 0, 2, 3, 7, 4]
        row = [row_order.index(i) for i in range(len(row_order))]
        # This is Ken's 2nd order (sequential)
        #row = range(8)
        # this is Ken's 3rd order (skip 3)
        #row = list( (np.arange(8) * 3) % 8 )
        unmix = DAQunmix(col[::-1], row, extra_col, ())

    # e.g. duke data from 4/26/2016
    elif (daq == '2t-as daq v1') and (headstage == 'zif26 to 2x uhdmi') and \
      (electrode == 'actv_64'):
        col_order = list( np.array([6, 9, 8, 7, 10, 1, 5, 4, 3, 2]) - 1 )
        if not len(row_order):
            row_order = list( np.array([1, 2, 3, 4, 8, 5, 6, 7]) - 1 )

        col = [col_order.index(i) for i in range(len(col_order))]
        extra_col = col[-2:]
        col = col[:-2]
        row = [row_order.index(i) for i in range(len(row_order))]
        unmix = DAQunmix(np.array(col[::-1]), np.array(row), extra_col, ())
    elif (daq == '2t-as daq v1') and (headstage == 'zif to 50mil') and \
      (electrode == 'cardiac v1'):
        col_order = np.array([12, 14, 17, 19, 5, 11, 13, 16, 18, 
                              20, 2, 4, 7, 9, 15, 10, 8, 6, 3, 1]) - 1
        if not len(row_order):
            row_order = np.array([16, 1, 6, 8, 4, 20, 2, 12, 14, 17, 9, 
                                  22, 21, 10, 13, 18, 3, 19, 7, 11, 15, 5]) - 1

        # reorder to my convention
        col = [list(col_order).index(i) for i in range(len(col_order))]
        # remove floating and ref channels
        extra_col = [4, 14]
        col.remove(4)
        col.remove(14)
        row = [list(row_order).index(i) for i in range(len(row_order))]
        unmix = DAQunmix(np.array(col[::-1]), np.array(row), extra_col, ())
    elif (daq == '2t-as daq v2') and (headstage == 'zif51_p4-50_demux-14c20r') \
      and (electrode == 'active_1008ch_sp_v2'):

        col_order = np.array([8, 7, 11, 14, 13, 12, -1, 1, 5,
                              4, 3, 2, 6, 10, 9, 28, 27, 22, 16, 18,
                              20, -1, 15, 23, 21, 19, 17, 25, 24, 26]) - 1
        col = [list(col_order).index(i) for i in np.sort( col_order[col_order>=0] )]

        if not len(row_order):
            row_order = np.array([8, 6, 2, 4, 18, 14, 16, 1, 3, 10, 12, 5, 7, 11,
                                  9, 17, 15, 13, 26, 24, 20, 22, 36, 32, 34, 19,
                                  21, 28, 30, 23, 25, 29, 27, 35, 33, 31]) - 1
        row = [list(row_order).index(i) for i in range(len(row_order))]
        extra_col = np.where(col_order < 0)[0]
        unmix = DAQunmix(np.array(col[::-1]), np.array(row), extra_col, ())
    elif daq.lower() == 'passthru':
        unmix = DAQunmix(slice(None), slice(None), (), ())
    else:
        err = ['Combination unknown:',
               'DAQ {0}'.format(daq),
               'Headstage {0}'.format(headstage),
               'Electrode {0}'.format(electrode)]
        raise NotImplementedError('\n'.join(err))
    return unmix


class ActiveLoader(FileLoader):
    transpose_array = True
    permissible_types = ['.mat', '.h5', '.hdf']

    def __init__(self, experiment_path, recording, electrode, daq, headstage, bnc=(), **kwargs):
        self.daq_type = daq
        self.headstage_type = headstage
        self.bnc_columns = bnc
        self.scale_to_uv = 1e6 / gain.get(self.daq_type, 1.0)
        super(ActiveLoader, self).__init__(experiment_path, recording, electrode, **kwargs)
        with h5py.File(self.data_file, 'r') as h5file:
            shape = h5file['data'].shape
            num_row = int(h5file['numRow'][()])
            num_chan = int(h5file['numChan'][()])
            total_channels = num_row * num_chan
            # if this is a downsample file, check for an extracted BNC array
            source_has_bnc = 'bnc' in h5file
        self.transpose_array = (shape[1] == total_channels)
        if bnc:
            if source_has_bnc:
                self.aligned_arrays = ['bnc']
            else:
                bnc_channels = np.concatenate([np.arange(bnc * num_row, (bnc + 1) * num_row) for bnc in bnc])
                self.aligned_arrays = [('bnc', bnc_channels)]


    def create_downsample_file(self, data_file, resample_rate, downsamp_file):
        # The parent method creates a channel-compatible source file with anti-aliased downsamples in the channel
        # array. For active electrode data with all external channels (e.g. logic levels) packed into the main data
        # array, a side effect is that the external channels will be anti-alias filtered as well.
        # However, the new source file will have a separate "bnc" array that is downsampled w/o filtering.
        new_file = super(ActiveLoader, self).create_downsample_file(data_file, resample_rate, downsamp_file)
        # add in the other metadata -- note that this assumes that create_downsample creates a mapped file,
        # which may change
        with h5py.File(data_file, 'r') as f1, h5py.File(new_file, 'r+') as f2:
            samp_rate = f1['Fs'][()]
            samp_rate[:] = resample_rate
            f2['Fs'] = samp_rate
            for k in f1.keys():
                if k not in (self.data_array, 'Fs', 'bnc'):
                    try:
                        f2[k] = f1[k][()]
                    except AttributeError:
                        pass
        # shorten this to the extracted BNC array
        self.aligned_arrays = ['bnc']
        return new_file

    def make_channel_map(self):
        unmix = get_daq_unmix(self.daq_type, self.headstage_type, self.electrode)
        with h5py.File(self.data_file, 'r') as h5file:
            nrow = int(h5file['numRow'][()])
            ncol = int(h5file['numCol'][()])

        pitch = pitch_lookup.get(self.electrode, 1.0)
        # go through channels,
        # if channel is data, put down the array matrix location
        # else, put down a disconnected channel
        data_rows = list(unmix.row)
        data_cols = list(unmix.col)
        # data_chans = np.array(data_cols) * nrow + np.array(data_rows)
        electrode_chans = []
        chan_map = []
        other_chans = []
        for c in range(nrow * ncol):
            col = c // nrow
            row = c % nrow
            if col in data_cols:
                arow = data_rows.index(row)
                acol = data_cols.index(col)
                chan_map.append(arow * len(data_cols) + acol)
                electrode_chans.append(c)
            else:
                other_chans.append(c)
        nr = len(unmix.row)
        nc = len(unmix.col)
        cm = ChannelMap(chan_map, (nr, nc), pitch=pitch, col_major=False)
        return cm, electrode_chans, other_chans, []

    def find_trigger_signals(self, data_file):
        bnc_columns = self.bnc_columns
        if not bnc_columns:
            return (), ()
        # If trigger index is an integer, proceed. If not and it evaluates false, then skip
        if not isinstance(self.trigger_idx, int) and not self.trigger_idx:
            return (), ()
        if not np.iterable(bnc_columns):
            bnc_columns = (bnc_columns,)

        trigger_idx = self.trigger_idx
        if np.iterable(trigger_idx):
            trigger_idx = trigger_idx[0]
        with h5py.File(data_file, 'r') as h5file:
            nrow = int(h5file['numRow'][()])
            # if this is a downsample file, it should be the case that a BNC array has been extracted and downsampled
            # without filtering
            if 'bnc' in h5file:
                bnc_data = h5file['bnc'][:].reshape(len(bnc_columns), nrow, -1)
            else:
                bnc_channels = np.concatenate([np.arange(bnc * nrow, (bnc + 1) * nrow) for bnc in bnc_columns])
                if self.transpose_array:
                    bnc_data = h5file['data'][:, bnc_channels].T
                else:
                    bnc_data = h5file['data'][bnc_channels, :]
                bnc_data = bnc_data.reshape(len(bnc_columns), nrow, -1)
            try:
                trigger_signal = bnc_data[trigger_idx]
                pos_edge = process_trigger(trigger_signal)[0]
            except (IndexError, ValueError) as e:
                tb = e.__traceback__
                msg = 'Trigger channels were specified but do not exist'
                if self.raise_on_glitch:
                    raise Exception(msg).with_traceback(tb)
                else:
                    warnings.warn(msg, RuntimeWarning)
        return trigger_signal, pos_edge


# def slice_style(cmd_str):
#     if cmd_str.find('skip') >= 0:
#         cmd1, cmd2 = cmd_str.split('-')
#         if cmd2 == 'odd':
#             return slice(None, None, 2)
#         if cmd2 == 'even':
#             return slice(1, None, 2)
#         else:
#             n = int(cmd1.replace('skip', ''))
#             idx = list(map(int, cmd2.split(',')))
#             select = np.setdiff1d(np.arange(n), np.array(idx))
#             return select
#     elif cmd_str.find('all') >= 0:
#         return slice(None)
#     else:
#         raise NotImplementedError('slicing not known')
#
#
# def rawload_active(
#         exp_path, test, gain, shm=False,
#         bnc=(), unmix=None, row_cmd=''
# ):
#     # splits the raw TDMS file into channel data and BNC data
#
#     try:
#         raw_load = load_bunch(os.path.join(exp_path, test + '.h5'), '/')
#     except IOError:
#         raw_load = load_bunch(os.path.join(exp_path, test + '.mat'), '/')
#
#     try:
#         Fs = raw_load.Fs
#     except:
#         Fs = raw_load.fs
#
#     shape = raw_load.data.shape
#     if shape[1] < shape[0]:
#         raw_load.data = raw_load.data.transpose()
#     nrow, ncol_load = list(map(int, (raw_load.numRow, raw_load.numCol)))
#     nchan = int(raw_load.numChan)
#     if raw_load.data.shape[0] < nchan * nrow:
#         # each row of data needs to be demuxed as (nsamp, nrow)
#         # since rows are serially sampled in every pass
#         demux = raw_load.data.reshape(nchan, -1, nrow).transpose(0, 2, 1)
#     else:
#         demux = raw_load.data.reshape(nchan, nrow, -1)
#
#     del raw_load['data']
#     if unmix is None:
#         extra = range(ncol_load, nchan)
#         unmix = DAQunmix(slice(0, ncol_load), slice(None), extra, ())
#     col_slice = unmix.col
#     row_slice = unmix.row
#     extra_col = unmix.extra_col
#     extra_row = unmix.extra_row  # currently unused
#
#     # get BNC channels (triggers and stims, etc) and any extra channels
#     bnc = list(map(int, bnc))
#     bnc_chans = demux[bnc].copy() if len(bnc) else ()
#     extra_chans = demux[extra_col].copy() if len(extra_col) else ()
#
#     # get electrode channels
#     cdata = demux[col_slice]
#     del demux
#     while gc.collect():
#         pass
#     cdata = cdata[:, row_slice, :]
#     f = row_cmd.find('avg')
#     if f >= 0:
#         n_avg = int(row_cmd[f + 3:])
#         # reshape the data into (n_col, n_row', n_avg, n_pts)
#         nrow = nrow / n_avg
#         shp = list(cdata.shape)
#         shp[1] = nrow
#         shp.insert(2, n_avg)
#         cdata = cdata.reshape(shp).mean(-2)
#     else:
#         nrow = cdata.shape[1]
#     if shm:
#         data = shared_copy(cdata)
#     else:
#         data = cdata.copy()
#     del cdata
#     while gc.collect():
#         pass
#     data.shape = (-1, data.shape[-1])
#     data /= gain
#     ncol = data.shape[0] / nrow
#     try:
#         info = tdms_info(raw_load.info)
#     except AttributeError:
#         info = None
#     return data, bnc_chans, extra_chans, Fs, (nrow, ncol), info
#
#
# def load_active(exp_path, name, electrode, daq, headstage,
#                 bandpass=(), notches=(), trigger=0,
#                 snip_transient=True, units='uV', save=False,
#                 row_order=(), bnc=(), **load_kws
#                 ):
#     """
#     Load a variety of active-electrode data formats.
#
#     * exp_path, name: the path and recording file name (without extension)
#     * electrode: name of electrode used
#     * daq: data-acquisition system (see below)
#     * other parameters straightforward
#
#     The DAQ label identifies a particular electrode-indexing scheme. In
#     principle columns and rows can be permuted in any order, and the DAQ
#     label is specific to a single order for a given electrode.
#
#     """
#
#     unmix = get_daq_unmix(daq, headstage, electrode, row_order=row_order)
#     data, bnc_chans, extra_chans, Fs, eshape, info = rawload_active(
#         exp_path, name, gain[daq.lower()],
#         shm=True, unmix=unmix, bnc=bnc, **load_kws
#         )
#
#     # get triggers
#     if len(bnc_chans):
#         pos_edge, trig = process_trigger(bnc_chans[int(trigger)])
#         # re-mux the BNC channels
#         #bnc_chans = bnc_chans.transpose(0, 1, 2)
#         #bnc_chans = bnc_chans.reshape(bnc_chans.shape[0], -1)
#     else:
#         pos_edge = ()
#         trig = None
#
#     # deal with extra chans
#     if len(extra_chans):
#         extra_chans = extra_chans.reshape(extra_chans.shape[0], -1)
#
#     # get electrode channel map
#     ii, jj = np.mgrid[:eshape[0], :eshape[1]]
#     # channels are ordered in column-major (i.e. rows count serially)
#     chan_map = ut.mat_to_flat(
#         eshape, ii.ravel('F'), jj.ravel('F'), col_major=False
#         )
#     # describe this order in row-major fashion
#     chan_map = ut.ChannelMap(chan_map, eshape, col_major=False,
#                              pitch=pitch_lookup.get(electrode, 1))
#
#     if units.lower() != 'v':
#         convert_scale(data, 'v', units)
#
#     # do highpass filtering for stationarity
#     if bandpass:
#         # remove DC from rows
#         if bandpass[0] > 0:
#             data -= data.mean(1)[:,None]
#         ft.filter_array(
#             data,
#             design_kwargs=dict(lo=bandpass[0], hi=bandpass[1], Fs=Fs),
#             filt_kwargs=dict(filtfilt=True)
#             )
#
#     if notches:
#         ft.notch_all(
#             data, Fs, lines=notches, inplace=True, filtfilt=True
#             )
#
#
#     if snip_transient:
#         if isinstance(snip_transient, bool):
#             snip_len = int( Fs * 5 )
#         else:
#             snip_len = int( Fs * snip_transient )
#         if len(pos_edge):
#             pos_edge -= snip_len
#             pos_edge = pos_edge[pos_edge > 0]
#             trig = trig[...,snip_len:].copy()
#         if len(bnc_chans):
#             f = bnc_chans.shape[-1] / data.shape[-1]
#             bnc_chans = bnc_chans[...,snip_len*f:].copy()
#         if len(extra_chans):
#             f = extra_chans.shape[-1] / data.shape[-1]
#             extra_chans = extra_chans[...,snip_len*f:].copy()
#
#         data = data[...,snip_len:].copy()
#
#     dset = ut.Bunch()
#     dset.pos_edge = pos_edge
#     dset.data = data
#     dset.extra_chans = extra_chans
#     dset.bnc = bnc_chans
#     dset.chan_map = chan_map
#     dset.Fs = Fs
#     while not os.path.split(exp_path)[1]:
#         exp_path = os.path.split(exp_path)[0]
#     dset.name = '.'.join( [os.path.split(exp_path)[1], name] )
#     dset.bandpass = bandpass
#     dset.trig = trig
#     dset.transient_snipped = snip_transient
#     dset.units = units
#     dset.notches = notches
#     dset.info = info
#
#     return dset
