from tempfile import NamedTemporaryFile  # , _TemporaryFileCloser
import numpy as np
import h5py
from scipy.signal import lfilter_zi
from numpy.linalg import LinAlgError
from tqdm import tqdm


from ecogdata.expconfig import load_params
from ecogdata.util import ToggleState
from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.parallel.split_methods import lfilter
from ecogdata.filt.time import filter_array, notch_all

from .basic import ElectrodeDataSource, calc_new_samples, PlainArraySource
from .array_abstractions import HDF5Buffer, slice_to_range


__all__ = ['MappedSource', 'MemoryBlowOutError']


# TODO: figure out how to create a temp file for writing that can be re-opened in read-only mode and still be deleted
#  after closing
# class ClosesAfterReopening(object):
#
#     def __init__(self, *args, **kwargs):
#         delete = kwargs.pop('delete', True)
#         self._will_delete = delete
#         kwargs['delete'] = False
#         self._args = args
#         self._kwargs = kwargs
#         self._open_count = 0
#         self.file = NamedTemporaryFile(*args, **kwargs)
#
#
#     def __enter__(self):
#         self._open_count += 1
#         if self._open_count == 2:
#             self.file.file = open(self.file.name, 'r')
#             self.file._closer = _TemporaryFileCloser(self.file.file, self.file.name, self._will_delete)
#         self.file.__enter__()
#         return self
#
#
#     def __exit__(self, exc, value, tb):
#         return self.file.__exit__(exc, value, tb)


class MemoryBlowOutError(Exception):
    """Raise this if a memory-mapped slice will be ginormous."""
    pass


class MappedSource(ElectrodeDataSource):

    def __init__(self, source_file, electrode_field, electrode_channels=None, channel_mask=None,
                 aligned_arrays=(), units_scale=None, transpose=False, raise_on_big_slice=True):
        """
        Provides a memory-mapped data source. This object should rarely be constructed by-hand, but it should not be
        overwhelmingly difficult to do so.

        Parameters
        ----------
        source_file: h5py.File or str
            An opened HDF5 file: the file access mode will be respected. If a file name (str) is given, then the file
            will be opened read-only.
        electrode_field: str
            The electrode recording channels are in source_file[electrode_field]. Mapped array may be Channel x Time
            or Time x Channel. Output slices from this object are always Channel x Time.
        electrode_channels: None or sequence
            The subset of channels in the electrode data array that contain electrode signals. A value of None
            indicates all channels contain electrode signals.
        channel_mask: boolean ndarray or None
            A binary mask the same length as electrode_channels or the number of rows in the signal array is
            electrode channels is None. The set of active electrodes is where the binary mask is True. The number of
            rows returned in a slice from this data source is sum(channel_mask). A value of None indicates all
            electrode channels are active.
        aligned_arrays: sequence
            Any other datasets in source_file that should be aligned with the electrode signal array. These fields
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source. Elements of the sequence can be dataset
            names in the mapped file, or (name, channel-list) pairs in the case that the aligned array is particular
            subset of channels in the same array (e.g. same as the electrode data).
        units_scale: float or 2-tuple
            Either the scaling value or (offset, scaling) values such that signal = (memmap + offset) * scaling
        transpose: bool
            The is the mapped array stored in transpose (Time x Channels)?
        raise_on_big_slice: bool
            If True (default), then raise an exception if the proposed read slice will be larger than the memory
            limit. If you really want to read-out anyway, use the `with source.big_slices(True)` context.
        """

        if isinstance(source_file, str):
            source_file = h5py.File(source_file, 'r')
            self.__close_source = True
        else:
            self.__close_source = False
        self._source_file = source_file
        self._electrode_field = electrode_field
        self.channel_mask = channel_mask
        self._electrode_array = self._source_file[self._electrode_field]
        # electrode_channels is a list of of data channel indices where electrode data are (immutable)
        if electrode_channels is None:
            n_channels = self._electrode_array.shape[1] if transpose else self._electrode_array.shape[0]
            self.__electrode_channels = list(range(n_channels))
        else:
            self.__electrode_channels = electrode_channels

        # As the "data" object, set up a array slicing cache with possible units conversion
        # This data cache will need special logic to expose only (active) electrode channels, depending on the
        # electrode_channels list and the current channel mask
        self._units_scale = units_scale
        self._data_buffer = HDF5Buffer(self._electrode_array, units_scale=units_scale)
        self._transpose = transpose
        self.dtype = self._data_buffer.dtype

        # channel_mask is the list of data indices for the active electrode set (mutable)
        self._active_channels = self.__electrode_channels
        self.set_channel_mask(channel_mask)

        # this is toggle than when called creates a context manager with the supplied state (ToggleState)
        self._allow_big_slicing = ToggleState(init_state=False)
        self._raise_on_big_slice = raise_on_big_slice
        # Normally self[:10] will return an array. With this state toggled, it will return a new MappedSource
        self._slice_channels_as_maps = ToggleState(init_state=False)

        # The other aux fields will be setattr'd like
        # setattr(self, field, source_file[field]) ??
        array_names = []
        for field in aligned_arrays:
            if isinstance(field, str):
                setattr(self, field, source_file[field])
            else:
                # this actually loads the channels
                channels = np.s_[:, field[1]] if self._transpose else np.s_[field[1], :]
                field = field[0]
                setattr(self, field, self._electrode_array[channels])
            array_names.append(field)
        # after this, only refer to these arrays by name
        self.aligned_arrays = array_names

    def __del__(self):
        if self.__close_source:
            self._source_file.close()

    def __len__(self):
        return self.shape[0]

    def close_source(self):
        """Close off the source file"""
        self._source_file.close()

    @property
    def shape(self):
        if self._transpose:
            return len(self._active_channels), self._data_buffer.shape[0]
        else:
            return len(self._active_channels), self._data_buffer.shape[1]

    @property
    def writeable(self):
        return self._data_buffer.writeable

    @property
    def big_slices(self):
        return self._allow_big_slicing

    @property
    def channels_are_maps(self):
        return self._slice_channels_as_maps

    @property
    def is_direct_map(self):
        all_active = self.binary_channel_mask.all()
        # quick short-circuit: False if there are masked channels
        if not all_active:
            return False
        num_disk_channels = self._data_buffer.shape[1] if self._transpose else self._data_buffer.shape[0]
        mapped_channels = np.array(self.__electrode_channels)
        all_mapped = np.array_equal(mapped_channels, np.arange(num_disk_channels))
        return all_mapped

    @property
    def binary_channel_mask(self):
        if not hasattr(self, '_bmask'):
            self._bmask = np.ones(len(self.__electrode_channels), '?')
        if self._bmask.sum() == len(self._active_channels):
            return self._bmask
        active_channels = set(self._active_channels)
        for n, c in enumerate(self.__electrode_channels):
            self._bmask[n] = c in active_channels
        return self._bmask

    @property
    def _auto_block_length(self):
        chunks = self._electrode_array.chunks
        block_size = chunk_size = chunks[0] if self._transpose else chunks[1]
        min_block_size = 10000
        while block_size < min_block_size:
            block_size += chunk_size
        return block_size

    def set_channel_mask(self, channel_mask):
        """
        Update the internal electrode channel mapping to exclude channel masked from an electrode vector binary mask.

        Parameters
        ----------
        channel_mask: binary ndarray or None
            A binary mask vector the same length of the number of electrode channels.

        """

        # currently this would not allow composite masking, such as
        # data1 = data(mask1)  len(mask1) == num_electrodes
        # data2 = data1(mask2)  len(mask2) == num_mask1_electrodes
        #
        # So mask resets will have to be stated in terms of the full electrode set
        n_electrodes = len(self.__electrode_channels)
        if channel_mask is None or not len(channel_mask):
            # (re)set to the full set of electrode channels
            self._active_channels = self.__electrode_channels
        elif len(channel_mask) != n_electrodes:
            raise ValueError('channel_mask must be length {}'.format(n_electrodes))
        else:
            self._active_channels = list()
            for n, c in enumerate(self.__electrode_channels):
                if channel_mask[n]:
                    self._active_channels.append(c)

    def _slice_logic(self, slicer):
        """Translate the slicing object to point to the correct data channels on disk"""
        if not np.iterable(slicer):
            slicer = (slicer,)
        if self.is_direct_map:
            return slicer[::-1] if self._transpose else slicer
        chan_range = slicer[0]
        if len(slicer[1:]):
            time_range = slicer[1]
        else:
            # If the slice is for channels only and we're in map-subset mode, then simply return the slice without
            # modifying the range
            if self.channels_are_maps.state:
                return slicer
            time_range = slice(None)
        if isinstance(chan_range, (int, np.integer)):
            get_chan = self._active_channels[chan_range]
            return (time_range, get_chan) if self._transpose else (get_chan, time_range)
        # Map the sliced channels to corresponding active channels
        if isinstance(chan_range, slice):
            r_max = len(self._active_channels)
            chan_range = list(slice_to_range(chan_range, r_max))
        get_chans = [self._active_channels[c] for c in chan_range]
        return (time_range, get_chans) if self._transpose else (get_chans, time_range)

    def _check_slice_size(self, slicer):
        if not self._raise_on_big_slice:
            return
        shape = self._data_buffer._get_output_array(slicer, only_shape=True)
        size = np.prod(shape) * self._data_buffer.dtype.itemsize
        state = self._allow_big_slicing.state
        if size > load_params().memory_limit and not state:
            raise MemoryBlowOutError('A read with shape {} will be *very* large. Use the big_slices context to '
                                     'proceed'.format(shape))

    def slice_subset(self, slicer):
        new_electrode_channels = np.array(self._active_channels)[slicer].tolist()
        return MappedSource(self._source_file, self._electrode_field, electrode_channels=new_electrode_channels,
                            aligned_arrays=self.aligned_arrays, units_scale=self._units_scale, transpose=self._transpose,
                            raise_on_big_slice=self._raise_on_big_slice)

    def __getitem__(self, slicer):
        """Return the sub-series of samples selected by slicer on (possibly a subset of) all channels"""
        # data goes out as [subset]channels x time
        slicer = self._slice_logic(slicer)
        if len(slicer) == 1 and self.channels_are_maps.state:
            return self.slice_subset(slicer)
        self._check_slice_size(slicer)
        # print('Slicing memmap as {}'.format(slicer))
        with self._data_buffer.transpose_reads(self._transpose):
            # What this context should mean is that the buffer is going to get sliced in the prescribed way and then
            # the output is going to get transposed. Handling the transpose logic in the buffer avoids some
            # unnecessary array copies
            data_slice = self._data_buffer[slicer]
        return data_slice

    def __setitem__(self, slicer, data):
        """Write the sub-series of samples selected by slicer (from possibly a subset of channels)"""
        # data comes in as [subset]channels x time
        if not self.writeable:
            print('Cannot write to this file')
            return
        # if both these conditions are try, something is probably wrong
        suspect_call = self.writeable and self._transpose
        # make sure subset mode is off
        with self.channels_are_maps(False):
            slicer = self._slice_logic(slicer)
        try:
            self._data_buffer[slicer] = data
        except IndexError as e:
            if suspect_call:
                tb = e.__traceback__
                msg = 'This mapped array is both writeable and in transpose mode, which probably is not what you mean '
                'to be doing and probably is the source of this error.'
                raise Exception(msg).with_traceback(tb)
            else:
                raise e

    def iter_channels(self, chans_per_block=None, use_max_memory=True, return_slice=False):
        # Just change the signature to use maximum memory by default
        return super(MappedSource, self).iter_channels(chans_per_block=chans_per_block,
                                                       use_max_memory=use_max_memory, return_slice=return_slice)

    def filter_array(self, **kwargs):
        kwargs['block_filter'] = bfilter
        # kwargs['inplace'] = True
        filter_array(self, **kwargs)
        if 'out' in kwargs:
            return kwargs['out']
        return self

    def notch_filter(self, **kwargs):
        kwargs['block_filter'] = bfilter
        # kwargs['inplace'] = True
        notch_all(self, **kwargs)
        if 'out' in kwargs:
            return kwargs['out']
        return self

    def mirror(self, new_rate_ratio=None, writeable=True, mapped=True, channel_compatible=False, filename='',
               copy='', **map_args):
        """
        Create an empty ElectrodeDataSource based on the current source, possibly with a new sampling rate and new
        access permissions.

        Parameters
        ----------
        new_rate_ratio: int or None
            Ratio of old to new sample rate for the mirrored array (> 1).
        writeable: bool
            Make any new MappedSource arrays writeable. This implies 1) datatype casting to floats, and 2) there is
            no more units conversion on the primary array.
        mapped: bool
            If False, mirror to a PlainArraySource (in memory). Else mirror into a new MappedSource.
        channel_compatible: bool
            If True, preserve the same number of raw data channels in a MappedSource. Otherwise, reduce the channels
            to just the set of active channels.
        filename: str
            Name of the new MappedSource. If empty, use a self-destroying temporary file.
        copy: str
            Code whether to copy any arrays, which is only valid when new_rate_ratio is None or 1. 'aligned' copies
            aligned arrays. 'electrode' copies electrode data: only valid if channel_compatible is False.
            'all' copies all arrays. By default, nothing is copied.
        map_args: dict
            Any other MappedSource arguments

        Returns
        -------
        datasource: ElectrodeDataSource subtype

        """

        T = self.shape[1]
        if new_rate_ratio:
            T = calc_new_samples(T, new_rate_ratio)

        copy_electrodes_coded = copy.lower() in ('all', 'electrode')
        copy_aligned_coded = copy.lower() in ('all', 'aligned')
        diff_rate = T != self.shape[1]
        if copy_electrodes_coded:
            if diff_rate or channel_compatible:
                copy_electrodes = False
                print('Not copying electrode channels. Diff rate '
                      '({}) or indirect channel map ({})'.format(diff_rate, channel_compatible))
            else:
                copy_electrodes = True
        else:
            copy_electrodes = False
        if copy_aligned_coded:
            if diff_rate:
                copy_aligned = False
                print('Not copying aligned arrays: different sample rate')
            else:
                copy_aligned = True
        else:
            copy_aligned = False

        if mapped:
            if channel_compatible:
                electrode_channels = self.__electrode_channels
                C = self._electrode_array.shape[1] if self._transpose else self._electrode_array.shape[0]
                channel_mask = self.binary_channel_mask
            else:
                C = self.shape[0]
                electrode_channels = None
                channel_mask = None
            if writeable:
                new_dtype = 'd'
                reopen_mode = 'r+'
                units_scale = None
            else:
                new_dtype = self._electrode_array.dtype
                reopen_mode = 'r'
                units_scale = self._units_scale
            tempfile = not filename
            if tempfile:
                with NamedTemporaryFile(mode='ab', dir='.', delete=False) as f:
                    # punt on the unlink-on-close issue for now with "delete=False"
                    # f.file.close()
                    filename = f.name
            with h5py.File(filename, 'w') as fw:
                # Create all new datasets as non-transposed
                fw.create_dataset(self._electrode_field, shape=(C, T), dtype=new_dtype, chunks=True)
                if copy_electrodes:
                    for block, sl in self.iter_blocks(return_slice=True):
                        fw[self._electrode_field][sl] = block
                for name in self.aligned_arrays:
                    arr = getattr(self, name)
                    if len(arr.shape) > 1:
                        dims = (arr.shape[1], T) if self._transpose else (arr.shape[0], T)
                    # is this correct ???
                    dtype = 'd' if writeable else arr.dtype
                    fw.create_dataset(name, shape=dims, dtype=dtype, chunks=True)
                    if copy_aligned:
                        aligned = getattr(self, name)[:]
                        fw[name][:] = aligned.T if self._transpose else aligned
            f_mapped = h5py.File(filename, reopen_mode)
            return MappedSource(f_mapped, self._electrode_field, electrode_channels=electrode_channels,
                                channel_mask=channel_mask, aligned_arrays=self.aligned_arrays, units_scale=units_scale,
                                transpose=False, **map_args)
        else:
            self._check_slice_size(np.s_[:, :])
            C = self.shape[0]
            new_source = shared_ndarray((C, T), 'd')
            if copy_electrodes:
                for block, sl in self.iter_blocks(return_slice=True):
                    new_source[sl] = block
            # Kind of tricky with aligned fields -- assume that transpose means the same thing for them?
            # But also un-transpose them on this mirroring step
            aligned_arrays = dict()
            for name in self.aligned_arrays:
                arr = getattr(self, name)
                if len(arr.shape) > 1:
                    dims = (arr.shape[1], T) if self._transpose else (arr.shape[0], T)
                aligned_arrays[name] = shared_ndarray(dims, 'd')
                if copy_aligned:
                    aligned = getattr(self, name)[:]
                    aligned_arrays[name][:] = aligned.T if self._transpose else aligned
            return PlainArraySource(new_source, **aligned_arrays)


def bfilter(b, a, x, out=None, filtfilt=False, verbose=False, **extra):
    """
    Apply linear filter inplace over array x streaming from disk.

    Parameters
    ----------
    b: ndarray
        Transfer function numerator polynomial coefficients.
    a: ndarray
        Transfer function denominator polynomial coefficients.
    x: MappedSource
        Mapped data source
    out: MappedSource
        Mapped source to store filter output, otherwise store in x.
    filtfilt: bool
        If True, make a second filtering in reverse time sequence to create zero phase.
    verbose: bool
        Make a progress bar.

    """
    try:
        zii = lfilter_zi(b, a)
    except LinAlgError:
        # the integrating filter doesn't have valid zi
        zii = np.array([0.0])

    zi_sl = [np.newaxis] * 2
    zi_sl[1] = slice(None)
    zi_sl = tuple(zi_sl)
    xc_sl = [slice(None)] * 2
    xc_sl[1] = slice(0, 1)
    xc_sl = tuple(xc_sl)
    # fir_size = len(b)
    zi = None
    itr = x.iter_blocks(return_slice=True)
    if verbose:
        itr = tqdm(itr, desc='Blockwise filtering', leave=True, total=len(itr))
    for xc, sl in itr:
        # xc = x[sl]
        if zi is None:
            zi = zii[zi_sl] * xc[xc_sl]
        xcf, zi = lfilter(b, a, xc, axis=1, zi=zi)
        if out is None:
            x[sl] = xcf
        else:
            out[sl] = xcf

    if not filtfilt:
        del xc
        del xcf
        return

    # Nullify initial conditions for reverse run (not sure if it's correct, but it's compatible with bfilter from
    # the ecogdata.filter.time.blocked_filter module
    zi = None
    if out is None:
        itr = x.iter_blocks(return_slice=True, reverse=True)
    else:
        itr = out.iter_blocks(return_slice=True, reverse=True)
    if verbose:
        itr = tqdm(itr, desc='Blockwise filtering (reverse)', leave=True, total=len(itr))
    for xc, sl in itr:
        # if out is None:
        #     xc = x[sl]
        # else:
        #     xc = out[sl]
        if zi is None:
            zi = zii[zi_sl] * xc[xc_sl]
        xcf, zi = lfilter(b, a, xc, axis=1, zi=zi)
        # write out with negative step slices (buffer will correct the write order)
        if out is None:
            x[sl] = xcf
        else:
            out[sl] = xcf
    del xc
    del xcf