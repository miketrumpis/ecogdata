from tempfile import NamedTemporaryFile  # , _TemporaryFileCloser
import numpy as np
import h5py

from ecogdata.expconfig import load_params
from ecogdata.util import ToggleState
from ecogdata.parallel.array_split import shared_ndarray, shared_copy

from .basic import ElectrodeDataSource, calc_new_samples, PlainArraySource
from .array_abstractions import HDF5Buffer, slice_to_range


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
                 aux_fields=(), units_scale=None, transpose=False, raise_on_big_slice=True):
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
        aux_fields: sequence
            Any other datasets in source_file that should be aligned with the electrode signal array. These fields
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source.
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
        self._aux_fields = aux_fields
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
        self._data_buffer = HDF5Buffer(self._electrode_array, units_scale=units_scale)
        self._transpose = transpose
        self.dtype = self._data_buffer.dtype

        # channel_mask is the list of data indices for the active electrode set (mutable)
        self._active_channels = self.__electrode_channels
        self.set_channel_mask(channel_mask)

        # this is toggle than when called creates a context manager with the supplied state (ToggleState)
        self._allow_big_slicing = ToggleState(init_state=False)
        self._raise_on_big_slice = raise_on_big_slice

        # The other aux fields will be setattr'd like
        # setattr(self, field, source_file[field]) ??
        for field in aux_fields:
            setattr(self, field, source_file[field])
        self._aligned_arrays = aux_fields

    def __del__(self):
        if self.__close_source:
            self._source_file.close()

    @property
    def writeable(self):
        return self._data_buffer.writeable

    @property
    def big_slices(self):
        return self._allow_big_slicing

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
        if self._transpose:
            self.data_shape = len(self._active_channels), self._data_buffer.shape[0]
        else:
            self.data_shape = len(self._active_channels), self._data_buffer.shape[1]

    @property
    def is_direct_map(self):
        num_disk_channels = self._data_buffer.shape[1] if self._transpose else self._data_buffer.shape[0]
        mapped_channels = np.array(self.__electrode_channels)
        all_mapped = np.array_equal(mapped_channels, np.arange(num_disk_channels))
        all_active = self.binary_channel_mask.all()
        return all_mapped and all_active

    @property
    def binary_channel_mask(self):
        bmask = np.ones(len(self.__electrode_channels), '?')
        active_channels = set(self._active_channels)
        for n, c in enumerate(self.__electrode_channels):
            if c not in active_channels:
                bmask[n] = False
        return bmask

    @property
    def _auto_block_length(self):
        chunks = self._electrode_array.chunks
        return chunks[0] if self._transpose else chunks[1]

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

    def __getitem__(self, slicer):
        """Return the sub-series of samples selected by slicer on (possibly a subset of) all channels"""
        # data goes out as [subset]channels x time
        slicer = self._slice_logic(slicer)
        self._check_slice_size(slicer)
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

    def iter_channels(self, chans_per_block=None, max_memory=None, return_slice=False):
        if max_memory is None:
            max_memory = load_params()['memory_limit']
        if self._transpose:
            # compensate for the necessary copy-to-transpose
            max_memory /= 2
        C, T = self.data_shape
        if chans_per_block is None:
            bytes_per_samp = self.dtype.itemsize
            chans_per_block = max(1, int(max_memory / T / bytes_per_samp))
        num_iter = C // chans_per_block
        if chans_per_block * num_iter < C:
            num_iter += 1
        for i in range(num_iter):
            start = i * chans_per_block
            stop = min(C, (i + 1) * chans_per_block)
            if return_slice:
                yield self[start:stop], np.s_[start:stop, :]
            else:
                yield self[start:stop]

            # why is this logic necessary? redundant no?
            # to_yield = self._active_channels[start:stop]
            # print('Getting channels', to_yield)
            # if self._transpose:
            #     out = shared_copy(self._data_buffer[:, to_yield].T)
            # else:
            #     out = self._data_buffer[to_yield, :]
            # if return_slice:
            #     yield out, np.s_[start:stop, :]
            # else:
            #     yield out

    def mirror(self, new_rate_ratio=None, writeable=True, mapped=True, channel_compatible=False, filename=''):
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

        Returns
        -------
        datasource: ElectrodeDataSource subtype

        """

        T = self.data_shape[1]
        if new_rate_ratio:
            T = calc_new_samples(T, new_rate_ratio)

        if mapped:
            if channel_compatible:
                electrode_channels = self.__electrode_channels
                C = self._electrode_array.shape[1] if self._transpose else self._electrode_array.shape[0]
                channel_mask = self.binary_channel_mask
            else:
                C = self.data_shape[0]
                electrode_channels = None
                channel_mask = None
            if writeable:
                new_dtype = 'd'
                reopen_mode = 'r+'
                units_scale = None
            else:
                new_dtype = self._electrode_array.dtype
                reopen_mode = 'r'
                units_scale = self._data_buffer._units_scale
            tempfile = not filename
            if tempfile:
                with NamedTemporaryFile(mode='ab', dir='.', delete=False) as f:
                    # punt on the unlink-on-close issue for now with "delete=False"
                    # f.file.close()
                    filename = f.name
            with h5py.File(filename, 'w') as fw:
                # Create all new datasets as non-transposed
                fw.create_dataset(self._electrode_field, shape=(C, T), dtype=new_dtype, chunks=True)
                for name in self._aux_fields:
                    arr = getattr(self, name)
                    if len(arr.shape) > 1:
                        dims = (arr.shape[1], T) if self._transpose else (arr.shape[0], T)
                    # is this correct ???
                    dtype = 'd' if writeable else arr.dtype
                    fw.create_dataset(name, shape=dims, dtype=dtype, chunks=True)
            f_mapped = h5py.File(filename, reopen_mode)
            return MappedSource(f_mapped, self._electrode_field, electrode_channels=electrode_channels,
                                channel_mask=channel_mask, aux_fields=self._aux_fields, units_scale=units_scale,
                                transpose=False)
        else:
            C = self.data_shape[0]
            new_source = shared_ndarray((C, T), 'd')
            # Kind of tricky with aux fields -- assume that transpose means the same thing for them?
            # But also un-transpose them on this mirroring step
            aux_fields = dict()
            for name in self._aux_fields:
                arr = getattr(self, name)
                if len(arr.shape) > 1:
                    dims = (arr.shape[1], T) if self._transpose else (arr.shape[0], T)
                aux_fields[name] = shared_ndarray(dims, 'd')
            return PlainArraySource(new_source, **aux_fields)
