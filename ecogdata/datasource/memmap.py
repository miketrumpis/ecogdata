import numpy as np
import h5py

from .basic import ElectrodeDataSource
from .array_abstractions import ReadCache, slice_to_range, range_to_slice

class MappedSource(ElectrodeDataSource):


    def __init__(self, source_file, samp_rate, electrode_field, electrode_channels=None, channel_mask=None,
                 aux_fields=(), units_scale=None, transpose=False):
        """
        Provides a memory-mapped data source. This object should rarely be constructed by-hand, but it should not be
        overwhelmingly difficult to do so.

        Parameters
        ----------
        source_file: h5py.File or str
            An opened HDF5 file: the file access mode will be respected. If a file name (str) is given, then the file
            will be opened read-only.
        samp_rate: float
            Sampling rate of electrode timeseries
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
            will be kept at the same sampling rate and index alignment as the electrode signal.
        units_scale: float or 2-tuple
            Either the scaling value or (offset, scaling) values such that signal = (memmap + offset) * scaling
        transpose: bool
            The is the mapped array stored in transpose (Time x Channels)?
        """

        if isinstance(source_file, str):
            source_file = h5py.File(source_file, 'r')
            self.__close_source = True
        else:
            self.__close_source = False
        self._source_file = source_file
        # Anything other than 'r' indicates some kind of write access
        self.__writeable = source_file.mode != 'r'
        self.Fs = samp_rate
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
        self._data_cache = ReadCache(self._electrode_array, units_scale=units_scale)
        self._transpose = transpose

        # channel_mask is the list of data indices for the active electrode set (mutable)
        self._channel_mask = self.__electrode_channels
        self.set_channel_mask(channel_mask)

        # The other aux fields will be setattr'd like
        # setattr(self, field, source_file[field]) ??
        for field in aux_fields:
            setattr(self, field, source_file[field])


    def __del__(self):
        if self.__close_source:
            self._source_file.close()


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
            self._channel_mask = self.__electrode_channels
        elif len(channel_mask) != n_electrodes:
            raise ValueError('channel_mask must be length {}'.format(n_electrodes))
        else:
            self._channel_mask = list()
            for n, c in enumerate(self.__electrode_channels):
                if channel_mask[n]:
                    self._channel_mask.append(c)
        if self._transpose:
            self.data_shape = len(self._channel_mask), self._data_cache.shape[0]
        else:
            self.data_shape = len(self._channel_mask), self._data_cache.shape[1]


    @property
    def binary_channel_mask(self):
        bmask = np.ones(len(self.__electrode_channels), '?')
        active_channels = set(self._channel_mask)
        for n, c in enumerate(self.__electrode_channels):
            if c not in active_channels:
                bmask[n] = False
        return bmask


    def _output_channel_subset(self, array_block):
        """Returns the subset of array channels defined by a channel mask"""
        # The subset of data channels that are array channels is defined in particular data source types
        if self._channel_mask is None:
            return array_block
        return array_block[self._channel_mask]


    def __getitem__(self, slicer):
        """Return the sub-series of samples selected by slicer on (possibly a subset of) all channels"""
        # data goes out as [subset]channels x time
        # assume 2D slicer
        chan_range, time_range = slicer
        if isinstance(chan_range, (int, np.integer)):
            get_chan = self._channel_mask[chan_range]
            if self._transpose:
                return self._data_cache[(time_range, get_chan)]
            else:
                return self._data_cache[(get_chan, time_range)]

        # if the channel range is not a single channel, then load the full slice anyway
        if self._transpose:
            chan_axis = 1
            slicer = (time_range, slice(None))
        else:
            chan_axis = 0
            slicer = (slice(None), time_range)
        full_slice = self._data_cache[slicer]
        if isinstance(chan_range, slice):
            r_max = self._data_cache.shape[1] if self._transpose else self._data_cache.shape[0]
            chan_range = slice_to_range(chan_range, r_max)
        # get_chans = [self._channel_mask[c] for c in chan_range]
        get_chans = [chan_range[c] for c in self._channel_mask]
        out = np.take(full_slice, get_chans, axis=chan_axis)
        return out.T.copy() if self._transpose else out


    def __setitem__(self, slicer, data):
        """Write the sub-series of samples selected by slicer (from possibly a subset of channels)"""
        # data comes in as [subset]channels x time
        if not self.__writeable:
            print('Cannot write to this file')
            return
        pass


    def iter_blocks(self, block_length, overlap=0, return_slice=False):
        """
        Yield data blocks with given length (in samples)

        Parameters
        ----------
        block_length: int
            Number of samples per block
        overlap: int
            Number of samples overlapping between blocks
        return_slice: bool
            If True return the ndarray block followed by the memmap array slice to yield this block. Helpful for
            pairing the yielded blocks with the same position in a follower array, or writing back transformed data
            to this memmap (if writeable).

        """

        L = block_length
        # # Blocks need to be even length
        # if L % 2:
        #     L += 1
        T = self.data_shape[1]
        N = T // (L - overlap)
        if (L - overlap) * N < T:
            N += 1
        for i in range(N):
            start = i * (L - overlap)
            if start >= T:
                raise StopIteration
            end = min(T, start + L)
            # if the tail block is odd-length, clip off the last point
            if (end - start) % 2:
                end -= 1
            sl = (slice(None), slice(start, end))
            if return_slice:
                yield self[sl], sl
            else:
                yield self[sl]



class DataSource(object):
    """Data source providing access from file-mapped LFP signals.

    >>> batch = data_source[a:b]
    This syntax returns an array timeseries in (channels, samps) shape, including only electrode array channels (but
    excluding any channels rejected by, e.g., manual inspection).

    >>> data_source[a:b] = filtered_batch
    If the DataSource was constructed with "saving=True", then this writes array channels into a file that is
    compatible with the geometry of the original source file.

    >>> data_source.channel_map
    This is a ChannelMap object, providing channel to electrode-site map information and methods.
    """

    def __init__(self, array, channel_map, samp_rate, exclude_channels=[],
                 is_transpose=False, saving=False, save_mod='clean'):
        self.array = array
        self.channel_map = channel_map
        # keep this in case of HDF5 export
        self.__full_channel_map = channel_map[:]
        self.samp_rate = samp_rate
        self.is_transpose = is_transpose
        self.saving = saving
        if len(exclude_channels):
            n_chan = len(channel_map)
            mask = np.ones(n_chan, '?')
            mask[exclude_channels] = False
            self.channel_map = channel_map.subset(mask)
            self._channel_mask = mask
        else:
            self._channel_mask = None
        self.series_length = self.array.shape[1 - int(is_transpose)]
        if saving:
            self._initialize_output(save_mod)


    def _initialize_output(self, file_mod='clean'):
        # Must be overloaded
        # defines self.output_file and self._write_array
        pass


    def _output_channel_subset(self, array_block):
        """Returns the subset of array channels defined by a channel mask"""
        # The subset of data channels that are array channels is defined in particular data source types
        if self._channel_mask is None:
            return array_block
        return array_block[self._channel_mask]


    def _full_channel_set(self, array_block):
        """Writes a subset of array channels into the full set of array channels"""
        if self._channel_mask is None:
            return array_block
        n_chan = len(self._channel_mask)
        full_block = np.zeros((n_chan, array_block.shape[1]))
        full_block[self._channel_mask] = array_block
        return full_block


    def __getitem__(self, slicer):
        """Return the sub-series of samples selected by slicer on (possibly a subset of) all channels"""
        # data goes out as [subset]channels x time
        if self.is_transpose:
            sub_array = self.array[slicer, :].T
        else:
            sub_array = self.array[:, slicer]
        return self._output_channel_subset(sub_array)


    def __setitem__(self, slicer, data):
        """Write the sub-series of samples selected by slicer (from possibly a subset of channels)"""
        # data comes in as [subset]channels x time
        if not self.saving:
            print('This object has no write file')
            return None
        full_data = self._full_channel_set(data)
        if self.is_transpose:
            self._write_array[slicer, :] = full_data.T
        else:
            self._write_array[:, slicer] = full_data


    def iter_blocks(self, block_length, overlap=0, return_slice=True):
        """Yield data blocks with given length (in samples)"""
        L = block_length
        # Blocks need to be even length
        if L % 2:
            L += 1
        T = self.series_length
        N = T // (L - overlap)
        if (L - overlap) * N < T:
            N += 1
        for i in range(N):
            start = i * (L - overlap)
            if start >= T:
                raise StopIteration
            end = min(T, start + L)
            # if the tail block is odd-length, clip off the last point
            if (end - start) % 2:
                end -= 1
            sl = slice(start, end)
            if return_slice:
                yield self[sl], sl
            else:
                yield self[sl]


    def write_parameters(self, **params):
        """Store processing parameters"""
        # must be overloaded!
        pass
