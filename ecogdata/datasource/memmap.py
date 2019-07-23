from tempfile import NamedTemporaryFile  # , _TemporaryFileCloser
import numpy as np
import h5py


from .basic import ElectrodeDataSource, calc_new_samples, PlainArraySource
from .array_abstractions import ReadCache, slice_to_range



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
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source.
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
        self._aligned_arrays = aux_fields


    def __del__(self):
        if self.__close_source:
            self._source_file.close()

    @property
    def writeable(self):
        return self.__writeable


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


    def mirror(self, new_rate=None, writeable=True, mapped=True, channel_compatible=False, filename=''):
        """
        Create an empty ElectrodeDataSource based on the current source, possibly with a new sampling rate and new
        access permissions.

        Parameters
        ----------
        new_rate: float or None
            New sample rate for the mirrored array.
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
        if new_rate:
            T = calc_new_samples(T, self.Fs, new_rate)
            samp_rate = new_rate
        else:
            samp_rate = self.Fs

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
                units_scale = self._data_cache._units_scale
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
            return MappedSource(f_mapped, samp_rate, self._electrode_field, electrode_channels=electrode_channels,
                                channel_mask=channel_mask, aux_fields=self._aux_fields, units_scale=units_scale,
                                transpose=False)
        else:
            C = self.data_shape[0]
            new_source = shared_ndarray((C, T), dtype='d')
            # Kind of tricky with aux fields -- assume that transpose means the same thing for them?
            # But also un-transpose them on this mirroring step
            aux_fields = dict()
            for name in self._aux_fields:
                arr = getattr(self, name)
                if len(arr.shape) > 1:
                    dims = (arr.shape[1], T) if self._transpose else (arr.shape[0], T)
                aux_fields[name] = shared_ndarray(dims, dtype='d')
            return PlainArraySource(new_source, samp_rate, **aux_fields)

