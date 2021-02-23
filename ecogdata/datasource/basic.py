from warnings import warn
import numpy as np
import h5py
# from tqdm import tqdm

from ecogdata.expconfig import load_params
from ecogdata.parallel.sharedmem import shared_copy, shared_ndarray
from ecogdata.filt.time import downsample, filter_array, notch_all
from ecogdata.filt.blocks import BlockSignalBase


__all__ = ['calc_new_samples', 'ElectrodeDataSource', 'PlainArraySource']


def calc_new_samples(N, rate_change):
    """
    Find number of points in a downsampled N-vector given a rate conversion new_rate:old_rate

    Parameters
    ----------
    N: int
        Original vector length
    rate_change: int
        The old-rate:new-rate ratio (must be an integer > 1)

    Returns
    -------
    P: int
        Downsampled vector length

    """

    if int(rate_change) != rate_change:
        raise ValueError('Rate change ratio is not an integer')
    rate_change = int(rate_change)
    num_pts = N // rate_change
    num_pts += int((N - num_pts * rate_change) > 0)
    return num_pts


class DataSourceBlockIter(BlockSignalBase):

    def __init__(self, datasource, block_length=None, overlap=0, start_offset=0,
                 axis=1, return_slice=False, reverse=False, partial_block=True, **kwargs):
        if block_length is None:
            L = datasource._auto_block_length
        else:
            L = block_length
        super(DataSourceBlockIter, self).__init__(datasource, L, overlap=overlap, axis=axis,
                                                  start_offset=start_offset, partial_block=partial_block,
                                                  reverse=reverse)
        self.datasource = datasource
        self._count = 0
        self.return_slice = return_slice
        self._itr = range(0, self._n_block)[::-1] if reverse else range(0, self._n_block)
        self._slice_cache_args = kwargs

    def __len__(self):
        return len(self._itr)

    def __iter__(self):
        self._count = 0
        return self

    def block(self, b):
        if self._itr.start <= b < self._itr.stop:
            sl = self._make_slice(b)
            return self.datasource[sl], sl if self.return_slice else self.datasource[sl]
        else:
            raise ValueError('Slice {} out of bounds'.format(b))

    def _make_slice(self, i):
        start = i * (self.L - self._overlap) + self.start_offset
        # this really should not happen
        # if start < 0 or start >= self.T:
        #     raise StopIteration
        # self.T already compensated by start_offset -- add back to find last valid index
        end = min(self.T + self.start_offset, start + self.L)
        if self.L == 1:
            # assume that a dimension-eating slice is wanted if block size is 1
            sl = (slice(None), start)
        else:
            if self._reverse:
                start -= 1
                if start < 0:
                    start = None
                sl = (slice(None), slice(end - 1, start, -1))
            else:
                sl = (slice(None), slice(start, end))
        if self.axis == 0:
            sl = sl[::-1]
        return sl

    def __next__(self):
        try:
            i = self._itr[self._count]
        except IndexError:
            raise StopIteration
        sl = self._make_slice(i)
        # Send data caching into motion on first iteration
        if self._count == 0:
            self.datasource.cache_slice(sl, **self._slice_cache_args)
        self._count += 1
        # Wait on previous cache
        output = self.datasource.get_cached_slice()
        if self._count < len(self):
            # Start caching the next load
            i_next = self._itr[self._count]
            self.datasource.cache_slice(self._make_slice(i_next), **self._slice_cache_args)
        if self.return_slice:
            return output, sl
        else:
            return output


class ElectrodeDataSource:
    """
    a parent class for all types
    """

    shape = ()
    dtype = None
    _auto_block_length = 20000
    writeable = True
    # data buffer should be array like, exposing: shape, ndim, len, dtype
    data_buffer = None
    _transpose = False

    def __len__(self):
        return len(self.data_buffer)

    @property
    def shape(self):
        return self.data_buffer.shape

    @property
    def ndim(self):
        return self.data_buffer.ndim

    def cache_slice(self, slicer, not_strided=False, sharedmem=False):
        """
        Caches a slice to yield during iteration. This takes place in a background thread for mapped sources.

        Parameters
        ----------
        slicer: slice
            Array __getitem__ slice spec.
        not_strided: bool
            If True, ensure that sliced array is not strided
        sharedmem: bool
            If True, cast the slice into a shared ctypes array

        """
        if sharedmem:
            self._cache_output = shared_copy(self[slicer])
        elif not_strided:
            output = self[slicer]
            if output.__array_interface__['strides']:
                self._cache_output = output.copy()
            else:
                self._cache_output = output
        else:
            self._cache_output = self[slicer]

    def get_cached_slice(self):
        return self._cache_output

    def iter_blocks(self, block_length=None, overlap=0, start_offset=0, return_slice=False, reverse=False,
                    partial_block=True, **kwargs):
        """
        Yield data blocks with given length (in samples)

        Parameters
        ----------
        block_length: int
            Number of samples per block
        overlap: int
            Number of samples overlapping between blocks
        start_offset: int
            Number of samples to skip before iteration.
        return_slice: bool
            If True return the ndarray block followed by the array slice to yield this block. Helpful for
            pairing the yielded blocks with the same position in a follower array, or writing back transformed data
            to this datasource (if writeable).
        reverse: bool
            If True, yield the blocks in reverse sequence.
        partial_block: bool
            If False, yield every full length block. If True, yield data until the end of the series,
            even if the last block is not full length.
        kwargs: dict
            Arguments for ElectrodeDataSource.cache_slice

        """

        return DataSourceBlockIter(self, axis=1, block_length=block_length, overlap=overlap, start_offset=start_offset,
                                   return_slice=return_slice, reverse=reverse, partial_block=partial_block, **kwargs)

    def iter_channels(self, chans_per_block=None, use_max_memory=False, return_slice=False, **kwargs):
        """
        Yield data channels.

        Parameters
        ----------
        chans_per_block: int
            Number of channels per iteration. The default value is either 16 or based on memory limit if 
            use_max_memory=True.
        use_max_memory: bool
            Set the number of channels based on the "memory_limit" config value.
        return_slice: bool
            If True return the ndarray block followed by the array slice to yield this block. Helpful for
            pairing the yielded blocks with the same position in a follower array, or writing back transformed data
            to this datasource (if writeable).
        kwargs: dict
            Arguments for ElectrodeDataSource.cache_slice

        """
        C, T = self.shape
        if chans_per_block is None:
            if use_max_memory:
                max_memory = load_params()['memory_limit'] if isinstance(use_max_memory, bool) else use_max_memory
                if self._transpose:
                    # compensate for the necessary copy-to-transpose
                    max_memory /= 2
                bytes_per_samp = self.dtype.itemsize
                chans_per_block = max(1, int(max_memory / T / bytes_per_samp))
            else:
                chans_per_block = 16
        return DataSourceBlockIter(self, axis=0, block_length=chans_per_block, return_slice=return_slice, **kwargs)

    def batch_change_rate(self, new_rate_ratio, new_source, antialias_aligned=False, aggregate_aligned=True,
                          verbose=False, filter_inplace=False):
        """
        Downsample a datasource into a new source. Preserves timing alignment of any aligned arrays.

        Parameters
        ----------
        new_rate_ratio: int
            The downsample rate: old_rate / new_rate
        new_source: ElectrodeDataSource
            A Datasource with the correct channel layouts
        antialias_aligned: bool
            Downsample aligned channels with anti-aliasing (may distort logic level signals). If False,
            then just decimate.
        aggregate_aligned: bool
            Reduce sampling rate by accumulate samples in aligned channels. Good choice to logic level, and provides
            some antialiasing.
        verbose: bool
            Use a progress bar.
        filter_inplace: bool
            Do anti-aliasing filtering in-place. TODO: should be set to True for mapped sources

        Returns
        -------

        """
        new_rate_ratio = int(new_rate_ratio)
        if new_source.shape[0] != self.shape[0]:
            raise ValueError('Output source has the wrong number of channels: {}'.format(new_source.shape[0]))
        if new_source.shape[1] != calc_new_samples(self.shape[1], new_rate_ratio):
            raise ValueError('Output source has the wrong series length: {}'.format(new_source.shape[1]))
        chan_itr = self.iter_channels(return_slice=True)
        # DISABLE tqdm for now -- it seems to hold onto a reference for the iteration variable (i.e. big array) and
        # prevents garbage collection
        # verbose = False
        # if verbose:
        #     chan_itr = tqdm(chan_itr, desc='Downsampling channels', leave=True, total=len(chan_itr))

        for raw_channels, sl in chan_itr:
            # kind of fake the sampling rate
            r = downsample(raw_channels, float(new_rate_ratio), r=new_rate_ratio, filter_inplace=filter_inplace)[0]
            new_source[sl] = r
            del raw_channels

        # now decimate aligned_channels with or without anti-aliasing -- *assuming* that a full load won't bust RAM
        for k_src, k_dst in zip(self.aligned_arrays, new_source.aligned_arrays):
            print('Downsampling array {}-->{}'.format(k_src, k_dst))
            a_src = getattr(self, k_src)[:, :]
            if self._transpose:
                # do copy b/c downsample would need c-contiguous
                a_src = a_src.T.copy()
            a_dst = getattr(new_source, k_dst)
            if antialias_aligned:
                a_dst[:, :] = downsample(a_src, float(new_rate_ratio), r=new_rate_ratio,
                                         filter_inplace=filter_inplace)[0]
            elif aggregate_aligned:
                T = new_rate_ratio * a_dst.shape[1]
                if T > a_src.shape[1]:
                    T -= new_rate_ratio
                a_src = a_src[:, :T].reshape(a_dst.shape[0], T // new_rate_ratio, -1).sum(axis=2)
                if a_src.shape[1] < a_dst.shape[1]:
                    a_src = np.c_[a_src, np.zeros(len(a_src))]
                    assert a_src.shape == a_dst.shape
                a_dst[:, :] = a_src
            else:
                # make some redundant slicing in case this is a mapped array:
                # strided reads from HDF5 are horribly slow!
                a_dst[:, :] = a_src[:, ::new_rate_ratio]

    # Implement a few common ndarray methods (with easy logic!)...
    # sum, mean, std, and var can be processed in stereotyped logic
    def _setup_reduce_method(self, axis, dtype, out):
        if axis is not None:
            while axis < 0:
                axis += self.ndim
        if axis in (None, 1):
            # reduction is either total or over time
            itr = self.iter_blocks()
        else:
            # reduction is over channels
            itr = self.iter_channels()
        if out is None:
            if dtype is None:
                # If self.data_buffer.dtype is an integer type, then summation happens with long or ulong.
                # But mean (and var and std) happens with float64
                if self.data_buffer.dtype in np.sctypes['int']:
                    dtype = np.dtype('l')
                elif self.data_buffer.dtype in np.sctypes['uint']:
                    dtype = np.dtype('L')
                else:
                    dtype = self.data_buffer.dtype
            if axis is None:
                outsize = ()
            else:
                outsize = (self.shape[self.ndim - 1 - axis],)
            out = np.zeros(outsize, dtype=dtype)
        else:
            # If out is given, then that governs the dtype
            dtype = out.dtype
            if out.ndim == self.ndim:
                # the intermediate reduction ops will actually eat a dimension,
                # but out will be reshaped later (presuming keepdims is True)
                out = out.squeeze()
        return itr, dtype, out

    def _end_reduce_method(self, out, axis, keepdims):
        # Replace any dimensions that got eaten if keepdims is True
        # Note that bool(np._NoValue) is True!! so have to evaluate T/F long-hand
        if keepdims is True:
            if axis is None:
                dims = (1, 1)
            else:
                dims = [len(out)]
                dims.insert(axis, 1)
            out = out.reshape(*dims)
        return out


    def min(self, axis=None, out=None, keepdims=False, **kwargs):
        if len(kwargs):
            warn('These arguments are not supported: {}'.format([k for k in kwargs]), UserWarning)
        itr, dtype, out = self._setup_reduce_method(axis, self.dtype, out)
        first_min = True
        for block in itr:
            b_min = block.min(axis=axis)
            if first_min:
                out.fill(np.iinfo(out.dtype).max)
                first_min = False
            out = np.minimum(b_min, out)
        return self._end_reduce_method(out, axis, keepdims)


    def max(self, axis=None, out=None, keepdims=False, **kwargs):
        if len(kwargs):
            warn('These arguments are not supported: {}'.format([k for k in kwargs]), UserWarning)
        itr, dtype, out = self._setup_reduce_method(axis, self.dtype, out)
        first_max = True
        for block in itr:
            b_max = block.max(axis=axis)
            if first_max:
                out.fill(np.iinfo(out.dtype).min)
                first_max = False
            out = np.maximum(b_max, out)
        return self._end_reduce_method(out, axis, keepdims)


    def sum(self, axis=None, dtype=None, out=None, keepdims=np._NoValue, **kwargs):
        if len(kwargs):
            warn('These arguments are not supported: {}'.format([k for k in kwargs]), UserWarning)
        itr, dtype, out = self._setup_reduce_method(axis, dtype, out)
        for block in itr:
            out += block.sum(axis=axis).astype(dtype, copy=False)
        return self._end_reduce_method(out, axis, keepdims)

    def mean(self, axis=None, dtype=None, out=None, keepdims=np._NoValue):
        if dtype is None and self.data_buffer.dtype in np.sctypes['int'] + np.sctypes['uint']:
            dtype = np.dtype('d')
        sigma = self.sum(axis=axis, dtype=dtype, out=out, keepdims=keepdims)
        if axis is None:
            n = self.shape[0] * self.shape[1]
        else:
            n = self.shape[axis]
        sigma /= n
        return sigma

    def var(self, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
        if dtype is None and self.data_buffer.dtype in np.sctypes['int'] + np.sctypes['uint']:
            dtype = np.dtype('d')
        itr, dtype, out = self._setup_reduce_method(axis, dtype, out)
        for block in itr:
            # TODO: just adding squares without mean subtraction??
            block **= 2
            out += block.sum(axis=axis).astype(dtype, copy=False)
        if axis is None:
            n = self.shape[0] * self.shape[1] - ddof
        else:
            n = self.shape[axis] - ddof
        out /= n
        return self._end_reduce_method(out, axis, keepdims)

    def std(self, axis=None, dtype=None, out=None, ddof=0, keepdims=np._NoValue):
        v = self.var(axis=axis, dtype=dtype, out=out, ddof=ddof, keepdims=keepdims)
        np.sqrt(v, v)
        return v

    def filter_array(self, **kwargs):
        # Needs overload
        pass

    def notch_filter(self, **kwargs):
        # Needs overload
        pass

    def join(self, other_source: 'ElectrodeDataSource') -> 'ElectrodeDataSource':
        """Return a new data source joining this source with another."""
        # Needs overload
        pass


class PlainArraySource(ElectrodeDataSource):

    """
    Will include in-memory data arrays from a raw data source. I.e. primary file(s) have been loaded and scaled and
    possibly filtered in arrays described here.

    """

    def __init__(self, data_matrix, shared_mem=False, **aligned_arrays):
        """

        Parameters
        ----------
        data_matrix: ndarray
            Channel x Time matrix, presumably floating point
        shared_mem: bool
            Is the data buffer in shared memory? Affects set_channel_mask
        aligned_arrays: dict
            Any other datasets that should be aligned with the electrode signal array. These arrays
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source.
        """

        self.data_buffer = data_matrix
        self.dtype = data_matrix.dtype
        for name in aligned_arrays:
            setattr(self, name, aligned_arrays[name])
        self.aligned_arrays = tuple(aligned_arrays.keys())
        self._shm = shared_mem

    def join(self, other_source: 'PlainArraySource') -> 'PlainArraySource':
        if not isinstance(other_source, PlainArraySource):
            raise ValueError('Cannot append source type {} to this PlainArraySource'.format(type(other_source)))
        if set(self.aligned_arrays) != set(other_source.aligned_arrays):
            raise ValueError('Mismatch in aligned arrays')
        # TODO: deal with shared memory
        new_array = np.concatenate([self.data_buffer, other_source.data_buffer], axis=1)
        new_aligned = dict()
        for k in self.aligned_arrays:
            new_aligned[k] = np.concatenate([getattr(self, k), getattr(other_source, k)], axis=1)
        return PlainArraySource(new_array, **new_aligned)


    def set_channel_mask(self, channel_mask):
        """Apply the binary channel mask to the current data matrix"""
        if self._shm:
            data_buffer = shared_ndarray((channel_mask.sum(), self.shape[1]))
            data_buffer[:] = self.data_buffer[channel_mask]
            self.data_buffer = data_buffer
        else:
            self.data_buffer = self.data_buffer[channel_mask]

    def to_map(self):
        """Creates a temp HDF5 file and returns a MappedSource for this datasource."""
        from .memmap import TempFilePool, MappedSource
        with TempFilePool(mode='ab') as tf:
            filename = str(tf)
        fp_precision = load_params().floating_point.lower()
        typecode = 'f' if fp_precision == 'single' else 'd'
        hdf = h5py.File(filename, 'w', libver='latest')
        tf.register_to_close(hdf)
        hdf.create_dataset('data', data=self.data_buffer.astype(typecode), chunks=True)
        for name in self.aligned_arrays:
            hdf.create_dataset(name, data=getattr(self, name).astype(typecode), chunks=True)
        return MappedSource.from_hdf_sources(hdf, 'data', aligned_arrays=self.aligned_arrays)

    def __getitem__(self, slicer):
        return self.data_buffer[slicer]

    def __setitem__(self, slicer, data):
        self.data_buffer[slicer] = data

    def filter_array(self, **kwargs):
        """
        Apply the `ecogdata.filt.time.proc.filter_array` method to this data matrix and return a new
        PlainArraySource. If `inplace=True` then the returned source will be this data source

        Parameters
        ----------
        kwargs: dict
            Options for `filter_array`

        Returns
        -------
        data: PlainArraySource

        """

        inplace = kwargs.get('inplace', True)
        f_arr = filter_array(self.data_buffer, **kwargs)
        if inplace:
            return self
        aligned_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        return PlainArraySource(f_arr, **aligned_arrays)

    def notch_filter(self, *args, **kwargs):
        """
        Apply the `ecogdata.filt.time.proc.notch_all` method to this data matrix and return a new PlainArraySource.
        If `inplace=True` then the returned source will be this data source

        Parameters
        ----------
        kwargs: dict
            Options for `notch_all`

        Returns
        -------
        data: PlainArraySource

        """

        inplace = kwargs.get('inplace', True)
        f_arr = notch_all(self.data_buffer, *args, **kwargs)
        if inplace:
            return self
        aligned_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        return PlainArraySource(f_arr, **aligned_arrays)

    # Simplify these (don't use iteration)
    def sum(self, **kwargs):
        return self.data_buffer.sum(**kwargs)

    def mean(self, **kwargs):
        return self.data_buffer.mean(**kwargs)

    def var(self, **kwargs):
        return self.data_buffer.var(**kwargs)

    # Simplify these (don't use iteration)
    def sum(self, **kwargs):
        return self.data_buffer.sum(**kwargs)

    def mean(self, **kwargs):
        return self.data_buffer.mean(**kwargs)

    def var(self, **kwargs):
        return self.data_buffer.var(**kwargs)

    def std(self, **kwargs):
        return self.data_buffer.std(**kwargs)

