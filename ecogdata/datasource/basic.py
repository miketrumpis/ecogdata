import numpy as np
from tqdm import tqdm

from ecogdata.expconfig import load_params
from ecogdata.parallel.array_split import shared_copy
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
                 axis=1, return_slice=False, reverse=False):
        if block_length is None:
            L = datasource._auto_block_length
        else:
            L = block_length
        super(DataSourceBlockIter, self).__init__(datasource, L, overlap=overlap, axis=axis,
                                                  start_offset=start_offset, partial_block=True,
                                                  reverse=reverse)
        self.datasource = datasource
        self._count = 0
        self.return_slice = return_slice
        self._itr = range(0, self._n_block)[::-1] if reverse else range(0, self._n_block)

    def __len__(self):
        return len(self._itr)

    def __iter__(self):
        self._count = 0
        return self

    def block(self, b):
        if self._itr.start <= b < self._itr.stop:
            sl = self._make_slice(b)
            return self.datasource[sl]
        else:
            raise ValueError('Slice {} out of bounds'.format(b))

    def _make_slice(self, i):
        start = i * (self.L - self._overlap) + self.start_offset
        # this really should not happen
        # if start < 0 or start >= self.T:
        #     raise StopIteration
        end = min(self.T + self.start_offset, start + self.L)
        if self._reverse:
            sl = (slice(None), slice(end - 1, start - 1, -1))
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
            self.datasource.cache_slice(sl)
        self._count += 1
        # Wait on previous cache
        output = self.datasource.get_cached_slice()
        if self._count < len(self):
            # Start caching the next load
            i_next = self._itr[self._count]
            self.datasource.cache_slice(self._make_slice(i_next))
        if self.return_slice:
            return output, sl
        else:
            return output


class ElectrodeDataSource(object):
    """
    a parent class for all types
    """

    shape = ()
    dtype = None
    _auto_block_length = 20000
    writeable = True
    _transpose = False

    def cache_slice(self, slicer):
        self._cache_output = self[slicer]

    def get_cache_slice(self):
        return self._cache_output

    def iter_blocks(self, block_length=None, overlap=0, start_offset=0, return_slice=False, reverse=False):
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
            If True return the ndarray block followed by the memmap array slice to yield this block. Helpful for
            pairing the yielded blocks with the same position in a follower array, or writing back transformed data
            to this memmap (if writeable).
        reverse: bool
            If True, yield the blocks in reverse sequence.

        """

        return DataSourceBlockIter(self, axis=1, block_length=block_length, overlap=overlap, start_offset=start_offset,
                                   return_slice=return_slice, reverse=reverse)

    def iter_channels(self, chans_per_block=None, use_max_memory=False, return_slice=False):
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
        return DataSourceBlockIter(self, axis=0, block_length=chans_per_block, return_slice=return_slice)

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
        verbose = False
        if verbose:
            chan_itr = tqdm(chan_itr, desc='Downsampling channels', leave=True, total=len(chan_itr))

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

    def filter_array(self, **kwargs):
        # Needs overload
        pass

    def notch_filter(self, **kwargs):
        # Needs overload
        pass


class PlainArraySource(ElectrodeDataSource):

    """
    Will include in-memory data arrays from a raw data source. I.e. primary file(s) have been loaded and scaled and
    possibly filtered in arrays described here.

    """

    def __init__(self, data_matrix, use_shared_mem=False, **aligned_arrays):
        """

        Parameters
        ----------
        data_matrix: ndarray
            Channel x Time matrix, presumably floating point
        samp_rate: float
            Sampling rate S/s
        use_shared_mem: bool
            If True, copy arrays to shared memory. Wasteful if arrays are already shared.
        aligned_arrays: dict
            Any other datasets that should be aligned with the electrode signal array. These arrays
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source.
        """

        self._data_matrix = shared_copy(data_matrix) if use_shared_mem else data_matrix
        self.shape = data_matrix.shape
        self.dtype = data_matrix.dtype
        for name in aligned_arrays:
            setattr(self, name, (shared_copy(aligned_arrays[name]) if use_shared_mem else aligned_arrays[name]))
        self.aligned_arrays = list(aligned_arrays.keys())

    def __getitem__(self, slicer):
        return self._data_matrix[slicer]

    def __setitem__(self, slicer, data):
        self._data_matrix[slicer] = data

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
        f_arr = filter_array(self._data_matrix, **kwargs)
        if inplace:
            return self
        aligned_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        # use_shared_mem = False b/c the array returned from filter_array is already shared mem.
        return PlainArraySource(f_arr, use_shared_mem=False, **aligned_arrays)

    def notch_filter(self, **kwargs):
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
        f_arr = notch_all(self._data_matrix, **kwargs)
        if inplace:
            return self
        aligned_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        # use_shared_mem = False b/c the array returned from filter_array is already shared mem.
        return PlainArraySource(f_arr, use_shared_mem=False, **aligned_arrays)


