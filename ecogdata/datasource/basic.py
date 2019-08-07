from tqdm import tqdm

from ecogdata.expconfig import load_params
from ecogdata.parallel.array_split import shared_copy
from ecogdata.filt.time import downsample, filter_array, notch_all


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


# TODO: make a process-based dual buffer that continues to read-out data (e.g. network or local disk reads) during
#  the time that one data block is yielded and processed in "iter_blocks".
# class _DualStreamBuffer(object):
#
#     def __init__(self, datasource, buffer_a, buffer_b):
#         pass


class DataSourceBlockIter(object):

    def __init__(self, datasource, block_length=None, overlap=0, start_offset=0,
                 axis=1, return_slice=False, reverse=False):
        if block_length is None:
            L = datasource._auto_block_length
        else:
            L = block_length
        T = datasource.shape[axis] - start_offset
        N = T // (L - overlap)
        # add in another block to trigger stop-iteration in forward mode
        if not reverse and (L - overlap) * N <= T:
            N += 1
        # if the advance size exactly divides T then adjust to start at N - 1 for reverse mode
        if reverse and (L - overlap) * N == T:
            N -= 1
        self.L = L
        self.T = T
        self.overlap = overlap
        self.datasource = datasource
        self.count = 0
        self.return_slice = return_slice
        self.reverse = reverse
        self.start_offset = start_offset
        self.axis = axis
        self.itr = range(N, -2, -1) if reverse else range(0, N)

    def __len__(self):
        return len(self.itr)

    def __iter__(self):
        self._count = 0
        return self

    def __next__(self):
        try:
            i = self.itr[self._count]
        except IndexError:
            raise StopIteration
        self._count += 1
        start = i * (self.L - self.overlap) + self.start_offset
        if start < 0 or start >= self.T:
            raise StopIteration
        end = min(self.T, start + self.L)
        if self.reverse:
            sl = (slice(None), slice(end - 1, start - 1, -1))
        else:
            sl = (slice(None), slice(start, end))
        if self.axis == 0:
            sl = sl[::-1]
        # print('Slicing datasource as {}'.format(sl))
        if self.return_slice:
            return self.datasource[sl], sl
        else:
            return self.datasource[sl]


class ElectrodeDataSource(object):
    """
    a parent class for all types
    """

    shape = ()
    dtype = None
    _auto_block_length = 20000
    writeable = True
    _transpose = False

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

    def batch_change_rate(self, new_rate_ratio, new_source, antialias_aux=False, verbose=False, filter_inplace=False):
        new_rate_ratio = int(new_rate_ratio)
        if new_source.shape[0] != self.shape[0]:
            raise ValueError('Output source has the wrong number of channels: {}'.format(new_source.shape[0]))
        if new_source.shape[1] != calc_new_samples(self.shape[1], new_rate_ratio):
            raise ValueError('Output source has the wrong series length: {}'.format(new_source.shape[1]))
        chan_itr = self.iter_channels(return_slice=True)
        if verbose:
            chan_itr = tqdm(chan_itr, desc='Downsampling channels', leave=True, total=len(chan_itr))
        # for raw_channels, sl in self.iter_channels(return_slice=True):
        for raw_channels, sl in chan_itr:
            # kind of fake the sampling rate
            r = downsample(raw_channels, float(new_rate_ratio), r=new_rate_ratio, filter_inplace=filter_inplace)[0]
            new_source[sl] = r

        # now decimate aux_channels with or without anti-aliasing -- *assuming* that a full load won't bust RAM
        for k_src, k_dst in zip(self.aligned_arrays, new_source.aligned_arrays):
            print('Downsampling array {}-->{}'.format(k_src, k_dst))
            a_src = getattr(self, k_src)
            a_dst = getattr(new_source, k_dst)
            if antialias_aux:
                a_dst[:, :] = downsample(a_src[:, :], float(new_rate_ratio), r=new_rate_ratio,
                                         filter_inplace=filter_inplace)[0]
            else:
                # make some redundant slicing in case this is a mapped array:
                # strided reads from HDF5 are horribly slow!
                a_dst[:, :] = a_src[:, :][:, ::new_rate_ratio]

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

    def __init__(self, data_matrix, use_shared_mem=False, **aux_arrays):
        """

        Parameters
        ----------
        data_matrix: ndarray
            Channel x Time matrix, presumably floating point
        samp_rate: float
            Sampling rate S/s
        use_shared_mem: bool
            If True, copy arrays to shared memory. Wasteful if arrays are already shared.
        aux_arrays: dict
            Any other datasets that should be aligned with the electrode signal array. These arrays
            will be kept at the same sampling rate and index alignment as the electrode signal. They will also be
            preserved if this source is mirrored or copied to another source.
        """

        self._data_matrix = shared_copy(data_matrix) if use_shared_mem else data_matrix
        self.shape = data_matrix.shape
        self.dtype = data_matrix.dtype
        for name in aux_arrays:
            setattr(self, name, (shared_copy(aux_arrays[name]) if use_shared_mem else aux_arrays[name]))
        self.aligned_arrays = list(aux_arrays.keys())

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
        aux_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        # use_shared_mem = False b/c the array returned from filter_array is already shared mem.
        return PlainArraySource(f_arr, use_shared_mem=False, **aux_arrays)

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
        aux_arrays = dict([(a, getattr(self, a)) for a in self.aligned_arrays])
        # use_shared_mem = False b/c the array returned from filter_array is already shared mem.
        return PlainArraySource(f_arr, use_shared_mem=False, **aux_arrays)


