import numpy as np
from ecogdata.parallel.array_split import shared_copy
from ecogdata.filt.time import downsample


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


class ElectrodeDataSource(object):
    """
    a parent class for all types
    """

    data_shape = ()
    dtype = None
    _auto_block_length = 20000
    writeable = True

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

        if block_length is None:
            L = self._auto_block_length
        else:
            L = block_length
        T = self.data_shape[1] - start_offset
        N = T // (L - overlap)
        # add in another block to trigger stop-iteration in forward mode
        if not reverse and (L - overlap) * N <= T:
            N += 1
        # if the advance size exactly divides T then adjust to start at N - 1 for reverse mode
        if reverse and (L - overlap) * N == T:
            N -= 1
        itr = range(N, -2, -1) if reverse else range(0, N)
        for i in itr:
            start = i * (L - overlap) + start_offset
            if start < 0 or start >= T:
                raise StopIteration
            end = min(T, start + L)
            sl = (slice(None), slice(start, end))
            if return_slice:
                yield self[sl], sl
            else:
                yield self[sl]

    def iter_channels(self, chans_per_block=None, return_slice=False):
        C, T = self.data_shape
        if chans_per_block is None:
            # is there any principled default?
            chans_per_block = 16
        num_iter = C // chans_per_block
        if chans_per_block * num_iter < C:
            num_iter += 1
        for i in range(num_iter):
            start = i * chans_per_block
            stop = min(C, (i + 1) * chans_per_block)
            if return_slice:
                yield self[start:stop, :], np.s_[start:stop, :]
            else:
                yield self[start:stop, :]

    def batch_change_rate(self, new_rate_ratio, new_source):
        if new_source.data_shape[0] != self.data_shape[0]:
            raise ValueError('Output source has the wrong number of channels: {}'.format(new_source.data_shape[0]))
        if new_source.data_shape[1] != calc_new_samples(self.data_shape[1], new_rate_ratio):
            raise ValueError('Output source has the wrong series length: {}'.format(new_source.data_shape[1]))
        for raw_channels, sl in self.iter_channels(return_slice=True):
            # kind of fake the sampling rate
            r = downsample(raw_channels, float(new_rate_ratio), r=new_rate_ratio)
            new_source[sl] = r




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
        self.data_shape = data_matrix.shape
        self.dtype = data_matrix.dtype
        for name in aux_arrays:
            setattr(self, name, (shared_copy(aux_arrays[name]) if use_shared_mem else aux_arrays[name]))
        self._aligned_arrays = list(aux_arrays.keys())

    def __getitem__(self, slicer):
        return self._data_matrix[slicer]

    def __setitem__(self, slicer, data):
        self._data_matrix[slicer] = data

