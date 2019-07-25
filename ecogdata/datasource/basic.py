import numpy as np
from ecogdata.parallel.array_split import shared_copy
from ecogdata.filt.time import downsample

def calc_new_samples(N, old_rate, new_rate):
    """
    Find number of points in a downsampled N-vector given a rate conversion new_rate:old_rate

    Parameters
    ----------
    N: int
        Original vector length
    old_rate:
        Original sample rate
    new_rate:
        New sample rate (must divide old_rate!)

    Returns
    -------
    P: int
        Downsampled vector length

    """

    r = old_rate / new_rate
    if int(r) != r:
        raise ValueError('New rate {} does not divide old rate {}'.format(new_rate, old_rate))
    r = int(r)
    num_pts = N // r
    num_pts += int((N - num_pts * r) > 0)
    return num_pts


class ElectrodeDataSource(object):
    """
    a parent class for all types
    """

    data_shape = ()
    dtype = None
    Fs = None
    _auto_block_length = 20000

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
        # # Blocks need to be even length
        # if L % 2:
        #     L += 1
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
            # if the tail block is odd-length, clip off the last point
            # if (end - start) % 2:
            #     end -= 1
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


    def batch_change_rate(self, new_rate, new_source):
        if new_source.Fs != new_rate:
            raise ValueError('Output source is set to the wrong rate')

        for raw_channels, sl in self.iter_channels(return_slice=True):
            r = downsample(raw_channels, self.Fs, r=int(self.Fs / new_rate))[0]
            new_source[sl] = r




class PlainArraySource(ElectrodeDataSource):

    """
    Will include in-memory data arrays from a raw data source. I.e. primary file(s) have been loaded and scaled and
    possibly filtered in arrays described here.

    """

    def __init__(self, data_matrix, samp_rate, use_shared_mem=False, **aux_arrays):
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
        self.Fs = samp_rate
        for name in aux_arrays:
            setattr(self, name, (shared_copy(aux_arrays[name]) if use_shared_mem else aux_arrays[name]))
        self._aligned_arrays = list(aux_arrays.keys())


    def __getitem__(self, slicer):
        return self._data_matrix[slicer]


    def __setitem__(self, slicer, data):
        self._data_matrix[slicer] = data


    @property
    def writeable(self):
        return True
