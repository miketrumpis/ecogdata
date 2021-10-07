import numpy as np
from numpy.lib.stride_tricks import as_strided


__all__ = ['BlockedSignal', 'BlockSignalBase', 'block_reduce', 'block_apply']


class BlockSignalBase:

    def __init__(self, array, block_length, overlap=0, axis=-1, start_offset=0, partial_block=True, reverse=False):
        block_length = int(block_length)
        shape = array.shape
        while axis < 0:
            axis += len(shape)
        T = shape[axis] - start_offset
        if isinstance(overlap, int) and overlap > 0:
            step = block_length - overlap
            self._overlap = overlap
        else:
            step = int(round((1 - overlap) * block_length))
            self._overlap = block_length - step
        # The last full block is (T - block size) // step length
        # So 0, ..., last_block is last_block + 1
        n_block = (T - block_length) // step + 1
        # if the next step starts before T and ends after T then a partial block is possible
        if partial_block and n_block * step < T < n_block * step + block_length:
            self._last_block_sz = T - step * n_block
            n_block += 1
        else:
            self._last_block_sz = block_length
        self._n_block = n_block
        self.axis = axis
        self.array_shape = shape
        self.L = block_length
        self.T = T
        self.start_offset = start_offset
        self._reverse = reverse

    def __len__(self):
        return self._n_block


class BlockedSignal(BlockSignalBase):
    """A class that transforms an N-dimension signal into multiple
    blocks along a given _axis. The resulting object can yield blocks
    in forward or _reverse sequence.
    """

    def __init__(self, x, bsize, overlap=0, axis=-1, partial_block=True, reverse=False):
        """
        Split a (possibly quite large) array into blocks along one _axis.

        Parameters
        ----------

        x : ndarray
          The signal to blockify.
        bsize : int
          The maximum blocksize for the given _axis.
        overlap : float 0 <= overlap <= 1 or int 0 < bsize
          The proportion of overlap between adjacent blocks. The (k+1)th
          block will begin at an offset of (1-overlap)*bsize points into
          the kth block. If overlap is an integer, it will be used literally.
        axis : int (optional)
          The _axis to split into blocks
        partial_block : bool
          If blocks don't divide the _axis length exactly, allow a partial
          block at the end (default True).

        """
        # if x is not contiguous then I think we're out of luck
        if not x.flags.c_contiguous:
            raise RuntimeError('The data to be blocked must be C-contiguous')
        # This object does not support start offset
        super(BlockedSignal, self).__init__(x, bsize, overlap=overlap, axis=axis, partial_block=partial_block,
                                            reverse=reverse, start_offset=0)
        shape = x.shape
        strides = x.strides
        bitdepth = x.dtype.itemsize
        # first reshape x to have shape (..., _n_block, bsize, ...),
        # where the (_n_block, bsize) pair replaces the _axis in kwargs
        nshape = shape[:self.axis] + (self._n_block, bsize) + shape[self.axis + 1:]
        # Assuming C-contiguous, strides were previously
        # (..., nx*ny, nx, 1) * bitdepth
        # Change the strides at _axis to reflect new shape
        b_offset = int(np.prod(shape[self.axis + 1:]) * bitdepth)
        lap = self.L - self._overlap
        nstrides = strides[:self.axis] + (lap * b_offset, b_offset) + strides[self.axis + 1:]
        self._x_blk = as_strided(x, shape=nshape, strides=nstrides)

    def __iter__(self):
        self._gen = self.bwd() if self._reverse else self.fwd()
        return self

    def __next__(self):
        return next(self._gen)

    def fwd(self):
        "Yield the blocks one at a time in forward sequence"
        # this object will be repeatedly modified in the following loop(s)
        blk_slice = [slice(None)] * self._x_blk.ndim
        for blk in range(self._n_block):
            blk_slice[self.axis] = blk
            if blk == self._n_block - 1:
                # VERY important! don't go out of bounds in memory!
                blk_slice[self.axis + 1] = slice(0, self._last_block_sz)
            else:
                blk_slice[self.axis + 1] = slice(None)
            xc = self._x_blk[tuple(blk_slice)]
            yield xc

    def bwd(self):
        "Yield the blocks one at a time in _reverse sequence"
        # loop through in _reverse order, slicing out _reverse-time blocks
        bsize = self._x_blk.shape[self.axis + 1]
        # this object will be repeatedly modified in the following loop(s)
        blk_slice = [slice(None)] * self._x_blk.ndim
        for blk in range(self._n_block - 1, -1, -1):
            blk_slice[self.axis] = blk
            if blk == self._n_block - 1:
                # VERY important! don't go out of bounds in memory!
                # confusing syntax.. but want to count down from the *negative*
                # index of the last good point: -(bsize+1-last_block_sz)
                # down to the *negative* index of the
                # beginning of the block: -(bsize+1)
                blk_slice[self.axis+1] = slice(-(bsize + 1) + self._last_block_sz, -(bsize + 1), -1)
            else:
                blk_slice[self.axis + 1] = slice(None, None, -1)
            xc = self._x_blk[tuple(blk_slice)]
            yield xc

    def block(self, b):
        "Yield the index b block"
        blk_slice = [slice(None)] * self._x_blk.ndim
        while b < 0:
            b += self._n_block
        if b >= self._n_block:
            raise IndexError
        blk_slice[self.axis] = b
        if b == self._n_block - 1:
            blk_slice[self.axis + 1] = slice(0, self._last_block_sz)
        else:
            blk_slice[self.axis + 1] = slice(None)
        return self._x_blk[tuple(blk_slice)]


def block_reduce(rfn, array, bsize, f_axis=1, **kwargs):
    bsig = BlockedSignal(array, bsize, **kwargs)
    reduced = list()
    for blk in bsig.fwd():
        reduced.append(rfn(blk, axis=f_axis))
    return np.array(reduced)


def block_apply(fn, bsize, args, block_arg=0, b_axis=-1, **kwargs):
    """
    Performs blockwise computation of an array operator 'fn'.

    Parameters
    ----------

    fn : operator method with one-to-one array input-output
    bsize : block size to operate over
    args : method arguments sequence
    block_arg : index of the operand in 'args' sequence
    b_axis : axis of array to block (currently must be last)
    kwargs : method keyword arguments

    Returns
    -------

    arr : output array of same shape and dtype as input array

    """
    array = args[block_arg]
    if not (b_axis == -1 or b_axis == array.ndim-1):
        raise ValueError('Currently only blocking on last axis')

    def _hotswap_block(blk):
        n_arg = len(args)
        a = [args[n] for n in range(n_arg) if n != block_arg]
        a.insert(block_arg, blk)
        return a
    
    b_sig = BlockedSignal(array, bsize, partial_block=True, axis=b_axis)
    a_proc = np.empty_like(array)
    b_proc = BlockedSignal(a_proc, bsize, partial_block=True, axis=b_axis)

    for b_in, b_out in zip(b_sig, b_proc):
        a = _hotswap_block(b_in)
        b_out[:] = fn(*a, **kwargs)
    return a_proc
