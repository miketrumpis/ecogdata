from itertools import product
from contextlib import contextmanager
import numpy as np
from h5py._hl.selections import select
from ecogdata.util import ToggleState
from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.expconfig import load_params


__all__ = ['slice_to_range', 'range_to_slice', 'unpack_ellipsis', 'tile_slices', 'MappedBuffer', 'HDF5Buffer']


def slice_to_range(slicer, r_max):
    """Convert a slice object to the corresponding index list"""
    step = 1 if slicer.step is None else slicer.step
    if slicer.start is None:
        start = r_max - 1 if step < 0 else 0
    else:
        start = slicer.start
    if slicer.stop is None:
        stop = -1 if step < 0 else r_max
    else:
        stop = slicer.stop
    # start = 0 if slicer.start is None else slicer.start
    # stop = r_max if slicer.stop is None else slicer.stop
    return range(start, stop, step)


def range_to_slice(range, offset=None):
    """Convert a range (or list) to a slice. Range must be periodic!!"""
    if not isinstance(range, (list, tuple)):
        range = list(range)
    if offset is None:
        offset = 0
    step = range[1] - range[0]
    start = range[0] - offset
    stop = range[-1] + 1 - offset
    return slice(start, stop, step)


def _abs_slicer(slicer, shape):
    """Convert a bunch of slices to be positive-step only in order to work with the "select" merthod."""
    if not np.iterable(slicer):
        slicer = (slicer,)
    fwd_slices = []
    for s, dim in zip(slicer, shape):
        if isinstance(s, slice) and s.step is not None and s.step < 0:
            s = range_to_slice(slice_to_range(s, dim)[::-1])
        fwd_slices.append(s)
    return tuple(fwd_slices)


def unpack_ellipsis(slicer, dims):
    """Transform e.g. (..., slice(0, 10)) --> (slice(None), slice(None), slice(0, 10)) for three dimensions"""
    if Ellipsis not in slicer:
        return slicer
    index = slicer.index(Ellipsis)
    remainder_slices = slicer[index + 1:]
    # if it's the last slicing object, then simply drop it
    if not len(remainder_slices):
        return slicer[:index]
    new_slices = slicer[:index] + (slice(None),) * (dims - index - len(remainder_slices)) + remainder_slices
    return new_slices


def tile_slices(slicers, shape, chunks):
    """
    Tile possibly complex array slicing if there are step sizes that exceed the data chunking sizes. Also correct
    array slices with negative steps to slice with positive step and write out with negative step. This makes all
    slicing permissible for h5py.DataSet objects.

    Parameters
    ----------
    slicers: tuple
        Vaild tuple (may include Ellipsis)
    shape: tuple
        Array dimensions
    chunks: tuple
        Array chunk size

    Returns
    -------
    array_slices: list
        Tiled array slices for the sliced array
    out_slices: list
        Tiled slicing for the output array

    Examples
    --------
    Slicing a hypothetical (20, 20)-shape array with (5, 5) chunks using a combination of big strides and
    discontiguous ranges:
    >>> i, o = aa.tile_slices(np.s_[::10, [0, 1, 10, 11]], (20, 20), (5, 5))
    >>> out_arr = np.zeros((2, 4))
    >>> for n, sl in enumerate(o):
    ...     out_arr[sl] = n
    ...
    >>> out_arr
    array([[0., 0., 1., 1.],
           [2., 2., 3., 3.]])
    >>> i
    [(0, [0, 1]), (0, [10, 11]), (10, [0, 1]), (10, [10, 11])]

    """
    if not isinstance(slicers, tuple):
        slicers = (slicers,)
    slicers = unpack_ellipsis(slicers, len(shape))
    dim = 0
    new_slicers = []
    out_slicers = []
    for slicer in slicers:
        if isinstance(slicer, int):
            new_slicers.append([slicer])
            # this is a dimension-reducing slice, so do not put an output slice??
            # out_slicers.append([slicer])
        elif isinstance(slicer, slice):
            # actually going one by one is faster than staggered slices (sometimes it is about the same)
            if slicer.step and abs(slicer.step) > chunks[dim]:
                if slicer.step < 0:
                    # read-out in permitted forward sequence and write in reverse
                    rng = slice_to_range(slicer, shape[dim])[::-1]
                    new_slicers.append(rng)
                    out_slicers.append(range(len(rng))[::-1])
                else:
                    rng = slice_to_range(slicer, shape[dim])
                    new_slicers.append(rng)
                    out_slicers.append(range(len(rng)))
            elif slicer.step and slicer.step < 0:
                # in this case, read-out the slice in permitted forward sequence, but load to memory in reverse
                # sequence
                fwd_slice = range_to_slice(slice_to_range(slicer, shape[dim])[::-1])
                new_slicers.append([fwd_slice])
                out_slicers.append([slice(None, None, -1)])
            else:
                new_slicers.append([slicer])
                # this is a full slice
                out_slicers.append([slice(None)])
        elif np.iterable(slicer):
            slicer = list(slicer)
            if (np.diff(slicer) < 0).any():
                raise ValueError("Can't process decreasing or permuted list-indexing: {}".format(slicer))
            # anywhere the sequence jumps to a new spot larger than chunk-size away, split it up
            jumps = np.where(np.diff(slicer) > chunks[dim])[0] + 1
            idx = 0
            slice_segs = []
            out_segs = []
            for j in np.r_[jumps, len(slicer)]:
                slice_segs.append(slicer[idx:j])
                out_segs.append(np.s_[idx:j])
                idx = j
            new_slicers.append(slice_segs)
            out_slicers.append(out_segs)
        dim += 1
    return list(product(*new_slicers)), list(product(*out_slicers))


class MappedBuffer(object):
    """
    Abstracts indexing from memmap file, with reads being converted to shared memory and possibly transformed to
    unit-ful values.
    """

    def __init__(self, array, units_scale=None, raise_bad_write=False):
        """

        Parameters
        ----------
        array: numpy.memmap
            A memory-mapped array. If using HDF5, take advantage of `HDF5Buffer`.
        units_scale: float or 2-tuple
            Either the scaling value or (offset, scaling) values such that signal = (memmap + offset) * scaling
        """
        self._array = array
        # Anything other than 'r' indicates some kind of write access
        if hasattr(array, 'mode'):
            mode = array.mode
            self.__writeable = mode != 'r'
        elif hasattr(array, 'file'):
            mode = array.file.mode
            self.__writeable = mode != 'r'
        else:
            print('Unknown array type -- assuming not writeable')
            self.__writeable = False
        self._current_slice = None
        self._current_seg = ()
        self._raw_offset = None
        self._units_scale = None
        if units_scale is not None:
            # Ignore the (0, 1) scaling to save cycles
            if np.iterable(units_scale):
                self._raw_offset = units_scale[0] if units_scale[0] != 0 else None
                self._units_scale = units_scale[1]
            else:
                self._units_scale = units_scale
            if self._units_scale == 1:
                self._units_scale = None
        if self._units_scale is None:
            self.dtype = array.dtype
        else:
            fp_precision = load_params().floating_point.lower()
            typecode = 'f' if fp_precision == 'single' else 'd'
            self.dtype = np.dtype(typecode)
        self.shape = array.shape
        self._raise_bad_write = raise_bad_write
        # This is a callable -- when called, a content manager is created and the state is toggled in-context
        self._transpose_state = ToggleState(init_state=False)
        # A private output array that can be set using a context manager, allowing array reads in that context to be
        # placed in this array
        self._read_output = None

    def __len__(self):
        return len(self._array)

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def transpose_reads(self):
        """
        This is a callable ToggleState object that can create a context in which this buffer will return __getitem__
        slices in transpose.
        """
        return self._transpose_state

    @property
    def file_array(self):
        return self._array

    @property
    def writeable(self):
        return self.__writeable

    def _scale_segment(self, x):
        if self._units_scale is None:
            return x
        if self._raw_offset is not None:
            x += self._raw_offset
        x *= self._units_scale
        return x

    @contextmanager
    def direct_read(self, output_array):
        self._read_output = output_array
        try:
            yield
        finally:
            self._read_output = None

    def get_output_array(self, slicer, only_shape=False):
        slicer = _abs_slicer(slicer, self.shape)
        out_shape = select(self.shape, slicer, 0).mshape
        # if the output will be transposed, then pre-create the array in the right order
        if self._transpose_state.state:
            out_shape = out_shape[::-1]
        if only_shape:
            return out_shape
        typecode = self.dtype.char
        if self._read_output is None:
            return shared_ndarray(out_shape, typecode)
        else:
            if self._read_output.shape != out_shape:
                raise ValueError('Direct read context was given an array with the wrong shape')
            return self._read_output

    def __getitem__(self, sl):
        out_arr = self.get_output_array(sl)
        # not sure what the most efficient way to slice is?
        if self._transpose_state.state:
            out_arr[:] = self._array[sl].transpose()
        else:
            out_arr[:] = self._array[sl]
        return self._scale_segment(out_arr)

    def __setitem__(self, sl, data):
        if self.writeable:
            # if the datatypes are a mismatch then don't try to fix it here
            self._array[sl] = data
        elif self._raise_bad_write:
            raise RuntimeError('Tried to write to a not-writeable MappedBuffer')


class HDF5Buffer(MappedBuffer):
    """Optimized mapped file reading for HDF5 files (h5py.File)"""

    def __getitem__(self, sl):
        # this function computes the output shape -- use ID=0 to explicitly *not* work for funky RegionReference slicing

        out_arr = self.get_output_array(sl)
        i_slices, o_slices = tile_slices(sl, self.shape, self._array.chunks)
        # print('Slicing buffer as {}'.format(i_slices))
        for isl, osl in zip(i_slices, o_slices):
            if self._transpose_state.state:
                # generally need to reverse the slice order
                if isinstance(osl, int):
                    osl = (osl,)
                osl = osl[::-1]
                if len(osl) < out_arr.ndim:
                    osl = (Ellipsis,) + osl
                # can't avoid making a copy here?
                out_arr[osl] = self._array[isl].T
            else:
                # try a direct read, but this will fail if there are negative steps going on
                try:
                    self._array.read_direct(out_arr, source_sel=isl, dest_sel=osl)
                except ValueError:
                    out_arr[osl] = self._array[isl]
        return self._scale_segment(out_arr)

    def __setitem__(self, sl, data):
        if not self.writeable:
            super(HDF5Buffer, self).__setitem__(sl, data)
        # this should work for writing too?
        i_slices, o_slices = tile_slices(sl, self.shape, self._array.chunks)
        # if broadcasting, then pull only the first data_dim slices from the output slicer?
        if isinstance(data, np.ndarray):
            data_dim = data.ndim
        else:
            data_dim = 0
        # print('Write slices', i_slices)
        # print('Data slices', [osl[(len(osl) - data_dim):] for osl in o_slices])
        for isl, osl in zip(i_slices, o_slices):
            osl = osl[(len(osl) - data_dim):]
            self._array[isl] = (data[osl] if len(osl) else data)


class ReadCache(object):
    # TODO -- re-enable read-caching (or change name/nature of the object)
    # TODO -- make mapped access compatible with numpy "memmap" arrays
    """
    Buffers row indexes from memmap or hdf5 file.

    --> For now just pass through slicing without cache. Perform scaling and return.

    Ignore this for now
    ---vvvvvvvvvvvvv---

    For cases where array[0, m:n], array[1, m:n], array[2, m:n] are
    accessed sequentially, this object buffers the C x (n-m)
    submatrix before yielding individual rows.

    Access such as array[p:q, m:n] is handled by the underlying
    array's __getitem__ method.

    http://docs.h5py.org/en/stable/high/dataset.html#reading-writing-data
    """

    def __init__(self, array, units_scale=None):
        self._array = array
        self._current_slice = None
        self._current_seg = ()
        self._raw_offset = None
        self._units_scale = None
        if units_scale is not None:
            if np.iterable(units_scale):
                self._raw_offset = units_scale[0]
                self._units_scale = units_scale[1]
            else:
                self._units_scale = units_scale
        if self._units_scale is None:
            self.dtype = array.dtype
        else:
            fp_precision = load_params().floating_point.lower()
            typecode = 'f' if fp_precision == 'single' else 'd'
            self.dtype = np.dtype(typecode)
        self.shape = array.shape

    def __len__(self):
        return len(self._array)

    @property
    def file_array(self):
        return self._array

    def _scale_segment(self, x):
        if self._units_scale is None:
            return x
        if self._raw_offset is not None:
            x += self._raw_offset
        x *= self._units_scale
        return x

    def __getitem__(self, sl):
        # this function computes the output shape -- use ID=0 to explicitly *not* work for funky RegionReference slicing
        out_shape = select(self.shape, sl, 0).mshape
        if self._units_scale is None:
            out_arr = shared_ndarray(out_shape, self._array.dtype.char)
            self._array.read_direct(out_arr, source_sel=sl)
            return out_arr
        else:
            # with self._array.astype('d'):
            #     out = self._array[sl]
            out_arr = shared_ndarray(out_shape, self.dtype.char)
            self._array.read_direct(out_arr, source_sel=sl)
            return self._scale_segment(out_arr)
        # indx, srange = sl
        # if not isinstance(indx, (np.integer, int)):
        #     # make sure to release a copy if no scaling happens
        #     if self._units_scale is None:
        #         # read creates copy
        #         return self._array[sl]
        #     return self._scale_segment(self._array[sl])
        # if self._current_slice != srange:
        #     all_sl = (slice(None), srange)
        #     self._current_seg = self._scale_segment(self._array[all_sl])
        #     self._current_slice = srange
        # # always return the full range after slicing with possibly
        # # complex original range
        # new_srange = slice(None)
        # new_sl = (indx, new_srange)
        # return self._current_seg[new_sl].copy()


# class CommonReferenceReadCache(ReadCache):
#     """Returns common-average re-referenced blocks"""
#
#     def __getitem__(self, sl):
#         indx, srange = sl
#         if not isinstance(indx, (np.integer, int)):
#             return self._array[sl].copy()
#         if self._current_slice != srange:
#             all_sl = (slice(None), srange)
#             if self.dtype in np.sctypes['int']:
#                 self._current_seg = self._array[all_sl].astype('d')
#             else:
#                 self._current_seg = self._array[all_sl].copy()
#             self._current_seg -= self._current_seg.mean(0)
#             self._current_slice = srange
#         # always return the full range after slicing with possibly
#         # complex original range
#         new_srange = slice(None)
#         new_sl = (indx, new_srange)
#         return self._current_seg[new_sl].copy()
#
#
# class FilteredReadCache(ReadCache):
#     """
#     Apply row-by-row filters to a ReadCache
#     """
#
#     def __init__(self, array, filters):
#         if not isinstance(filters, (tuple, list)):
#             f = filters
#             filters = [f] * len(array)
#         self.filters = filters
#         super(FilteredReadCache, self).__init__(array)
#
#     def __getitem__(self, sl):
#         idx = sl[0]
#         x = super(FilteredReadCache, self).__getitem__(sl)
#         if isinstance(idx, int):
#             return self.filters[idx](x)
#         y = np.empty_like(x)
#         for x_, y_, f in zip(x[idx], y[idx], self.filters[idx]):
#             y_[:] = f(x_)
#         return y
#
#
# def _make_subtract(z):
#     def _f(x):
#         return x - z
#
#     return _f
#
#
# class DCOffsetReadCache(FilteredReadCache):
#     """
#     A filtered read cache with a simple offset subtraction.
#     """
#
#     def __init__(self, array, offsets):
#         # filters = [lambda x: x - off for off in offsets]
#         filters = [_make_subtract(off) for off in offsets]
#         super(DCOffsetReadCache, self).__init__(array, filters)
#         self.offsets = offsets
