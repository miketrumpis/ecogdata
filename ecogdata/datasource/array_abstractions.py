import numpy as np


def slice_to_range(slice, r_max):
    """Convert a slice object to the corresponding index list"""
    start = 0 if slice.start is None else slice.start
    stop = r_max if slice.stop is None else slice.stop
    step = 1 if slice.step is None else slice.step
    return list(range(start, stop, step))


def range_to_slice(range):
    """Convert a range (or list) to a slice. Range must be periodic!!"""
    range = list(range)
    step = range[1] - range[0]
    start = range[0]
    stop = range[-1] + 1
    return slice(start, stop, step)


class ReadCache(object):
    # TODO -- enable catch for full slicing or fancy (list) slicing
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
        self.dtype = array.dtype
        self.shape = array.shape

    def __len__(self):
        return len(self._array)

    @property
    def file_array(self):
        return self._array

    def _scale_segment(self, x):
        if self._units_scale is None:
            return x
        # Note -- this type conversion is about as time consuming as reading out the disk
        x = x.astype('d')
        if self._raw_offset is not None:
            x += self._raw_offset
        x *= self._units_scale
        return x

    def __getitem__(self, sl):
        if self._units_scale is None:
            return self._array[sl]
        else:
            return self._scale_segment(self._array[sl])
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
