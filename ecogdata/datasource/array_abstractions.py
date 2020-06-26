from itertools import product
from contextlib import contextmanager, ExitStack
from typing import Sequence
import numpy as np
import h5py
from h5py._hl.selections import select
from ecogdata.util import ToggleState
from ecogdata.parallel.sharedmem import shared_ndarray, SharedmemManager
from ecogdata.parallel.mproc import Process
from ecogdata.expconfig import load_params


__all__ = ['slice_to_range', 'range_to_slice', 'unpack_ellipsis', 'tile_slices',
           'BufferBase', 'MappedBuffer', 'HDF5Buffer', 'BufferBinder', 'slice_data_buffer',
           'BackgroundRead']


_integer_types = tuple(np.sctypes['int']) + (int,)


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
    stop = range[-1] + np.sign(step) - offset
    return slice(start, stop, step)


def _abs_slicer(slicer, shape):
    """Convert a bunch of slices to be positive-step only in order to work with the "select" method."""
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
    if not np.iterable(slicer):
        slicer = (slicer,)
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
        if isinstance(slicer, _integer_types):
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


def slice_data_buffer(buffer, slicer, transpose=False, output=None):
    """
    Slicing helper that can be run in a subprocess.
    Parameters
    ----------
    buffer: BufferType
        Read-access data buffer
    slicer:
        slice object
    transpose: bool
        read output should be transposed
    output: ndarray
        Output array

    Returns
    -------
    data_slice: ndarray
        Slice output

    """
    if output is None:
        with buffer.transpose_reads(transpose):
            # What this context should mean is that the buffer is going to get sliced in the prescribed way and then
            # the output is going to get transposed. Handling the transpose logic in the buffer avoids some
            # unnecessary array copies
            data_slice = buffer[slicer]
    else:
        # Interesting ... transpose needs to be stated BEFORE direct_reads (at least to prevent an error regarding
        # the output shapes)
        with buffer.transpose_reads(transpose), buffer.direct_read(slicer, output):
            buffer[slicer]
            data_slice = output
    return data_slice


class BufferBase:

    # A Buffer exposes these attributes a la ndarray
    dtype = None  # a dtype or typecode character
    shape = ()  # dimension sizes
    ndim = None  # number of dims (i.e. len(shape))

    # Also these attributes related to the mapped data
    map_dtype = None  # a dtype or typecode character
    units_scale = None  # a scaling value or an (offset, scale) pair
    writeable = False  # is the map read-only or read-write?
    file_array = None   # the underlying mapped array object
    filename = None  # File name of mapped data
    chunks = 20000  # Optimal "chunking" block size (useful for HDF5 maps). This default is mutable for other types

    # Do a no-argument constructor to initialize some object-specific stuff
    def __init__(self):
        # This is a callable -- when called, a content manager is created and the state is toggled in-context
        self._transpose_state = ToggleState(init_state=False)
        # A private output array that can be set using a context manager, allowing array reads in that context to be
        # placed in this array
        self._read_output = None

    @property
    def transpose_reads(self):
        """
        This is a callable ToggleState object that can create a context in which this buffer will return __getitem__
        slices in transpose.
        """
        return self._transpose_state

    @contextmanager
    def direct_read(self, slicer, output_array):
        """
        Set up a context where the slicer read on this buffer will be placed into a pre-defined array.

        Parameters
        ----------
        slicer: slice
            The read slice.
        output_array: ndarray
            Stuff output into this array

        """
        expected_shape = self.get_output_array(slicer, only_shape=True)
        if output_array.shape != expected_shape:
            s = output_array.shape
            raise ValueError('Wrong shape for output_array ({}), expected {}'.format(s, expected_shape))
        self._read_output = output_array
        try:
            yield
        finally:
            self._read_output = None

    def get_output_array(self, slicer, only_shape=False):
        """
        Allocate an array for read outs (or just the output shape).

        Parameters
        ----------
        slicer: slice
            Read slice
        only_shape: bool
            If True, just return the output shape

        Returns
        -------
        output_array: ndarray
            Allocated array for output (unless only_shape is given).

        """
        # Currently this is used to
        # 1) check a slice size
        # 2) create an output array for slice caching
        # **NOTE** that the output needs to respect `transpose_state`
        slicer = _abs_slicer(slicer, self.shape)
        out_shape = select(self.shape, slicer, 0).mshape
        # if the output will be transposed, then pre-create the array in the right order
        if self._transpose_state:
            out_shape = out_shape[::-1]
        if only_shape:
            return out_shape
        typecode = self.dtype.char
        if self._read_output is None:
            return shared_ndarray(out_shape, typecode)
        else:
            return self._read_output

    def close_source(self):
        raise NotImplementedError('Only implemented in subclasses')


class MappedBuffer(BufferBase):
    """
    Abstracts indexing from memmap file, with reads being converted to shared memory and possibly transformed to
    unit-ful values.
    """

    def __init__(self, array, units_scale=None, raise_bad_write=False):
        """

        Parameters
        ----------
        array: file-mapped array
            A memory-mapped array supporting slice syntax. This class assumes numpy.memmap. If using HDF5,
            construct with `HDF5Buffer`.
        units_scale: float or 2-tuple
            Either the scaling value or (offset, scaling) values such that signal = (memmap + offset) * scaling
        """
        super(MappedBuffer, self).__init__()
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
        self.units_scale = None
        if units_scale is not None:
            # Ignore the (0, 1) scaling to save cycles
            if np.iterable(units_scale):
                self._raw_offset = units_scale[0] if units_scale[0] != 0 else None
                self.units_scale = units_scale[1]
            else:
                self.units_scale = units_scale
            if self.units_scale == 1:
                self.units_scale = None
        if self.units_scale is None:
            self.dtype = array.dtype
        else:
            fp_precision = load_params().floating_point.lower()
            typecode = 'f' if fp_precision == 'single' else 'd'
            self.dtype = np.dtype(typecode)
        self.map_dtype = array.dtype
        self.shape = array.shape
        self._raise_bad_write = raise_bad_write

    def __len__(self):
        return len(self._array)

    @property
    def filename(self):
        return self._array.filename

    @property
    def ndim(self):
        return self._array.ndim

    @property
    def file_array(self):
        return self._array

    @property
    def writeable(self):
        return self.__writeable

    def describe(self):
        """
        Return sufficient info to reconstruct this buffer.

        Returns
        -------
        h5_name: str
            file name of the HDF5
        h5_path: str
            path info of this buffer's array

        """
        mm_name = self.filename
        mm_dtype = self.dtype
        mm_shape = self.shape
        return (mm_name, mm_dtype, mm_shape, self.units_scale, self._raise_bad_write)

    @classmethod
    def reopen(cls, buffer_info):
        mm_ro = np.memmap(buffer_info[0], mode='r', dtype=buffer_info[1], shape=buffer_info[2])
        return cls(mm_ro, units_scale=buffer_info[3], raise_bad_write=buffer_info[4])

    def close_source(self):
        self._array.flush()
        # Maybe this is a little seg-faulty?
        # self._array.base.close()
        del self._array

    def _scale_segment(self, x):
        if self.units_scale is None:
            return x
        if self._raw_offset is not None:
            x += self._raw_offset
        x *= self.units_scale
        return x

    def __getitem__(self, sl):
        out_arr = self.get_output_array(sl)
        # not sure what the most efficient way to slice is?
        if self._transpose_state:
            out_arr[:] = self._array[sl].transpose()
        else:
            out_arr[:] = self._array[sl]
        return self._scale_segment(out_arr)

    def __setitem__(self, sl, data):
        if self.writeable:
            # if the datatypes are a mismatch then don't try to fix it here
            self._array[sl] = data
            self._array.flush()
        elif self._raise_bad_write:
            raise RuntimeError('Tried to write to a not-writeable MappedBuffer')
        else:
            return


class HDF5Buffer(MappedBuffer):
    """Optimized mapped file reading for HDF5 files (h5py.File)"""

    @property
    def filename(self):
        return self._array.file.filename

    @property
    def chunks(self):
        return self._array.chunks

    def close_source(self):
        self._array.file.flush()
        self._array.file.close()
        del self._array

    def describe(self):
        """
        Return sufficient info to reconstruct this buffer.

        Returns
        -------
        h5_name: str
            file name of the HDF5
        h5_path: str
            path info of this buffer's array

        """
        h5_name = self.filename
        h5_path = self._array.name
        return (h5_name, h5_path, self.units_scale, self._raise_bad_write)

    @classmethod
    def reopen(cls, buffer_info):
        h5_ro = h5py.File(buffer_info[0], 'r', libver='latest', swmr=True)
        arr = h5_ro[buffer_info[1]]
        return cls(arr, units_scale=buffer_info[2], raise_bad_write=buffer_info[3])

    def __getitem__(self, sl):
        # this function computes the output shape -- use ID=0 to explicitly *not* work for funky RegionReference slicing
        out_arr = self.get_output_array(sl)
        i_slices, o_slices = tile_slices(sl, self.shape, self.chunks)
        # print('Slicing buffer as {}'.format(i_slices))
        for isl, osl in zip(i_slices, o_slices):
            if self._transpose_state:
                # generally need to reverse the slice order
                if isinstance(osl, _integer_types):
                    osl = (osl,)
                osl = osl[::-1]
                if len(osl) < out_arr.ndim:
                    osl = (Ellipsis,) + osl
                # can't avoid making a copy here?
                out_arr[osl] = self._array[isl].T
            else:
                # try a direct read, but this will fail if
                # 1) there are negative steps going on (ValueError)
                # 2) output is not C-contiguous (TypeError)
                try:
                    self._array.read_direct(out_arr, source_sel=isl, dest_sel=osl)
                except (ValueError, TypeError):
                    out_arr[osl] = self._array[isl]
        return self._scale_segment(out_arr)

    def __setitem__(self, sl, data):
        if not self.writeable:
            super(HDF5Buffer, self).__setitem__(sl, data)
            return
        # this should work for writing too?
        i_slices, o_slices = tile_slices(sl, self.shape, self.chunks)
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
        self._array.flush()


class BufferBinder(BufferBase):
    """We've got binders full of buffers!"""

    def __init__(self, buffers: Sequence[MappedBuffer], axis=-1):
        super(BufferBinder, self).__init__()
        if not np.iterable(buffers):
            buffers = (buffers,)
        # raise exception on any heterogeneity
        # 1) number of dimensions
        ndim = set([b.ndim for b in buffers])
        if len(ndim) > 1:
            raise ValueError('Mixture of dims: {}'.format(ndim))
        self._ndim = ndim.pop()
        # 2 array shapes: I think all but "axis" dimension need to be equal size?
        while axis < 0:
            axis += self._ndim
        dim_sizes = []
        for d in range(self._ndim):
            if d == axis:
                # this dimension can vary
                continue
            sizes_this_dim = set([b.shape[d] for b in buffers])
            dim_sizes.append(len(sizes_this_dim))
        if max(dim_sizes) > 1:
            raise ValueError('Inconsistent dimension sizes all first {} axes'.format(self._ndim - 1))
        self._buffer_lengths = [b.shape[axis] for b in buffers]
        self._total_length = sum(self._buffer_lengths)
        other_dims = list(buffers[0].shape)
        other_dims.pop(axis)
        self._other_dims = tuple(other_dims)
        self._concat_axis = axis
        # 3) write mode
        write_mode = [b.writeable for b in buffers]
        if any(write_mode) and not all(write_mode):
            raise ValueError('Some writeable buffers, but not all')
        self._writeable = all(write_mode)
        # 4) dtype -- but maybe this could be relaxed if it becomes a burden
        dtypes = set([b.dtype for b in buffers])
        if len(dtypes) > 1:
            raise ValueError('Mixture of dtypes: {}'.format(dtypes))
        self.dtype = dtypes.pop()
        # assume underlying maps have the same dtype if everything else is consistent
        self.map_dtype = buffers[0].map_dtype
        self.units_scale = buffers[0].units_scale

        self._buffers = buffers

    def __add__(self, other):
        buffers = self._buffers[:]
        if isinstance(other, MappedBuffer):
            buffers.append(other)
        elif isinstance(other, BufferBinder):
            if other._concat_axis != self._concat_axis:
                raise ValueError('Can only join BufferBinders with equal concatenation axis')
            buffers.extend(other._buffers)
        else:
            raise ValueError('Cannot extend buffer with type {}'.format(type(other)))
        return BufferBinder(buffers, axis=self._concat_axis)

    @property
    def filename(self):
        return [b.filename for b in self._buffers]

    @property
    def file_array(self):
        return [b.file_array for b in self._buffers]

    @property
    def chunks(self):
        return [b.chunks for b in self._buffers]

    @property
    def ndim(self):
        return self._ndim

    @property
    def writeable(self):
        return self._writeable

    @property
    def shape(self):
        dims = list(self._other_dims)
        dims.insert(self._concat_axis, self._total_length)
        return tuple(dims)

    def close_source(self):
        for b in self._buffers:
            b.close_source()

    def describe(self):
        # return each buffer's info and sneak in the type of the buffer as well
        info = tuple([(type(b),) + b.describe() for b in self._buffers])
        return (self._concat_axis,) + info

    @classmethod
    def reopen(cls, buffer_info):
        buffers = []
        axis = buffer_info[0]
        for info in buffer_info[1:]:
            # split info into buffer class and actual info
            b_cls, b_info = info[0], info[1:]
            buffers.append(b_cls.reopen(b_info))
        return cls(buffers, axis=axis)

    @contextmanager
    def transpose_reads(self, status=None):
        # slight modification -- need to make a context that puts all buffers in transpose state
        try:
            with self._transpose_state(status), ExitStack() as stack:
                for b in self._buffers:
                    stack.enter_context(b.transpose_reads(status))
                # ToggleState yields None, so also yield None here
                yield
        finally:
            for b in self._buffers:
                assert b._transpose_state != status

    def target_sources(self, slicer):
        # This will parse the slicer into a list of (source, sub-slice) pairs.
        # Reminder to self: the requested slices *always* have the correct dimension order,
        # regardless of transpose status

        running_length = np.cumsum(self._buffer_lengths)
        slicer = unpack_ellipsis(slicer, self.ndim)
        if len(slicer) < self.ndim:
            slicer = slicer + (slice(None),) * (self.ndim - len(slicer))
        slicer = list(slicer)
        axis = self._concat_axis
        skip_slice = slicer[axis]
        # handle the chance that this is just a singleton slice
        if isinstance(skip_slice, _integer_types):
            buffer = running_length.searchsorted(skip_slice, side='right') - 1
            slicer[axis] = skip_slice - running_length[buffer]
            return [(self._buffers[buffer], slicer)]
        step = 1 if skip_slice.step is None else skip_slice.step
        rev_slice = step < 0
        skip_slice = _abs_slicer(skip_slice, self.shape[axis:axis + 1])[0]
        start = 0 if skip_slice.start is None else skip_slice.start
        stop = self._total_length if skip_slice.stop is None else skip_slice.stop
        step = skip_slice.step
        for n in range(len(self._buffers)):
            if start < running_length[n]:
                start_buffer = n
                break
        buffer = start_buffer
        if buffer > 0:
            start_pt = start - running_length[buffer - 1]
        else:
            start_pt = start
        sources_and_slices = []
        while True:
            this_slice = slicer[:]
            if stop <= running_length[buffer]:
                # stop at this buffer
                if buffer > 0:
                    stop_pt = stop - running_length[buffer - 1]
                else:
                    stop_pt = stop
            else:
                # otherwise slice thru the end of this buffer and find the correct start point in the next buffer
                stop_pt = self._buffer_lengths[buffer]
            sl = slice(start_pt, stop_pt, step)
            if rev_slice:
                sl = range_to_slice(slice_to_range(sl, self.shape[axis])[::-1])
            this_slice[axis] = sl
            sources_and_slices.append((self._buffers[buffer], tuple(this_slice)))
            if stop <= running_length[buffer]:
                break
            if step is not None:
                # this is the number of points to skip into the next buffer:
                # (stop_pt - 1 - start_pt) % step is the remainder of points from this buffer
                start_pt = step - (stop_pt - 1 - start_pt) % step - 1
            else:
                start_pt = 0
            buffer += 1
        if rev_slice:
            return sources_and_slices[::-1]
        else:
            return sources_and_slices

    def _correct_concat_axis(self, slicer):
        # logic to apply a correction to the concatenating/splitting axis for slices that eat dimensions
        slicer = unpack_ellipsis(slicer, self.ndim)
        if len(slicer) < self.ndim:
            slicer = slicer + (slice(None),) * (self.ndim - len(slicer))
        correction = 0
        for n, sl in enumerate(slicer):
            # need to reduce the concat axis for every dimension-eating slice that occurs before it
            if n < self._concat_axis and isinstance(sl, _integer_types):
                correction -= 1
        return self._concat_axis + correction

    @contextmanager
    def direct_read(self, slicer, output_array):
        # This is going to need to assign partitions of the output_array to one or more buffers,
        # and then tell the buffers to clear the _read_output in the "exit/finally" clause
        buffer_slices = self.target_sources(slicer)
        # All buffers are in the same transpose status as the binder, so their output shapes will
        # split across the transposed concat axis.
        output_sizes = [b.get_output_array(s, only_shape=True) for b, s in buffer_slices]
        output_dim = len(output_sizes[0])

        # transpose manipulations only matter if the read slice addresses more than 1 axis
        # and the read slice crosses buffers
        #
        # Example... we have a binder with shapes [(10, 4, 50), (10, 4, 30)] and concat axis 2
        # and the slice is [:, 0, :10] --> output shape (10, 10) (or trans) but all from the same buffer.
        # So the "split" axis here is computed as 2 - 2 - 1 = -1 if transposed, else 2.
        # Both values are wrong, but the direct read array should not be split anyway.
        #
        # Example 2.. same binder and the slice is [:, 0, 40:60] --> output shape (10, 20) (or trans) across buffers.
        # Corrected concat axis is 2 - 1 = 1
        # Transpose Split axis 2 - 1 - 1 = 0 (should be 0)
        # Regular split axis is 1 (should be 1)
        #
        # Example 3.. shapes [(10, 4, 50), (10, 4, 30)] and concat axis 0
        # Slice [5:15, 0, :20] -> shape (10, 20) (or trans) across two buffers.
        # Corrected concat axis is 0
        # Transpose split axis is 2 - 0 - 1 = 1 (should be 1) (shape (20, 10) -> (20, 5) + (20, 5)).
        # Regular split axis is 0 (should be 0) (shape (10, 20) -> (5, 20) + (5, 20))
        #
        # Example 4.. shapes [(10, 4, 50), (10, 4, 30)] and concat axis 1
        # Slice [0, 2:5, :10] -> shape (3, 10) (or trans) across two buffers.
        # Corrected concat axis is 0
        # Transpose split axis is 2 - 0 - 1 = 1 (should be 1) (shape (10, 2) + (10, 1))
        # Regular split axis is 0 (should be 0) (shape (2, 10) + (1, 10))

        if len(buffer_slices) > 1:
            if output_dim > 1:
                axis = self._correct_concat_axis(slicer)
                if self._transpose_state:
                    axis = output_dim - axis - 1
            else:
                axis = 0
            output_sizes = [o[axis] for o in output_sizes]
            break_points = np.cumsum(output_sizes)[:-1]
            splits = np.split(output_array, break_points, axis=axis)
        else:
            splits = [output_array]
        try:
            self._read_output = output_array
            with ExitStack() as stack:
                for b, arr in zip(buffer_slices, splits):
                    buf, sl = b
                    stack.enter_context(buf.direct_read(sl, arr))
                yield
        finally:
            self._read_output = None
            pass

    def __getitem__(self, slicer):
        buffer_slices = self.target_sources(slicer)
        output = []
        for b, s in buffer_slices:
            # Do not create a transpose context here. The binder-buffers transpose contexts
            # are now synced with an ExitStack.
            output.append(b[s])
        if self._read_output is not None:
            return self._read_output
        elif len(output) == 1:
            return output[0]
        else:
            output_dim = output[0].ndim
            if output_dim > 1:
                axis = self._correct_concat_axis(slicer)
                if self._transpose_state:
                    axis = output_dim - axis - 1
            else:
                axis = 0
            return np.concatenate(output, axis)

    def __setitem__(self, slicer, array):
        buffer_slices = self.target_sources(slicer)
        # TODO: need to handle broadcasting here
        array = np.asanyarray(array)
        axis = self._concat_axis
        # # If the array can broadcast to every buffer, then don't use array splitting..
        # # This would be true if the number of dims after the concatenate axis is >= array.ndim
        # if array.ndim < (self.ndim - axis):
        #     # just copy the reference a few times
        #     splits = [array] * len(buffer_slices)
        # else:
        #     # Otherwise, the concatenate axis splits over an axis in array
        new_dims = (1,) * (self.ndim - array.ndim) + array.shape
        full_array = array.reshape(*new_dims)
        if full_array.shape[axis] > 1:
            output_sizes = [b.get_output_array(s, only_shape=True) for b, s in buffer_slices]
            if len(output_sizes[0]) == 1:
                # if the slicing only accesses a single dimension, then the
                # shape and split axis has to be zero
                axis = 0
            output_sizes = [o[axis] for o in output_sizes]
            break_points = np.cumsum(output_sizes)[:-1]
            splits = np.split(array, break_points, axis=axis)
        else:
            splits = [array] * len(buffer_slices)
        for bs, sub_arr in zip(buffer_slices, splits):
            buffer, sl = bs
            buffer[sl] = sub_arr


class BackgroundRead(Process):
    """
    A Process that can re-open mapped files in read-only mode and slice out some data in the background
    """

    def __init__(self, buffer: BufferBase, slicer: slice, transpose: bool=False, output=None):
        super(BackgroundRead, self).__init__()
        self.slicer = slicer
        self.transpose = transpose
        if output is None:
            output = buffer.get_output_array(slicer)
        self.output = SharedmemManager(output)
        self.buffer_type = type(buffer)
        self.buffer_info = buffer.describe()

    def run(self):
        ro_buffer = self.buffer_type.reopen(self.buffer_info)
        try:
            with self.output.get_ndarray() as output:
                slice_data_buffer(ro_buffer, self.slicer, transpose=self.transpose, output=output)
        finally:
            ro_buffer.close_source()
