from nose.tools import assert_true, assert_equal, raises
import numpy as np
import h5py
from tempfile import NamedTemporaryFile

from ecogdata.parallel.mproc import Process
from ecogdata.datasource.memmap import TempFilePool
from ecogdata.datasource.array_abstractions import slice_to_range, range_to_slice, unpack_ellipsis, tile_slices, \
    HDF5Buffer, BufferBinder, slice_data_buffer


def _create_hdf5(n_rows=20, n_cols=1000, extra_dims=(), rand=False,
                 transpose=False, aux_arrays=(), chunks=True, dtype='i'):

    with NamedTemporaryFile(mode='ab', dir='.') as f:
        f.file.close()
        fw = h5py.File(f.name, 'w', libver='latest')
        arrays = ('data',) + tuple(aux_arrays)
        array_shape = (n_rows, n_cols) + extra_dims
        if transpose:
            disk_shape = array_shape[::-1]
        else:
            disk_shape = array_shape
        for name in arrays:
            y = fw.create_dataset(name, shape=disk_shape, dtype=dtype, chunks=chunks)
            if rand:
                arr = np.random.randint(0, 2 ** 13, size=array_shape).astype(dtype)
                y[:] = arr.T if transpose else arr
            else:
                # test pattern
                arr = np.arange(np.prod(array_shape), dtype=dtype).reshape(array_shape)
                y[:] = arr.T if transpose else arr
    return fw, f.file


def _create_buffer(units_scale=None, **kwargs):
    hdf_file = _create_hdf5(**kwargs)[0]
    buffer = HDF5Buffer(hdf_file['data'], units_scale=units_scale)
    data = hdf_file['data'][:]
    return buffer, data


def _create_binder(num_buffers=3, axis=0, units_scale=None, **kwargs):
    hdf_files = [_create_hdf5(**kwargs)[0] for i in range(num_buffers)]
    buffers = [HDF5Buffer(f['data'], units_scale=units_scale) for f in hdf_files]
    data = np.concatenate([b[:] for b in buffers], axis=axis)
    buffer_binder = BufferBinder(buffers, axis=axis)
    return buffer_binder, data


def test_slice_range_conv():
    slice1 = np.s_[5::2]
    rng = slice_to_range(slice1, 25)
    assert_equal(list(rng), list(range(5, 25, 2)), 'slice to range failed')
    slice2 = range_to_slice(rng)
    # need to test the functional equivalence
    r = np.arange(25)
    assert_true(np.all(r[slice1] == r[slice2]), 'slice to range to slice failed')
    slice3 = range_to_slice(rng, offset=5)
    assert_true(np.all(r[slice3] == r[0:20:2]), 'range to slice with offset failed')


def test_unpack_ellipsis():
    dims = 4
    sl = np.s_[:4, ..., 0]
    assert_equal(unpack_ellipsis(sl, dims), np.s_[:4, :, :, 0], 'ellipsis failed (1)')
    sl = np.s_[..., 0]
    assert_equal(unpack_ellipsis(sl, dims), np.s_[:, :, :, 0], 'ellipsis failed (2)')


def test_tile_slice():
    i, o = tile_slices(np.s_[::10, [0, 1, 10, 11]], (20, 20), (5, 5))
    assert_true(len(i) == len(o), 'slice lists should be the same name')
    out_arr = np.zeros((2, 4))
    for n, sl in enumerate(o):
        out_arr[sl] = n
    should_be = np.array([[0., 0., 1., 1.],
                          [2., 2., 3., 3.]])
    assert_true(np.all(out_arr == should_be), 'tiled slicing failed (output)')
    should_be = [(0, [0, 1]), (0, [10, 11]), (10, [0, 1]), (10, [10, 11])]
    assert_equal(i, should_be, 'tiled slicing failed (input)')


def test_tile_slice2():
    r = np.random.rand(20, 20, 20)
    s1 = np.s_[::2, ::2, ::2]  # simple
    s2 = np.s_[::10, 4, [0, 1, 2, 10, 12, 14]]  # complicated
    s3 = np.s_[..., [0, 1, 2, 10, 12, 14]]  # with ellipse
    s4 = np.s_[:5]  # implicit slices
    for slicer in (s1, s2, s3, s4):
        i, o = tile_slices(slicer, r.shape, (5, 5, 5))
        assert_true(len(i) == len(o), 'tile slices should be equal length')
        regular_array = r[slicer]
        tiled_array = regular_array.copy()
        for isl, osl in zip(i, o):
            tiled_array[osl] = r[isl]
        assert_true(np.all(regular_array == tiled_array), 'tiled array not correctly tiled')


def test_tiles_negative_step():
    r = np.random.rand(20, 20, 20)
    s1 = np.s_[::-2, ::2, ::-2]  # simple
    s2 = np.s_[::-10, 4, [0, 1, 2, 10, 12, 14]]  # complicated
    s3 = np.s_[..., ::-1]  # with ellipse
    s4 = np.s_[5::-1]  # implicit slices
    for slicer in (s1, s2, s3, s4):
        i, o = tile_slices(slicer, r.shape, (5, 5, 5))
        assert_true(len(i) == len(o), 'tile slices should be equal length')
        regular_array = r[slicer]
        tiled_array = regular_array.copy()
        for isl, osl in zip(i, o):
            tiled_array[osl] = r[isl]
        assert_true(np.all(regular_array == tiled_array), 'tiled array not correctly tiled')


def test_hdf_buffer_basic():
    buf, data = _create_buffer(n_rows=20, n_cols=1000, dtype='i')
    assert_true(np.all(data[:5, :] == buf[:5, :]), 'basic buffer failed')
    assert_true(np.all(data[:, ::2] == buf[:, ::2]), 'basic buffer failed')
    buf = _create_buffer(units_scale=(1000, 0.5), n_rows=20, n_cols=1000, dtype='i')[0]
    data = (data.astype('d') + 1000) * 0.5
    assert_true(np.all(data[:5, :] == buf[:5, :]), 'basic buffer with scale failed')
    assert_true(np.all(data[:, ::2] == buf[:, ::2]), 'basic buffer with scale failed')


def test_hdf_buffer_tiled():
    # chunking should kick-in on the first two axes
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    assert_true(np.all(data[::10, ::20] == buf[::10, ::20]), 'tiled buffer failed')
    assert_true(np.all(data[::10, ::20, 0] == buf[::10, ::20, 0]), 'tiled buffer failed')
    # chunking should kick-in on the first and third axes
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))
    assert_true(np.all(data[::10, ::20] == buf[::10, ::20]), 'tiled buffer failed')
    assert_true(np.all(data[::10, ::20, 0] == buf[::10, ::20, 0]), 'tiled buffer failed')


def test_hdf_buffer_in_transpose():
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(20, 100, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true(np.all(data[::10, ::20].T == seg1), 'tiled buffer failed in transpose')
    assert_true(np.all(data[::10, ::20, 2].T == seg2), 'tiled buffer failed in transpose')


def test_hdf_buffer_in_transpose_tiled():
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true(np.all(data[::10, ::20].T == seg1), 'tiled buffer failed in transpose')
    assert_true(np.all(data[::10, ::20, 2].T == seg2), 'tiled buffer failed in transpose')


def test_hdf_buffer_tiled_negative_step():
    # chunking should kick-in on the first two axes
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    assert_true(np.all(data[::10, ::-20] == buf[::10, ::-20]), 'neg step tiled buffer failed')
    assert_true(np.all(data[::10, ::-20, 0] == buf[::10, ::-20, 0]), 'neg step tiled buffer failed')
    # chunking should kick-in on the first and third axes
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))
    assert_true(np.all(data[::-10, ::20] == buf[::-10, ::20]), 'neg step tiled buffer failed')
    assert_true(np.all(data[::-10, ::20, 0] == buf[::-10, ::20, 0]), 'neg step tiled buffer failed')


def test_hdf_buffer_transpose_negative_step():
    buf, data = _create_buffer(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::-20]
        seg2 = buf[::10, ::-20, 2]
    assert_true(np.all(data[::10, ::-20].T == seg1), 'neg step tiled buffer failed in transpose')
    assert_true(np.all(data[::10, ::-20, 2].T == seg2), 'neg step tiled buffer failed in transpose')


@raises(RuntimeError)
def test_hdf_buffer_raise_write_fail():
    # Create a buffer long-hand with a temp file that can be re-opened
    with TempFilePool(mode='ab') as f:
        filename = f.name
    with h5py.File(filename, 'w') as hdf:
        hdf.create_dataset('data', shape=(10, 10), dtype='i')
    # now re-open in read mode
    with h5py.File(filename, 'r') as hdf:
        buf = HDF5Buffer(hdf['data'], raise_bad_write=True)
        buf[:] = 0


def test_hdf_buffer_silent_write_fail():
    # Create a buffer long-hand with a temp file that can be re-opened
    with TempFilePool(mode='ab') as f:
        filename = f.name
    with h5py.File(filename, 'w') as hdf:
        hdf.create_dataset('data', data=np.ones((10, 10)))
    # now re-open in read mode
    with h5py.File(filename, 'r') as hdf:
        buf = HDF5Buffer(hdf['data'], raise_bad_write=False)
        buf[:] = 0
        assert_true(np.all(buf[:] == 1), 'silent pass for read-only buffer failed')


def test_hdf_buffer_basic_write():
    buf = _create_buffer(n_rows=20, n_cols=1000, dtype='i')[0]
    assert_true(buf.writeable, 'Should be write mode')
    buf[:5, :] = 0
    assert_true(not buf[:5, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_hdf_buffer_tiled_write():
    buf = _create_buffer(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    buf[::6, :] = 0
    assert_true(not buf[::6, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_hdf_buffer_broadcast_write():
    buf = _create_buffer(n_rows=20, n_cols=1000, dtype='i')[0]
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[:5, :] = rand_pattern
    assert_true(np.all(buf[:5, :] == rand_pattern), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::20]), 'Should have written out')


def test_hdf_buffer_broadcast_tiled_write():
    buf = _create_buffer(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, :] = rand_pattern
    assert_true(np.all(buf[::6, :] == rand_pattern), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::20]), 'Should have written out')


def test_hdf_buffer_broadcast_tiled_write_negative_step():
    buf = _create_buffer(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, ::-1] = rand_pattern
    assert_true(np.all(buf[::6, :] == rand_pattern[::-1]), 'Should have written')
    buf[::6, ::-20] = rand_pattern[::-20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::-20]), 'Should have written out')


def test_hdf_buffer_direct_read():
    buf, data = _create_buffer(n_rows=20, n_cols=100, dtype='i')
    sl = np.s_[2:7, 20:30]
    direct_out = buf.get_output_array(sl)
    with buf.direct_read(sl, direct_out):
        out1 = buf[sl]
    assert_true(out1 is direct_out, 'direct read returned indirect array')
    out1 = data[sl]
    assert_true(np.all(out1 == direct_out), 'direct read returned wrong values')


def test_hdf_buffer_direct_read_transpose():
    buf, data = _create_buffer(n_rows=20, n_cols=100, dtype='i')
    sl = np.s_[2:7, 20:30]
    with buf.transpose_reads(True):
        direct_out = buf.get_output_array(sl)
    with buf.transpose_reads(True), buf.direct_read(sl, direct_out):
        out1 = buf[sl]
    assert_true(out1 is direct_out, 'direct read returned indirect array')
    out1 = data[sl].T
    assert_true(np.all(out1 == direct_out), 'direct read returned wrong values')


# All these tests should pass with BufferBinders

def test_buffer_binder_basic():
    buf, data = _create_binder(axis=1, n_rows=20, n_cols=1000, dtype='i')
    assert_true(np.all(data[:5, :] == buf[:5, :]), 'basic binder failed')
    assert_true(np.all(data[:, ::2] == buf[:, ::2]), 'basic binder failed')
    buf, data = _create_binder(axis=1, units_scale=(1000, 0.5), n_rows=20, n_cols=1000, dtype='i')
    assert_true(np.all(data[:5, :] == buf[:5, :]), 'basic binder with scale failed')
    assert_true(np.all(data[:, ::2] == buf[:, ::2]), 'basic binder with scale failed')


def test_buffer_binder_tiled():
    # chunking should kick-in on the first two axes
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    assert_true(np.all(data[::10, ::20] == buf[::10, ::20]), 'tiled binder failed')
    assert_true(np.all(data[::10, ::20, 0] == buf[::10, ::20, 0]), 'tiled binder failed')
    # chunking should kick-in on the first and third axes
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))
    assert_true(np.all(data[::10, ::20] == buf[::10, ::20]), 'tiled binder failed')
    assert_true(np.all(data[::10, ::20, 0] == buf[::10, ::20, 0]), 'tiled binder failed')


def test_buffer_binder_in_transpose():
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(20, 100, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true(np.all(data[::10, ::20].T == seg1), 'tiled binder failed in transpose')
    assert_true(np.all(data[::10, ::20, 2].T == seg2), 'tiled binder failed in transpose')


def test_buffer_binder_in_transpose_tiled():
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true(np.all(data[::10, ::20].T == seg1), 'tiled binder failed in transpose')
    assert_true(np.all(data[::10, ::20, 2].T == seg2), 'tiled binder failed in transpose')


def test_buffer_binder_tiled_negative_step():
    # chunking should kick-in on the first two axes
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    assert_true(np.all(data[::10, ::-20] == buf[::10, ::-20]), 'neg step tiled binder failed')
    assert_true(np.all(data[::10, ::-20, 0] == buf[::10, ::-20, 0]), 'neg step tiled binder failed')
    # chunking should kick-in on the first and third axes
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))
    assert_true(np.all(data[::-10, ::20] == buf[::-10, ::20]), 'neg step tiled binder failed')
    assert_true(np.all(data[::-10, ::20, 0] == buf[::-10, ::20, 0]), 'neg step tiled binder failed')


def test_buffer_binder_transpose_negative_step():
    buf, data = _create_binder(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::-20]
        seg2 = buf[::10, ::-20, 2]
    assert_true(np.all(data[::10, ::-20].T == seg1), 'neg step tiled binder failed in transpose')
    assert_true(np.all(data[::10, ::-20, 2].T == seg2), 'neg step tiled binder failed in transpose')


def test_buffer_binder_basic_write():
    buf = _create_binder(n_rows=20, n_cols=1000, dtype='i')[0]
    assert_true(buf.writeable, 'Should be write mode')
    buf[:5, :] = 0
    assert_true(not buf[:5, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_buffer_binder_tiled_write():
    buf = _create_binder(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    buf[::6, :] = 0
    assert_true(not buf[::6, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_buffer_binder_broadcast_write():
    buf = _create_binder(n_rows=20, n_cols=1000, dtype='i')[0]
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[:5, :] = rand_pattern
    assert_true(np.all(buf[:5, :] == rand_pattern), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::20]), 'Should have written out')


def test_buffer_binder_broadcast_tiled_write():
    buf = _create_binder(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, :] = rand_pattern
    assert_true(np.all(buf[::6, :] == rand_pattern), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::20]), 'Should have written out')


def test_buffer_binder_broadcast_tiled_write_negative_step():
    buf = _create_binder(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, ::-1] = rand_pattern
    assert_true(np.all(buf[::6, :] == rand_pattern[::-1]), 'Should have written')
    buf[::6, ::-20] = rand_pattern[::-20]
    assert_true(np.all(buf[::6, ::20] == rand_pattern[::-20]), 'Should have written out')


def test_buffer_binder_direct_read():
    buf, data = _create_binder(n_rows=20, n_cols=100, dtype='i')
    sl = np.s_[2:37, 20:30]
    direct_out = buf.get_output_array(sl)
    with buf.direct_read(sl, direct_out):
        out1 = buf[sl]
    assert_true(out1 is direct_out, 'direct read returned indirect array')
    out1 = data[sl]
    assert_true(np.all(out1 == direct_out), 'direct read returned wrong values')


def test_buffer_binder_direct_read_transpose():
    buf, data = _create_binder(n_rows=20, n_cols=100, dtype='i')
    sl = np.s_[2:37, 20:30]
    with buf.transpose_reads(True):
        direct_out = buf.get_output_array(sl)
    with buf.transpose_reads(True), buf.direct_read(sl, direct_out):
        out1 = buf[sl]
    assert_true(out1 is direct_out, 'direct read returned indirect array')
    out1 = data[sl].T
    assert_true(np.all(out1 == direct_out), 'direct read returned wrong values')


def test_subprocess_caching():
    buf, data = _create_binder(n_rows=10, n_cols=100, axis=1, dtype='i')
    # 1) test a slice into a single buffer
    sl = np.s_[3:6, 20:60]
    output = buf.get_output_array(sl)
    p = Process(target=slice_data_buffer, args=(buf, sl), kwargs=dict(output=output))
    p.start()
    p.join()
    # compare output with data array
    assert_true(np.all(output == data[sl]), 'Subprocess slicing failed for single buffer')
    # 2) test a slice across buffers
    sl = np.s_[3:6, 80:130]
    output = buf.get_output_array(sl)
    p = Process(target=slice_data_buffer, args=(buf, sl), kwargs=dict(output=output))
    p.start()
    p.join()
    # compare output with data array
    assert_true(np.all(output == data[sl]), 'Subprocess slicing failed across buffers')


def test_subprocess_cachingT():
    buf, data = _create_binder(n_rows=10, n_cols=100, axis=1, dtype='i')
    # 1) test a slice into a single buffer
    sl = np.s_[3:6, 20:60]
    with buf.transpose_reads(True):
        output = buf.get_output_array(sl)
    p = Process(target=slice_data_buffer, args=(buf, sl), kwargs=dict(output=output, transpose=True))
    p.start()
    p.join()
    # compare output with data array
    assert_true(np.all(output == data[sl].T), 'Subprocess slicing failed for single buffer (transpose)')
    # 2) test a slice across buffers
    sl = np.s_[3:6, 80:130]
    with buf.transpose_reads(True):
        output = buf.get_output_array(sl)
    p = Process(target=slice_data_buffer, args=(buf, sl), kwargs=dict(output=output, transpose=True))
    p.start()
    p.join()
    # compare output with data array
    assert_true(np.all(output == data[sl].T), 'Subprocess slicing failed across buffers (transpose)')
