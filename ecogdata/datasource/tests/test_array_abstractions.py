from nose.tools import assert_true, assert_equal
import numpy as np

from ecogdata.datasource.array_abstractions import slice_to_range, range_to_slice, unpack_ellipsis, tile_slices, \
    HDF5Buffer

from .test_mapped_source import _create_hdf5

def test_slice_range_conv():
    slice = np.s_[5::2]
    rng = slice_to_range(slice, 25)
    assert_equal(list(rng), list(range(5, 25, 2)), 'slice to range failed')
    slice2 = range_to_slice(rng)
    # need to test the functional equivalence
    r = np.arange(25)
    assert_true((r[slice] == r[slice2]).all(), 'slice to range to slice failed')
    slice3 = range_to_slice(rng, offset=5)
    assert_true((r[slice3] == r[0:20:2]).all(), 'range to slice with offset failed')


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
    assert_true((out_arr == should_be).all(), 'tiled slicing failed (output)')
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
        assert_true((regular_array == tiled_array).all(), 'tiled array not correctly tiled')


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
        assert_true((regular_array == tiled_array).all(), 'tiled array not correctly tiled')


def test_hdf_buffer_basic():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i')[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    assert_true((data[:5, :] == buf[:5, :]).all(), 'basic buffer failed')
    assert_true((data[:, ::2] == buf[:, ::2]).all(), 'basic buffer failed')
    buf = HDF5Buffer(f['data'], (1000, 0.5))
    data = (data.astype('d') + 1000) * 0.5
    assert_true((data[:5, :] == buf[:5, :]).all(), 'basic buffer with scale failed')
    assert_true((data[:, ::2] == buf[:, ::2]).all(), 'basic buffer with scale failed')


def test_hdf_buffer_tiled():
    # chunking should kick-in on the first two axes
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    assert_true((data[::10, ::20] == buf[::10, ::20]).all(), 'tiled buffer failed')
    assert_true((data[::10, ::20, 0] == buf[::10, ::20, 0]).all(), 'tiled buffer failed')
    # chunking should kick-in on the first and third axes
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    assert_true((data[::10, ::20] == buf[::10, ::20]).all(), 'tiled buffer failed')
    assert_true((data[::10, ::20, 0] == buf[::10, ::20, 0]).all(), 'tiled buffer failed')


def test_hdf_buffer_in_transpose():
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(20, 100, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true((data[::10, ::20].T == seg1).all(), 'tiled buffer failed in transpose')
    assert_true((data[::10, ::20, 2].T == seg2).all(), 'tiled buffer failed in transpose')


def test_hdf_buffer_in_transpose_tiled():
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::20]
        seg2 = buf[::10, ::20, 2]
    assert_true((data[::10, ::20].T == seg1).all(), 'tiled buffer failed in transpose')
    assert_true((data[::10, ::20, 2].T == seg2).all(), 'tiled buffer failed in transpose')


def test_hdf_buffer_tiled_negative_step():
    # chunking should kick-in on the first two axes
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    assert_true((data[::10, ::-20] == buf[::10, ::-20]).all(), 'neg step tiled buffer failed')
    assert_true((data[::10, ::-20, 0] == buf[::10, ::-20, 0]).all(), 'neg step tiled buffer failed')
    # chunking should kick-in on the first and third axes
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 50, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    assert_true((data[::-10, ::20] == buf[::-10, ::20]).all(), 'neg step tiled buffer failed')
    assert_true((data[::-10, ::20, 0] == buf[::-10, ::20, 0]).all(), 'neg step tiled buffer failed')


def test_hdf_buffer_transpose_negative_step():
    f = _create_hdf5(n_rows=20, n_cols=100, extra_dims=(5, 10), dtype='i', chunks=(5, 10, 5, 10))[0]
    buf = HDF5Buffer(f['data'])
    data = f['data'][:]
    with buf.transpose_reads(True):
        # slice the buffer in the same way as norm, but the output should be transposed
        seg1 = buf[::10, ::-20]
        seg2 = buf[::10, ::-20, 2]
    assert_true((data[::10, ::-20].T == seg1).all(), 'neg step tiled buffer failed in transpose')
    assert_true((data[::10, ::-20, 2].T == seg2).all(), 'neg step tiled buffer failed in transpose')



# Can't open the file since it is already "deleted" by the OS
# @raises(RuntimeError)
# def test_hdf_buffer_basic_write_fail():
#     f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i')[0]
#     f_ro = h5py.File(f.file.filename, 'r')
#     buf = HDF5Buffer(f_ro['data'])
#     buf[:] = 0


def test_hdf_buffer_basic_write():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i')[0]
    buf = HDF5Buffer(f['data'])
    assert_true(buf.writeable, 'Should be write mode')
    buf[:5, :] = 0
    assert_true(not buf[:5, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_hdf_buffer_tiled_write():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    buf = HDF5Buffer(f['data'])
    assert_true(buf.writeable, 'Should be write mode')
    buf[::6, :] = 0
    assert_true(not buf[::6, :].any(), 'Should have zeroed out')
    buf[:, ::2] = 0
    assert_true(not buf[:, ::2].any(), 'Should have zeroed out')


def test_hdf_buffer_broadcast_write():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i')[0]
    buf = HDF5Buffer(f['data'])
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[:5, :] = rand_pattern
    assert_true((buf[:5, :] == rand_pattern).all(), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true((buf[::6, ::20] == rand_pattern[::20]).all(), 'Should have written out')


def test_hdf_buffer_broadcast_tiled_write():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    buf = HDF5Buffer(f['data'])
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, :] = rand_pattern
    assert_true((buf[::6, :] == rand_pattern).all(), 'Should have written')
    buf[::6, ::20] = rand_pattern[::20]
    assert_true((buf[::6, ::20] == rand_pattern[::20]).all(), 'Should have written out')


def test_hdf_buffer_broadcast_tiled_write_negative_step():
    f = _create_hdf5(n_rows=20, n_cols=1000, dtype='i', chunks=(5, 10))[0]
    buf = HDF5Buffer(f['data'])
    assert_true(buf.writeable, 'Should be write mode')
    rand_pattern = np.random.randint(0, 2 ** 14, size=1000)
    buf[::6, ::-1] = rand_pattern
    assert_true((buf[::6, :] == rand_pattern[::-1]).all(), 'Should have written')
    buf[::6, ::-20] = rand_pattern[::-20]
    assert_true((buf[::6, ::20] == rand_pattern[::-20]).all(), 'Should have written out')


