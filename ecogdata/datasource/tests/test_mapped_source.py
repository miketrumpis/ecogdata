from nose.tools import assert_true, assert_equal
import numpy as np
import h5py
from tempfile import NamedTemporaryFile
from ecogdata.datasource.memmap import MappedSource

def _create_hdf5(n_rows=20, n_cols=1000, rand=False, transpose=False, aux_arrays=(), dtype='i'):

    with NamedTemporaryFile(mode='ab', dir='.') as f:
        f.file.close()
        fw = h5py.File(f.name, 'w')
        arrays = ('data',) + tuple(aux_arrays)
        disk_shape = (n_cols, n_rows) if transpose else (n_rows, n_cols)
        for name in arrays:
            y = fw.create_dataset(name, shape=disk_shape, dtype=dtype)
            if rand:
                arr = np.random.randint(0, 2 ** 13, size=(n_rows, n_cols)).astype(dtype)
                y[:] = arr.T if transpose else arr
            else:
                # test pattern
                arr = np.arange(n_rows * n_cols, dtype=dtype).reshape(n_rows, n_cols)
                y[:] = arr.T if transpose else arr
    return fw, f.file


def test_construction():
    aux_arrays = ('test1', 'test2')
    f, filename = _create_hdf5(aux_arrays=aux_arrays)
    data_shape = f['data'].shape
    map_source = MappedSource(f, 1, 'data', aux_fields=aux_arrays)
    print(data_shape, map_source.data_shape)
    assert_equal(map_source.data_shape, data_shape, 'Shape wrong')
    assert_equal(map_source.binary_channel_mask.sum(), data_shape[0], 'Wrong number of active channels')
    for field in aux_arrays:
        assert_true(hasattr(map_source, field), 'Aux field {} not preserved'.format(field))
    # repeat for transpose
    map_source = MappedSource(f, 1, 'data', aux_fields=aux_arrays, transpose=True)
    assert_equal(map_source.data_shape, data_shape[::-1], 'Shape wrong in transpose')
    assert_equal(map_source.binary_channel_mask.sum(), data_shape[1], 'Wrong number of active channels in transpose')


def test_scaling():
    f, filename = _create_hdf5()
    float_data = f['data'][:, 500:1000].astype('d')
    map_source = MappedSource(f, 1, 'data', units_scale=2.0)
    assert_true(np.all(map_source[:, 500:1000] == float_data * 2).all(), 'scalar scaling wrong')
    map_source = MappedSource(f, 1, 'data', units_scale=(-100, 2.0))
    assert_true(np.all(map_source[:, 500:1000] == (float_data - 100) * 2).all(), 'affine scaling wrong')


def test_electrode_subset():
    f, filename = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels)
    data = f['data'][:, :][electrode_channels]
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed')


def test_electrode_subsetT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels, transpose=True)
    data = f['data'][:, :][:, electrode_channels].T
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed in transpose')


def test_channel_map():
    f, filename = _create_hdf5()
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels)
    map_source.set_channel_mask(binary_mask)
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong')
    data = f['data'][:, :][electrode_channels, :]
    assert_true(np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed')


def test_channel_mapT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels, transpose=True)
    map_source.set_channel_mask(binary_mask)
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong in transpose')
    data = f['data'][:, :][:, electrode_channels].T
    assert_true(np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed in transpose')


def test_iter():
    f, filename = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:]
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong')


def test_iterT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:].T
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose')
