from nose.tools import assert_true, assert_equal
import os
import numpy as np
import h5py
from tempfile import NamedTemporaryFile
from ecogdata.datasource.memmap import MappedSource
from ecogdata.datasource.basic import PlainArraySource

def _create_hdf5(n_rows=20, n_cols=1000, rand=False, transpose=False, aux_arrays=(), chunks=True, dtype='i'):

    with NamedTemporaryFile(mode='ab', dir='.') as f:
        f.file.close()
        fw = h5py.File(f.name, 'w')
        arrays = ('data',) + tuple(aux_arrays)
        disk_shape = (n_cols, n_rows) if transpose else (n_rows, n_cols)
        for name in arrays:
            y = fw.create_dataset(name, shape=disk_shape, dtype=dtype, chunks=chunks)
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
    blocks = list(map_source.iter_blocks(block_size, reverse=True))
    assert_true((data[electrode_channels][:, block_size:] == blocks[0]).all(), 'first rev block wrong')
    assert_true((data[electrode_channels][:, :block_size] == blocks[1]).all(), 'second rev block wrong')


def test_iter_overlap():
    f, filename = _create_hdf5(n_cols=100)
    data = f['data'][:]
    block_size = 20
    overlap = 10
    map_source = MappedSource(f, 1, 'data')
    blocks = list(map_source.iter_blocks(block_size, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[1]).all(), 'second block wrong')
    # last block is a partial, starting at index 90
    assert_true((data[:, -10:] == blocks[-1]).all(), 'last block wrong')
    blocks = list(map_source.iter_blocks(block_size, reverse=True, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[-1]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[-2]).all(), 'second block wrong')
    assert_true((data[:, -10:] == blocks[0]).all(), 'last block wrong')


def test_iterT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:].T
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource(f, 1, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose')


def _clean_up_hdf_files(temp_files):
    for f in temp_files:
        name = f.filename
        f.close()
        if os.path.exists(name):
            os.unlink(name)


def test_basic_mirror():
    try:
        f, filename = _create_hdf5(n_rows=25, n_cols=500)
        electrode_channels = [2, 4, 6, 8]
        map_source = MappedSource(f, 40, 'data', electrode_channels=electrode_channels)
        temp_files = []
        clone1 = map_source.mirror(new_rate=None, writeable=True, mapped=True, channel_compatible=False, filename='foo.h5')
        temp_files.append(clone1._source_file)
        assert_true(clone1.data_shape == (len(electrode_channels), 500), 'wrong # of channels')
        assert_true(clone1.writeable, 'Should be writeable')
        assert_true(isinstance(clone1, MappedSource), 'Clone is not a MappedSource')
        clone2 = map_source.mirror(new_rate=None, mapped=False, channel_compatible=False)
        assert_true(isinstance(clone2, PlainArraySource), 'Not-mapped file should be PlainArraySource')
    except Exception as e:
        raise e
    finally:
        _clean_up_hdf_files(temp_files)


def test_mirror_modes():
    try:
        f, filename = _create_hdf5(n_rows=25, n_cols=500)
        electrode_channels = [2, 4, 6, 8]
        map_source = MappedSource(f, 40, 'data', electrode_channels=electrode_channels)
        temp_files = []
        clone1 = map_source.mirror(writeable=True, mapped=True, channel_compatible=False)
        temp_files.append(clone1._source_file)
        assert_true(clone1.data_shape == (len(electrode_channels), 500), 'wrong # of samples')
        clone2 = map_source.mirror(writeable=True, mapped=True, channel_compatible=True)
        temp_files.append(clone2._source_file)
        assert_true(clone2._electrode_array.shape == (25, 500), 'wrong # of channels for channel-compat')
        f, filename = _create_hdf5(n_rows=25, n_cols=500, transpose=True)
        map_source = MappedSource(f, 40, 'data', electrode_channels=electrode_channels, transpose=True)
        clone3 = map_source.mirror(mapped=True, channel_compatible=True)
        temp_files.append(clone3._source_file)
        assert_true(clone3._electrode_array.shape == (25, 500), 'mapped mirror did not reverse the source transpose')
    except Exception as e:
        raise e
    finally:
        _clean_up_hdf_files(temp_files)

