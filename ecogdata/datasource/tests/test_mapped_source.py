import os
import pytest
import numpy as np
from ecogdata.datasource.array_abstractions import HDF5Buffer
from ecogdata.datasource.memmap import MappedSource, MemoryBlowOutError
from ecogdata.datasource.basic import PlainArraySource

from .test_array_abstractions import _create_hdf5, _create_buffer, _create_binder


def test_basic_construction():
    aux_arrays = ('test1', 'test2')
    buffer, data = _create_buffer(aux_arrays=aux_arrays)
    # hacky way to get h5py.File object...
    hdf = buffer.file_array.file
    aligned = dict([(k, HDF5Buffer(hdf[k])) for k in aux_arrays])
    map_source = MappedSource(buffer, aligned_arrays=aligned)
    shape = data.shape
    assert map_source.shape == shape, 'Shape wrong'
    assert map_source.binary_channel_mask.sum() == shape[0], 'Wrong number of active channels'
    for field in aux_arrays:
        assert hasattr(map_source, field), 'Aux field {} not preserved'.format(field)
        assert getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field)
    # repeat for transpose
    map_source = MappedSource(buffer, aligned_arrays=aligned, transpose=True)
    assert map_source.shape == shape[::-1], 'Shape wrong in transpose'
    assert map_source.binary_channel_mask.sum() == shape[1], 'Wrong number of active channels in transpose'


def test_basic_construction_binder():
    buffer, data = _create_binder(axis=1)
    map_source = MappedSource(buffer)
    shape = data.shape
    assert map_source.shape == shape, 'Shape wrong'
    assert map_source.binary_channel_mask.sum() == shape[0], 'Wrong number of active channels'
    # repeat for transpose
    map_source = MappedSource(buffer, transpose=True)
    assert map_source.shape == shape[::-1], 'Shape wrong in transpose'
    assert map_source.binary_channel_mask.sum() == shape[1], 'Wrong number of active channels in transpose'


def test_construction_from_single_source():
    aux_arrays = ('test1', 'test2')
    f = _create_hdf5(aux_arrays=aux_arrays)
    shape = f['data'].shape
    map_source = MappedSource.from_hdf_sources(f, 'data', aligned_arrays=aux_arrays)
    assert map_source.shape == shape, 'Shape wrong'
    assert map_source.binary_channel_mask.sum() == shape[0], 'Wrong number of active channels'
    for field in aux_arrays:
        assert hasattr(map_source, field), 'Aux field {} not preserved'.format(field)
        assert getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field)
    # repeat for transpose
    map_source = MappedSource.from_hdf_sources(f, 'data', aligned_arrays=aux_arrays, transpose=True)
    assert map_source.shape == shape[::-1], 'Shape wrong in transpose'
    assert map_source.binary_channel_mask.sum() == shape[1], 'Wrong number of active channels in transpose'


def test_construction_from_sources():
    aux_arrays = ('test1', 'test2')
    files = [_create_hdf5(aux_arrays=aux_arrays) for i in range(3)]
    shape = files[0]['data'].shape
    shape = (shape[0], 3 * shape[1])
    map_source = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays)
    assert map_source.shape == shape, 'Shape wrong'
    assert map_source.binary_channel_mask.sum() == shape[0], 'Wrong number of active channels'
    for field in aux_arrays:
        assert hasattr(map_source, field), 'Aux field {} not preserved'.format(field)
        assert getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field)
    # repeat for transpose: now sources are stacked on axis=0, but the resulting shape is transposed per vector
    # timeseries convention (channels X samples)
    shape = files[0]['data'].shape
    shape = (shape[0] * 3, shape[1])
    map_source = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays, transpose=True)
    assert map_source.shape == shape[::-1], 'Shape wrong in transpose'
    assert map_source.binary_channel_mask.sum() == shape[1], 'Wrong number of active channels in transpose'
    for field in aux_arrays:
        assert hasattr(map_source, field), 'Aux field {} not preserved'.format(field)
        assert getattr(map_source, field).shape[0] == shape[0], 'aligned field {} wrong length'.format(field)


def test_joining():
    aux_arrays = ('test1', 'test2')
    files = [_create_hdf5(aux_arrays=aux_arrays) for i in range(3)]
    map_source1 = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays)
    next_file = _create_hdf5(aux_arrays=aux_arrays)
    map_source2 = MappedSource.from_hdf_sources(next_file, 'data', aligned_arrays=aux_arrays)
    full_map = map_source1.join(map_source2)
    assert full_map.shape == (len(map_source1), map_source1.shape[1] + map_source2.shape[1]), 'binder to buffer appending failed'
    full_map = map_source2.join(map_source1)
    assert full_map.shape == (len(map_source1), map_source1.shape[1] + map_source2.shape[1]), 'buffer to binder appending failed'


def test_joiningT():
    aux_arrays = ('test1', 'test2')
    files = [_create_hdf5(aux_arrays=aux_arrays) for i in range(3)]
    map_source1 = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays, transpose=True)
    next_file = _create_hdf5(aux_arrays=aux_arrays)
    map_source2 = MappedSource.from_hdf_sources(next_file, 'data', aligned_arrays=aux_arrays, transpose=True)
    full_map = map_source1.join(map_source2)
    assert full_map.shape == (len(map_source1), map_source1.shape[1] + map_source2.shape[1]), 'binder to buffer appending failed'
    full_map = map_source2.join(map_source1)
    assert full_map.shape == (len(map_source1), map_source1.shape[1] + map_source2.shape[1]), 'buffer to binder appending failed'


def test_direct_mapped():
    f = _create_hdf5()
    mapped_source = MappedSource.from_hdf_sources(f, 'data')
    assert mapped_source.is_direct_map, 'direct map should be true'
    mapped_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=range(4))
    assert not mapped_source.is_direct_map, 'direct map should be false'
    # for transposed disk arrays
    f = _create_hdf5(transpose=True)
    mapped_source = MappedSource.from_hdf_sources(f, 'data', transpose=True)
    assert mapped_source.is_direct_map, 'direct map should be true'
    mapped_source = MappedSource.from_hdf_sources(f, 'data', transpose=True, electrode_channels=range(4))
    assert not mapped_source.is_direct_map, 'direct map should be false'


def test_scaling():
    f = _create_hdf5()
    float_data = f['data'][:, 500:1000].astype('d')
    map_source = MappedSource.from_hdf_sources(f, 'data', units_scale=2.0)
    assert np.all(map_source[:, 500:1000] == float_data * 2).all(), 'scalar scaling wrong'
    map_source = MappedSource.from_hdf_sources(f, 'data', units_scale=(-100, 2.0))
    assert np.all(map_source[:, 500:1000] == (float_data - 100) * 2).all(), 'affine scaling wrong'


def test_electrode_subset():
    f = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    data = f['data'][:, :][electrode_channels]
    assert np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed'


def test_electrode_subsetT():
    f = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    data = f['data'][:, :][:, electrode_channels].T
    assert np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed in transpose'


def test_channel_map():
    f = _create_hdf5()
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    map_source.set_channel_mask(binary_mask)
    assert (map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong'
    data = f['data'][:, :][electrode_channels, :]
    assert np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed'
    # unmask
    map_source.set_channel_mask(None)
    binary_mask[:] = True
    assert (map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong'
    data = f['data'][:, :][electrode_channels, :]
    assert np.all(data[:, 100:200] == map_source[:, 100:200]), 'channel masking failed'


def test_channel_mapT():
    f = _create_hdf5(transpose=True)
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    map_source.set_channel_mask(binary_mask)
    assert (map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong in transpose'
    data = f['data'][:, :][:, electrode_channels].T
    assert np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed in transpose'
    # unmask
    map_source.set_channel_mask(None)
    binary_mask[:] = True
    assert (map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong'
    data = f['data'][:, :][:, electrode_channels].T
    assert np.all(data[:, 100:200] == map_source[:, 100:200]), 'channel masking failed'


def test_channel_slicing():
    f = _create_hdf5()
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, units_scale=5.0)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert isinstance(first_channels, MappedSource), 'slice did not return new map'
    assert np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped'
    first_channels = map_source[:3]
    assert isinstance(first_channels, np.ndarray), 'slice-as-array failed'
    assert np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data'


def test_channel_slicingT():
    f = _create_hdf5(transpose=True)
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True, units_scale=5.0)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert isinstance(first_channels, MappedSource), 'slice did not return new map'
    assert np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped'
    first_channels = map_source[:3]
    assert isinstance(first_channels, np.ndarray), 'slice-as-array failed'
    assert np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data'


def test_channel_slicing_with_mask():
    f = _create_hdf5()
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    mask = map_source.binary_channel_mask
    mask[:5] = False
    map_source.set_channel_mask(mask)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert isinstance(first_channels, MappedSource), 'slice did not return new map'
    assert np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped'
    first_channels = map_source[:3]
    assert isinstance(first_channels, np.ndarray), 'slice-as-array failed'
    assert np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data'


def test_big_slicing_exception():
    import ecogdata.expconfig.global_config as globalconfig
    f = _create_hdf5()
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data')
    with pytest.raises(MemoryBlowOutError):
        try:
            map_source[:, :]
        except Exception as e:
            raise e
        finally:
            globalconfig.OVERRIDE.pop('memory_limit')


def test_big_slicing_allowed():
    import ecogdata.expconfig.global_config as globalconfig
    f = _create_hdf5()
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data')
    try:
        with map_source.big_slices(True):
            _ = map_source[:, :]
    except MemoryBlowOutError as e:
        assert False, 'Big slicing context failed'
    finally:
        globalconfig.OVERRIDE.pop('memory_limit')


def test_big_slicing_allowed_always():
    import ecogdata.expconfig.global_config as globalconfig
    f = _create_hdf5()
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data', raise_on_big_slice=False)
    try:
        _ = map_source[:, :]
    except MemoryBlowOutError as e:
        assert False, 'Big slicing context failed'
    finally:
        globalconfig.OVERRIDE.pop('memory_limit')


def test_write():
    f = _create_hdf5()
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    shp = map_source.shape
    rand_pattern = np.random.randint(0, 100, size=(2, shp[1]))
    map_source[:2] = rand_pattern
    # use full-slice syntax to get data
    assert np.array_equal(map_source[:2, :], rand_pattern), 'write failed (map subset)'
    map_source.set_channel_mask(binary_mask)
    # write again
    map_source[:2] = rand_pattern
    assert np.array_equal(map_source[:2, :], rand_pattern), 'write failed (map subset and mask)'


def test_write_to_binder():
    files = [_create_hdf5() for i in range(3)]
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(files, 'data', electrode_channels=electrode_channels)
    # make a write that spans buffers
    single_length = files[0]['data'].shape[1]
    rand_pattern = np.random.randint(0, 100, size=(2, 205))
    sl = np.s_[:2, single_length - 100: single_length + 105]
    map_source[sl] = rand_pattern
    # use full-slice syntax to get data
    assert np.array_equal(map_source[sl], rand_pattern), 'write failed to binder (map subset)'
    map_source.set_channel_mask(binary_mask)
    # write again
    map_source[sl] = rand_pattern
    assert np.array_equal(map_source[sl], rand_pattern), 'write failed to binder (map subset and mask)'


def test_iter():
    f = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:]
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    blocks = list(map_source.iter_blocks(block_size))
    assert (data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong'
    assert (data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong'
    blocks = list(map_source.iter_blocks(block_size, reverse=True))
    assert (data[electrode_channels][:, block_size:][:, ::-1] == blocks[0]).all(), 'first rev block wrong'
    assert (data[electrode_channels][:, :block_size][:, ::-1] == blocks[1]).all(), 'second rev block wrong'


def test_iter_binder():
    files = [_create_hdf5(n_cols=100) for i in range(3)]
    electrode_channels = [2, 4, 6, 8]
    data = np.concatenate([f['data'][:] for f in files], axis=1)
    block_size = data.shape[1] // 2 + 20
    map_source = MappedSource.from_hdf_sources(files, 'data', electrode_channels=electrode_channels)
    blocks = list(map_source.iter_blocks(block_size))
    assert (data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong'
    assert (data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong'
    blocks = list(map_source.iter_blocks(block_size, reverse=True))
    assert (data[electrode_channels][:, block_size:][:, ::-1] == blocks[0]).all(), 'first rev block wrong'
    assert (data[electrode_channels][:, :block_size][:, ::-1] == blocks[1]).all(), 'second rev block wrong'


def test_iter_overlap():
    f = _create_hdf5(n_cols=100)
    data = f['data'][:]
    block_size = 20
    overlap = 10
    map_source = MappedSource.from_hdf_sources(f, 'data')
    blocks = list(map_source.iter_blocks(block_size, overlap=overlap))
    assert (data[:, :block_size] == blocks[0]).all(), 'first block wrong'
    assert (data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[1]).all(), 'second block wrong'
    # last block is a partial, starting at index 90
    assert (data[:, -10:] == blocks[-1]).all(), 'last block wrong'
    blocks = list(map_source.iter_blocks(block_size, reverse=True, overlap=overlap))
    assert (data[:, :block_size] == blocks[-1][:, ::-1]).all(), 'first block wrong'
    assert (data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[-2][:, ::-1]).all(), 'second block wrong'
    assert (data[:, -10:] == blocks[0][:, ::-1]).all(), 'last block wrong'


def test_iter_overlap_binder():
    files = [_create_hdf5(n_cols=100) for i in range(3)]
    data = np.concatenate([f['data'][:] for f in files], axis=1)
    block_size = 20
    overlap = 10
    map_source = MappedSource.from_hdf_sources(files, 'data')
    blocks = list(map_source.iter_blocks(block_size, overlap=overlap))
    assert (data[:, :block_size] == blocks[0]).all(), 'first block wrong'
    assert (data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[1]).all(), 'second block wrong'
    # last block is a partial, starting at index 90
    assert (data[:, -10:] == blocks[-1]).all(), 'last block wrong'
    blocks = list(map_source.iter_blocks(block_size, reverse=True, overlap=overlap))
    assert (data[:, :block_size] == blocks[-1][:, ::-1]).all(), 'first block wrong'
    assert (data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[-2][:, ::-1]).all(), 'second block wrong'
    assert (data[:, -10:] == blocks[0][:, ::-1]).all(), 'last block wrong'


def test_iterT():
    f = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:].T
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert (data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose'
    assert (data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose'


def test_iterT_binder():
    files = [_create_hdf5(transpose=True, n_cols=100) for i in range(3)]
    data = np.concatenate([f['data'][:] for f in files], axis=0).T
    electrode_channels = [2, 4, 6, 8]
    block_size = data.shape[1] // 2 + 20
    map_source = MappedSource.from_hdf_sources(files, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert (data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose'
    assert (data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose'


def test_iter_channels():
    f = _create_hdf5(n_rows=10, n_cols=100)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=[2, 4, 6, 8, 9])
    data = f['data'][:]
    channel_blocks = []
    for chans in map_source.iter_channels(chans_per_block=2):
        channel_blocks.append(chans)
    for n, chans in enumerate(np.array_split(data[[2, 4, 6, 8, 9]], 3)):
        assert np.array_equal(channel_blocks[n], chans), 'channel block {} not equal'.format(n)


def test_iter_channelsT():
    f = _create_hdf5(n_rows=10, n_cols=100, transpose=True)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=[2, 4, 6, 8, 9], transpose=True)
    data = f['data'][:].T
    channel_blocks = []
    for chans in map_source.iter_channels(chans_per_block=2):
        channel_blocks.append(chans)
    for n, chans in enumerate(np.array_split(data[[2, 4, 6, 8, 9]], 3)):
        assert np.array_equal(channel_blocks[n], chans), 'channel block {} not equal'.format(n)


def _clean_up_hdf_files(temp_files):
    for f in temp_files:
        name = f.filename
        f.close()
        if os.path.exists(name):
            os.unlink(name)


def test_basic_mirror():
    try:
        f = _create_hdf5(n_rows=25, n_cols=500)
        electrode_channels = [2, 4, 6, 8]
        map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
        temp_files = []
        clone1 = map_source.mirror(new_rate_ratio=None, writeable=True, mapped=True, channel_compatible=False,
                                   filename='foo.h5')
        temp_files.append(clone1.data_buffer._array.file)
        assert clone1.shape == (len(electrode_channels), 500), 'wrong # of channels'
        assert clone1.writeable, 'Should be writeable'
        assert isinstance(clone1, MappedSource), 'Clone is not a MappedSource'
        clone2 = map_source.mirror(new_rate_ratio=None, mapped=False, channel_compatible=False)
        assert isinstance(clone2, PlainArraySource), 'Not-mapped file should be PlainArraySource'
    except Exception as e:
        raise e
    finally:
        _clean_up_hdf_files(temp_files)


def test_mirror_modes():
    f = _create_hdf5(n_rows=25, n_cols=500)
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    clone1 = map_source.mirror(writeable=True, mapped=True, channel_compatible=False)
    assert clone1.shape == (len(electrode_channels), 500), 'wrong # of samples'
    clone2 = map_source.mirror(writeable=True, mapped=True, channel_compatible=True)
    assert clone2.data_buffer.shape == (25, 500), 'wrong # of channels for channel-compat'
    f = _create_hdf5(n_rows=25, n_cols=500, transpose=True)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    clone3 = map_source.mirror(mapped=True, channel_compatible=True)
    assert clone3.data_buffer.shape == (25, 500), 'mapped mirror did not reverse the source transpose'


