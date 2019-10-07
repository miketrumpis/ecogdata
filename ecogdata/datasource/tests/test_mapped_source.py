import os
from nose.tools import assert_true, assert_equal, raises
import numpy as np
from ecogdata.datasource.array_abstractions import HDF5Buffer, BufferBinder
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
    assert_equal(map_source.shape, shape, 'Shape wrong')
    assert_equal(map_source.binary_channel_mask.sum(), shape[0], 'Wrong number of active channels')
    for field in aux_arrays:
        assert_true(hasattr(map_source, field), 'Aux field {} not preserved'.format(field))
        assert_true(getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field))
    # repeat for transpose
    map_source = MappedSource(buffer, aligned_arrays=aligned, transpose=True)
    assert_equal(map_source.shape, shape[::-1], 'Shape wrong in transpose')
    assert_equal(map_source.binary_channel_mask.sum(), shape[1], 'Wrong number of active channels in transpose')


def test_basic_construction_binder():
    buffer, data = _create_binder(axis=1)
    map_source = MappedSource(buffer)
    shape = data.shape
    assert_equal(map_source.shape, shape, 'Shape wrong')
    assert_equal(map_source.binary_channel_mask.sum(), shape[0], 'Wrong number of active channels')
    # repeat for transpose
    map_source = MappedSource(buffer, transpose=True)
    assert_equal(map_source.shape, shape[::-1], 'Shape wrong in transpose')
    assert_equal(map_source.binary_channel_mask.sum(), shape[1], 'Wrong number of active channels in transpose')


def test_construction_from_single_source():
    aux_arrays = ('test1', 'test2')
    f = _create_hdf5(aux_arrays=aux_arrays)[0]
    shape = f['data'].shape
    map_source = MappedSource.from_hdf_sources(f, 'data', aligned_arrays=aux_arrays)
    assert_equal(map_source.shape, shape, 'Shape wrong')
    assert_equal(map_source.binary_channel_mask.sum(), shape[0], 'Wrong number of active channels')
    for field in aux_arrays:
        assert_true(hasattr(map_source, field), 'Aux field {} not preserved'.format(field))
        assert_true(getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field))
    # repeat for transpose
    map_source = MappedSource.from_hdf_sources(f, 'data', aligned_arrays=aux_arrays, transpose=True)
    assert_equal(map_source.shape, shape[::-1], 'Shape wrong in transpose')
    assert_equal(map_source.binary_channel_mask.sum(), shape[1], 'Wrong number of active channels in transpose')


def test_construction_from_sources():
    aux_arrays = ('test1', 'test2')
    files = [_create_hdf5(aux_arrays=aux_arrays)[0] for i in range(3)]
    shape = files[0]['data'].shape
    shape = (shape[0], 3 * shape[1])
    map_source = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays)
    assert_equal(map_source.shape, shape, 'Shape wrong')
    assert_equal(map_source.binary_channel_mask.sum(), shape[0], 'Wrong number of active channels')
    for field in aux_arrays:
        assert_true(hasattr(map_source, field), 'Aux field {} not preserved'.format(field))
        assert_true(getattr(map_source, field).shape[1] == shape[1], 'aligned field {} wrong length'.format(field))
    # repeat for transpose: now sources are stacked on axis=0, but the resulting shape is transposed per vector
    # timeseries convention (channels X samples)
    shape = files[0]['data'].shape
    shape = (shape[0] * 3, shape[1])
    map_source = MappedSource.from_hdf_sources(files, 'data', aligned_arrays=aux_arrays, transpose=True)
    assert_equal(map_source.shape, shape[::-1], 'Shape wrong in transpose')
    assert_equal(map_source.binary_channel_mask.sum(), shape[1], 'Wrong number of active channels in transpose')
    for field in aux_arrays:
        assert_true(hasattr(map_source, field), 'Aux field {} not preserved'.format(field))
        assert_true(getattr(map_source, field).shape[0] == shape[0], 'aligned field {} wrong length'.format(field))


def test_direct_mapped():
    f = _create_hdf5()[0]
    mapped_source = MappedSource.from_hdf_sources(f, 'data')
    assert_true(mapped_source.is_direct_map, 'direct map should be true')
    mapped_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=range(4))
    assert_true(not mapped_source.is_direct_map, 'direct map should be false')
    # for transposed disk arrays
    f = _create_hdf5(transpose=True)[0]
    mapped_source = MappedSource.from_hdf_sources(f, 'data', transpose=True)
    assert_true(mapped_source.is_direct_map, 'direct map should be true')
    mapped_source = MappedSource.from_hdf_sources(f, 'data', transpose=True, electrode_channels=range(4))
    assert_true(not mapped_source.is_direct_map, 'direct map should be false')


def test_scaling():
    f, filename = _create_hdf5()
    float_data = f['data'][:, 500:1000].astype('d')
    map_source = MappedSource.from_hdf_sources(f, 'data', units_scale=2.0)
    assert_true(np.all(map_source[:, 500:1000] == float_data * 2).all(), 'scalar scaling wrong')
    map_source = MappedSource.from_hdf_sources(f, 'data', units_scale=(-100, 2.0))
    assert_true(np.all(map_source[:, 500:1000] == (float_data - 100) * 2).all(), 'affine scaling wrong')


def test_electrode_subset():
    f, filename = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    data = f['data'][:, :][electrode_channels]
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed')


def test_electrode_subsetT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    data = f['data'][:, :][:, electrode_channels].T
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'electrode subset failed in transpose')


def test_channel_map():
    f, filename = _create_hdf5()
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    map_source.set_channel_mask(binary_mask)
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong')
    data = f['data'][:, :][electrode_channels, :]
    assert_true(np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed')
    # unmask
    map_source.set_channel_mask(None)
    binary_mask[:] = True
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong')
    data = f['data'][:, :][electrode_channels, :]
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'channel masking failed')


def test_channel_mapT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    map_source.set_channel_mask(binary_mask)
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong in transpose')
    data = f['data'][:, :][:, electrode_channels].T
    assert_true(np.all(data[5:, 100:200] == map_source[:, 100:200]), 'channel masking failed in transpose')
    # unmask
    map_source.set_channel_mask(None)
    binary_mask[:] = True
    assert_true((map_source.binary_channel_mask == binary_mask).all(), 'binary mask wrong')
    data = f['data'][:, :][:, electrode_channels].T
    assert_true(np.all(data[:, 100:200] == map_source[:, 100:200]), 'channel masking failed')


def test_channel_slicing():
    f, filename = _create_hdf5()
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, units_scale=5.0)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert_true(isinstance(first_channels, MappedSource), 'slice did not return new map')
    assert_true(np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped')
    first_channels = map_source[:3]
    assert_true(isinstance(first_channels, np.ndarray), 'slice-as-array failed')
    assert_true(np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data')


def test_channel_slicingT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True, units_scale=5.0)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert_true(isinstance(first_channels, MappedSource), 'slice did not return new map')
    assert_true(np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped')
    first_channels = map_source[:3]
    assert_true(isinstance(first_channels, np.ndarray), 'slice-as-array failed')
    assert_true(np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data')


def test_channel_slicing_with_mask():
    f, filename = _create_hdf5()
    electrode_channels = list(range(6, 17))
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    mask = map_source.binary_channel_mask
    mask[:5] = False
    map_source.set_channel_mask(mask)
    data_first_channels = map_source[:3, :]
    with map_source.channels_are_maps(True):
        first_channels = map_source[:3]
    assert_true(isinstance(first_channels, MappedSource), 'slice did not return new map')
    assert_true(np.array_equal(data_first_channels, first_channels[:, :]), 'new map data mis-mapped')
    first_channels = map_source[:3]
    assert_true(isinstance(first_channels, np.ndarray), 'slice-as-array failed')
    assert_true(np.array_equal(data_first_channels, first_channels), 'slice-as-array wrong data')


@raises(MemoryBlowOutError)
def test_big_slicing_exception():
    import ecogdata.expconfig._globalconfig as globalconfig
    f = _create_hdf5()[0]
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data')
    try:
        map_source[:, :]
    except Exception as e:
        raise e
    finally:
        globalconfig.OVERRIDE.pop('memory_limit')


def test_big_slicing_allowed():
    import ecogdata.expconfig._globalconfig as globalconfig
    f = _create_hdf5()[0]
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data')
    try:
        with map_source.big_slices(True):
            _ = map_source[:, :]
    except MemoryBlowOutError as e:
        assert_true(False, 'Big slicing context failed')
    finally:
        globalconfig.OVERRIDE.pop('memory_limit')


def test_big_slicing_allowed_always():
    import ecogdata.expconfig._globalconfig as globalconfig
    f = _create_hdf5()[0]
    data = f['data']
    globalconfig.OVERRIDE['memory_limit'] = data.size * data.dtype.itemsize / 2.0
    map_source = MappedSource.from_hdf_sources(f, 'data', raise_on_big_slice=False)
    try:
        _ = map_source[:, :]
    except MemoryBlowOutError as e:
        assert_true(False, 'Big slicing context failed')
    finally:
        globalconfig.OVERRIDE.pop('memory_limit')


def test_write():
    f, filename = _create_hdf5()
    electrode_channels = list(range(10))
    binary_mask = np.ones(10, '?')
    binary_mask[:5] = False
    # so channels 5, 6, 7, 8, 9 should be active
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    shp = map_source.shape
    rand_pattern = np.random.randint(0, 100, size=(2, shp[1]))
    map_source[:2] = rand_pattern
    # use full-slice syntax to get data
    assert_true(np.array_equal(map_source[:2, :], rand_pattern), 'write failed (map subset)')
    map_source.set_channel_mask(binary_mask)
    # write again
    map_source[:2] = rand_pattern
    assert_true(np.array_equal(map_source[:2, :], rand_pattern), 'write failed (map subset and mask)')


def test_write_to_binder():
    files = [_create_hdf5()[0] for i in range(3)]
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
    assert_true(np.array_equal(map_source[sl], rand_pattern), 'write failed to binder (map subset)')
    map_source.set_channel_mask(binary_mask)
    # write again
    map_source[sl] = rand_pattern
    assert_true(np.array_equal(map_source[sl], rand_pattern), 'write failed to binder (map subset and mask)')


def test_iter():
    f, filename = _create_hdf5()
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:]
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong')
    blocks = list(map_source.iter_blocks(block_size, reverse=True))
    assert_true((data[electrode_channels][:, block_size:][:, ::-1] == blocks[0]).all(), 'first rev block wrong')
    assert_true((data[electrode_channels][:, :block_size][:, ::-1] == blocks[1]).all(), 'second rev block wrong')


def test_iter_binder():
    files = [_create_hdf5(n_cols=100)[0] for i in range(3)]
    electrode_channels = [2, 4, 6, 8]
    data = np.concatenate([f['data'][:] for f in files], axis=1)
    block_size = data.shape[1] // 2 + 20
    map_source = MappedSource.from_hdf_sources(files, 'data', electrode_channels=electrode_channels)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong')
    blocks = list(map_source.iter_blocks(block_size, reverse=True))
    assert_true((data[electrode_channels][:, block_size:][:, ::-1] == blocks[0]).all(), 'first rev block wrong')
    assert_true((data[electrode_channels][:, :block_size][:, ::-1] == blocks[1]).all(), 'second rev block wrong')


def test_iter_overlap():
    f, filename = _create_hdf5(n_cols=100)
    data = f['data'][:]
    block_size = 20
    overlap = 10
    map_source = MappedSource.from_hdf_sources(f, 'data')
    blocks = list(map_source.iter_blocks(block_size, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[1]).all(), 'second block wrong')
    # last block is a partial, starting at index 90
    assert_true((data[:, -10:] == blocks[-1]).all(), 'last block wrong')
    blocks = list(map_source.iter_blocks(block_size, reverse=True, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[-1][:, ::-1]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[-2][:, ::-1]).all(),
                'second block wrong')
    assert_true((data[:, -10:] == blocks[0][:, ::-1]).all(), 'last block wrong')


def test_iter_overlap_binder():
    files = [_create_hdf5(n_cols=100)[0] for i in range(3)]
    data = np.concatenate([f['data'][:] for f in files], axis=1)
    block_size = 20
    overlap = 10
    map_source = MappedSource.from_hdf_sources(files, 'data')
    blocks = list(map_source.iter_blocks(block_size, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[0]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[1]).all(), 'second block wrong')
    # last block is a partial, starting at index 90
    assert_true((data[:, -10:] == blocks[-1]).all(), 'last block wrong')
    blocks = list(map_source.iter_blocks(block_size, reverse=True, overlap=overlap))
    assert_true((data[:, :block_size] == blocks[-1][:, ::-1]).all(), 'first block wrong')
    assert_true((data[:, (block_size - overlap):(2 * block_size - overlap)] == blocks[-2][:, ::-1]).all(),
                'second block wrong')
    assert_true((data[:, -10:] == blocks[0][:, ::-1]).all(), 'last block wrong')


def test_iterT():
    f, filename = _create_hdf5(transpose=True)
    electrode_channels = [2, 4, 6, 8]
    data = f['data'][:].T
    block_size = data.shape[1] // 2 + 100
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose')


def test_iterT_binder():
    files = [_create_hdf5(transpose=True, n_cols=100)[0] for i in range(3)]
    data = np.concatenate([f['data'][:] for f in files], axis=0).T
    electrode_channels = [2, 4, 6, 8]
    block_size = data.shape[1] // 2 + 20
    map_source = MappedSource.from_hdf_sources(files, 'data', electrode_channels=electrode_channels, transpose=True)
    blocks = list(map_source.iter_blocks(block_size))
    assert_true((data[electrode_channels][:, :block_size] == blocks[0]).all(), 'first block wrong in transpose')
    assert_true((data[electrode_channels][:, block_size:] == blocks[1]).all(), 'second block wrong in transpose')


def test_iter_channels():
    f, filename = _create_hdf5(n_rows=10, n_cols=100)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=[2, 4, 6, 8, 9])
    data = f['data'][:]
    channel_blocks = []
    for chans in map_source.iter_channels(chans_per_block=2):
        channel_blocks.append(chans)
    for n, chans in enumerate(np.array_split(data[[2, 4, 6, 8, 9]], 3)):
        assert_true(np.array_equal(channel_blocks[n], chans), 'channel block {} not equal'.format(n))


def test_iter_channelsT():
    f, filename = _create_hdf5(n_rows=10, n_cols=100, transpose=True)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=[2, 4, 6, 8, 9], transpose=True)
    data = f['data'][:].T
    channel_blocks = []
    for chans in map_source.iter_channels(chans_per_block=2):
        channel_blocks.append(chans)
    for n, chans in enumerate(np.array_split(data[[2, 4, 6, 8, 9]], 3)):
        assert_true(np.array_equal(channel_blocks[n], chans), 'channel block {} not equal'.format(n))


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
        map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
        temp_files = []
        clone1 = map_source.mirror(new_rate_ratio=None, writeable=True, mapped=True, channel_compatible=False,
                                   filename='foo.h5')
        temp_files.append(clone1.data_buffer._array.file)
        assert_true(clone1.shape == (len(electrode_channels), 500), 'wrong # of channels')
        assert_true(clone1.writeable, 'Should be writeable')
        assert_true(isinstance(clone1, MappedSource), 'Clone is not a MappedSource')
        clone2 = map_source.mirror(new_rate_ratio=None, mapped=False, channel_compatible=False)
        assert_true(isinstance(clone2, PlainArraySource), 'Not-mapped file should be PlainArraySource')
    except Exception as e:
        raise e
    finally:
        _clean_up_hdf_files(temp_files)


def test_mirror_modes():
    f, filename = _create_hdf5(n_rows=25, n_cols=500)
    electrode_channels = [2, 4, 6, 8]
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels)
    clone1 = map_source.mirror(writeable=True, mapped=True, channel_compatible=False)
    assert_true(clone1.shape == (len(electrode_channels), 500), 'wrong # of samples')
    clone2 = map_source.mirror(writeable=True, mapped=True, channel_compatible=True)
    assert_true(clone2.data_buffer.shape == (25, 500), 'wrong # of channels for channel-compat')
    f, filename = _create_hdf5(n_rows=25, n_cols=500, transpose=True)
    map_source = MappedSource.from_hdf_sources(f, 'data', electrode_channels=electrode_channels, transpose=True)
    clone3 = map_source.mirror(mapped=True, channel_compatible=True)
    assert_true(clone3.data_buffer.shape == (25, 500), 'mapped mirror did not reverse the source transpose')


