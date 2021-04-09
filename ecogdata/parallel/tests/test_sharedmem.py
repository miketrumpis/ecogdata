import multiprocess as mp
import numpy as np
from ecogdata.parallel.sharedmem import ForkSharedmemManager, SpawnSharedmemManager, SharedmemTool
from . import skip_win


def init_shared_array(shared_manager: SharedmemTool):
    global shared_array
    with shared_manager.get_ndarray() as arr:
        shared_array = arr


def write_to_array(i):
    shared_array[i, :] = i


@skip_win
def test_forked_sharedmem():
    array = ForkSharedmemManager.shared_ndarray((10, 3), 'd')
    mm = ForkSharedmemManager(array)
    ctx = mp.get_context('fork')

    with ctx.Pool(processes=max(1, mp.cpu_count() - 3), initializer=init_shared_array, initargs=(mm,)) as p:
        p.map(write_to_array, range(len(array)))
    p.close()
    p.join()

    expected_result = np.repeat(np.arange(array.shape[0]), array.shape[1]).reshape(array.shape)
    assert np.all(expected_result == array), 'Forked shared-memory writing failed'


def test_spawned_sharedmem():
    array = SpawnSharedmemManager.shared_ndarray((10, 3), 'd')
    mm = SpawnSharedmemManager(array)
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=max(1, mp.cpu_count() - 3), initializer=init_shared_array, initargs=(mm,)) as p:
        p.map(write_to_array, range(len(array)))
    p.close()
    p.join()

    expected_result = np.repeat(np.arange(array.shape[0]), array.shape[1]).reshape(array.shape)
    assert np.all(expected_result == array), 'Forked shared-memory writing failed'


