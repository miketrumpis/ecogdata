import sys
from nose.tools import assert_true
from nose import SkipTest
import random
import numpy as np

import ecogdata.parallel.mproc as mp
from ecogdata.parallel.array_split import shared_ndarray, shared_copy, split_at, divy_slices, ForkSharedmemManager, \
    SpawnSharedmemManager, SharedmemTool

# This should UNDO the Arena.__init__ monkey patching that might make heaps unpickle-able under spawning
from importlib import reload
import multiprocessing.heap as heap
reload(heap)


def check_platform():
    if sys.platform == 'win32':
        raise SkipTest('Cannot test forked processses on Windows')


def init_shared_array(shared_manager: SharedmemTool):
    global shared_array
    with shared_manager.get_ndarray() as arr:
        shared_array = arr


def write_to_array(i):
    shared_array[i, :] = i


def sum_columns(x):
    return np.sum(x, axis=1)


def sum_columns_and_write(x, out):
    out[:] = np.sum(x, axis=1)


def test_divy_slices():
    def eval_slices(dim, divs, slices):
        # evaluate the right-ness of sub-slices
        correct_length = divs == len(slices)
        start = 0
        correct_index = True
        for sl in slices:
            correct_index = correct_index and (start == sl.start)
            start = sl.stop
        correct_index = correct_index and (start == dim)
        return correct_length and correct_index

    for _ in range(4):
        dim = random.randint(100, 200)
        divs = random.randint(1, 10)
        passed = eval_slices(dim, divs, divy_slices(dim, divs))
        assert_true(passed, 'wrong slicing {} {}'.format(dim, divs))


def test_forked_sharedmem():
    check_platform()
    array = ForkSharedmemManager.shared_ndarray((10, 3), 'd')
    mm = ForkSharedmemManager(array)
    ctx = mp.get_context('fork')

    with ctx.Pool(processes=max(1, mp.cpu_count() - 3), initializer=init_shared_array, initargs=(mm,)) as p:
        p.map(write_to_array, range(len(array)))
    p.close()
    p.join()

    expected_result = np.repeat(np.arange(array.shape[0]), array.shape[1]).reshape(array.shape)
    assert_true(np.all(expected_result == array), 'Forked shared-memory writing failed')


def test_spawned_sharedmem():
    array = SpawnSharedmemManager.shared_ndarray((10, 3), 'd')
    mm = SpawnSharedmemManager(array)
    ctx = mp.get_context('spawn')

    with ctx.Pool(processes=max(1, mp.cpu_count() - 3), initializer=init_shared_array, initargs=(mm,)) as p:
        p.map(write_to_array, range(len(array)))
    p.close()
    p.join()

    expected_result = np.repeat(np.arange(array.shape[0]), array.shape[1]).reshape(array.shape)
    assert_true(np.all(expected_result == array), 'Forked shared-memory writing failed')


def test_ro_split():
    # I guess there's no good way to test both fork and spawn modes, since split_at()
    # is only aware of one type of SharedmemManager at runtime.
    array = np.arange(20 * 30).reshape(20, 30)
    sh_array = shared_copy(array)

    split_fn = split_at()(sum_columns)

    # this method should work equally for read-only and shared access
    res_1 = split_fn(array)
    res_2 = split_fn(sh_array)
    res_3 = sum_columns(array)

    assert_true(np.all(res_1 == res_3), 'read-only does not match ref')
    assert_true(np.all(res_2 == res_3), 'shm does not match ref')
    assert_true(np.all(res_1 == res_2), 'para results do not match')


def test_rw_split():
    # I guess there's no good way to test both fork and spawn modes, since split_at()
    # is only aware of one type of SharedmemManager at runtime.
    array = np.arange(20 * 30).reshape(20, 30)
    sh_array = shared_copy(array)
    out = np.zeros(20)
    sh_out = shared_copy(out)

    split_fn = split_at(split_arg=(0, 1))(sum_columns_and_write)

    # this call should not touch out
    split_fn(array, out)
    assert_true(np.all(out == 0), 'non-shared output modified?')
    # this call should not touch out
    split_fn(sh_array, out)
    assert_true(np.all(out == 0), 'non-shared output modified?')
    # this call should write to shm
    split_fn(array, sh_out)
    assert_true(np.all(sh_out == sum_columns(array)), 'shared mem did not get written?')
    sh_out.fill(0)
    # this call should write to shm
    split_fn(sh_array, sh_out)
    assert_true(np.all(sh_out == sum_columns(array)), 'shared mem did not get written?')


def very_weird_fun(shared_array, split_array):
    # this method exercises a few features
    # 1. split array and returned array (splicing array) are in position 1 not 0
    # 2. there is a shared array that is *not* split, but is available to all jobs
    # 3. the split array is updated inplace

    # need to access shared array first
    shared_array = shared_array[:]
    shared_size = shared_array.size
    split_array += shared_array.var()
    return shared_size, split_array.sum(axis=1)


def test_complex_input_output_pattern():
    shared_arg = np.random.randn(100)
    split_arg = shared_ndarray((50, 20))
    split_arg[:] = np.arange(20 * 50).reshape(50, 20)
    test_arg = split_arg.copy()

    split_fn = split_at(split_arg=(1,), splice_at=(1,), shared_args=(0,), concurrent=True)(very_weird_fun)

    size, sum = split_fn(shared_arg, split_arg)

    assert_true(size == shared_arg.size, 'single return value wrong')
    assert_true(np.all(sum == split_arg.sum(axis=1)))
    assert_true(np.all(split_arg == test_arg + shared_arg.var()))
