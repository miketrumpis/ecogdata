import random
import numpy as np

from ecogdata.parallel.mproc import parallel_context
from ecogdata.parallel.array_split import split_at, divy_slices
from . import with_start_methods


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
        assert passed, 'wrong slicing {} {}'.format(dim, divs)


@with_start_methods
def test_ro_split():
    # I guess there's no good way to test both fork and spawn modes, since split_at()
    # is only aware of one type of SharedmemManager at runtime.
    array = np.arange(20 * 30).reshape(20, 30)
    sh_array = parallel_context.shared_copy(array)

    split_fn = split_at()(sum_columns)

    # this method should work equally for read-only and shared access
    res_1 = split_fn(array)
    res_2 = split_fn(sh_array)
    res_3 = sum_columns(array)

    assert np.all(res_1 == res_3), 'read-only does not match ref'
    assert np.all(res_2 == res_3), 'shm does not match ref'
    assert np.all(res_1 == res_2), 'para results do not match'


@with_start_methods
def test_rw_split():
    # I guess there's no good way to test both fork and spawn modes, since split_at()
    # is only aware of one type of SharedmemManager at runtime.
    array = np.arange(20 * 30).reshape(20, 30)
    sh_array = parallel_context.shared_copy(array)
    out = np.zeros(20)
    sh_out = parallel_context.shared_copy(out)

    split_fn = split_at(split_arg=(0, 1))(sum_columns_and_write)

    # this call should not touch out
    split_fn(array, out)
    assert np.all(out == 0), 'non-shared output modified?'
    # this call should not touch out
    split_fn(sh_array, out)
    assert np.all(out == 0), 'non-shared output modified?'
    # this call should write to shm
    split_fn(array, sh_out)
    assert np.all(sh_out == sum_columns(array)), 'shared mem did not get written?'
    sh_out.fill(0)
    # this call should write to shm
    split_fn(sh_array, sh_out)
    assert np.all(sh_out == sum_columns(array)), 'shared mem did not get written?'


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


@with_start_methods
def test_complex_input_output_pattern():
    shared_arg = np.random.randn(100)
    split_arg = parallel_context.shared_ndarray((50, 20))
    split_arg[:] = np.arange(20 * 50).reshape(50, 20)
    test_arg = split_arg.copy()

    split_fn = split_at(split_arg=(1,), splice_at=(1,), shared_args=(0,), concurrent=True)(very_weird_fun)

    size, sum = split_fn(shared_arg, split_arg)

    assert size == shared_arg.size, 'single return value wrong'
    assert np.all(sum == split_arg.sum(axis=1))
    assert np.all(split_arg == test_arg + shared_arg.var())

