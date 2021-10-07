import pytest
import numpy as np
import multiprocess as mp
from ecogdata.parallel.jobrunner import JobRunner, ParallelWorker
from ecogdata.parallel.mproc import parallel_context
from . import with_start_methods


@with_start_methods
def test_process_types():
    jr = JobRunner(np.var)
    # create 3 workers and check type
    jr._renew_workers(3)
    if parallel_context.context_name == 'spawn':
        assert isinstance(jr.workers[0], mp.context.SpawnProcess)
    elif parallel_context.context_name == 'fork':
        assert isinstance(jr.workers[0], mp.context.ForkProcess)


@with_start_methods
def test_simple_method():
    arrays = [np.arange(10) for _ in range(20)]
    sums = JobRunner(np.sum, n_workers=4).run_jobs(inputs=arrays, progress=False)
    assert sums.dtype not in np.sctypes['others'], 'simple results dtype is not numeric'
    assert (sums == 9 * 5).all(), 'wrong sums'


@with_start_methods
def test_submitting_context():
    jr = JobRunner(np.sum, n_workers=4)
    with jr.submitting_jobs(progress=False):
        for _ in range(20):
            jr.submit(np.arange(10))
    sums = jr.output_from_submitted
    assert sums.dtype not in np.sctypes['others'], 'simple results dtype is not numeric'
    assert (sums == 9 * 5).all(), 'wrong sums'


def nonnumeric_input_output(a: int, b: str, c: list):
    return c, b, a


@with_start_methods
def test_nonnumeric():
    from string import ascii_letters
    n = 20
    # inputs types are (int, str, list)
    inputs = list(zip(range(n),
                      ascii_letters,
                      [list(range(np.random.randint(low=1, high=6))) for _ in range(n)]))
    runner = JobRunner(nonnumeric_input_output, n_workers=4)
    outputs = runner.run_jobs(inputs=inputs, progress=False)
    assert outputs.dtype == object, 'did not return object array'
    assert isinstance(outputs[0], tuple), 'individual ouputs have wrong type'
    assert all([o[::-1] == i for o, i in zip(outputs, inputs)]), 'wrong output values'


class ArraySum(ParallelWorker):
    """
    A fancier simple function
    """
    para_method = staticmethod(np.sum)

    def map_job(self, job):
        """
        Create arguments and keywords to call self.para_method(*args, **kwargs)

        "job" is of the form (i, job_spec) where i is a place keeper.

        """
        i, arr = job
        return i, (arr,), dict()


@with_start_methods
def test_custom_worker():
    plain_arrays = [np.arange(100) for _ in range(25)]
    jobs = JobRunner(ArraySum, n_workers=4)
    sums = jobs.run_jobs(inputs=plain_arrays, progress=False)
    assert (sums == 99 * 50).all(), 'wrong sums'


class SharedarraySum(ArraySum):
    """
    A fancier simple function that uses shared memory
    """

    def __init__(self, shm_managers):
        # list of SharedmemManager objects
        self.shm_managers = shm_managers

    def map_job(self, job):
        # job is only the job number
        i = job
        # use the get_ndarray() context manager to simply get the array
        with self.shm_managers[i].get_ndarray() as arr:
            pass
        return i, (arr,), dict()


@with_start_methods
def test_shm_worker():
    mem_man = parallel_context.SharedmemManager
    shared_arrays = [mem_man(np.arange(100), use_lock=False) for _ in range(25)]
    # worker constructor now takes shared mem pointers
    jobs = JobRunner(SharedarraySum, n_workers=4, w_args=(shared_arrays,))
    # and run-mode just creates indices to the pointers
    sums = jobs.run_jobs(n_jobs=len(shared_arrays), progress=False)
    assert (sums == 99 * 50).all(), 'wrong sums'


def hates_eights(n):
    if n == 8:
        raise ValueError("n == 8, what did you think would happen?!")
    return n


@with_start_methods
def test_skipped_excpetions():
    jobs = JobRunner(hates_eights, n_workers=4)
    r, e = jobs.run_jobs(np.arange(10), reraise_exceptions=False, progress=False, return_exceptions=True)
    assert len(e) == 1, 'exception not returned'
    assert np.isnan(r[8]), 'exception not interpolated'
    assert all([r[i] == i for i in range(10) if i != 8]), 'non-exceptions not returned correctly'


@with_start_methods
def test_raised_exceptions():
    # testing raising after the fact
    with pytest.raises(ValueError):
        jobs = JobRunner(hates_eights, n_workers=4)
        r = jobs.run_jobs(np.arange(10), reraise_exceptions=True, progress=False)
    # testing raise-immediately
    with pytest.raises(ValueError):
        jobs = JobRunner(hates_eights, n_workers=1, single_job_in_thread=False)
        r = jobs.run_jobs(np.arange(10), reraise_exceptions=True, progress=False)
