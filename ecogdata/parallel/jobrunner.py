from tblib import pickling_support
pickling_support.install()
from .mproc import parallel_context, make_stderr_logger
import sys
from contextlib import contextmanager
import numpy as np
from tqdm.auto import tqdm
from tqdm.std import tqdm as tqdm_T
from time import sleep
import queue
import inspect


class Jobsdone(Exception):
    pass


# pickling business to work in spawn mode
def wrapper_unpickler(factory, func, reconstructor, *args):
    return reconstructor(*((factory(func),) + args))


def make_worker(func):
    class RunsFunc(ParallelWorker):
        para_method = staticmethod(func)

        def __reduce__(self):
            r = super(RunsFunc, self).__reduce__()
            return (wrapper_unpickler,
                    ((make_worker, func, r[0]) + r[1][1:])) + r[2:]
    return RunsFunc


class ParallelWorker:
    input_q: parallel_context.JoinableQueue
    output_q: parallel_context.Queue
    error_q: parallel_context.Queue
    para_method: callable

    def set_queues(self,
                   input_q: parallel_context.JoinableQueue,
                   output_q: parallel_context.Queue,
                   error_q: parallel_context.Queue):
        self.input_q = input_q
        self.output_q = output_q
        self.error_q = error_q

    def map_job(self, job):
        """
        This method must translate the job spec into args and kwargs for the para_method.
        It returns (i, args, kwargs) where i is the task order number for keeping sequence in the Runner

        """
        if np.iterable(job):
            i, args = job
        else:
            i = job
            args = (i,)
        # Do some helpful logging
        info = parallel_context.get_logger().info
        info('Got job {}'.format(i))
        return i, args, dict()

    def check_job(self, endtask=True):
        job = self.input_q.get()
        if job is None:
            info = parallel_context.get_logger().info
            info('Exit code, doing task-done: {}'.format(endtask))
            if endtask:
                self.input_q.task_done()
            raise Jobsdone
        return job

    def run(self, raise_immediately=False):
        info = parallel_context.get_logger().info
        info('Starting jobs')
        while True:
            try:
                job = self.check_job(endtask=True)
            except Jobsdone:
                break
            try:
                i, args, kwargs = self.map_job(job)
                r = self.para_method(*args, **kwargs)
                self.output_q.put((i, r))
            except Exception as e:
                # This is helpful for debugging (in single-thread mode)
                if raise_immediately:
                    raise e
                err = parallel_context.get_logger().error
                err('Exception: {}'.format(repr(e)))
                err_info = sys.exc_info()
                self.error_q.put(err_info)
                i = job[0]
                self.output_q.put((i, np.nan))
            finally:
                self.input_q.task_done()


class JobRunner:

    def __init__(self, worker: callable, n_workers: int=None,
                 w_args: tuple=(), w_kwargs: dict=dict(),
                 single_job_in_thread: bool=False):
        # temporary values
        self.input_q = self.output_q = self.error_q = None
        self.n_workers = parallel_context.cpu_count() if n_workers is None else n_workers
        self._threaded = self.n_workers > 1 or single_job_in_thread
        self.workers = list()
        if not (inspect.isclass(worker) and issubclass(worker, ParallelWorker)):
            worker = make_worker(worker)
        self._worker_constructor = worker
        self._w_args = w_args
        self._w_kwargs = w_kwargs
        self._stale_workers = True
        self._submit_queue = list()
        self.output_from_submitted = None

    def _renew_workers(self, n_workers, **kwargs):
        """
        Regenerate a ParallelWorker, multiprocessing queues, and worker processes.
        Worker types are context-dependent. The process "target" is the bound
        method "run" from a new ParallelWorker constructed here.

        Parameters
        ----------
        n_workers : int
            Number of workers
        kwargs : dict
            Extra parameters for the ParallelWorker.run() method

        """
        self.workers = list()
        # refresh error queue (why not refresh all queues?)
        ctx = parallel_context.ctx
        self.input_q = ctx.JoinableQueue()
        self.output_q = ctx.Queue()
        self.error_q = ctx.Queue()

        # for _ in range(n_workers):
        #     w = self._worker_constructor(*self._w_args, **self._w_kwargs)
        #     w.set_queues(self.input_q, self.output_q, self.error_q)
        #     self.workers.append(w)

        # Instead try single worker with a bound method as target method
        worker_target = self._worker_constructor(*self._w_args, **self._w_kwargs)
        worker_target.set_queues(self.input_q, self.output_q, self.error_q)
        for _ in range(n_workers):
            w = parallel_context.Process(target=worker_target.run, kwargs=kwargs)
            self.workers.append(w)

        self._stale_workers = False

    def run_jobs(self, inputs: np.ndarray=None, n_jobs: int=None, output_shape: tuple=(),
                 output_dtype: np.dtype=None, timeout: float=20e3, progress: bool=True,
                 loglevel: str='error', return_exceptions: bool=False, reraise_exceptions: bool=True):
        """
        Run jobs in threads (or in this thread).

        Parameters
        ----------
        inputs : ndarray or sequence
            The sequence of generically typed inputs. Each input will be enqueued as a single argument,
            unless the type of inputs[i] is a tuple, in which case it method(*inputs[i]) will apply.
        n_jobs : int
            Alternate run-mode if inputs=None: run a number of jobs with argument i=0, ..., n_jobs - 1
        output_shape : tuple:
            If given, form the output array with this shape (otherwise len(inputs) is used)
        output_dtype : dtype
            If given, formt the output array with this type (otherwise an object array is used)
        timeout : float
            Multiprocessing timeout in milliseconds
        progress : bool
            Show a progress bar (T/F). A pre-configured "tqdm" instance can also be given.
        loglevel : str
            Run workers with this log level on the stdout
        return_exceptions : bool
            In addition to output values, return any exceptions + tracebacks encountered
        reraise_exceptions : bool
            If True, raise on the first exception encountered in the output queue. If jobs are run
            in single-thread mode, then the exception will be raised directly in the worker rather.

        Returns
        -------
        outputs: ndarray
            Results sequence. The output can be simplified with a pre-determined shape and dtype, or
            using numpy.row_stack, if possible.
        exceptions: list
            If return_exceptions, these are exceptions encountered in the worker processes.

        """
        if inputs is None and n_jobs is None:
            print("Can't do anything without inputs or the number of jobs.")
            return
        push_inputs = True
        if inputs is None:
            inputs = range(n_jobs)
            push_inputs = False
        with make_stderr_logger(loglevel):
            if self._stale_workers:
                workers = min(len(inputs), self.n_workers)
                raises = reraise_exceptions and not self._threaded
                self._renew_workers(workers, raise_immediately=raises)
            if not output_shape:
                if not isinstance(inputs, np.ndarray):
                    output_shape = (len(inputs),)
                else:
                    output_shape = inputs.shape
            if not output_dtype:
                output_dtype = object  # Try to shim into typed array later
            outputs = np.empty(output_shape, dtype=output_dtype)
            if self._threaded:
                for w in self.workers:
                    w.start()
            self._stale_workers = True
            info = parallel_context.get_logger().info
            info('Queueing inputs')
            for i, x in enumerate(inputs):
                if push_inputs:
                    # Enqueueing the input to enable method(*x) ---
                    # If x is a tuple, then enqueue it directly. Otherwise x
                    # is treated as a single argument
                    if isinstance(x, tuple):
                        self.input_q.put((i, x))
                    else:
                        self.input_q.put((i, (x,)))
                else:
                    # just send which job the worker is supposed to do
                    self.input_q.put(i)
            for _ in range(len(self.workers)):
                self.input_q.put(None)
            if not self._threaded:
                if len(self.workers) > 1:
                    print('This clause should not happen for > 1 workers')
                info('Running single worker')
                # self.workers[0].run(raise_immediately=reraise_exceptions)
                self.workers[0].run()
            # Do not use timeout until a first output is cleared (??)
            wait_time = None
            if progress:
                # make it possible to provide a pre-made progress bar -- helpful for inner-loops
                if isinstance(progress, tqdm_T):
                    pbar = progress
                else:
                    pbar = tqdm(total=len(inputs), desc='Jobs progress')
            n = 0
            while True:
                try:
                    i, y = self.output_q.get_nowait()
                except queue.Empty:
                    sleep(0.001)
                    continue
                n += 1
                outputs[i] = y
                if wait_time != timeout:
                    wait_time = timeout
                if progress:
                    pbar.update()
                if n == len(inputs):
                    break
                sleep(0.001)
            if progress:
                pbar.close()
            exceptions = []
            if not self.error_q.empty():
                while not self.error_q.empty():
                    exceptions.append(self.error_q.get())
        # Avoid turning an array of (out_1, ..., out_m) into a (len(inputs), m) shaped object array.
        # If the outputs are tuples, assume that separate return values are intended
        if not isinstance(outputs[0], tuple):
            try:
                outputs = np.row_stack(outputs).squeeze()
            except ValueError as e:
                info("Can't simplify output array: {}".format(repr(e)))
        if exceptions and reraise_exceptions:
            e = exceptions[0]
            raise e[1].with_traceback(e[2])
        if return_exceptions:
            return outputs, exceptions
        return outputs

    @contextmanager
    def submitting_jobs(self, **kwargs):
        try:
            self._submit_queue = list()
            yield
        finally:
            # have to do this in two steps to prevent numpy trying to broadcast
            inputs = np.empty(len(self._submit_queue), dtype=object)
            inputs[:] = self._submit_queue
            self._submit_queue = list()
            self.output_from_submitted = self.run_jobs(inputs, **kwargs)

    def submit(self, job_spec):
        self._submit_queue.append(job_spec)
