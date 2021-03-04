from tblib import pickling_support
pickling_support.install()
from .mproc import Queue, JoinableQueue, Process, make_stderr_logger, cpu_count, get_logger
import sys
from contextlib import contextmanager
import numpy as np
from tqdm.auto import tqdm
from time import sleep
import queue
import inspect


class Jobsdone(Exception):
    pass


def make_worker(func):
    class RunsFunc(ParallelWorker):
        para_method = staticmethod(func)

        def map_job(self, job):
            """
            Create arguments and keywords to call self.para_method(*args, **kwargs)

            "job" is of the form (i, job_spec) where i is a place keeper.

            """
            i, arr = job
            # Do some helpful logging
            info = get_logger().info
            info('Got job {}'.format(i))
            return i, (arr,), dict()
    return RunsFunc


class ParallelWorker(Process):
    input_q: JoinableQueue
    output_q: Queue
    error_q: Queue
    para_method: callable

    def set_queues(self, input_q: JoinableQueue, output_q: Queue, error_q: Queue):
        self.input_q = input_q
        self.output_q = output_q
        self.error_q = error_q

    def map_job(self, job):
        """
        This method must translate the job spec into args and kwargs for the para_method.
        It returns (i, args, kwargs) where i is the task order number for keeping sequence in the Runner

        """
        i = job
        return i, (i,), dict()

    def check_job(self, endtask=True):
        job = self.input_q.get()
        if job is None:
            info = get_logger().info
            info('Exit code, doing task-done: {}'.format(endtask))
            if endtask:
                self.input_q.task_done()
            raise Jobsdone
        return job

    def run(self, raise_immediately=False):
        info = get_logger().info
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
                err = get_logger().error
                err('Exception: {}'.format(repr(e)))
                err_info = sys.exc_info()
                self.error_q.put(err_info)
                self.output_q.put((i, np.nan))
            finally:
                self.input_q.task_done()


class JobRunner:

    def __init__(self, worker: callable, n_workers: int=None,
                 w_args: tuple=(), w_kwargs: dict=dict(),
                 single_job_in_thread: bool=False):
        self.input_q = JoinableQueue()
        self.output_q = Queue()
        self.error_q = Queue()
        self.n_workers = cpu_count() if n_workers is None else n_workers
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

    def _renew_workers(self, n_workers):
        self.workers = list()
        # refresh error queue
        self.error_q = Queue()
        for _ in range(n_workers):
            w = self._worker_constructor(*self._w_args, **self._w_kwargs)
            w.set_queues(self.input_q, self.output_q, self.error_q)
            self.workers.append(w)
        self._stale_workers = False

    def run_jobs(self, inputs: np.ndarray=None, n_jobs: int=None, output_shape: tuple=(),
                 output_dtype: np.dtype=None, timeout: float=20e3, progress: bool=True,
                 loglevel: str='error', return_exceptions: bool=False, reraise_exceptions: bool=True):
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
                self._renew_workers(workers)
            if not output_shape:
                if not isinstance(inputs, np.ndarray):
                    output_shape = (len(inputs),)
                else:
                    output_shape = inputs.shape
            if not output_dtype:
                output_dtype = np.object  # Try to shim into typed array later
            outputs = np.empty(output_shape, dtype=output_dtype)
            if self._threaded:
                for w in self.workers:
                    w.start()
            self._stale_workers = True
            info = get_logger().info
            info('Queueing inputs')
            for i, x in enumerate(inputs):
                if push_inputs:
                    self.input_q.put((i, x))
                else:
                    # just send which job the worker is supposed to do
                    self.input_q.put(i)
            for _ in range(len(self.workers)):
                self.input_q.put(None)
            if not self._threaded:
                if len(self.workers) > 1:
                    print('This clause should not happen for > 1 workers')
                info('Running single worker')
                self.workers[0].run(raise_immediately=reraise_exceptions)
            # Do not use timeout until a first output is cleared (??)
            wait_time = None
            if progress:
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
        # try:
        #     dtypes = set([o.dtype for o in outputs])
        #     if len(dtypes) == 1:
        #         try:
        #             outputs = np.row_stack(outputs).squeeze()
        #         except Exception as e:
        #             print('Tried to astype: {}'.format(repr(e)))
        # except AttributeError:
        #     print("Can't simplify output array")
        #     pass
        try:
            outputs = np.row_stack(outputs).squeeze()
        except Exception as e:
            print("Can't simplify output array: {}".format(repr(e)))
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
            inputs = np.empty(len(self._submit_queue), dtype=np.object)
            inputs[:] = self._submit_queue
            self._submit_queue = list()
            self.output_from_submitted = self.run_jobs(inputs, **kwargs)

    def submit(self, job_spec):
        self._submit_queue.append(job_spec)
