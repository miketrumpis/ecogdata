from .mproc import Queue, JoinableQueue, Process, make_stderr_logger, cpu_count, get_logger
from contextlib import contextmanager
import numpy as np


class Jobsdone(Exception):
    pass


class ParallelWorker(Process):
    input_q: JoinableQueue
    output_q: Queue
    para_method: callable

    def set_queues(self, input_q: JoinableQueue, output_q: Queue):
        self.input_q = input_q
        self.output_q = output_q

    def map_job(self, job):
        """
        This method must translate the job spec into args and kwargs for the para_method.
        It returns (i, args, kwargs) where i is the task order number for keeping sequence in the Runner

        """
        pass

    def check_job(self, endtask=True):
        job = self.input_q.get()
        if job is None:
            info = get_logger().info
            info('Exit code, doing task-done: {}'.format(endtask))
            if endtask:
                self.input_q.task_done()
            raise Jobsdone
        return job

    def run(self):
        info = get_logger().info
        info('Starting jobs')
        while True:
            try:
                job = self.check_job(endtask=True)
            except Jobsdone:
                break
            try:
                i, args, kwargs = self.map_job(job)
                # print(i, len(args))
                # print(type(kwargs), kwargs)
                # print(type(args), args)
                r = self.para_method(*args, **kwargs)
                self.output_q.put((i, r))
            except Exception as e:
                # print('doing error value for exception {}'.format(str(e)))
                err = get_logger().error
                err('Exception: {}'.format(repr(e)))
                self.output_q.put((i, np.nan))
            finally:
                # print('doing task done')
                self.input_q.task_done()


class JobRunner:

    def __init__(self, worker: callable, n_jobs: int=None,
                 w_args: tuple=(), w_kwargs: dict=dict(),
                 single_job_in_thread: bool=False):
        self.input_q = JoinableQueue()
        self.output_q = Queue()
        self.n_jobs = cpu_count() if n_jobs is None else n_jobs
        self._threaded = self.n_jobs > 1 or single_job_in_thread
        self.workers = list()
        self._worker_constructor = worker
        self._w_args = w_args
        self._w_kwargs = w_kwargs
        self._stale_workers = True
        self._submit_queue = list()
        self.output_from_submitted = None

    def _renew_workers(self):
        self.workers = list()
        for _ in range(self.n_jobs):
            w = self._worker_constructor(*self._w_args, **self._w_kwargs)
            w.set_queues(self.input_q, self.output_q)
            self.workers.append(w)
        self._stale_workers = False

    def run_jobs(self, inputs: np.ndarray, output_shape=(), output_dtype=None, timeout=20, loglevel='error'):
        with make_stderr_logger(loglevel):
            if self._stale_workers:
                self._renew_workers()
            if not output_shape:
                output_shape = inputs.shape
            if not output_dtype:
                output_dtype = inputs.dtype
            outputs = np.empty(output_shape, dtype=output_dtype)
            if self._threaded:
                for w in self.workers:
                    w.start()
            self._stale_workers = True
            for i, x in enumerate(inputs):
                self.input_q.put((i, x))
            for _ in range(len(self.workers)):
                self.input_q.put(None)
            if self._threaded:
                self.input_q.join()
            else:
                if len(self.workers) > 1:
                    print('This clause should not happen for > 1 workers')
                self.workers[0].run()
            for _ in range(len(inputs)):
                i, y = self.output_q.get(True, timeout)
                outputs[i] = y
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
