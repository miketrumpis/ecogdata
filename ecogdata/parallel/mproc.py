from contextlib import contextmanager
from collections import defaultdict
import platform
from multiprocess import get_context, log_to_stderr
from ecogdata.util import ToggleState


__all__ = ['parallel_controller', 'parallel_context', 'make_stderr_logger', 'timestamp']


parallel_controller = ToggleState(name='Parallel Controller')


class _pcontext:
    # could potentially screen for win32 and limit to spawn, but
    # maybe better not to cover for bad choices here.
    contexts = ('fork', 'spawn', 'forkserver')

    def __init__(self, default_starter='fork'):
        if platform.system() == 'Windows':
            default_starter = 'spawn'
        self._registry = defaultdict(list)
        self.ctx = default_starter

    def __getattribute__(self, attr):
        try:
            return super().__getattribute__(attr)
        except AttributeError:
            return getattr(self.ctx, attr)

    def register_context_dependent_namespace(self, context, object, altname=None, reload_context=True):
        if context not in self.contexts:
            raise ValueError('not a context: {}'.format(context))
        self._registry[context].append((object, altname))
        if reload_context:
            self.ctx = self.context_name

    @property
    def ctx(self):
        return self._ctx

    @ctx.setter
    def ctx(self, starter):
        self._ctx = get_context(starter)
        self._ctxname = starter
        registered_objects = self._registry[starter]
        for obj, name in registered_objects:
            if name is None:
                name = obj.__name__
            setattr(self._ctx, name, obj)

    @property
    def context_name(self):
        return self._ctxname

    @contextmanager
    def switch_context(self, starter):
        oldstarter = self.context_name
        self.ctx = starter
        try:
            yield
        finally:
            self.ctx = oldstarter


parallel_context = _pcontext()
_stderr_logger = None


def timestamp():
    from datetime import datetime
    return datetime.now().strftime('%H-%M-%S-%f')


@contextmanager
def make_stderr_logger(level='info'):
    # if None, then use "NOTSET" level
    import logging
    # log_to_stderr() should already prevent duplicate loggers, but it *does* add subsequent handlers
    global _stderr_logger
    if _stderr_logger is None:
        _stderr_logger = log_to_stderr()
    logger = _stderr_logger
    logger.propagate = False
    base_level = logger.level
    if level is None:
        logger.setLevel(logging.NOTSET)
    elif level.lower() == 'info':
        logger.setLevel(logging.INFO)
    elif level.lower() in ('warn', 'warning'):
        logger.setLevel(logging.WARNING)
    elif level.lower() == 'error':
        logger.setLevel(logging.ERROR)
    elif level.lower() == 'critical':
        logger.setLevel(logging.CRITICAL)

    try:
        yield logger
    finally:
        logger.setLevel(base_level)
