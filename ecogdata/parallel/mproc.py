"""Safely set numexpr.set_num_threads(1) before attempting multiprocessing"""

import numexpr
numexpr.set_num_threads(1)

from contextlib import contextmanager
import platform
from multiprocessing import *
import multiprocessing.sharedctypes
sharedctypes = multiprocessing.sharedctypes


if __name__ == '__main__':
    if platform.system() == 'Darwin':
        try:
            set_start_method('fork')
        except RuntimeError:
            # already set, perhaps by parent process
            pass

_stderr_logger = None


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


if platform.system().lower().find('windows') < 0 and get_start_method() != 'spawn':
    # Do monkey-patch suggested for Python Issue 30919
    # https://bugs.python.org/issue30919
    # This issue was *addressed* in Python 3.7, but performance is still poor.

    import mmap
    from multiprocessing.heap import Arena

    def anonymous_arena_init(self, size, fd=-1):
        "Create Arena using an anonymous memory mapping."
        self.size = size
        self.fd = fd  # still kept but is not used !
        self.buffer = mmap.mmap(-1, self.size)

    Arena.__init__ = anonymous_arena_init
    if __name__ == '__main__':
        set_start_method('fork')
