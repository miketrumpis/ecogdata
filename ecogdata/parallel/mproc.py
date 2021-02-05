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
