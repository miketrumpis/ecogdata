"""Safely set numexpr.set_num_threads(1) before attempting multiprocessing"""

import numexpr
numexpr.set_num_threads(1)

# import this first to inject shared memory stuff into namespace
from . import sharedmem as shm
from .array_split import *
from .jobrunner import *
from .mproc import *
