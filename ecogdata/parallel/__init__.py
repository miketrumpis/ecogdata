"""Safely set numexpr.set_num_threads(1) before attempting multiprocessing"""

import numexpr
numexpr.set_num_threads(1)

from .array_split import *
from .jobrunner import *
from .mproc import *
from . import sharedmem as shm