"""Safely set numexpr.set_num_threads(1) before attempting multiprocessing"""

import numexpr
numexpr.set_num_threads(1)

import platform
from multiprocessing import *
import multiprocessing.sharedctypes
sharedctypes = multiprocessing.sharedctypes

if platform.system().lower().find('windows') < 0:
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
