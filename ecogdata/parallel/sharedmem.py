"""
This module defines a shared memory tool (abstract base class `SharedmemTool`) that can be used to

1. create shared memory from multiprocessing.sharedctypes
2. wrap shared memory as ndarrays
3. convert ndarrays to shared memory

The way that translations 2 & 3 work depends on the "start method" of multiprocessing. The `SharedmemManager` is
conditionally defined based on the start method.

Memory pointers are more straightforward for forked processes--processes inherit memory.
Copying to shared memory is generally required for spawned processes, so that the shared pointer can be pickled
correctly. For any shared memory explicity created by the SharedmemTool (i.e. operation 1 above), a ndarray pointer
to shared array pointer lookup cache is retained to prevent memory copies.

"""
from abc import ABC
import sys
from ecogdata.parallel.mproc import parallel_context as pctx
from multiprocessing.sharedctypes import synchronized
import numpy as np
from contextlib import contextmanager, ExitStack
from ecogdata.util import ToggleState


__all__ = ['SharedmemManager', 'shared_copy', 'shared_ndarray']


# from the "array" module docstring

# This module defines an object type which can efficiently represent
# an array of basic values: characters, integers, floating point
# numbers.  Arrays are sequence types and behave very much like lists,
# except that the type of objects stored in them is constrained.  The
# type is specified at object creation time by using a type code, which
# is a single character.  The following type codes are defined:
#
#     Type code   C Type             Minimum size in bytes
#     'c'         character          1
#     'b'         signed integer     1
#     'B'         unsigned integer   1
#     'u'         Unicode character  2
#     'h'         signed integer     2
#     'H'         unsigned integer   2
#     'i'         signed integer     2
#     'I'         unsigned integer   2
#     'l'         signed integer     4
#     'L'         unsigned integer   4
#     'f'         floating point     4
#     'd'         floating point     8

# convert dtypes to this ctype if the codes differ
dtype_maps_to = dict([('?', 'b')])

# Two-way lookups for handling complex types (F, D, G)
dtype_ctype = dict((('F', 'f'), ('D', 'd'), ('G', 'g')))
ctype_dtype = dict(((v, k) for k, v in dtype_ctype.items()))


class SharedmemTool(ABC):

    def __init__(self, shm_object, shape, dtype_code, use_lock=False):
        self.shm = shm_object
        self.shape = shape
        self.dtype = dtype_code
        self.use_lock = ToggleState(init_state=use_lock)

    @classmethod
    def sharedctypes_from_shape_and_code(cls, shape, typecode='d'):
        N = int(shape[0])
        for dim in shape[1:]:
            N *= int(dim)
        # check for complex type -- needs to be flattened into real pairs
        if typecode in dtype_ctype:
            N *= 2
            ctypecode = dtype_ctype[typecode]
        else:
            ctypecode = typecode

        shm = pctx.Array(ctypecode, N)
        return shm

    @classmethod
    def shared_ndarray(cls, shape, typecode='d'):
        ctype_array = cls.sharedctypes_from_shape_and_code(shape, typecode=typecode)
        # this should actually always return np.frombuffer(ctypes_array.get_obj(), ...)
        return np.frombuffer(ctype_array.get_obj(), dtype=typecode).reshape(shape)

    @classmethod
    def shared_copy(cls, x):
        typecode = dtype_maps_to.get(x.dtype.char, x.dtype.char)
        y = cls.shared_ndarray(x.shape, typecode=typecode)
        y[:] = x.astype(typecode, copy=False)
        return y

    @contextmanager
    def get_ndarray(self):
        cls = type(self)
        with ExitStack() as stack:
            if self.use_lock:
                stack.enter_context(self.shm.get_lock())
            yield cls.tonumpyarray(self.shm, dtype=self.dtype, shape=self.shape)

    @classmethod
    def tonumpyarray(cls, mp_arr, dtype=float, shape=None):
        if shape is None:
            # global would be gotten from pool start-up
            global shape_
            shape = shape_
        return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)


class ForkSharedmemManager(SharedmemTool):

    def __init__(self, np_array, use_lock=False):
        dtype = np_array.dtype.char
        shape = np_array.shape
        if dtype in dtype_ctype:
            ctype_view = dtype_ctype[dtype]
            shm = synchronized(
                np.ctypeslib.as_ctypes(np_array.view(ctype_view))
            )
        else:
            shm = synchronized(
                np.ctypeslib.as_ctypes(np_array)
            )
        super(ForkSharedmemManager, self).__init__(shm, shape, dtype, use_lock=use_lock)


_spawning_mem_cache = dict()

# TODO: the timing of creating a large shared array is ridic!
class SpawnSharedmemManager(SharedmemTool):
    """
    Basically the same but we have to hold shared mem as a plain sharedctypes.Array
    and COPY a numpy ndarray into the array. However, if this class was already used to
    create a shared array (shared_ndarray), then the sharedctypes.Array should be found
    in the memory cache dictionary.
    """

    def __init__(self, np_array, use_lock=False):
        dtype = np_array.dtype.char
        shape = np_array.shape
        ptr = np_array.__array_interface__['data'][0]
        if ptr in _spawning_mem_cache:
            shared_ctypes = _spawning_mem_cache[ptr]
            existing_shared_mem = True
        else:
            # instead of creating a ctypes "view" a la numpy.ctypeslib.as_ctypes, we
            # need to make a whole new shared ctypes Array that can be pickled (hopefully the pointer only?)
            shared_ctypes = SharedmemTool.sharedctypes_from_shape_and_code(shape, typecode=dtype)
            existing_shared_mem = False
        super(SpawnSharedmemManager, self).__init__(shared_ctypes, shape, dtype, use_lock=use_lock)
        # shm = mp.sharedctypes.synchronized(shared_ctypes)
        # must copy np_array into the new array
        if not existing_shared_mem:
            with self.get_ndarray() as ndarray:
                ndarray[:] = np_array
                # also add this pointer to the cache?
                ptr = ndarray.__array_interface__['data'][0]
                _spawning_mem_cache[ptr] = shared_ctypes

    @classmethod
    def shared_ndarray(cls, shape, typecode='d'):
        ctype_array = cls.sharedctypes_from_shape_and_code(shape, typecode=typecode)
        # this should actually always return np.frombuffer(ctypes_array.get_obj(), ...)
        ndarray = np.frombuffer(ctype_array.get_obj(), dtype=typecode).reshape(shape)
        ptr = ndarray.__array_interface__['data'][0]
        _spawning_mem_cache[ptr] = ctype_array
        return ndarray


# register these to change with the parallel context namespace
pctx.register_context_dependent_namespace('spawn', SpawnSharedmemManager, altname='SharedmemManager')
pctx.register_context_dependent_namespace('spawn', SpawnSharedmemManager.shared_ndarray)
pctx.register_context_dependent_namespace('spawn', SpawnSharedmemManager.shared_copy)
pctx.register_context_dependent_namespace('fork', ForkSharedmemManager, altname='SharedmemManager')
pctx.register_context_dependent_namespace('fork', ForkSharedmemManager.shared_ndarray)
pctx.register_context_dependent_namespace('fork', ForkSharedmemManager.shared_copy)


# dynamic (context-dependent) lookup of the previous items
def __getattr__(name):
    if name in __all__:
        return getattr(pctx.ctx, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# nice tool to backport the __getattr__ trick to earlier Python
# https://github.com/facelessuser/pep562
PY37 = sys.version_info >= (3, 7)
if not PY37:
    from .pep562 import Pep562
    Pep562(__name__)
