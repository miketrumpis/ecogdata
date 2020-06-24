import ecogdata.parallel.mproc as mp
from contextlib import closing, contextmanager
import warnings
from decorator import decorator
import numpy as np
from ecogdata.util import ToggleState
from datetime import datetime


def timestamp():
    return datetime.now().strftime('%H-%M-%S-%f')


parallel_controller = ToggleState(name='Parallel Controller')

# from the "array" module docstring
"""
This module defines an object type which can efficiently represent
an array of basic values: characters, integers, floating point
numbers.  Arrays are sequence types and behave very much like lists,
except that the type of objects stored in them is constrained.  The
type is specified at object creation time by using a type code, which
is a single character.  The following type codes are defined:

    Type code   C Type             Minimum size in bytes
    'c'         character          1
    'b'         signed integer     1
    'B'         unsigned integer   1
    'u'         Unicode character  2
    'h'         signed integer     2
    'H'         unsigned integer   2
    'i'         signed integer     2
    'I'         unsigned integer   2
    'l'         signed integer     4
    'L'         unsigned integer   4
    'f'         floating point     4
    'd'         floating point     8

"""

# can only think of booleans
dtype_maps_to = dict([('?', 'b')])

dtype_ctype = dict((('F', 'f'), ('D', 'd'), ('G', 'g')))
ctype_dtype = dict(((v, k) for k, v in dtype_ctype.items()))


class SharedmemTool:

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

        shm = mp.Array(ctypecode, N)
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
        if self.use_lock.state:
            with self.shm.get_lock():
                yield cls.tonumpyarray(self.shm, dtype=self.dtype, shape=self.shape)
        else:
            yield cls.tonumpyarray(self.shm, dtype=self.dtype, shape=self.shape)

    @classmethod
    def tonumpyarray(cls, mp_arr, dtype=float, shape=None):
        if shape is None:
            shape = shape_
        return np.frombuffer(mp_arr.get_obj(), dtype=dtype).reshape(shape)


class ForkSharedmemManager(SharedmemTool):

    def __init__(self, np_array, use_lock=False):
        dtype = np_array.dtype.char
        shape = np_array.shape
        if dtype in dtype_ctype:
            ctype_view = dtype_ctype[dtype]
            shm = mp.sharedctypes.synchronized(
                np.ctypeslib.as_ctypes(np_array.view(ctype_view))
            )
        else:
            shm = mp.sharedctypes.synchronized(
                np.ctypeslib.as_ctypes(np_array)
            )
        super(ForkSharedmemManager, self).__init__(shm, shape, dtype, use_lock=use_lock)


_spawning_mem_cache = dict()


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




if mp.get_start_method() in ('spawn', 'forkserver'):
    SharedmemManager = SpawnSharedmemManager
else:
    SharedmemManager = ForkSharedmemManager


shared_ndarray = SharedmemManager.shared_ndarray
shared_copy = SharedmemManager.shared_copy


def parallel_runner(method, static_args, split_arg, shared_split_arrays, shared_args, shared_full_arrays,
                    splice_args, kwargs, n_jobs, info_logger):
    """
    Runs method in a Pool with split and full shared memory.

    Parameters
    ----------
    method: callable
    static_args: tuple
        Any static arguments in method's signature (not split nor shared)
    split_arg: tuple
        Indices of the split arguments in method's signature
    shared_split_arrays: tuple
        SharedmemManagers of the split arguments
    shared_args: tuple
        Indices of the full shared arrays in method's signature
    shared_full_arrays: tuple
        SharedmemManagers of the read-only shared arrays
    splice_args: tuple
        Indices of return values to splice from split inputs
    kwargs: dict
        method's keyword arguments
    n_jobs: int
        Number of Pool processes
    info_logger: logger

    Returns
    -------
    r: result of method(*args, **kwargs)

    """
    split_shapes = [x.shape for x in shared_split_arrays]
    init_args = (split_arg,
                 shared_split_arrays,
                 split_shapes,
                 shared_args,
                 shared_full_arrays,
                 method,
                 static_args,
                 kwargs)
    mp.freeze_support()
    info_logger('{} Creating pool'.format(timestamp()))
    with closing(mp.Pool(processes=n_jobs, initializer=_init_globals, initargs=init_args)) as p:
        dim_size = shared_split_arrays[0].shape[0]
        # map the jobs
        job_slices = divy_slices(dim_size, len(p._pool))
        info_logger('{} Mapping jobs'.format(timestamp()))
        res = p.map_async(_global_method_wrap, job_slices)

    p.join()
    if res.successful():
        info_logger('{} Joining results'.format(timestamp()))
        res = splice_results(res.get(), splice_args)
        # res = res.get()
    else:
        # raises exception ?
        res.get()
    # gc.collect()
    info_logger('{} Wrap done'.format(timestamp()))
    return res


def split_at(split_arg=(0,), splice_at=(0,), shared_args=(), split_over=None, n_jobs=-1, concurrent=False):
    """
    Decorator that enables parallel dispatch over multiple blocks of input ndarrays. Input arrays are cast to shared
    memory pointers using `as_ctypes` from numpy.ctypeslib. These arrays can be modified by subproceses if they were
    originally shared memory Arrays cast to ndarray using `frombuffer` (see `shared_ndarray`). Otherwise the arrays
    are read-only. Returned arrays may be spliced together if appropriate.

    Parameters
    ----------
    split_arg: tuple
        Position(s) of any arguments in the method's signature that should be split for dispatch
    splice_at: tuple
        Position(s) of any return arguments that should be combined for output
    shared_args: tuple
        Position of any array arguments that should be available to all workers as shared memory
    split_over: int
        If not None, only split the array(s) if they're over this many MBs
    n_jobs: int
        Number of workers (if -1 then use all cpus)
    concurrent: bool
        If True, then the shared memory will be accessed with with thread locks.

    """
    info = mp.get_logger().info
    info('{} Starting wrap'.format(timestamp()))
    # normalize inputs
    if not np.iterable(splice_at):
        splice_at = (splice_at,)
    if not np.iterable(split_arg):
        split_arg = (split_arg,)
    if n_jobs < 0:
        n_jobs = mp.cpu_count()

    splice_at = tuple(splice_at)
    split_arg = tuple(split_arg)
    shared_args = tuple(shared_args)

    @decorator
    def inner_split_method(method, *args, **kwargs):
        # check if the arrays are too small to bother with subprocesses
        if split_over is not None:
            mx_size = 0
            for p in split_arg:
                arr = args[p]
                size = arr.size * arr.dtype.itemsize
                if size > mx_size:
                    mx_size = size
            if mx_size / 1024 / 1024 < split_over:
                return method(*args, **kwargs)
        # check if the parallel context is false
        if not parallel_controller.state:
            return method(*args, **kwargs)
        pop_args = sorted(split_arg + shared_args)
        shared_array_shm = list()
        n = 0
        args = list(args)
        split_array_shm = list()
        for pos in pop_args:
            pos = pos - n
            a = args.pop(pos)
            info('{} Wrapping shared memory size {} MB'.format(timestamp(), a.size * a.dtype.itemsize / 1024. / 1000.))
            x = SharedmemManager(a, use_lock=concurrent)
            if pos + n in split_arg:
                split_array_shm.append(x)
            else:
                shared_array_shm.append(x)
            n += 1
        static_args = tuple(args)
        shared_array_shm = tuple(shared_array_shm)

        split_lens = set([x.shape[0] for x in split_array_shm])
        if len(split_lens) > 1:
            raise ValueError(
                'all of the arrays to split must have the same length '
                'on the first axis'
            )
        return parallel_runner(method, static_args, split_arg, split_array_shm, shared_args, shared_array_shm,
                               splice_at, kwargs, n_jobs, info)

    return inner_split_method


def divy_slices(dim_size, n_div):
    # if there are less or equal dims as procs, then split it up 1 per
    # otherwise, balance it with some procs having
    # N=ceil(dims / procs) dims, and the rest having N-1

    max_dims = int(np.ceil(float(dim_size) / n_div))
    job_dims = [max_dims] * n_div
    n = -1
    # step back and subtract job size until sum matches total size
    while np.sum(job_dims) > dim_size:
        m = job_dims[n]
        job_dims[n] = m - 1
        n -= 1
    # filter out any proc with zero size
    job_dims = [_f for _f in job_dims if _f]
    n = 0
    job_slices = list()
    # now form the data slicing to map out to the jobs
    for dims in job_dims:
        job_slices.extend([slice(n, n + dims)])
        n += dims
    return job_slices


def splice_results(map_list, splice_at):
    if [x for x in map_list if x is None]:
        return
    if isinstance(map_list[0], np.ndarray):
        res = np.concatenate(map_list, axis=0)
        return res
    splice_at = sorted(splice_at)

    res = tuple()
    pres = 0
    res = tuple()
    for sres in splice_at:
        res = res + map_list[0][pres:sres]
        arr_list = [m[sres] for m in map_list]
        res = res + (np.concatenate(arr_list, axis=0),)
        pres = sres + 1
    res = res + map_list[0][pres:]

    return res

# --- the following are initialized in the global state of the subprocesses


class shared_readonly:
    def __init__(self, mem_mgr):
        self.mem_mgr = mem_mgr

    def __getitem__(self, idx):
        # TODO: is a copy really needed here? will any subsequent access to this slice, even if read-only, interfere
        #  with other processes? I guess probably yes?
        with self.mem_mgr.get_ndarray() as shm_ndarray:
            return shm_ndarray[idx].copy()


def _init_globals(
        split_arg, shm, shm_shape,
        shared_args, sh_arg_mem,
        method, args, kwdict
):
    """
    Initialize the pool worker global state with the method to run and shared arrays + info + other args and kwargs

    Parameters
    ----------
    split_arg: sequence
        sequence of positions of argument that are split over
    shm: sequence
        the shared memory managers for these split args
    shm_shape: sequence
        the shapes of the underlying ndarrays of the split args
    shared_args: sequence
        positions for readonly, non-split shared memory args
    sh_arg_mem: sequence
        shared mem managers for any readonly shared memory arguments
    method: function
        function to call
    args:
        other positional argument
    kwdict:
        keyword arguments
    """

    # globals for primary shared array
    global shared_arr_
    shared_arr_ = shm
    global shape_
    shape_ = shm_shape
    global split_arg_
    split_arg_ = split_arg

    # globals for secondary shared memory arguments
    global shared_args_
    shared_args_ = shared_args
    global shared_args_mem_
    shared_args_mem_ = tuple([shared_readonly(mm) for mm in sh_arg_mem])

    # globals for pickled method and other arguments
    global method_
    method_ = method
    global args_
    args_ = args
    ## info = mp.get_logger().info
    ## info(repr(map(type, args)))

    global kwdict_
    kwdict_ = kwdict

    info = mp.get_logger().info
    info('{} Initialized globals'.format(timestamp()))


def _global_method_wrap(aslice):
    arrs = []
    # cycle through shared arrays and translate to ndarray (possibly in a locking context)
    for arr_ in shared_arr_:
        with arr_.get_ndarray() as array:
            arrs.append(array)
    info = mp.get_logger().info

    spliced_in = list(zip(
        split_arg_ + shared_args_,
        [arr_[aslice] for arr_ in arrs] + list(shared_args_mem_)
    ))
    spliced_in = sorted(spliced_in, key=lambda x: x[0])
    # assemble argument order correctly
    args = list()
    n = 0
    l_args = list(args_)
    while l_args:
        if spliced_in and spliced_in[0][0] == n:
            args.append(spliced_in[0][1])
            spliced_in.pop(0)
        else:
            args.append(l_args.pop(0))
        n += 1
    args.extend([spl[1] for spl in spliced_in])
    args = tuple(args)
    #info(repr(map(type, args)))

    info('{} Applying method {} to slice {} at position {}'.format(timestamp(), method_, aslice, split_arg_))
    then = datetime.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = method_(*args, **kwdict_)
    time_lapse = (datetime.now() - then).total_seconds()
    info('{} method {} slice {} elapsed time: {}'.format(timestamp(), method_, aslice, time_lapse))
    return r
