from .mproc import timestamp
from .mproc import parallel_controller, parallel_context as pctx
from contextlib import closing
import warnings
from functools import wraps
import numpy as np
from ecogdata.util import get_default_args
from datetime import datetime


def parallel_runner(method, static_args, split_arg, shared_split_arrays, shared_args, shared_full_arrays,
                    splice_args, kwargs, n_jobs, info_logger):
    """
    Runs method in a Pool with split and full shared memory.

    Initialization of the Pool creates a method wrapper in each subprocess. By mapping out slices to the
    subprocesses, the wrapper function calls the original method on slices of the input array(s).

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
    n_jobs = min(n_jobs, split_shapes[0][0])
    init_args = (split_arg,
                 shared_split_arrays,
                 split_shapes,
                 shared_args,
                 shared_full_arrays,
                 method,
                 static_args,
                 kwargs)
    ctx = pctx.ctx
    ctx.freeze_support()
    info_logger('{} Creating pool'.format(timestamp()))
    with closing(ctx.Pool(processes=n_jobs, initializer=_init_globals, initargs=init_args)) as p:
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


def split_at(split_arg=(0,), splice_at=(0,), shared_args=(), split_kwargs=(),
             split_over=None, n_jobs=-1, concurrent=False):
    """
    Returns a decorator that enables parallel dispatch over multiple blocks of input ndarrays.

    The decorator returned here splits one or more inputs on their first dimension across multiple processes for
    automatic parallelization of row-by-row operations. Other arrays may be shared by all processes. Zero or more
    return values are spliced together after the jobs are done.

    Input ndarrays are wrapped with shared memory pointers (method depends on whether processes are "fork" or "spawn"
    mode). These arrays can be modified by subproceses if they were originally defined as shared memory ctypes Arrays
    a la `ecogdata.parallel.sharedmem.shared_ndarray`. For "fork" mode, regular ndarray pointers in main memory can be
    read-only accessed. For "spawn" mode, these arrays need to be copied to shared memory.

    Parameters
    ----------
    split_arg: tuple
        Position(s) of any arguments in the method's signature that should be split for dispatch
    splice_at: tuple
        Position(s) of any return arguments that should be combined for output
    shared_args: tuple
        Position of any array arguments that should be available to all workers as shared memory
    split_kwargs: tuple
        Position of potential arrays to split, if they are present in the keyword arguments
    split_over: int
        If not None, only split the array(s) if they're over this many MBs
    n_jobs: int
        Number of workers (if -1 then use all cpus)
    concurrent: bool
        If True, then the shared memory (indexed by shared_args) will be accessed with with thread locks.

    """
    info = pctx.get_logger().info
    info('{} Starting wrap'.format(timestamp()))
    # normalize inputs
    if not np.iterable(splice_at):
        splice_at = (splice_at,)
    if not np.iterable(split_arg):
        split_arg = (split_arg,)

    splice_at = tuple(splice_at)
    split_arg = tuple(split_arg)
    shared_args = tuple(shared_args)
    if not isinstance(split_kwargs, tuple):
        split_kwargs = (split_kwargs,)
    if n_jobs < 0:
        n_jobs = pctx.cpu_count()

    def check_parallel_usage(*args, **kwargs):
        # check if there are even multiple rows in the array!!
        arr = args[split_arg[0]]
        if arr.ndim == 1 or arr.shape[0] == 1:
            return False
        # check if the parallel context is false
        if not parallel_controller.state:
            return False
        # check if the arrays are too small to bother with subprocesses
        if split_over is not None:
            mx_size = 0
            for p in split_arg:
                arr = args[p]
                size = arr.size * arr.dtype.itemsize
                if size > mx_size:
                    mx_size = size
            if mx_size / 1024 / 1024 < split_over:
                return False
        return True

    #@decorator
    #def inner_split_method(method, *args, **kwargs):

    def run_parallel(method):

        @wraps(method)
        def inner_split_method(*args, **kwargs):

            # This decorator scans the arguments to create shared memory wrappers of arrays
            # and then calls the parallel runner for dispatch of the method

            only_check = kwargs.pop('check_parallel', False)
            para = check_parallel_usage(*args, **kwargs)
            if only_check:
                return para
            elif not para:
                return method(*args, **kwargs)
            # Check if there are arrays present in the split_kwargs potential splits
            # Filter split_kwargs for actually present data
            really_split = tuple([k for k in split_kwargs if isinstance(kwargs.get(k, None), np.ndarray)])
            if really_split:
                # modify method to enable splitting
                run_method = fix_signature_for_split(*really_split, cancel_output=False)(method)
                # global split_arg
                full_split_arg = split_arg + tuple([len(args) + i for i in range(len(really_split))])
                # split_arg = new_split_arg
                args = args + tuple([kwargs.pop(k) for k in really_split])
            else:
                full_split_arg = split_arg
                run_method = method
            # print('kwargs:', list(kwargs.keys()))
            # print('split_kwargs:', split_kwargs)
            # print('really splitting:', really_split)
            # print('number of args', len(args), full_split_arg)
            pop_args = sorted(full_split_arg + shared_args)
            shared_array_shm = list()
            n = 0
            args = list(args)
            split_array_shm = list()
            for pos in pop_args:
                pos = pos - n
                a = args.pop(pos)
                info('{} Wrapping shared memory size {} MB'.format(timestamp(), a.size * a.dtype.itemsize / 1024. / 1000.))
                x = pctx.SharedmemManager(a, use_lock=concurrent)
                if pos + n in full_split_arg:
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
            return parallel_runner(run_method, static_args, full_split_arg, split_array_shm, shared_args,
                                   shared_array_shm, splice_at, kwargs, n_jobs, info)

        return inner_split_method

    # Work in progress -- figure out how to attach checker as an attribute
    # inner_split_method.uses_parallel = check_parallel_usage

    return run_parallel


def to_void(method, clear_position=0):
    """
    Turn a single-output method into a void method by appending a positional argument
    to hold output. Now input & output arrays can be split shared memory.

    """

    @wraps(method)
    def void_method(*args, **kwargs):
        output = args[-1]
        method_args = args[:-1]
        r = method(*method_args, **kwargs)
        # gobble cleared output position, return others if applicable
        if isinstance(r, tuple):
            output[:] = r[clear_position]
            return r[:clear_position] + r[clear_position + 1:]
        output[:] = r

    return void_method


def fix_signature_for_split(*kwargs_to_args, cancel_output=False):
    """
    Create a decorator that returns a method that puts selected kwargs
    into positional arg slots. For n original args, the new call looks like this:

    method_expand(arg_1, ..., arg_n, arg_n+1, ..., **new_kwargs)

    For the common case where the optional argument is an output buffer,
    then it also makes sense to cancel output values that would need to be
    serialized in multiprocessing.

    Parameters
    ----------
    kwargs_to_args :
    cancel_output :

    """
    # print(kwargs_to_args, isinstance(kwargs_to_args, tuple))
    if not isinstance(kwargs_to_args, tuple):
        kwargs_to_args = (kwargs_to_args,)
    # print(kwargs_to_args)
    def expands_signature(method):
        if not len(kwargs_to_args):
            return method

        @wraps(method)
        def expanded(*args, **kwargs):
            n_expanded = len(kwargs_to_args)
            n_orig = len(args) - n_expanded
            # print('expanded args:', n_expanded, 'orig args:', n_orig)
            orig_args = args[:n_orig]
            # print('kw to replace:', kwargs_to_args)
            for i, kw in enumerate(kwargs_to_args):
                kwargs[kw] = args[n_orig + i]
            # print('kw:', len(kwargs), kwargs[kw].shape)
            # print('args:', len(orig_args), [o.shape for o in orig_args])
            r = method(*orig_args, **kwargs)
            if cancel_output:
                # if multiple outputs, assume that first output should be hidden
                if isinstance(r, tuple):
                    r = r[1:]
                else:
                    r = None
            return r
        return expanded

    return expands_signature


def allocate_output(call_args: tuple, call_kwargs: dict, split_kwargs: dict,
                    inplace: bool=False, shared: bool=True, same_shape: bool=True, calc_shape: callable=None):
    """
    Allocate memory for output from parallel jobs

    """
    if not (same_shape or calc_shape):
        raise ValueError
    default_args = get_default_args(split_at)
    split_input = call_args[split_kwargs.get('split_arg', default_args['split_arg'])[0]]
    if same_shape:
        shape = split_input.shape
        dtype = split_input.dtype
    else:
        shape, dtype = calc_shape(call_args, call_kwargs)
    if inplace:
        # print('using input for output')
        # the output will be placed in the split arg
        # (or the first split arg, if that makes sense?)
        out = split_input
    elif shared:
        # print('creating shared output')
        out = pctx.shared_ndarray(shape, typecode=dtype.char)
    else:
        # print('creating plain output')
        # don't bother with shared
        out = np.empty(shape, dtype=dtype)
    return out


def fix_output(canceled_output, other_output):
    """
    Restore the full output when the main output was blocked in processes

    """
    if other_output is None:
        return canceled_output
    if isinstance(other_output, tuple):
        return (canceled_output,) + other_output
    return (canceled_output, other_output)


def split_optional_output(output_kwarg, runs_inplace=False, same_shape: bool=True, calc_shape: callable=None,
                          **split_kwargs):
    """

    Parameters
    ----------
    output_kwarg : basestring
        Name of the optional output argument
    runs_inplace : bool
        If the method changes input data inplace under the default output, set runs_inplace=True.
    split_kwargs : dict
        Parameters for the split_at function

    """
    # split_arg = split_kwargs.pop('split_arg', (0,))

    def run_para_method(method):
        @wraps(method)
        def run_as_void(*args, **kwargs):
            # method_has_output_arg = True
            default_args = get_default_args(method)
            default_out = default_args[output_kwarg]
            out = kwargs.pop(output_kwarg, default_out)
            inplace = kwargs.pop('inplace', False)
            void_method = fix_signature_for_split(output_kwarg, cancel_output=True)(method)
            # print('out was', type(out))
            # If the method normally runs inplace and the output arg is None,
            # then run the void wrapped method with just the normal array splitting.
            # In all other cases, split the output variable
            if runs_inplace and out is default_out:
                para_method = split_at(**split_kwargs)(method)
                full_args = args
            else:
                void_split_args = split_kwargs.copy()
                # split the last arg that is added by to_void()
                split_arg = split_kwargs.get('split_arg', get_default_args(split_at)['split_arg'])
                full_split_arg = split_arg + (len(args),)
                void_split_args['split_arg'] = full_split_arg
                # split and run as void, then return the output
                para_method = split_at(**void_split_args)(void_method)
                if out is default_out:
                    use_shared = para_method(*args, None, check_parallel=True)
                    try:
                        out = allocate_output(args, kwargs, void_split_args, inplace=inplace, shared=use_shared,
                                              same_shape=same_shape, calc_shape=calc_shape)
                    except ValueError:
                        raise ValueError('output argument "out" is required')

                # if out is default_out:
                #     split_input = args[split_arg[0]]
                #     if para_method(*args, None, check_parallel=True):
                #         print('creating shared output')
                #         shape = split_input.shape
                #         tcode = split_input.dtype.char
                #         out = pctx.shared_ndarray(shape, typecode=tcode)
                #     else:
                #         print('creating plain output')
                #         # don't bother with shared
                #         out = np.empty_like(split_input)
                full_args = args + (out,)
            # Run the method and return the output (or None, if output was not set)
            r = para_method(*full_args, **kwargs)
            return fix_output(out, r)

        return run_as_void

    return run_para_method


def split_output(position=0, same_shape: bool=True, calc_shape: callable=None, **split_kwargs):
    """
    Split a single-output method by turning it into a void method and splitting
    the input-output arrays. The wrapped method also supports new keyword arguments:

    * out: output shared array
    * inplace (bool): use the input shared array in place of out

    If same_shape==True, then shared output will be created if out is None.
    Otherwise, out is considered required and a ValueError will be raised if out is None.

    Parameters
    ----------
    position: int
        Position of output to disable
    split_kwargs : dict
        Parameters for the split_at function

    """
    # split_arg = split_kwargs.pop('split_arg', (0,))

    def para_void_method(method):
        @wraps(method)
        def run_para_void(*args, **kwargs):
            out = kwargs.pop('out', None)
            void_method = to_void(method, clear_position=position)
            # print('out was', type(out))
            inplace = kwargs.pop('inplace', False)

            void_split_args = split_kwargs.copy()
            # split the last arg that is added by to_void()
            split_arg = split_kwargs.get('split_arg', get_default_args(split_at)['split_arg'])
            full_split_arg = split_arg + (len(args),)
            void_split_args['split_arg'] = full_split_arg
            # split and run as void, then return the output
            para_method = split_at(**void_split_args)(void_method)
            if out is None:
                use_shared = para_method(*args, None, check_parallel=True)
                try:
                    out = allocate_output(args, kwargs, split_kwargs, inplace=inplace, shared=use_shared,
                                          same_shape=same_shape, calc_shape=calc_shape)
                except ValueError:
                    raise ValueError('output argument "out" is required')
                #
                # print('out is None')
                # if not (same_shape or calc_shape):
                #     raise ValueError('output argument "out" is required')
                # split_input = args[split_arg[0]]
                # if same_shape:
                #     shape = split_input.shape
                #     dtype = split_input.dtype
                # else:
                #     shape, dtype = calc_shape(*args, **kwargs)
                # if inplace:
                #     print('using input for output')
                #     # the output will be placed in the split arg
                #     # (or the first split arg, if that makes sense?)
                #     out = split_input
                # elif para_method(*args, None, check_parallel=True):
                #     print('creating shared output')
                #     out = pctx.shared_ndarray(shape, typecode=dtype.char)
                # else:
                #     print('creating plain output')
                #     # don't bother with shared
                #     out = np.empty(shape, dtype=dtype)
            full_args = args + (out,)
            r = para_method(*full_args, **kwargs)
            return fix_output(out, r)

        return run_para_void

    return para_void_method





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
    # Check if None was returned (weird way to do it...)
    # If this was a void function then nothing to splice.
    if [x for x in map_list if x is None]:
        return
    # if there is only a single array returned, concatenate it and return
    if isinstance(map_list[0], np.ndarray):
        res = np.concatenate(map_list, axis=0)
        return res
    # If here, multiple values were returned: look for arrays to splice
    splice_at = sorted(splice_at)
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
    """
    Very light wrapper of SharedmemManager allowing slice-like access.
    Lock access determined by the given SharedmemManager.
    """

    # TODO: only has array like *access*, not any shape or dtype (etc) info
    def __init__(self, mem_mgr: pctx.SharedmemManager):
        self.mem_mgr = mem_mgr

    def __getitem__(self, idx):
        # TODO: is a copy really needed here? will any subsequent access to this slice, even if read-only, interfere
        #  with other processes? I guess probably yes?
        with self.mem_mgr.get_ndarray() as shm_ndarray:
            return shm_ndarray[idx].copy()


def _init_globals(split_args, split_mem, split_arr_shape, shared_args, shared_mem, method, args, kwdict):
    """
    Initialize the pool worker global state with the method to run and shared arrays + info + other args and kwargs

    Parameters
    ----------
    split_args: sequence
        sequence of positions of argument that are split over
    split_mem: sequence
        the shared memory managers for these split args
    solit_arr_shape: sequence
        the shapes of the underlying ndarrays of the split args
    shared_args: sequence
        positions for readonly, non-split shared memory args
    shared_mem: sequence
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
    shared_arr_ = split_mem
    global shape_
    shape_ = split_arr_shape
    global split_args_
    split_args_ = split_args

    # globals for secondary shared memory arguments
    global shared_args_
    shared_args_ = shared_args
    global shared_args_mem_
    shared_args_mem_ = tuple([shared_readonly(mm) for mm in shared_mem])

    # globals for pickled method and other arguments
    global method_
    method_ = method
    global args_
    args_ = args
    global kwdict_
    kwdict_ = kwdict

    info = pctx.get_logger().info
    info('{} Initialized globals'.format(timestamp()))


def _global_method_wrap(aslice):
    """
    Wrapper method depending on this job's slice. This is the target of Pool.map.
    The method and all inputs are in global namespace.

    Parameters
    ----------
    aslice: slice
        Slice of the input array(s) to feed to the method.

    Returns
    -------
    method results

    """
    arrs = []
    # cycle through split arrays and translate to ndarray (possibly in a locking context)
    # TODO: to truly lock, the method should be called under an ExitStack that stacks
    #  array locking contexts
    for arr_ in shared_arr_:
        with arr_.get_ndarray() as array:
            arrs.append(array)
    info = pctx.get_logger().info

    # create (arg_idx, arr) pairs for all split arrays (now properly sliced) and all shared arrays
    spliced_in = list(zip(
        split_args_ + shared_args_,
        [arr_[aslice] for arr_ in arrs] + list(shared_args_mem_)
    ))
    # sort these pairs by the argument index
    spliced_in = sorted(spliced_in, key=lambda x: x[0])
    # assemble argument order correctly
    args = list()
    n = 0
    l_args = list(args_)
    # step through arguments and drop in shared mem (case 1) or original arg (case 2)
    while l_args:
        if spliced_in and spliced_in[0][0] == n:
            args.append(spliced_in[0][1])
            spliced_in.pop(0)
        else:
            args.append(l_args.pop(0))
        n += 1
    # add any more shared ararys (why would there be extra??)
    args.extend([spl[1] for spl in spliced_in])
    args = tuple(args)
    # time to drive the method
    info('{} Applying method {} to slice {} at position {}'.format(timestamp(), method_, aslice, split_args_))
    then = datetime.now()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = method_(*args, **kwdict_)
    time_lapse = (datetime.now() - then).total_seconds()
    info('{} method {} slice {} elapsed time: {}'.format(timestamp(), method_, aslice, time_lapse))
    return r
