from __future__ import print_function
from builtins import zip
from builtins import map
from builtins import range
import numpy as np
import gc
import inspect

from ecogdata.util import ChannelMap, map_intersection, mat_to_flat

from ecogdata.expconfig import *
from ecogdata.expconfig.exp_descr import join_experiments
from .load import *
from .load.util import convert_tdms

from ecogdata.parallel.array_split import shared_ndarray
from ecogdata.expconfig.config_decode import Parameter, TypedParam, BoolOrNum, NSequence, NoneOrStr, uniform_bunch_case

_loading = dict(
    wireless=load_wireless,
    blackrock=load_blackrock,
    ddc=load_ddc,
    afe=load_afe,
    afe_aug21=load_afe_aug21,
    ddc_oephys=load_openephys_ddc,
    oephys=load_open_ephys
    )


for hs in mux_headstages:
    _loading[hs] = load_mux


for hs in ('active',) + active_headstages:
    _loading[hs] = load_active


_converts_tdms = ('stim_mux64', 'mux3', 'mux4',
                  'mux5', 'mux6', 'mux7', 'mux7_lg', 'active') + \
                  active_headstages


# The keys for this look-up must be lower-case
params_table = {
    # common args
    'exp_path' : Path,
    'test' : Parameter,
    'electrode' : Parameter,
    # (mostly) common kwargs
    'bandpass' : NSequence,
    'notches' : NSequence,
    'trigger' : TypedParam.from_type(int),
    'snip_transient' : BoolOrNum,
    'units' : Parameter,
    'save' : BoolOrNum,
    'bnc' : NSequence,
    # active
    'daq' : Parameter,
    'headstage' : Parameter,
    'row_order' : NSequence,
    # afe 
    'n_data' : TypedParam.from_type(int),
    'range_code' : TypedParam.from_type(int),
    'cycle_rate' : TypedParam.from_type(float),
    # mux-ish
    'mux_version' : Parameter,
    'mux_notches' : NSequence,
    'mux_connectors' : NSequence,
    'ni_daq_variant' : Parameter,
    # blackrock
    'page_size' : TypedParam.from_type(int),    
    'connections' : NSequence,
    'lowpass_ord' : TypedParam.from_type(int),
    # ddc
    'drange' : TypedParam.from_type(int),
    'fs' : TypedParam.from_type(float),
    # open ephys -- this case is tricky in general, but can be ducked here
    'rec_num' : NoneOrStr,
    'downsamp' : TypedParam.from_type(int),
    'trigger_idx' : NSequence,
    'usefs' : TypedParam.from_type(float),
    'save_downsamp' : BoolOrNum,
    'store_path' : Path,
    'use_stored' : BoolOrNum,
    'memmap' : BoolOrNum,
    'connectors' : NSequence,
    }



post_load_params = {
    'car': BoolOrNum,
    'local_ref': BoolOrNum,
}


def load_experiment_auto(session, test, **load_kwargs):
    """
    Loads a recording from the session database system. Hardware and
    multiple other parameters are interpreted/parsed from the database 
    config file. Any arguments specified in load_kwargs take precedence
    and must be literal (e.g. already parsed).

    Parameters
    ----------

    session : str
        Name of recording session in 'group/session-name' syntax
    test : str
        Base name (no extension) of recording. If this is also a section
        in the config file, then further information is taken from that
        section.
    
    """

    if np.iterable(test) and not isinstance(test, str):
        return append_datasets(session, test, **load_kwargs)

    cfg = session_conf(session, params_table=params_table)
    test_info = cfg.session
    # fill in session info with any specific instructions for the test
    test_info.update(cfg.get(test, {}))
    # normalize all test_info parameter keys to be lower case so that
    # they will be detected for any case
    test_info = uniform_bunch_case(test_info)

    electrode = test_info.electrode
    headstage = test_info.headstage
    if os.name == 'nt':
        if test_info.exp_path[0] == '/':
            test_info.exp_path = test_info.exp_path[1:]
        if test_info.nwk_path[0] == '/':
            test_info.nwk_path = test_info.nwk_path[1:]
    test_info.exp_path = test_info.exp_path.replace('//', '/')

    # finally update test info with kwargs which have top priority
    test_info.update(load_kwargs)

    load_fn = _loading[headstage]

    # try to parse some args
    a = inspect.getargspec(load_fn)
    args = a.args
    vals = a.defaults
    n_pos = len(args) - len(vals)

    # first three arguments are known (standard), find any others
    extra_pos_names = args[3:n_pos]
    try:
        extra_pos_args = list()
        for n in extra_pos_names:
            extra_pos_args.append(test_info[n.lower()])
    except KeyError:
        raise ValueError('A required load argument is missing: {}'.format(n))

    # now get keyword arguments
    kws = dict(zip(args[n_pos:], vals))
    for n in kws.keys():
        if n.lower() in test_info:
            kws[n] = test_info.get(n)
    # check to see if any meta-load parameters are present in the given kwargs or the session file
    for n in post_load_params.keys():
        if n.lower() in test_info:
            kws[n] = test_info.get(n)

    if headstage in _converts_tdms:
        clear = params.clear_temp_converted
        post_fn = convert_tdms(
            test, test_info.nwk_path, test_info.exp_path,
            accepted=('.h5', '.mat'), clear=clear
        )

    try:
        exp_path = test_info.exp_path
        dset = load_experiment_manual(
            exp_path, test, headstage, electrode, *extra_pos_args, **kws
        )

    except (IOError, DataPathError) as e:
        try:
            exp_path = test_info.nwk_path
            dset = load_experiment_manual(
                exp_path, test, headstage, electrode, *extra_pos_args, **kws
            )
        except (IOError, DataPathError) as e:
            raise DataPathError('Recording not found')

    if headstage in _converts_tdms and post_fn is not None:
        post_fn()

    dset.exp = build_experiment(session, test, dset.pos_edge)
    dset.name = '.'.join((session, test))  # this should be a the unique ID (?)
    dset.headstage = headstage
    return dset


def load_experiment_manual(
        exp_path, test, headstage, electrode, *load_args, **load_kwargs
        ):
    """
    Loads a recording given a directory and test name and other labels
    identifying the hardware.  Depending on hardware, further information
    must be given in the load_args sequence. Any load keyword arguments
    must be literal (e.g. already parsed).

    Parameters
    ----------

    exp_path : str
        Path on file system where recordings live

    test : str
        Base name (no extension) of recording. 

    headstage : str
        Designated name of headstage.

    electrode : str
        Designated name of electrode.

    """

    load_fn = _loading[headstage]
    load_args = (exp_path, test, electrode) + load_args
    post_proc_args = dict()
    for k in post_load_params:
        post_proc_args[k] = load_kwargs.pop(k, None)

    dset = load_fn(*load_args, **load_kwargs)
    # experiment will have to be constructed separately,
    # or go through session database system
    com_avg = post_proc_args.pop('car', False)
    if com_avg:
        mn = dset.data.mean(0)
        dset.data -= mn

    # Local ref either goes to reference data (if the electrode has reference
    # channels), or it can be supplied as a channel number
    local_ref = post_proc_args.pop('local_ref', None)
    if isinstance(local_ref, bool):
        if local_ref:
            if 'ref_chans' in dset:
                ref = np.atleast_2d(dset.ref_chans).mean(0)
            else:
                print('Local re-ref triggered, but no reference channels available')
                ref = None
        else:
            # need to reset this b/c isinstance(False, int) evals to true!!
            local_ref = None
            ref = None
    elif isinstance(local_ref, int):
        ref = dset.data[local_ref]
    else:
        ref = None
    if ref is not None:
        dset.data -= ref

    dset.exp = None
    dset.name = '.'.join((os.path.basename(exp_path), test))
    dset.headstage = headstage
    return dset


def append_datasets(session, tests, **load_kwargs):
    """
    Append multiple data sets end-to-end to form a single set.
    If StimulatedExperiments are associated with these sets,
    then also join those experiments to reflect the union of all
    conditions presented.

    Parameters
    ----------

    session : name of session in group/session format
    tests : sequence of recording names
    load_kwargs : any further loading options

    Returns
    -------

    joined data set
    
    """

    
    if isinstance(tests, str):
        tests = (tests,)
    if isinstance(tests, (list, tuple)):
        try:
            tests = list(map(int, tests))
            tests = ['test_%03d'%t for t in tests]
        except:
            # assume it's already good
            pass
    
    all_sets = [load_experiment_auto(session, test, **load_kwargs) for test in tests]
    return join_datasets(all_sets)


def join_datasets(all_sets, popdata=True, rasterize=True, shared_mem=True):
    """Append multiple pre-loaded datasets end-to-end to form a single set.
    If StimulatedExperiments are associated with these sets,
    then also join those experiments to reflect the union of all
    conditions presented. If channel maps differ between datasets, only
    the intersection of all channels is retained in the joined set.

    Parameters
    ----------

    all_sets : sequence of dataset Bunches
    pop_data : {True/False} pop each datasets data array (may reduce
               memory consumption)
    rasterize : {True/False} re-index arrays to be in array raster order
    shaerd_mem : {True/False} combine data into a shared memory array

    Returns
    -------

    joined data set
    
    """
    if len(all_sets) == 1:
        return all_sets[0]
    all_len = [dset.data.shape[-1] for dset in all_sets]
    d_len = np.sum(all_len)
    full_map = map_intersection([d.chan_map for d in all_sets])
    nchan = full_map.sum()
    full_exp = join_experiments([dset.exp for dset in all_sets], np.cumsum(all_len))
    ii, jj = full_map.nonzero()

    if shared_mem:
        full_data = shared_ndarray((nchan, d_len))
    else:
        full_data = np.empty((nchan, d_len))
    offsets = np.r_[0, np.cumsum(all_len)]
    for n in range(len(all_sets)):
        data = all_sets[n].pop('data') if popdata else all_sets[n].data
        # need to find the set of channels in this set that is
        # consistent with the entire data set. Might as well
        # put them in "order" here
        cmap = all_sets[n].chan_map
        # this should be the raster order of the full_map
        # NO -- this is wrong, it should be
        # ii, jj = full_map.nonzero()
        # [ cmap.lookup(i,j) for i,j in zip( ii, jj ) ]
        if rasterize:
            #idx = [cmap.index(i) for i in cmap.subset(full_map)]
            idx = [cmap.lookup(i,j) for i,j in zip( ii, jj )]
        else:
            idx = [i for i in range(len(data)) if full_map[cmap.rlookup(i)]]
        full_data[:, offsets[n]:offsets[n+1]] = data[idx]
        del data
        gc.collect()

    
    bandpasses = set([dset.bandpass for dset in all_sets])
    assert len(bandpasses) == 1, 'Warning: data sets processed under different bandpasses'

    full_set = Bunch()

    non_data_series = ('bnc', 'ground_chans', 'trig', 'pos_edge')
    for timeseries in non_data_series:
        if timeseries in all_sets[0]:
            full_set[timeseries] = np.concatenate(
                [dset.get(timeseries) for dset in all_sets], axis=-1
                )
    
    full_set.data = full_data
    full_set.exp = full_exp
    if rasterize:
        pitch = all_sets[0].chan_map.pitch
        full_set.chan_map = ChannelMap(
            mat_to_flat(
                full_map.shape, ii, jj, col_major=False
                ),
            full_map.shape, pitch=pitch, col_major=False
            )
    else:
        full_set.chan_map = all_sets[0].chan_map.subset(full_map)
    session = all_sets[0].name.split('.')[0]
    tests = [s.name.split('.')[1] for s in all_sets]
    full_set.name = session + '.' + ','.join(tests)

    # get remaining keys after 
    # {'data', 'bnc', 'exp', 'ground_chans', 'name'}
    keys = set(all_sets[0].keys())
    keys.difference_update(('data', 'exp', 'name', 'chan_map') + non_data_series)
    for k in keys:
        print(k)
        full_set[k] = all_sets[0].get(k)

    del dset
    del all_sets
    gc.collect()
    return full_set
