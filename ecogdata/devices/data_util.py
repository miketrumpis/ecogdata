import os
import numpy as np
import gc
import inspect
from warnings import warn
from ..channel_map import map_intersection
from ..expconfig import session_conf, load_params
from ..expconfig.config_decode import (Parameter,
                                       Path,
                                       TypedParam,
                                       BoolOrNum,
                                       NSequence,
                                       NoneOrStr,
                                       uniform_bunch_case)
from ..expconfig.exp_descr import join_experiments, build_experiment
from .load import (load_wireless,        # -- Several of these might be dropped
                   load_blackrock,
                   load_ddc,
                   load_afe,
                   load_afe_aug21,
                   load_openephys_ddc,
                   load_active,
                   load_mux,
                   load_open_ephys,
                   load_rhd,
                   active_headstages,
                   mux_headstages,
                   FileLoader,
                   OpenEphysLoader,
                   translate_legacy_config_options,
                   RHDLoader,
                   ActiveLoader,
                   DataPathError)
from .load.util import convert_tdms
from ..util import Bunch
from ..parallel import sharedmem as shm
from ..datasource import ElectrodeDataSource, MappedSource, PlainArraySource


_loading = dict(
    wireless=load_wireless,
    blackrock=load_blackrock,
    ddc=load_ddc,
    afe=load_afe,
    afe_aug21=load_afe_aug21,
    ddc_oephys=load_openephys_ddc,
    oephys=load_open_ephys,
    intan_rhd=load_rhd
    )


for hs in mux_headstages:
    _loading[hs] = load_mux


for hs in ('active',) + active_headstages:
    _loading[hs] = load_active

# A table that returns the correct loader and (potentially) an argument translator
_loader_classes = {
    'oephys': (OpenEphysLoader, translate_legacy_config_options),
    'intan_rhd': (RHDLoader, None),
    'active': (ActiveLoader, None)
}

for hs in active_headstages:
    _loader_classes[hs] = (ActiveLoader, None)

_converts_tdms = ('stim_mux64', 'mux3', 'mux4',
                  'mux5', 'mux6', 'mux7', 'mux7_lg', 'active') + \
                  active_headstages


# The keys for this look-up must be lower-case
params_table = {
    # common args
    'exp_path': Path,
    'nwk_path': Path,
    'test': Parameter,
    'electrode': Parameter,
    # FileLoader args
    'bandpass': NSequence,
    'causal_filtering': TypedParam.from_type(bool),
    'notches': NSequence,
    'units': Parameter,
    'load_channels': NSequence,
    'trigger_idx': NSequence,
    'mapped': TypedParam.from_type(bool),
    'resample_rate': TypedParam.from_type(int),
    'use_stored': BoolOrNum,
    'save_downsamp': BoolOrNum,
    'store_path': Path,
    'raise_on_glitch': BoolOrNum,
    # (mostly) common kwargs
    'trigger': TypedParam.from_type(int),
    'snip_transient': BoolOrNum,
    'save': BoolOrNum,
    # active
    'daq': Parameter,
    'headstage': Parameter,
    'row_order': NSequence,
    'bnc': NSequence,
    # afe
    'n_data': TypedParam.from_type(int),
    'range_code': TypedParam.from_type(int),
    'cycle_rate': TypedParam.from_type(float),
    # mux-ish
    'mux_version': Parameter,
    'mux_notches': NSequence,
    'mux_connectors': NSequence,
    'ni_daq_variant': Parameter,
    # blackrock
    'page_size': TypedParam.from_type(int),
    'connections': NSequence,
    'lowpass_ord': TypedParam.from_type(int),
    # ddc
    'drange': TypedParam.from_type(int),
    'fs': TypedParam.from_type(float),
    # open ephys
    'rec_num': NoneOrStr,
    'downsamp': TypedParam.from_type(int),
    'usefs': TypedParam.from_type(float),
    'memmap': BoolOrNum,
    'connectors': NSequence,
    # RHD (mostly shares options with open ephys load)
    }


post_load_params = {
    'car': BoolOrNum,
    'local_ref': BoolOrNum,
}


def parse_load_arguments(session, test, **load_kwargs):
    """
    Combine loader instructions from the combination of config-file settings and call arguments.

    Parameters
    ----------
    session : str
        Name of recording session file
    test : str
        Name of recording to prepare to load
    load_kwargs : dict
        Optional load arguments to overwrite and/or supplement config-file settings

    Returns
    -------
    headstage: str
    electrode: str
    paths: sequence
    extra_pos_arg: tuple
    loader_kwargs: dict

    """
    cfg = session_conf(session, params_table=params_table)
    test_info = cfg.session
    # fill in session info with any specific instructions for the test
    test_info.update(cfg.get(test, {}))

    electrode = test_info.electrode
    headstage = test_info.headstage
    if os.name == 'nt':
        if test_info.exp_path and test_info.exp_path[0] == '/':
            test_info.exp_path = test_info.exp_path[1:]
        if test_info.nwk_path and test_info.nwk_path[0] == '/':
            test_info.nwk_path = test_info.nwk_path[1:]
    test_info.exp_path = test_info.exp_path.replace('//', '/')

    # finally update test info with kwargs which have top priority
    test_info.update(load_kwargs)
    # normalize all test_info parameter keys to be lower case so that
    # they will be detected for any case
    test_info = uniform_bunch_case(test_info)

    load_fn = _loading[headstage]

    # try to parse some args, starting with FileLoader arguments
    a = inspect.getfullargspec(FileLoader)
    vals = a.defaults
    loader_kwargs = dict(zip(a.args[-len(vals):], vals))

    # now go to the module-specific loader function
    a = inspect.getfullargspec(load_fn)
    args = a.args
    vals = a.defaults
    n_pos = len(args) - len(vals)
    # any keyword argument over-rides here are respected
    loader_kwargs.update(dict(zip(a.args[n_pos:], vals)))

    # first three arguments are known (standard), find any others
    extra_pos_names = args[3:n_pos]
    try:
        extra_pos_args = list()
        for n in extra_pos_names:
            extra_pos_args.append(test_info[n.lower()])
    except KeyError:
        raise ValueError('A required load argument is missing: {}'.format(n))

    # now get any keyword arguments from the test info config file
    for n in loader_kwargs.keys():
        if n.lower() in test_info:
            loader_kwargs[n] = test_info.get(n.lower())
    # check to see if any meta-load parameters are present in the given kwargs or the session file
    for n in post_load_params.keys():
        if n.lower() in test_info:
            loader_kwargs[n] = test_info.get(n.lower())
    paths = (test_info.exp_path, test_info.nwk_path)

    return headstage, electrode, paths, extra_pos_args, loader_kwargs


def find_loadable_files(session, recording, downsampled=False):
    """
    Find the first available loadable primary file for a session & recording

    Parameters
    ----------
    session : str
        Session config file ("ini" format)
    recording : str
        Recording name
    downsampled : bool
        If true, and if the config file has a downsampling setting,
        look for a downsampled source file

    Returns
    -------
    source_path: str
        The data file path (or None if nothing was found)

    """
    headstage, electrode, paths, pos_args, opt_args = parse_load_arguments(session, recording)
    if headstage not in _loader_classes:
        return 'unknown daq type {}'.format(headstage)
    load_cls, opt_parser = _loader_classes[headstage]
    if opt_parser:
        putative_args = (paths[0], recording, electrode)
        opt_args = opt_parser(*putative_args, **opt_args)
    if not downsampled:
        opt_args['resample_rate'] = None
    else:
        # Return None (or n/a or unknown) if a downsampled file is asked for,
        # but was not specified in the settings
        if opt_args.get('resample_rate', None) is None:
            return None
    for location in paths:
        args = (location, recording, electrode)
        try:
            loader = load_cls(*args, **opt_args)
            if downsampled:
                if loader.new_downsamp_file is None:
                    return loader.data_file
                else:
                    return None
            else:
                if loader.can_load_primary_data_file:
                    return loader.primary_data_file
                else:
                    return None
        except DataPathError:
            pass
    return None


def load_experiment_auto(session, test, **load_kwargs):
    """
    Loads a recording from the session database system. Hardware and
    multiple other parameters are interpreted/parsed from the database 
    config file. Any arguments specified in load_kwargs take precedence
    and must be literal (e.g. already parsed).

    Parameters
    ----------

    session: str
        Name of recording session in 'group/session-name' syntax
    test: str
        Base name (no extension) of recording. If this is also a section
        in the config file, then further information is taken from that
        section.

    Returns
    -------
    dataset: ElectrodeDataSource
    
    """

    if np.iterable(test) and not isinstance(test, str):
        return load_datasets(session, test, load_kwargs=load_kwargs)

    headstage, electrode, paths, extra_pos_args, loader_kwargs = parse_load_arguments(session, test, **load_kwargs)
    remote_path, local_path = paths
    params = load_params()
    if headstage in _converts_tdms:
        clear = params.clear_temp_converted
        post_fn = convert_tdms(
            test, remote_path, local_path,
            accepted=('.h5', '.mat'), clear=clear
        )

    try:
        exp_path = local_path
        dset = load_experiment_manual(
            exp_path, test, headstage, electrode, *extra_pos_args, **loader_kwargs
        )

    except (IOError, DataPathError) as e:
        try:
            exp_path = remote_path
            dset = load_experiment_manual(
                exp_path, test, headstage, electrode, *extra_pos_args, **loader_kwargs
            )
        except (IOError, DataPathError) as e:
            raise DataPathError('Recording not found')

    if headstage in _converts_tdms and post_fn is not None:
        post_fn()

    dset.exp = build_experiment(session, test, dset.pos_edge)
    dset.name = '.'.join((session, test))  # this should be a the unique ID (?)
    dset.headstage = headstage
    return dset


def load_experiment_manual(exp_path, test, headstage, electrode, *load_args, **load_kwargs):
    """
    Loads a recording given a directory and test name and other labels
    identifying the hardware.  Depending on hardware, further information
    must be given in the load_args sequence. Any load keyword arguments
    must be literal (e.g. already parsed).

    Parameters
    ----------

    exp_path: str
        Path on file system where recordings live
    test: str
        Base name (no extension) of recording.
    headstage: str
        Designated name of headstage.
    electrode: str
        Designated name of electrode.

    Returns
    -------
    dataset: ElectrodeDataSource

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


def load_datasets(session, tests, load_kwargs=dict(), **join_kwargs):
    """
    Append multiple data sets end-to-end to form a single set.
    If StimulatedExperiments are associated with these sets,
    then also join those experiments to reflect the union of all
    conditions presented.

    Parameters
    ----------
    session: str
        name of session in group/session format
    tests: sequence
        sequence of recording names
    load_kwargs: dict
        any further loading options
    join_kwargs: dict
        arguments for join_datasets

    Returns
    -------
    dataset: Bunch
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
    return join_datasets(all_sets, **join_kwargs)


def join_datasets(all_sets, popdata=True, shared_mem=True, source_type=''):
    """Append multiple pre-loaded datasets end-to-end to form a single set.
    If StimulatedExperiments are associated with these sets,
    then also join those experiments to reflect the union of all
    conditions presented. If channel maps differ between datasets, only
    the intersection of all channels is retained in the joined set. Note for
    joining mapped datasources, the underlying data layout (i.e. channel order) of
    each dataset needs to match. However, channel order permutation is supported
    with source_type='loaded'. Original dataset Bunches may be modified in
    this method.

    Parameters
    ----------
    all_sets: Sequence
        Sequene of dataset Bunches
    popdata: bool
        Pop each datasets data array from original Bunch (may reduce memory consumption)
    shared_mem: bool
        Combine data into a shared memory array (if not mapped)
    source_type: str {'mapped', 'loaded'}
        Force the joined datasource to be a MappedSource ('mapped') or a PlainArraySource ('loaded').
        If empty (''), then the source type will match that of the original sources, or will be a MappedSource
        in the case of a mixture.

    Returns
    -------
    dataset: Bunch
        Joined data set
    
    """
    if len(all_sets) == 1:
        return all_sets[0]
    bandpasses = set([dset.bandpass for dset in all_sets])
    if len(bandpasses) > 1:
        warn('Warning: data sets processed under different bandpasses', RuntimeWarning)
    all_len = [d.data.shape[-1] for d in all_sets]
    d_len = np.sum(all_len)
    full_map = map_intersection([d.chan_map for d in all_sets])
    nchan = full_map.sum()
    # find out if there are any leading datasets without experiments,
    # would need to add these samples to the final experiment timestamps
    extra_points = 0
    first_exp = 0
    for n, dataset in enumerate(all_sets):
        if dataset.exp is not None:
            first_exp = n
            break
        extra_points += dataset.data.shape[1]
    experiments = []
    offsets = []
    off = 0
    for dataset in all_sets[n:]:
        # the offsets sequence leads the experiments sequence
        off += dataset.data.shape[1]
        if dataset.exp is None:
            continue
        offsets.append(off)
        experiments.append(dataset.exp)
    # throw away the last offset
    if len(experiments):
        full_exp = join_experiments(experiments, offsets[:-1])
        full_exp.time_stamps += extra_points
    else:
        full_exp = None

    input_type = set([type(d.data) for d in all_sets])
    if len(input_type) > 1:
        if source_type.lower() == 'mapped' or not len(source_type):
            mapped = True
        else:
            mapped = False
    else:
        if source_type.lower() == 'loaded':
            mapped = False
        else:
            input_type = input_type.pop()
            mapped = input_type is MappedSource

    if mapped:
        # if source is to be mapped, then loop through sources and
        # * convert to map (if needed)
        # * apply intersecting channel map mask
        # * join sources (just quick list manipulations)
        for n in range(len(all_sets)):
            dataset = all_sets[n]
            # TODO: there is a problem if loaded dataset was pre-masked -- the mapped source after to_map() will be a
            #  direct map, and probably won't have a buffer that matches other mapped sources...
            #  might have to mirror all sources to direct maps, which would be slow and storage intensive
            if isinstance(dataset.data, PlainArraySource):
                dataset.data = dataset.data.to_map()
            idx = [i for i in range(len(dataset.data)) if full_map[dataset.chan_map.rlookup(i)]]
            mask = np.zeros(len(dataset.data.binary_channel_mask), '?')
            mask[idx] = True
            dataset.data.set_channel_mask(mask)
        joined_data = all_sets[0].data.join(all_sets[1].data)
        for dataset in all_sets[2:]:
            joined_data = joined_data.join(dataset.data)
        joined_set = Bunch(data=joined_data)
    else:
        # if source is to be loaded, then pre-allocate the array and loop through sources:
        # * apply intersecting channel map
        # * place channels into memory
        new_data = dict()
        if shared_mem:
            array_create = shm.shared_ndarray
        else:
            array_create = np.empty
        new_data['data_buffer'] = array_create((nchan, d_len))
        for name in all_sets[0].data.aligned_arrays:
            chans = len(getattr(all_sets[0], name))
            new_data[name] = array_create((chans, d_len))
        offsets = np.r_[0, np.cumsum(all_len)]
        for n in range(len(all_sets)):
            data = all_sets[n].pop('data') if popdata else all_sets[n].data
            channel_map = all_sets[n].chan_map
            idx = [i for i in range(len(data)) if full_map[channel_map.rlookup(i)]]
            data_slice = np.s_[:, offsets[n]:offsets[n + 1]]
            if isinstance(data, MappedSource):
                mask = np.zeros(len(data.binary_channel_mask), '?')
                mask[idx] =  True
                data.set_channel_mask(mask)
                # mirror this mapped set directly into the pre-allocated array
                sources = dict()
                for name in new_data:
                    sources[name] = new_data[name][data_slice]
                data = data.mirror(mapped=False, copy='all', new_sources=sources)
            else:
                new_data['data_buffer'][data_slice] = np.take(data, idx, axis=0)
                for name in data.aligned_arrays:
                    new_data[name][data_slice] = getattr(data, name)
        joined_data = new_data.pop('data_buffer')
        joined_set = Bunch()
        joined_set.data = PlainArraySource(joined_data, shared_mem=True, **new_data)
        del new_data
        del data

    # Now fix up the dataset
    # 1) promote aligned arrays to top-level
    for name in joined_set.data.aligned_arrays:
        joined_set[name] = getattr(joined_set.data, name)
    # 2) join names
    session = all_sets[0].name.split('.')[0]
    tests = [s.name.split('.')[1] for s in all_sets]
    joined_set.name = session + '.' + ','.join(tests)
    # 3) touch up trigger events
    joined_set.exp = full_exp
    if full_exp is not None:
        joined_set.pos_edge = full_exp.time_stamps
    # 4) append intersecting ChannelMap
    joined_set.chan_map = all_sets[0].chan_map.subset(full_map)
    # last) copy other keys
    dataset = all_sets[0]
    for key in dataset:
        if key in joined_set:
            continue
        joined_set[key] = dataset[key]

    # Attempt to garbage collect any crud
    del all_sets
    while gc.collect():
        pass

    return joined_set
