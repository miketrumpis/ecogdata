import os
import os.path as osp
import pandas as pd
import numpy as np
from glob import glob
import warnings
from .global_config import new_SafeConfigParser, cfg_to_bunch, load_params


__all__ = ['session_groups', 'available_sessions', 'find_conf',
           'session_conf', 'session_info', 'locate_old_session',
           'sessions_to_delta', 'session_table']

params = load_params()
# _cpath is the single session config database path with
# multiple sub-directories indicating different recording groups
if 'user_sessions' not in params or not params.user_sessions:
    warnings.warn('A session config path was not found.', UserWarning)
    # fall back on this path, which is probably empty
    _cpath = osp.join(osp.dirname(__file__), 'sessions')
else:
    _cpath = osp.abspath(params.user_sessions)


def find_conf(conf, extra_paths=None):
    """
    Session config "conf" is given in "group/session" syntax.
    Use "extra_paths" to specify anywhere else the "session" file
    might be found.
    """

    group, session = conf.split('/')
    if extra_paths is None:
        extra_paths = list()
    extra_paths.insert(0, osp.join(_cpath, group))
    session, _ = osp.splitext(session)

    for test_path in extra_paths:
        test_path = os.path.expanduser(test_path)
        p1 = osp.join(test_path, session + '.txt')
        if osp.exists(p1):
            return p1
        p2 = osp.join(test_path, session + '_conf.txt')
        if osp.exists(p2):
            return p2

    raise IOError('config file not found: ' + conf)


def __isdir(p):
    """Returns False for hidden directories"""
    p_part = osp.split(osp.abspath(p))[1]
    if p_part[0] == '.':
        return False
    return osp.isdir(p)


def session_groups():
    return [name for name in os.listdir(_cpath)
            if __isdir(osp.join(_cpath, name))]


def available_sessions(group=''):
    groups = [group] if len(group) else session_groups()
    txt_files = []
    for g in groups:
        g_files = glob(osp.join(osp.join(_cpath, g), '*.txt'))
        txt_files.extend(sorted(g_files))
    conf_files = list()
    cp = new_SafeConfigParser()
    for txt in txt_files:
        try:
            cp.read(txt)
            conf_files.append(txt)
        except BaseException:
            pass

    def _two_path(x):
        p1, x = osp.split(x)
        _, p2 = osp.split(p1)
        return '/'.join((p2, x))
    sessions = list(map(_two_path, conf_files))
    return [s.replace('.txt', '').replace('_conf', '') for s in sessions]


def locate_old_session(session, use_first=False):
    session = session.strip('.txt').replace('_conf', '')
    sessions = available_sessions()
    possibles = [s for s in sessions if s.endswith(session)]
    if len(possibles):
        if use_first:
            return possibles[0]
        print('Possible matches:')
        for n, s in enumerate(possibles):
            print('\t{0}\t{1}'.format(n, s))
        choice = input('Enter session choice (number [0]): ')
        if not choice.strip():
            choice = '0'
        return possibles[int(choice)]
    else:
        print('No matches')


def session_conf(session, params_table=None):
    """Return a Bunch-ified config file reporting the entire session"""
    return cfg_to_bunch(find_conf(session), params_table=params_table)


def session_info(session, params_table=None):
    """Return default info for a session"""
    cfg = session_conf(session, params_table=params_table)
    return cfg.session


def sessions_to_delta(sessions, reference=None, num=False, sortable=False):
    """
    Convert multiple session dates to day numbers. Days count relative to
    the first given session, or to a "reference" date. The date substring
    MUST be strictly in YYYY-MM-DD format.

    Paramters
    ---------

    sessions : sequence of session strings
    reference : (default None) optional reference date
    num : (default False) return numbers (as opposed to "Day X" strings)
    sortable : (default False) return date strings that sort correctly
    """

    import re
    from datetime import datetime
    fmt = '%Y-%m-%d'
    dates = list()
    # strict pattern YYYY-MM-DD
    pattern = re.compile(r'(\d{4})-(\d{2})-(\d{2})')
    for s in sessions:
        m = re.search(pattern, s)
        if m is None:
            raise ValueError('Session {0} not in YYYY-MM-DD format'.format(s))
        dates.append('-'.join(m.groups()))
    if reference is not None:
        m = re.search(pattern, reference)
        if m is None:
            raise ValueError(
                'Reference {0} not in YYYY-MM-DD format'.format(reference)
            )
        s0 = '-'.join(m.groups())
    else:
        s0 = dates[0]
    d0 = datetime.strptime(s0, fmt)
    deltas = [(datetime.strptime(s, fmt) - d0).days for s in dates]
    if num:
        return deltas
    if sortable and len(deltas) > 1:
        dig = int(np.log10(max(deltas))) + 1
        deltas = ['Day {day:{width}}'.format(day=d, width=dig)
                  for d in deltas]
    else:
        deltas = ['Day {0}'.format(d) for d in deltas]
    return deltas


# Framework for filtering some of the option values
# (e.g. to decode that tones_tab = "1000" codes for clicks)
class ValueFilter:

    def __call__(self, s):
        return s


class DefaultFilter(ValueFilter):
    pass


class SwapVal(ValueFilter):

    def __init__(self, input_val, output_val):
        self.in_str = input_val.lower()
        self.out_str = output_val

    def __call__(self, s):
        if s.lower() == self.in_str:
            return self.out_str
        return s


_value_filters = {
    'tones_tab': SwapVal('1000', 'clicks')
}


def filter_values(lookup):
    for key in lookup:
        new_val = _value_filters.get(key, DefaultFilter())(lookup[key])
        lookup[key] = new_val
    return lookup


def session_table(session, global_params=False, source_info=True, path_info=True):
    """
    Create a Pandas DataFrame that tabulates the information in a session file.
    Parameters
    ----------
    session : str
        Session config file ("ini" syntax)
    global_params : bool
        If true, report all inherited global settings
    source_info : bool
        If true, report data file locations (if found)
    path_info : bool
        If true, report all path related settings

    Returns
    -------
    tab: pd.DataFrame
        Session info in table form

    """
    from ecogdata.devices.data_util import find_loadable_files
    conf = session_conf(session)
    top_level_info = conf.pop('session')
    named_recordings = sorted([s for s in conf.sections if s != 'session'])
    keys = set(top_level_info.keys())
    for r in named_recordings:
        keys.update(conf[r].keys())
    if not global_params:
        gp = set(load_params().keys())
        keys.difference_update(gp)
    if not path_info:
        path_keys = {'nwk_path', 'exp_path', 'network_exp', 'local_exp', 'store_path'}
        keys.difference_update(path_keys)
    required_keys = ['headstage', 'electrode', 'exp_type']
    other_keys = list(keys.difference(required_keys))
    if source_info:
        columns = list(required_keys) + ['primary_file', 'downsampled_file'] + list(other_keys)
    else:
        columns = list(required_keys) + list(other_keys)
    tab = pd.DataFrame(columns=columns)
    for r in named_recordings:
        rec = top_level_info.copy()
        rec.update(conf[r])
        tab_row = dict([(k, rec.get(k, 'unknown')) for k in required_keys + other_keys])
        tab_row = filter_values(tab_row)
        if source_info:
            tab_row['primary_file'] = find_loadable_files(session, r, downsampled=False)
            tab_row['downsampled_file'] = find_loadable_files(session, r, downsampled=True)
        tab = tab.append(pd.DataFrame(index=[r], data=tab_row))
    return tab
