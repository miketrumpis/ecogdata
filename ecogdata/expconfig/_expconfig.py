from __future__ import print_function
from __future__ import absolute_import
from future import standard_library
standard_library.install_aliases()
from builtins import input
from builtins import map
import os
import os.path as osp
import numpy as np
from glob import glob
from ._globalconfig import new_SafeConfigParser, cfg_to_bunch, load_params
import warnings

__all__ = ['session_groups', 'available_sessions', 'find_conf',
           'session_conf', 'session_info', 'locate_old_session',
           'sessions_to_delta']

params = load_params()
# _cpath is the single session config database path with
# multiple sub-directories indicating different recording groups
if not 'user_sessions' in params or not params.user_sessions:
    warnings.warn('A session config path was not found.', UserWarning)
    # fall back on this path, which is probably empty
    _cpath = osp.join(osp.dirname(__file__), 'sessions')
else:
    _cpath = osp.abspath(params.user_sessions)
    
def find_conf(conf, extra_paths=None):
    '''
    Session config "conf" is given in "group/session" syntax.
    Use "extra_paths" to specify anywhere else the "session" file
    might be found.
    '''
    
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

    raise IOError('config file not found: '+conf)

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
        g_files = glob( osp.join(osp.join(_cpath, g), '*.txt' ) )
        #txt_files.extend( ['/'.join( (g, f) ) for f in sorted(g_files)] )
        txt_files.extend( sorted(g_files) )
    conf_files = list()
    cp = new_SafeConfigParser()
    for txt in txt_files:
        try:
            cp.read(txt)
            conf_files.append(txt)
        except:
            pass

    def _two_path(x):
        p1, x = osp.split(x)
        _, p2 = osp.split(p1)
        return '/'.join( (p2, x) )
    sessions = list(map(_two_path, conf_files))
    return [s.strip('.txt').replace('_conf', '') for s in sessions]

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
        return possibles[ int(choice) ]
    else:
        print('No matches')

def session_conf(session):
    "Return a Bunch-ified config file reporting the entire session"
    return cfg_to_bunch(find_conf(session))

def session_info(session):
    "Return default info for a session"
    cfg = session_conf(session)
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
        dates.append( '-'.join(m.groups()) )
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
    deltas = [ (datetime.strptime(s, fmt) - d0).days for s in dates ]
    if num:
        return deltas
    if sortable and len(deltas) > 1:
        dig = int(np.log10(max(deltas))) + 1
        deltas = ['Day {day:{width}}'.format(day=d, width=dig) 
                  for d in deltas]
    else:
        deltas = ['Day {0}'.format(d) for d in deltas]
    return deltas
