from __future__ import print_function
from builtins import zip
from builtins import map
import os.path as osp
from itertools import chain
from glob import glob
import numpy as np
from ecogdata.expconfig import session_conf, session_info, find_conf
from .base_exp import StimulatedExperiment
from .audio_exp import TonotopyExperiment
from .expo_exp import get_expo_experiment

__all__ = ['build_experiment']

# Key words for experiment properties
# * tone_onset, tone_width -- tonotopy experiments only
# * onset_delay (potential time lag btwn time stamp and event
#                synonymous with tone_onset)
# * ...

def _gen_params_table():
    param_to_type = {
        'tone_onset' : ('froemke tonotopy', 'tonotopy', 'puretones_amtones', 
                        'noise_amnoise', 'qt_amnoise', 'qt_amtones',
                        'puretonebursts', 'qt_puretones'),
        'tone_width' : ('froemke tonotopy', 'tonotopy', 'puretones_amtones', 
                        'noise_amnoise', 'qt_amnoise', 'qt_amtones',
                        'puretonebursts', 'qt_puretones'),
        'eye' : ('movshon exp',),
        }

    all_types = set( chain( *list(param_to_type.values()) ) )
    type_to_params = dict()
    for t in all_types:
        # add 'onset_delay' to all types
        type_to_params[t] = ['onset_delay']
        for p in list(param_to_type.keys()):
            if t in param_to_type[p]:
                type_to_params[t].append( p )

    return type_to_params

_params_table = _gen_params_table()
_condition_order_table = {
    'froemke tonotopy' : ('tones', 'amps'),
    'tonotopy' : ('tones', 'amps'),
    'puretones_amtones' : ('carrier_tone', 'mod_freq', 'am_depth'),
    'noise_amnoise' : ('mod_freq', 'am_depth')[::-1],
    'qt_amnoise' : ('mod_freq', 'am_depth')[::-1],
    'qt_amtones' :  ('carrier_tone', 'mod_freq', 'am_depth'),
    'puretonebursts' : ('tone', 'as_amp'),
    'qt_puretones' : ('tone', 'as_amp')
    }

def build_tonotopy(session, test, event_times):
    cfg = session_conf(session)
    exp_info = cfg.session
    exp_info.update(cfg[test])
    
    tone_tab = exp_info.tone_tab
    amp_tab = exp_info.amp_tab
    tone_onset = float(exp_info.get('tone_onset', 0))
    tone_width = float(exp_info.get('tone_width', 0))
    onset_delay = float(exp_info.get('onset_delay', tone_onset))

    paths = [exp_info.exp_path, exp_info.nwk_path]
    group = session.split('/')[0]

    try:
        tone_tab = find_conf('/'.join([group, tone_tab]), extra_paths=paths)
        tone_tab = np.loadtxt(tone_tab)
        tone_tab = tone_tab[ tone_tab > 0 ]        
    except IOError:
        try:
            tone_tab = np.array( list(map(float, tone_tab.split(','))) )
        except ValueError:
            # the tone_tab was not recognized, or was
            # coded to indicate no-stimulation
            return None

    n_tones = len( tone_tab )

    try:
        amp_tab = find_conf('/'.join([group, amp_tab]), extra_paths=paths)
    except IOError:
        amp_tab = osp.split(amp_tab)[1]
        amp_tab = np.array( list(map(int, amp_tab.split(','))) )
        amp_tab = np.tile(amp_tab[:,None], (1, n_tones)).ravel()

    exp = TonotopyExperiment.from_repeating_sequences(
        event_times, dict(tones=tone_tab, amps=amp_tab),
        tone_onset=tone_onset, tone_width=tone_width,
        onset_delay=onset_delay
        )

    if exp_info.tone_tab.find('rot') >= 0:
        # circular shift the amplitudes 1 step backwards
        exp.amps = np.r_[exp.amps[1:], exp.amps[:1]]
    
    return exp

def build_expo(session, test, event_times):
    cfg = session_conf(session)
    exp_info = cfg.session
    exp_info.update(cfg[test])

    xml_path = exp_info.xml_path
    eye = exp_info.eye
    # Get up to last character to fudge mistake in nov2013 test..
    # all expo files should be sufficiently distinct based on
    # session directory
    xml_prefix = exp_info.movshon_prefix[:-1]
    try:
        expo_num = exp_info.movshon_exp
    except:
        return None

    xml_path = glob(
        osp.join(xml_path, xml_prefix)+'*#'+expo_num+'*.xml'
        )
    if not xml_path:
        return None
    xml_path = xml_path[0]
    print('Generating experiment from', osp.split(xml_path)[1])
    exp_tab = get_expo_experiment(xml_path, event_times)
    exp_tab.stim_props.eye = eye
    return exp_tab

def build_simple(
        session, test, event_times, event_value=1.0, event_name='stim'
        ):
    # event_name needs to be a valid "attribute" name
    cfg = session_conf(session)
    exp_info = cfg.session
    exp_info.update(cfg[test])

    onset_delay = float(exp_info.get('onset_delay', 0))

    exp = StimulatedExperiment.from_repeating_sequences(
        event_times, {event_name : [event_value]}, 
        condition_order=(event_name,), onset_delay=onset_delay
        )
    return exp

def build_auto(session, test, event_times, exp_type=None):
    cfg = session_conf(session)
    exp_info = cfg.session
    exp_info.update(cfg[test])

    if 'exp_type' not in exp_info:
        if exp_type is None:
            raise RuntimeError('Could not determine the experiment type')
        exp_info.exp_type = exp_type
    
    extra_params = _params_table.get(exp_info.exp_type, ())
    p_values = [ float(exp_info.get(p, 0)) for p in extra_params ]
    extra_params = dict( zip(extra_params, p_values) )

    group = session.split('/')[0]
    # get table names and values
    tables = [s for s in cfg[test].keys() if s.endswith('_tab')]
    ev_tables = list()
    for tab in tables:
        tabv = exp_info.get(tab)
        # try to determine if it's a tab file or a literal table sequence
        try:
            paths = [cfg.session.exp_path, cfg.session.nwk_path]
            tabv = find_conf('/'.join([group, tabv]), extra_paths=paths)
            tabv = np.loadtxt(tabv)
        except IOError:
            # why this?
            tabv = osp.split(tabv)[1]
            tabv = np.array( list(map(float, tabv.split(','))) )

        ev_tables.append( [tab.replace('_tab', ''), tabv] )

    condition_order = _condition_order_table.get(exp_info.exp_type, ())
        
    exp = StimulatedExperiment.from_repeating_sequences(
        event_times, dict(ev_tables),
        condition_order=condition_order, **extra_params
        )
    return exp

def build_experiment(session, test, event_times, **kwargs):

    if not len(event_times):
        return None
    
    cfg = session_conf(session)
    exp_info = cfg.session
    try:
        exp_info.update(cfg[test])
    except KeyError:
        return

    # check for
    # 1) "legacy" experiments (e.g. froemke tonotopy or movshon)
    # 2) no "tab" info -- make simple experiment?
    # 3) otherwise try to auto-build

    if 'movshon_exp' in exp_info:
        # build Expo tables
        return build_expo(session, test, event_times)

    tables = [p for p in exp_info.keys() if p.endswith('_tab')]
    if len(tables):
        if 'exp_type' in exp_info:
            # If the type is named but it's froemke tonotopy,
            # still build tonotopy (the table names are wrong)
            if exp_info.exp_type == 'froemke tonotopy':
                return build_tonotopy(session, test, event_times)
            return build_auto(session, test, event_times)
        elif 'tone_tab' in tables and 'amp_tab' in tables:
            # if there is NO name but there are tables, then
            # default to "legacy" mode
            
            return build_tonotopy(session, test, event_times)
        else:
            # if no type is named and there are valid tables,
            # then it makes sense to build_auto (is exp_type
            # even needed?)
            return build_auto(session, test, event_times, exp_type='anon')

    # finally, try building a simple experiment
    return build_simple(session, test, event_times, **kwargs)
