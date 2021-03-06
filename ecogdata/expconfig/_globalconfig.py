import os
from configparser import ConfigParser
from ecogdata.util import Bunch
from .config_decode import *


SafeConfigParser = ConfigParser


all_keys = {
    'local_exp': Path.with_default(os.path.expanduser('~/')),
    'network_exp': Path.with_default(os.path.expanduser('~/')),
    'stash_path': Path.with_default(os.path.expanduser('~/')),
    'user_sessions': Path.with_default(os.path.expanduser('~/')),
    'clear_temp_converted': BoolOrNum.with_default('false'),
    'floating_point': Parameter.with_default('single'),  # just a string
    'memory_limit': TypedParam.from_type(float, default=1e9),  # 1 GB default
    'channel_mask': Path.with_default(os.path.expanduser('~/'))
}

# Kind of hacky, but the param values can be set/reset at run-time by
# using OVERRIDE[key] = new_value (helpful for test cases)
OVERRIDE = dict()


def load_params(as_string=False):
    cfg = ConfigParser()
    # Look for custom global config in ~/.mjt_exp_conf.txt
    # If nothing found, use a default one here
    cpath = os.path.expanduser('~/.mjt_exp_conf.txt')
    if not os.path.exists(cpath):
        cpath = os.path.split(os.path.abspath(__file__))[0]
        cpath = os.path.join(cpath, 'global_config.txt')
    cfg.read(cpath)

    params = Bunch()
    for opt in cfg.options('globals'):
        if as_string:
            params[opt] = OVERRIDE.get(opt, cfg.get('globals', opt))
        else:
            val = OVERRIDE.get(opt, cfg.get('globals', opt))
            params[opt] = parse_param(opt, val, all_keys)
    for k in all_keys:
        params.setdefault(k, '')
    return params


def new_SafeConfigParser():
    cp = SafeConfigParser(defaults=load_params(as_string=True))
    # this hot-swap will preserve case in option names
    cp.optionxform = str
    return cp


def data_path():
    return load_params().local_exp


def network_path():
    return load_params().network_exp


def cfg_to_bunch(cfg_file, section='', params_table=None):
    """Return session config info in Bunch (dictionary) form with interpolations
    from the master config settings. Perform full evaluation on parameters known
    here and leave subsequent evaluation downstream.
    """
    cp = new_SafeConfigParser()
    cp.read(cfg_file)
    sections = [section] if section else cp.sections()
    b = Bunch()
    if params_table is None:
        params_table = {}
    params_table.update(all_keys)
    for sec in sections:
        bsub = Bunch()
        opts = cp.options(sec)
        param_pairs = [(o, parse_param(o, cp.get(sec, o), params_table)) for o in opts]
        bsub.update(param_pairs)
        b[sec] = bsub
    b.sections = sections
    return b
