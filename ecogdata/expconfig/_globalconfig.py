from future import standard_library
standard_library.install_aliases()
import os
import six
if six.PY3:
    from configparser import ConfigParser, SafeConfigParser
else:
    from ConfigParser import ConfigParser, SafeConfigParser
from ecogdata.util import Bunch
from .config_decode import *


all_keys = {
    'local_exp': Path,
    'network_exp': Path,
    'stash_path': Path,
    'user_sessions': Path,
    'clear_temp_converted': BoolOrNum,
    'memory_limit': TypedParam.from_type(float),
    'channel_mask': Path
}


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
            params[opt] = cfg.get('globals', opt)
        else:
            params[opt] = parse_param(opt, cfg.get('globals', opt), all_keys)
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


def cfg_to_bunch(cfg_file, section=''):
    """Return session config info in Bunch (dictionary) form with interpolations
    from the master config settings. Perform full evaluation on parameters known
    here and leave subsequent evaluation downstream.
    """
    cp = new_SafeConfigParser()
    cp.read(cfg_file)
    sections = [section] if section else cp.sections()
    b = Bunch()
    for sec in sections:
        bsub = Bunch()
        opts = cp.options(sec)
        param_pairs = [(o, parse_param(o, cp.get(sec, o), all_keys)) for o in opts]
        bsub.update(param_pairs)
        b[sec] = bsub
    b.sections = sections
    return b
