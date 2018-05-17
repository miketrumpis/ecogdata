from future import standard_library
standard_library.install_aliases()
import os
from configparser import ConfigParser, SafeConfigParser
from ecogdata.util import Bunch

def load_params():
    cfg = ConfigParser()
    # Look for custom global config in ~/.mjt_exp_conf.txt
    # If nothing found, use a default one here
    cpath = os.path.join(
        os.path.expanduser('~'), '.mjt_exp_conf.txt'
        )
    if not os.path.exists(cpath):
        cpath = os.path.split(os.path.abspath(__file__))[0]
        cpath = os.path.join(cpath, 'global_config.txt')
    cfg.read(cpath)

    _all_keys = ('local_exp', 'network_exp', 'stash_path', 'user_sessions',
                 'clear_temp_converted', 'memory_limit', 'channel_mask')
    # XXX: need to specify transform table for the various global data types
    params = Bunch(
        **dict([ (opt, cfg.get('globals', opt)) 
                 for opt in cfg.options('globals') ])
        )
    for k in _all_keys:
        params.setdefault(k, '')
    return params

def new_SafeConfigParser():
    cp = SafeConfigParser(defaults=load_params())
    # this hot-swap will preserve case in option names
    cp.optionxform = str
    return cp

def data_path():
    return load_params().local_exp

def network_path():
    return load_params().network_path

def project_path():
    return load_params().local_proj

def cfg_to_bunch(cfg_file, section=''):
    cp = new_SafeConfigParser()
    cp.read(cfg_file)
    sections = [section] if section else cp.sections()
    b = Bunch()
    for sec in sections:
        bsub = Bunch()
        opts = cp.options(sec)
        bsub.update(( (o, cp.get(sec, o)) for o in opts ))
        b[sec] = bsub
    b.sections = sections
    return b
