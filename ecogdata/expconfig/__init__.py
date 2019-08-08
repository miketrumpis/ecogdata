from ._globalconfig import *
from ._expconfig import *
from .config_decode import *
from .exp_descr import build_experiment

params = load_params()

def reload_params():
    globals()['params'] = load_params()
