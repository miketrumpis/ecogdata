from .global_config import *
from .config_tools import *
from .config_decode import *
from .exp_descr import build_experiment

params = load_params()

def reload_params():
    globals()['params'] = load_params()
