import warnings
with warnings.catch_warnings() as w:
    warnings.simplefilter("always")
    warnings.warn('This module has been renamed: import from ecogdata.expconfig.config_tools', DeprecationWarning)
from .config_tools import *