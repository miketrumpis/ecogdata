import warnings
with warnings.catch_warnings() as w:
    warnings.simplefilter("always")
    warnings.warn('This module has been renamed: import from ecogdata.expconfig.global_config', DeprecationWarning)
from .global_config import *