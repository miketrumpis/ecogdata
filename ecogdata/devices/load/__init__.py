"""
This package implements data loading/mapping for various acquisition systems. Common logic for loading &
preprocessing is implemented in the file2data.FileLoader class.
"""


class DataPathError(Exception):
    pass


from .mux import load_mux, mux_headstages
from .blackrock import load_blackrock
from .wireless import load_wireless
from .ddc import load_ddc, load_openephys_ddc
from .afe import load_afe, load_afe_aug21
from .active_electrodes import load_active, active_headstages
from .open_ephys import load_open_ephys
from .nanoz import *
from .intan import *


