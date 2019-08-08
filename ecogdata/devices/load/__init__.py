"""
This module should eventually house device specific code, e.g.

* current sensing AFE headstage
* mux Vx headstages
* blackrock headstage

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