from builtins import chr
import os
from ecogdata.datastore import load_bunch
from ecogdata.util import mkdir_p

from .tdms import tdms_to_hdf5
from . import DataPathError


def try_saved(exp_path, test, bandpass):
    try:
        dset = load_bunch(os.path.join(exp_path, test+'_proc.h5'), '/')
        #h5 = tables.open_file(os.path.join(exp_path, test+'_proc.h5'))
    except IOError as exc:
        raise DataPathError

    if dset.bandpass != bandpass:
        del dset
        raise DataPathError(
            'saved data bandpass does not match requested bandpass'
            )
    return dset


def convert_tdms(test, remote, local, accepted=('.h5',), clear=False):
    "Convert from TDMS files on-demand, i.e. if no acceptable file is found."

    exists = [os.path.exists( os.path.join(local, test+e ) ) 
              for e in accepted]
    if any(exists):
        return

    exists = [os.path.exists( os.path.join(remote, test+e ) ) 
              for e in accepted]
    if any(exists):
        return

    mkdir_p(local)
    h5_file = os.path.join(local, test+'.h5')
    r_tdms_file = os.path.join(remote, test+'.tdms')
    l_tdms_file = os.path.join(local, test+'.tdms')
    if not os.path.exists( r_tdms_file ):
        if not os.path.exists( l_tdms_file ):
            msg = 'No TDMS data in {0}, {1}'.format(r_tdms_file, l_tdms_file)
            raise DataPathError(msg)
        tdms_file = l_tdms_file
    else:
        tdms_file = r_tdms_file
    
    tdms_to_hdf5(tdms_file, h5_file, memmap=False)

    def cleanup():
        if clear:
            os.unlink(h5_file)
        return
    return cleanup


def tdms_info(info):
    "Recover strings from MATLAB-converted TDMS files"
    def _arr_to_str(arr):
        return ''.join( [chr(a) for a in arr] )

    str_keys = ('RowSelect', 'Note', 'location', 
                'PXICard', 'AnalogOut', 'name', 'ConvertVer')
    clean_info = info.deepcopy()
    for field, arr in info.items():
        for sk in str_keys:
            if sk.lower() in field.lower() and not isinstance(arr, str):
                clean_info[field] = _arr_to_str(arr)
    return clean_info

