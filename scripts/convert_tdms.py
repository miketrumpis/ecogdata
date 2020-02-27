#!/usr/bin/env python
import os
from glob import glob
try:
    import resource
    RES_DEFED = True
except ImportError:
    RES_DEFED = False

from ecogdata.devices.load.tdms import tdms_to_hdf5

if __name__ == '__main__':
    import argparse
    prs = argparse.ArgumentParser(description='Convert TDMS to HDF5')
    prs.add_argument('tdms_file', nargs='+', help='path to the TDMS file', type=str)
    prs.add_argument('h5_file', nargs='+', help='name of the HDF5 file to create', type=str)
    prs.add_argument('-p', '--permutation', type=str, default='', help='file with table of channel permutations')
    prs.add_argument('-m', '--memmap', help='use disk mapping for large files', action='store_true')
    prs.add_argument('-z', '--compression', type=int, default=0, help='use zlib level # compression in HDF5')
    prs.add_argument('-b', '--batch', help='Batch process all matching files', action='store_true')
    args = prs.parse_args()
    if args.batch:
        hp = args.h5_file[0]
        if not os.path.isdir(hp):
            (hp, pf) = os.path.split(hp)
        else:
            pf = ''
        if len(args.tdms_file) > 1:
            # shell has globbed *.tdms
            all_tdms = args.tdms_file
        else:
            tdms = args.tdms_file[0]
            (tp, _) = os.path.split(tdms)
            all_tdms = glob(os.path.join(tp, '*.tdms'))
        all_h5 = list()
        for tdms in all_tdms:
            (_, tf) = os.path.split(tdms)
            (tf, ext) = os.path.splitext(tf)
            conv_file = os.path.join(hp, pf+tf+'.h5')
            if not os.path.exists(conv_file):
                all_h5.append(conv_file)
            else:
                all_h5.append(None)
    else:
        all_tdms = args.tdms_file
        all_h5 = args.h5_file

    if RES_DEFED:
        (flim_soft, flim_hard) = resource.getrlimit(resource.RLIMIT_NOFILE)
        # assume 500 channels per file
        flim_needed = len(all_tdms) * 500
        if flim_needed > flim_soft:
            print('boosting file limits', 3 * flim_needed)
            resource.setrlimit(resource.RLIMIT_NOFILE, (3 * flim_needed, flim_hard))

    for tf, hf in zip(all_tdms, all_h5):
        if hf:
            print(tf, '\t', hf)
            tdms_to_hdf5(tf, hf, chan_map=args.permutation, memmap=args.memmap, compression_level=args.compression)
