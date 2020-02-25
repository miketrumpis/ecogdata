#!/usr/bin/env python

import os
from glob import glob
from ecogdata.devices.load.open_ephys import hdf5_open_ephys_channels
from ecogdata.expconfig import session_conf


def convert_directory(path, downsamp, recordings=(), store_path=''):
    if not recordings:
        recordings = []
        for d in os.listdir(path):
            full_path = os.path.join(path, d)
            # if the sub-item is itself a directory and has continuous files, then convert it
            if os.path.isdir(full_path) and glob(os.path.join(full_path, '*.continuous')):
                # skip the AFP droppings
                if d == '.AppleDouble':
                    continue
                print('Adding {} to convert'.format(d))
                recordings.append(d)
        # if the recordings list is still empty, check to see if the path given is the full data path
        if not recordings:
            if glob(os.path.join(path, '*.continuous')):
                if path[-1] == os.path.sep:
                    path = path[:-1]
                path, rec = os.path.split(path)
                recordings = (rec,)

    for rec in recordings:
        hdf5_open_ephys_channels(path, rec, os.path.join(store_path, rec + '.h5'), downsamp=downsamp)


def run_convert(args):
    if args.config:
        conf = session_conf(args.config)
        path = conf.session.nwk_path
        recordings = [k for k in conf.keys() if k not in ('sections', 'session')]
        if args.downsamp == 1 and 'usefs' in conf.session:
            # assume 20 kS/s full rate
            downsamp = 2e4 // conf.session.usefs
        else:
            downsamp = args.downsamp
    elif args.path:
        path = args.path
        recordings = ()
        downsamp = args.downsamp
    else:
        raise RuntimeError('Neither config nor path specified')

    convert_directory(path, downsamp, recordings=recordings, store_path=args.store)


if __name__ == '__main__':
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Open-Ephys data conversion tool')
    parser.add_argument('-c', '--config', type=str, default='', help='session config file')
    parser.add_argument('-p', '--path', type=str, default='', help='convert recordings on this path')
    parser.add_argument('--downsamp', type=float, default=1, help='downsample ratio (1 for full rate)')
    parser.add_argument('--store', type=str, default='', help='store results on separate path')
    args = parser.parse_args()
    try:
        run_convert(args)
    except RuntimeError as e:
        print(e)
        sys.exit(1)
