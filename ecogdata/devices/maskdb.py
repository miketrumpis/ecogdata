import os
import warnings
import h5py
from tables import NoSuchNodeError
from ecogdata.util import Bunch
from ecogdata.datastore import load_bunch, save_bunch
from ecogdata.expconfig import params as global_params

__all__ = ['MaskDB', 'merge_db']


def _walk_paths(root, level=0):
    keys = sorted(root.keys())
    for k in keys:
        v = root[k]
        print('\t' * level + k)
        if 'chan_mask' not in v:
            _walk_paths(v, level=level + 1)


def merge_db(source, dest):
    src = load_bunch(source, '/', load=False)
    dst = load_bunch(dest, '/', load=False)
    def _merge_level(s_, d_, level='/'):

        if 'chan_mask' in s_:
            print('end-node')
            src_mask = s_.chan_mask
            dst_mask = d_.chan_mask
            print('Source mask: {0} channels'.format(src_mask.sum()))
            print('Dest mask: {0} channels'.format(dst_mask.sum()))
            choice = 'Overwrite destination? ([y]/n) '
            if choice.lower() in ('y', ''):
                # print 'would write', d_, 'at path:', level
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # remove trailing slash
                    k = level[:-1]
                    s_load = load_bunch(source, k)
                    save_bunch(dest, k, s_load, overwrite_paths=True)
            return

        src_keys = set(s_.keys())
        dst_keys = set(d_.keys())

        new_keys = src_keys.difference(dst_keys)
        for nk in new_keys:
            # print 'would write', s_[nk], 'at path:', '/'+nk
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                s_load = load_bunch(source, level + nk)
                save_bunch(dest, level + nk, s_load)
        sim_keys = src_keys.intersection(dst_keys)
        for sk in sim_keys:
            _merge_level(s_[sk], d_[sk], level + sk + '/')

    _merge_level(src, dst)


class MaskDB:

    dbfile = 'channel_mask_database'

    def __init__(self, local=False, dbfile=None):
        if dbfile is None:
            dbfile = os.path.join(
                global_params.channel_mask, MaskDB.dbfile
            ) + '.h5'
            if local or not os.path.exists(dbfile):
                dbfile = os.path.join(
                    global_params.stash_path, MaskDB.dbfile
                ) + '.h5'
        if not os.path.exists(dbfile):
            with h5py.File(dbfile, 'w'):
                pass
        self.dbfile = dbfile

    def _dset_to_path(self, dset_name):
        session, rec = os.path.splitext(dset_name)
        session = '/' + session
        rec = '/' + rec[1:]
        return session + rec

    def lookup(self, dset_name):
        path = self._dset_to_path(dset_name)
        try:
            node = load_bunch(self.dbfile, path)
        except IOError:
            return Bunch()
        except NoSuchNodeError:
            return Bunch()
        return node

    def list_paths(self):
        b = load_bunch(self.dbfile, '/', load=False)
        _walk_paths(b)

    def stash(self, dset_name, chan_mask=(), time_mask=(), overwrite=False):
        node = self.lookup(dset_name)
        if not overwrite:
            if len(chan_mask) and 'chan_mask' in node:
                raise RuntimeError('Channel mask already exists')
            if len(time_mask) and 'time_mask' in node:
                raise RuntimeError('Time mask already exists')
        if len(chan_mask):
            node.chan_mask = chan_mask
        if len(time_mask):
            node.time_mask = time_mask
        path = self._dset_to_path(dset_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            save_bunch(self.dbfile, path, node, overwrite_paths=overwrite)
