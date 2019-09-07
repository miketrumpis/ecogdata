#!/usr/bin/env python
import os.path as p
from contextlib import closing
from configparser import ConfigParser
from ecogdata.expconfig import params
try:
    from traits.api import HasTraits, on_trait_change, Code, Button, String, Bool, Directory
    from traitsui.api import Item, UItem, View, Group, VGroup, HGroup, Label
except ImportError:
    print('This tool needs the packages "traits" and "traitsui" to be installed.')
    import sys
    sys.exit(0)


def backup_config(fpath):
    # create a file with suffix .bk{n} where n is the next available number
    # not already used
    if not p.exists(fpath):
        return
    n = 0
    while True:
        fpath_bk = fpath + '.bk{0}'.format(n)
        n += 1
        if not p.exists(fpath_bk):
            break
    with closing(open(fpath)) as fo, closing(open(fpath_bk, 'w')) as fb:
        fb.write(fo.read())


class GlobalConfig(HasTraits):
    local_exp = Directory
    network_exp = Directory
    stash_path = Directory
    user_sessions = Directory
    clear_temp_converted = String
    memory_limit = String
    channel_mask = Directory

    # these two will disappear eventually, so don't bother setting
    filter_highpass = String

    saving = Button('Save config')
    status = String('unsaved')
    show_config = Bool(False)
    config_txt = Code

    def __init__(self, **traits):
        these = params.copy()
        these.update(traits)
        super(GlobalConfig, self).__init__(**these)

    def write(self):
        cpath = p.expanduser('~/.mjt_exp_conf.txt')
        backup_config(cpath)
        cfg = ConfigParser()
        cfg.add_section('globals')
        for k in params.keys():
            cfg.set('globals', k, getattr(self, k))
        with closing(open(cpath, 'w')) as f:
            cfg.write(f)
        self.status = 'saved to ' + cpath

    def _saving_fired(self):
        self.write()

    @on_trait_change('show_config', 'saving')
    def _load_config(self):
        if self.show_config:
            try:
                cpath = p.expanduser('~/.mjt_exp_conf.txt')
                self.config_txt = open(cpath).read()
            except IOError:
                self.config_txt = ''
                self.show_config_txt = False
        else:
            self.config_txt = ''

    view = View(
        VGroup(
            VGroup(
                Label("'local_exp': A local path (on your system's drive) "
                      "where recording sessions are saved:"),
                UItem('local_exp', style='simple'),
                Label("'network_exp': A network path where recording "
                      "sessions are saved:"),
                UItem('network_exp', style='simple'),
                Label("'stash_path': A local path where computational "
                      "resuls can be stored:"),
                UItem('stash_path', style='simple'),
                Label("'user_sessions': A local path where session "
                      "configuration files can be found:"),
                UItem('user_sessions', style='simple'),
                Label("'clear_temp_converted': Save or delete temporarily "
                      "converted recordings pulled from the network? "
                      "(True/False)"),
                UItem('clear_temp_converted'),
                Label("'memory_limit': Some memory intensive methods "
                      "will attempt to limit RAM footprint (put a limit "
                      "in bytes, e.g. 2e9)"),
                UItem('memory_limit'),
                Label("'channel_mask': Directory to find master channel "
                      "mask database"),
                UItem('channel_mask'),
                label='Global Configuration Parameters',
                show_border=True,
                padding=10
            ),
            HGroup(
                UItem('saving'),
                Item('show_config', label='Show config file'),
                Item('status', label='Status:', style='readonly')
            ),
            Group(
                Item(
                    'config_txt', style='readonly', label='Config',
                    enabled_when='show_config'
                ),
                visible_when='show_config'
            )
        ),
        title='Configuration Tool',
        resizable=True
    )


if __name__ == '__main__':
    gc = GlobalConfig()
    gc.configure_traits()
