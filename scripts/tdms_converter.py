#!/usr/bin/env python
import os
import os.path as p
from glob import glob
from math import log
from ecogdata.devices.load.tdms import tdms_to_hdf5
try:
    from traits.api import HasTraits, on_trait_change, Button, Str, Bool, Directory, Property, List, Instance
    from traitsui.api import Item, UItem, View, VGroup, Label, TableEditor
    from traitsui.table_column import ObjectColumn
    from traitsui.extras.checkbox_column import CheckboxColumn
except ImportError:
    print('This tool needs the packages "traits" and "traitsui" to be installed.')
    import sys
    sys.exit(0)


class TDMSFile(HasTraits):
    full_path = Str
    name = Property(fget=lambda self: p.split(self.full_path)[1],
                    depends_on='full_path')
    converted = Property
    skip_convert = Bool(False)
    fsize = Property  # (depends_on='full_path')

    def _get_converted(self):
        fp = p.splitext(self.full_path)[0]
        return p.exists(fp + '.h5')

    def _get_fsize(self):
        fs = os.stat(self.full_path)
        size = log(fs.st_size, 2.0)
        prefix = ('', 'k', 'M', 'G', 'T')
        n = int(size // 10)
        size = 2 ** (size - n * 10.0)
        order = prefix[n] + 'b'
        return '{0:.2f} {1}'.format(size, order)

    def convert_tdms(self):
        if self.converted:
            return
        if self.skip_convert:
            return
        h5_file = p.splitext(self.full_path)[0] + '.h5'
        print(self.full_path, h5_file)
        try:
            tdms_to_hdf5(self.full_path, h5_file, memmap=True, load_data=True)
        except:
            pass


class FileColumn(ObjectColumn):

    def get_text_color(self, object):
        return ['red', 'black'][object.converted]


tab_editor = TableEditor(
    columns=[
        FileColumn(name='name', label='TDMS File', editable=False),
        FileColumn(name='fsize', label='Size', editable=False),
        FileColumn(name='converted', label='Converted to HDF5', editable=False, width=0.1),
        CheckboxColumn(name='skip_convert', label='Skip convert?', editable=True, width=0.1)
    ],
    auto_size=False,
    row_factory=TDMSFile
)


def get_files(dir):
    tdms_files = sorted(glob(p.join(dir, '*.tdms')))
    files = [TDMSFile(full_path=fp) for fp in tdms_files]
    files[-1].skip_convert = True
    return files


class ConvertReport(HasTraits):
    dir = Str
    files = List(TDMSFile)
    convert = Button
    refresh = Button

    def __init__(self, dir_, **traits):
        if 'files' not in traits:
            traits['files'] = get_files(dir_)
        HasTraits.__init__(self, dir=dir_, **traits)

    def _convert_fired(self):
        for f in self.files:
            f.convert_tdms()

    def _refresh_fired(self):
        self.files = get_files(self.dir)

    view = View(
        VGroup(
            Item('files', id='table', editor=tab_editor),
            Item('refresh', label='Refresh List'),
            Item('convert', label='Convert TDMS'),
            show_labels=False
        ),
        title='TDMS to HDF5 Conversion',
        width=500,
        resizable=True
    )


class TDMSDirectory(HasTraits):
    dir = Directory
    report = Instance(ConvertReport)

    @on_trait_change('dir')
    def new_report(self):
        self.report = ConvertReport(self.dir)

    view = View(
        VGroup(
            Label('Recording directory'),
            UItem('dir', style='simple'),
            UItem('report')
        ),
        title='TDMS to HDF5 Conversion',
        width=300,
        resizable=True
    )


if __name__ == '__main__':
    tdmsdir = TDMSDirectory()
    tdmsdir.configure_traits()
