import os
import warnings
import h5py
import numpy as np
import json

from ecogdata.util import Bunch
from ecogdata.datasource import MappedSource
from ecogdata.devices.electrode_pinouts import get_electrode_map
from ecogdata.devices.units import convert_scale
from ecogdata.trigger_fun import process_trigger
from . import DataPathError


def load_rhd(experiment_path, test, electrode, load_channels=None, units='uV', bandpass=(), notches=(), useFs=None,
             trigger_idx=(), mapped=True, save_downsamp=True, use_stored=True, store_path=None,
             raise_on_glitches=False):
    """
    Load a combined RHD file recording converted to HDF5.

    Parameters
    ----------
    experiment_path: str
        Directory holding recording data.
    test: str
        Recording file name.
    electrode:
        Name of the channel-to-electrode map.
    load_channels: sequence
        If only a subset of channels should be loaded, list them here. For example, to load channels from only one
        port, use `load_channels=range(128)`. Otherwise all channels are used.
    units: str
        Scale data to these units, e.g. pv, uv, mv, v (default micro-volts).
    bandpass: 2-tuple
        Bandpass specified as (low-corner, high-corner). Use (-1, fc) for lowpass and (fc, -1) for highpass.
    notches: sequence
        List of notch filters to apply.
    useFs: float
        Downsample to this frequency (must divide the full sampling frequency).
    trigger_idx: int or sequence
        The index/indices of a logic-level trigger signal in the "ADC" array.
    mapped: bool
        Final datasource will be mapped if True or loaded if False.
    save_downsamp: bool
        If True, save the downsampled data source to a named file.
    use_stored: bool
        If True, load pre-computed downsamples if available. This implies that a store_path is used,
        but the experiment path will be searched if necessary.
    store_path: str
        If given, store the downsampled datasource here. Otherwise store it in the experiment_path directory.
    raise_on_glitches: bool
        Raise exceptions if any expected auxilliary channels are not loaded, otherwise emit warnings.

    Returns
    -------
    dataset: Bunch
        A attribute-full dictionary with an ElectrodeDataSource and ChannelMap and other metadata.

    """

    channel_map, ground_chans, ref_chans = get_electrode_map(electrode)

    # Determine the source file, which would either be the full resolution recording of a previously computed
    # downsample file. Both would be opened in read-only mode.
    # + Also tag whether a new downsample is required.
    # + Saved downsamples are in micro-volts units and full sources are in ADC units. Determine the correct units
    #   scaling for both cases.
    data_file = os.path.join(experiment_path, test + '.h5')
    units_scale = convert_scale(.195, 'uv', units)

    # Find the full sample rate, if the full resolution file even exists (otherwise it's still possible to load a
    # downsample file).
    if os.path.exists(data_file):
        with h5py.File(data_file, 'r') as h5file:
            header = json.loads(h5file.attrs['JSON_header'])
            full_samp_rate = header['sample_rate']
    else:
        full_samp_rate = -1

    # For a new sample rate (that is different than the full rate), determine if a pre-saved file can be used.
    if useFs and useFs != full_samp_rate:
        downsamp_file = test + '_Fs{:d}'.format(int(useFs))
        if use_stored:
            search_dirs = [experiment_path]
            if store_path:
                search_dirs.insert(0, store_path)
            data_files = [os.path.join(d, downsamp_file) + '.h5' for d in search_dirs]
            data_files_exist = [os.path.exists(p) for p in data_files]
            if any(data_files_exist):
                data_file = data_files[data_files_exist.index(True)]
                needs_downsamp = False
                units_scale = convert_scale(1, 'uv', units)
            else:
                needs_downsamp = True
                downsamp_file = os.path.join(search_dirs[0], downsamp_file) + '.h5'
        else:
            needs_downsamp = True
    else:
        needs_downsamp = False

    # check if the settled-on data file exists
    if not os.path.exists(data_file):
        raise DataPathError('RHD converted file {} not found'.format(data_file))

    file_is_temp = False
    if needs_downsamp:
        # 1. Do downsample conversion in a subroutine
        # 2. Save to a named file if save_downsamp is True
        # 3. Determine if the new source file is writeable (i.e. a temp file) or not
        downsamp_file = (downsamp_file if save_downsamp else '')
        print('Downsampling to {} Hz from file {} to file {}'.format(useFs, data_file, downsamp_file))
        data_file = make_rhd_downsample(data_file, useFs, downsamp_file=downsamp_file)
        file_is_temp = not save_downsamp
        units_scale = convert_scale(1, 'uv', units)

    open_mode = 'r+' if file_is_temp else 'r'
    print('Opening source file {} in mode {}'.format(data_file, open_mode))
    h5file = h5py.File(data_file, open_mode)
    n_amplifier_channels = h5file['amplifier_data'].shape[0]
    header = json.loads(h5file.attrs['JSON_header'])
    Fs = header['sample_rate']
    if useFs:
        Fs = useFs

    # Find the full set of expected electrode channels within the "amplifier_data" array
    electrode_channels = [n for n in range(n_amplifier_channels) if n not in ground_chans + ref_chans]
    if load_channels:
        # If load_channels is specified, need to modify the electrode channel list and find the channel map subset
        sub_channels = list()
        sub_indices = list()
        for i, n in enumerate(electrode_channels):
            if n in load_channels:
                sub_channels.append(n)
                sub_indices.append(i)
        electrode_channels = sub_channels
        channel_map = channel_map.subset(sub_indices)
        ground_chans = [n for n in ground_chans if n in load_channels]
        ref_chans = [n for n in ref_chans if n in load_channels]

    print('Creating mapped sources')
    datasource = MappedSource(h5file, 'amplifier_data', electrode_channels=electrode_channels,
                              aligned_arrays=('board_adc_data',), units_scale=units_scale)
    if ground_chans:
        ground_chans = MappedSource(h5file, 'amplifier_data', electrode_channels=ground_chans,
                                    units_scale=units_scale)
    if ref_chans:
        ref_chans = MappedSource(h5file, 'amplifier_data', electrode_channels=ref_chans, units_scale=units_scale)

    # Promote to a writeable and possibly RAM-loaded array here if either the final source should be loaded,
    # or if the mapped source is not writeable.
    def swap_handles(source, dest):
        return dest, None

    needs_load = not mapped
    needs_writeable = (bandpass or notches) and not datasource.writeable
    if needs_load or needs_writeable:
        # Need to make writeable copies of these data sources. If the final source is to be loaded, then mirror
        # to memory here.
        print('Creating writeable mirrored sources')
        copy_mode = 'aligned' if mapped else 'all'
        datasource_w = datasource.mirror(mapped=mapped, writeable=True, copy=copy_mode)
        if ground_chans:
            ground_chans_w = ground_chans.mirror(mapped=mapped, writeable=True, copy=copy_mode)
        if ref_chans:
            ref_chans_w = ref_chans.mirror(mapped=mapped, writeable=True, copy=copy_mode)
        if not mapped:
            # swap handles of these objects
            datasource, datasource_w = swap_handles(datasource, datasource_w)
            if ground_chans:
                ground_chans, ground_chans_w = swap_handles(ground_chans, ground_chans_w)
            if ref_chans:
                ref_chans, ref_chans_w = swap_handles(ref_chans, ref_chans_w)

    # For the filter blocks...
    # If mapped, then datasource and datasource_w will be identical.
    # If loaded, then datasource_w is None and datasource is filtered in-place
    if bandpass:
        filter_kwargs = dict(ftype='butterworth',
                             inplace=not mapped,
                             design_kwargs=dict(lo=bandpass[0], hi=bandpass[1], Fs=Fs),
                             filt_kwargs=dict(filtfilt=True))
        if mapped:
            # make "verbose" filtering with progress bar if we're filtering a mapped source
            filter_kwargs['filt_kwargs']['verbose'] = True
        print('Bandpass filtering')
        datasource = datasource.filter_array(out=datasource_w, **filter_kwargs)
        if ground_chans:
            ground_chans = ground_chans.filter_array(out=ground_chans_w, **filter_kwargs)
        if ref_chans:
            ref_chans = ref_chans.filter_array(out=ref_chans_w, **filter_kwargs)
    if notches:
        print('Notch filtering')
        notch_kwargs = dict(inplace=True, lines=notches, filtfilt=True, verbose=True)
        datasource = datasource.notch_filter(out=datasource_w, **notch_kwargs)
        if ground_chans:
            ground_chans = ground_chans.notch_filter(out=datasource_w, **notch_kwargs)
        if ref_chans:
            ref_chans = ref_chans.notch_filter(out=ref_chans_w, **notch_kwargs)

    # Now process triggers and build the Bunch, etc...
    if not np.iterable(trigger_idx):
        trigger_idx = (trigger_idx,)
    if len(trigger_idx):
        try:
            trigger_signal = datasource.board_adc_data[list(trigger_idx), :]
            pos_edge = process_trigger(trigger_signal)[0]
        except (IndexError, ValueError) as e:
            tb = e.__traceback__
            msg = 'Trigger channels were specified but do not exist'
            if raise_on_glitches:
                raise Exception(msg).with_traceback(tb)
            else:
                warnings.warn(msg, RuntimeWarning)
    else:
        trigger_signal = ()
        pos_edge = ()

    # Viventi lab convention: stim signal would be on the next available ADC channel... skip explicitly loading
    # this, because the "board_adc_data" array is cojoined with the main datasource

    dataset = Bunch()
    dataset.data = datasource
    for arr in datasource.aligned_arrays:
        dataset[arr] = getattr(datasource, arr)
    dataset.chan_map = channel_map
    dataset.Fs = Fs
    dataset.pos_edge = pos_edge
    dataset.trig_chan = trigger_signal
    dataset.bandpass = bandpass
    dataset.notches = notches
    dataset.units = units
    dataset.transient_snipped = False
    dataset.ground_chans = ground_chans
    dataset.ref_chans = ref_chans
    return dataset


def make_rhd_downsample(filename, new_fs, downsamp_file=''):
    """
    Create a channel compatible file with all RHD amplifier and ADC channels.

    Parameters
    ----------
    filename: str
        Mapped RHD data
    new_fs: float
        New sampling rate
    downsamp_file: str
        If not empty, then store the new file here.

    Returns
    -------
    new_filename: str
        File name for downsampled map file.

    """

    with h5py.File(filename, 'r') as h5file:
        header = json.loads(h5file.attrs['JSON_header'])
        Fs = header['sample_rate']
        downsamp_rate = int(Fs / new_fs)
        # *Always* downsample at micro-volts scaling
        print('Creating mapped primary source {}'.format(filename))
        try:
            datasource = MappedSource(h5file, 'amplifier_data', aligned_arrays=('board_adc_data',), units_scale=0.195)
            print('Mirroring at 1 / {} rate'.format(downsamp_rate))
            downsamp = datasource.mirror(new_rate_ratio=downsamp_rate, mapped=True, channel_compatible=True,
                                         filename=downsamp_file)
            ds_filename = downsamp._source_file.filename
            datasource.batch_change_rate(downsamp_rate, downsamp, verbose=True, filter_inplace=True)
            delete_file = False
        except:
            delete_file = True
            raise
        finally:
            if delete_file:
                os.unlink(ds_filename)
        print('Putting header and closing file')
        header['sample_rate'] = new_fs
        downsamp._source_file.attrs['JSON_header'] = json.dumps(header)
        # this should close the file
        downsamp.close_source()
    return ds_filename
