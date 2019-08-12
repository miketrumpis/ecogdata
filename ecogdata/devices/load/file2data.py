import os
import h5py
import numpy as np
import warnings

from ecogdata.util import Bunch
from ecogdata.datasource import MappedSource
from ecogdata.devices.electrode_pinouts import get_electrode_map
from ecogdata.devices.units import convert_scale
from ecogdata.trigger_fun import process_trigger
from . import DataPathError


__all__ = ['FileLoader']


class FileLoader:

    # normal path, electrode, and units related stuff
    # self.experiment_path
    # self.test
    # self.units

    # filtering
    # self.bandpass
    # self.notches

    # acq system specific channel mapping
    # self.load_channels
    # self.electrode

    # sample rate conversion stuff
    # self.full_sample_rate -- need full sample rate a priori
    # self.final_sample_rate
    # self.use_stored
    # self.save_downsamp
    # self.store_path

    # creating a dataset has these components
    # 0. preparing HDF5 file with possibly downsampled acq channels
    # 1. mapping file data and/or promoting to memory
    # 2. filtering electrode channels
    # 3. parsing external triggers
    #
    # Steps 0 and 1 + 2 should be reasonably generic and implemented in separate methods subject to overloading
    # Step 3 may be less generic and definitely subject to overloading

    # multiplier to scale raw data units to micro-volts
    scale_to_uv = 1.0
    # name (key) of the dataset in the HDF5 file
    data_array = 'data'
    # name of the dataset where a trigger signal may be found
    trigger_array = 'data'
    # name(s) of other auxiliary to keep track of
    aligned_arrays = ()
    # transpose_array is True if the data matrix is Time x Channels
    transpose_array = False
    # allowed file extensions
    permissible_types = ['.h5', '.hdf']

    def __init__(self, experiment_path, recording, electrode, bandpass=None, notches=None, units='uV',
                 load_channels=None, trigger_idx=(), mapped=False, resample_rate=None, use_stored=True,
                 save_downsamp=True, store_path=None, raise_on_glitch=False):
        """
        Data file mapping/loading. Supports downsampling.

        Parameters
        ----------
        experiment_path: str
            File system path where recordings are found
        recording: str
            Name of recording in path
        electrode: str
            Identifier of the channel map from `ecogdata.devices.electrode_pinouts`
        bandpass: sequence
            Bandpass edges (lo, hi) -- use -1 in either place for one-sided bands
        notches: sequence
            Sequence of line frequencies to notch
        load_channels: sequence
            If only a subset of channels should be loaded, list them here. For example, to load channels from only one
            port, use `load_channels=range(128)`. Otherwise all channels are used.
        trigger_idx: int or sequence
            The index/indices of a logic-level trigger signal in this class's trigger_array
        mapped: bool
            If True, leave the dataset mapped to file. Otherwise load to memory.
        resample_rate: float
            Downsample recording to this rate (must evenly divide the raw sample rate)
        use_stored: bool
            If True, look for a pre-computed downsampled dataset
        save_downsamp: bool
            If True, save a new downsampled dataset
        store_path: str
            Save/load downsampled datasets at this path, rather than `experiment_path`
        raise_on_glitch: bool
            If True, raise exceptions on unexpected events. Otherwise try to proceed with warnings.
        """

        self.experiment_path = experiment_path
        self.recording = recording
        self.electrode = electrode
        self.bandpass = bandpass
        self.notches = notches
        self.units = units
        self.load_channels = load_channels
        self.trigger_idx = trigger_idx
        self.mapped = mapped
        self.resample_rate = resample_rate
        self.use_stored = use_stored
        self.save_downsamp = save_downsamp
        self.store_path = store_path
        self.raise_on_glitch = raise_on_glitch
        self.data_file, self.new_downsamp_file, self.units_scale = self.find_source_files()

    @property
    def raw_data_file(self):
        """Return any existing files of possible types. Else return the preferred but nonexisting file name."""
        data_files = [os.path.join(self.experiment_path, self.recording + ext) for ext in self.permissible_types]
        existing = [os.path.exists(f) for f in data_files]
        if any(existing):
            return data_files[existing.index(True)]
        else:
            return data_files[0]

    def raw_sample_rate(self):
        """
        Return full sampling rate (or -1 if there is no raw data file)
        """

        if os.path.exists(self.raw_data_file):
            with h5py.File(self.raw_data_file, 'r') as h5file:
                samp_rate = h5file['Fs'][()]
                if np.iterable(samp_rate):
                    samp_rate = float(samp_rate)
        else:
            samp_rate = -1
        return samp_rate

    def find_source_files(self):
        # Determine the source file, which would either be the full resolution recording of a previously computed
        # downsample file. Both would be opened in read-only mode.
        # + Also tag whether a new downsample is required.
        # + Saved downsamples are in micro-volts units and full sources are in ADC units. Determine the correct units
        #   scaling for both cases.

        data_file = self.raw_data_file
        units_scale = convert_scale(self.scale_to_uv, 'uv', self.units)

        # Find the full sample rate, if the full resolution file even exists (otherwise it's still possible to load a
        # downsample file).
        full_samp_rate = self.raw_sample_rate()

        # For a new sample rate (that is different than the full rate), determine if a pre-saved file can be used.
        if self.resample_rate and self.resample_rate != full_samp_rate:
            new_downsamp_file = self.recording + '_Fs{:d}'.format(int(self.resample_rate))
            search_dirs = [self.experiment_path]
            if self.store_path:
                search_dirs.insert(0, self.store_path)
            # if there is a stored file then use that and do not make a new file
            if self.use_stored:
                data_files = [os.path.join(d, new_downsamp_file + e)
                              for d in search_dirs
                              for e in self.permissible_types]
                data_files_exist = [os.path.exists(p) for p in data_files]
                if any(data_files_exist):
                    data_file = data_files[data_files_exist.index(True)]
                    # don't need a new file
                    new_downsamp_file = None
                    units_scale = convert_scale(1, 'uv', self.units)
                else:
                    # use the preferred path with the preferred extension (always .h5)
                    new_downsamp_file = os.path.join(search_dirs[0], new_downsamp_file) + '.h5'
        else:
            new_downsamp_file = None

        # check if the settled-on data file exists
        if not os.path.exists(data_file):
            raise DataPathError('Data source file {} not found'.format(data_file))
        return data_file, new_downsamp_file, units_scale

    def make_channel_map(self):
        channel_map, grounded, reference = get_electrode_map(self.electrode)
        with h5py.File(self.data_file, 'r') as h5file:
            n_data_channels = h5file[self.data_array].shape[int(self.transpose_array)]
        electrode_chans = [n for n in range(n_data_channels) if n not in grounded + reference]
        return channel_map, electrode_chans, grounded, reference

    def find_trigger_signals(self, data_file):
        trigger_idx = self.trigger_idx
        with h5py.File(data_file, 'r') as h5file:
            if not np.iterable(trigger_idx):
                trigger_idx = (trigger_idx,)
            if len(trigger_idx):
                try:
                    trigger_signal = h5file[self.trigger_array][list(trigger_idx), :]
                    pos_edge = process_trigger(trigger_signal)[0]
                except (IndexError, ValueError) as e:
                    tb = e.__traceback__
                    msg = 'Trigger channels were specified but do not exist'
                    if self.raise_on_glitch:
                        raise Exception(msg).with_traceback(tb)
                    else:
                        warnings.warn(msg, RuntimeWarning)
            else:
                trigger_signal = ()
                pos_edge = ()
        return trigger_signal, pos_edge

    def create_downsample_file(self, data_file, resample_rate, downsamp_file, antialias_aligned=False,
                               aggregate_aligned=True):
        """
        Create a downsampled datasource, possibly in a temporary file.

        TODO: if not saving a downsample and the final source is not to be mapped then there's no need to create a
         temporary file here only to load it later -- could resample into a PlainArraySource here.

        Parameters
        ----------
        data_file: str
            Path of the full resolution data
        resample_rate: float
            New sampling rate
        downsamp_file: str
            New downsample file. If empty then a temp file is made.
        antialias_aligned: bool
        aggregate_aligned: bool
            See docstring for ElectrodeDataSource.batch_change_rate

        Returns
        -------
        ds_filename: str
            File name of the downsample dataset.

        """

        Fs = self.raw_sample_rate()
        downsamp_ratio = int(Fs / resample_rate)
        with h5py.File(data_file, 'r') as h5file:
            # *Always* downsample at micro-volts scaling
            print('Creating mapped primary source {}'.format(data_file))
            try:
                # Map the raw data source using all channels (leave default electrode_channels=None)
                datasource = MappedSource(h5file, self.data_array, aligned_arrays=self.aligned_arrays,
                                          units_scale=self.scale_to_uv, transpose=self.transpose_array)
                print('Mirroring at 1 / {} rate'.format(downsamp_ratio))
                downsamp = datasource.mirror(new_rate_ratio=downsamp_ratio, mapped=True, channel_compatible=True,
                                             filename=downsamp_file)
                ds_filename = downsamp._source_file.filename
                datasource.batch_change_rate(downsamp_ratio, downsamp, verbose=True, filter_inplace=True,
                                             antialias_aligned=antialias_aligned,
                                             aggregate_aligned=aggregate_aligned)
                delete_file = False
            except:
                delete_file = True
                raise
            finally:
                if delete_file:
                    os.unlink(ds_filename)
            # this should close the file
            downsamp.close_source()
        self.transpose_array = False
        return ds_filename

    def create_dataset(self):

        channel_map, electrode_chans, ground_chans, ref_chans = self.make_channel_map()

        data_file = self.data_file
        file_is_temp = False
        if self.new_downsamp_file is not None:
            # 1. Do downsample conversion in a subroutine
            # 2. Save to a named file if save_downsamp is True
            # 3. Determine if the new source file is writeable (i.e. a temp file) or not
            downsamp_file = (self.new_downsamp_file if self.save_downsamp else '')
            print('Downsampling to {} Hz from file {} to file {}'.format(self.resample_rate, self.data_file,
                                                                         downsamp_file))
            # data_file = make_rhd_downsample(self.data_file, self.resample_rate, downsamp_file=downsamp_file)
            data_file = self.create_downsample_file(self.data_file, self.resample_rate, downsamp_file)
            file_is_temp = not self.save_downsamp
            self.units_scale = convert_scale(1, 'uv', self.units)

        open_mode = 'r+' if file_is_temp else 'r'
        print('Opening source file {} in mode {}'.format(data_file, open_mode))
        h5file = h5py.File(data_file, open_mode)
        # n_amplifier_channels = h5file['amplifier_data'].shape[0]
        # header = json.loads(h5file.attrs['JSON_header'])
        # n_amplifier_channels = h5file[self.data_array].shape[int(self.transpose_array)]
        Fs = self.raw_sample_rate()
        if self.resample_rate:
            Fs = self.resample_rate

        # Find the full set of expected electrode channels within the "amplifier_data" array
        # electrode_channels = [n for n in range(n_amplifier_channels) if n not in ground_chans + ref_chans]
        if self.load_channels:
            # If load_channels is specified, need to modify the electrode channel list and find the channel map subset
            sub_channels = list()
            sub_indices = list()
            for i, n in enumerate(electrode_chans):
                if n in self.load_channels:
                    sub_channels.append(n)
                    sub_indices.append(i)
            electrode_chans = sub_channels
            channel_map = channel_map.subset(sub_indices)
            ground_chans = [n for n in ground_chans if n in self.load_channels]
            ref_chans = [n for n in ref_chans if n in self.load_channels]

        print('Creating mapped sources')
        datasource = MappedSource(h5file, self.data_array, electrode_channels=electrode_chans,
                                  aligned_arrays=self.aligned_arrays, units_scale=self.units_scale,
                                  transpose=self.transpose_array)
        if ground_chans:
            ground_chans = MappedSource(h5file, self.data_array, electrode_channels=ground_chans,
                                        units_scale=self.units_scale, transpose=self.transpose_array)
        if ref_chans:
            ref_chans = MappedSource(h5file, self.data_array, electrode_channels=ref_chans,
                                     units_scale=self.units_scale, transpose=self.transpose_array)

        # Promote to a writeable and possibly RAM-loaded array here if either the final source should be loaded,
        # or if the mapped source is not writeable.
        needs_load = not self.mapped
        needs_writeable = (self.bandpass or self.notches) and not datasource.writeable
        if needs_load or needs_writeable:
            # Need to make writeable copies of these data sources. If the final source is to be loaded, then mirror
            # to memory here. Copy everything to memory if not mapped, otherwise copy only aligned arrays.
            print('Creating writeable mirrored sources')
            copy_mode = 'aligned' if self.mapped else 'all'
            datasource_w = datasource.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if ground_chans:
                ground_chans_w = ground_chans.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if ref_chans:
                ref_chans_w = ref_chans.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if not self.mapped:
                # swap handles of these objects
                datasource = datasource_w; datasource_w = None
                if ground_chans:
                    ground_chans = ground_chans_w; ground_chans_w = None
                if ref_chans:
                    ref_chans = ref_chans_w; ref_chans_w = None

        # For the filter blocks...
        # If mapped, then datasource and datasource_w will be identical.
        # If loaded, then datasource_w is None and datasource is filtered in-place
        if self.bandpass:
            filter_kwargs = dict(ftype='butterworth',
                                 inplace=not self.mapped,
                                 design_kwargs=dict(lo=self.bandpass[0], hi=self.bandpass[1], Fs=Fs),
                                 filt_kwargs=dict(filtfilt=True))
            if self.mapped:
                # make "verbose" filtering with progress bar if we're filtering a mapped source
                filter_kwargs['filt_kwargs']['verbose'] = True
            print('Bandpass filtering')
            datasource = datasource.filter_array(out=datasource_w, **filter_kwargs)
            if ground_chans:
                ground_chans = ground_chans.filter_array(out=ground_chans_w, **filter_kwargs)
            if ref_chans:
                ref_chans = ref_chans.filter_array(out=ref_chans_w, **filter_kwargs)
        if self.notches:
            print('Notch filtering')
            notch_kwargs = dict(inplace=True, lines=self.notches, filtfilt=True, verbose=True)
            datasource = datasource.notch_filter(out=datasource_w, **notch_kwargs)
            if ground_chans:
                ground_chans = ground_chans.notch_filter(out=datasource_w, **notch_kwargs)
            if ref_chans:
                ref_chans = ref_chans.notch_filter(out=ref_chans_w, **notch_kwargs)

        # TODO: when create_downsample is changed to downsample directly into a plain source it raises the
        #  possibility that the residual data_file will be out-of-sync with the data array. This was the concern
        #  behind putting external signals in the form of "aligned_arrays". But for some datasets (nat'l
        #  instruments), all of the external and data channels go into the same array.
        #  Option 1: create a new or linked dataset with the exctracted channels (would requires modifying source file)
        #  Option 2: extract channels that can be accessed with the same syntax as a second array
        trigger_signal, pos_edge = self.find_trigger_signals(data_file)

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
        dataset.bandpass = self.bandpass
        dataset.notches = self.notches
        dataset.units = self.units
        dataset.transient_snipped = False
        dataset.ground_chans = ground_chans
        dataset.ref_chans = ref_chans
        return dataset
