import os
import h5py
import numpy as np
import warnings

from ecogdata.util import Bunch, mkdir_p
from ecogdata.datasource import MappedSource, TempFilePool, downsample_and_load
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
        mapped: bool or str
            If True, leave the dataset mapped to file. Otherwise load to memory. If the (mode == 'r+') then the
            mapped source will be writeable. (This will make a copy of the primary or downsampled datasource.)
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

        self.experiment_path = os.path.expanduser(experiment_path)
        self.recording = recording
        self.electrode = electrode
        self.bandpass = bandpass
        self.notches = notches
        self.units = units
        self.load_channels = load_channels
        self.trigger_idx = trigger_idx
        if isinstance(mapped, str):
            if mapped.lower() != 'r+':
                print('mapped value {} is not "r+", but proceeding with "r+" mode anyway'.format(mapped))
            self.mapped = True
            self.ensure_writeable = True
        else:
            self.mapped = mapped
            self.ensure_writeable = False
        self.resample_rate = resample_rate
        self.use_stored = use_stored
        self.save_downsamp = save_downsamp
        self.store_path = store_path
        self.raise_on_glitch = raise_on_glitch
        self.data_file, self.new_downsamp_file, self.units_scale = self.find_source_files()

    @property
    def primary_data_file(self):
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

        if os.path.exists(self.primary_data_file):
            with h5py.File(self.primary_data_file, 'r') as h5file:
                samp_rate = h5file['Fs'][()]
                if np.iterable(samp_rate):
                    samp_rate = float(samp_rate)
        else:
            samp_rate = -1
        return samp_rate

    def find_source_files(self):
        """
        Determine the source file, which would either be the full resolution recording or a previously computed
        downsample file. Both would be opened in read-only mode. This file is returned as "data_file".

        Also tag whether a new downsample is required. This is indicated by returning "new_downsamp_file"
        that is not None.

        Saved downsample data are in micro-volts units and full sources are in system-specific units. Determine the
        correct units scaling to self.units for both cases and return this as "units_scale"

        Returns
        -------
        data_file: str
            Data file to use for the beginning of the load sequence.
        new_downsamp_file: str or None
            Calculated name for the downsample file that needs to be created, or None if no downsampling is needed.
        units_scale:
            Scale multiplier to get values in data_file into self.units.

        """

        data_file = self.primary_data_file
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
                    # create the path if necessary
                    if not os.path.exists(search_dirs[0]):
                        mkdir_p(search_dirs[0])
        else:
            new_downsamp_file = None

        # check if the settled-on data file exists
        if not os.path.exists(data_file):
            raise DataPathError('Data source file {} not found'.format(data_file))
        return data_file, new_downsamp_file, units_scale

    def make_channel_map(self):
        """
        Return a ChannelMap and vectors of data array channels corresponding to electrode, grounded inputs,
        and reference electrodes.

        Returns
        -------
        channel_map: ChannelMap
            data-channel to electrode-grid map
        electrode_chans: list
            data channels that are electrodes
        grounded: list
            data channels that are grounded input (possibly empty)
        reference: list
            data channels that are reference electrodes (possibly empty)

        """

        channel_map, grounded, reference = get_electrode_map(self.electrode)
        with h5py.File(self.data_file, 'r') as h5file:
            n_data_channels = h5file[self.data_array].shape[int(self.transpose_array)]
        electrode_chans = [n for n in range(n_data_channels) if n not in grounded + reference]
        return channel_map, electrode_chans, grounded, reference

    def find_trigger_signals(self, data_file):
        """
        Extract trigger timing information from the data_file using `ecogdata.trigger_fun.process_triggers`. This
        will almost definitely be overloaded for each acquisition system.

        Parameters
        ----------
        data_file: str
            Raw data file where external input data should be found

        Returns
        -------
        trigger_signal: ndarray
            binarized trigger signal
        pos_edge: ndarray
            Vector of index ticks marking the "rising edge" (or timestamps) of triggered events

        """

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
        Create a downsampled datasource, possibly in a temporary file. The downsampled source is created by mapping
        and mirroring *all* channels of the raw source. The resulting data file is a channel-compatible source with
        downsampled array(s). The downsampled source *always* has floating point arrays with values in micro-volts
        units.

        The core method does not handle any metadata: this should be handled by overloading in this manner:

        ds_file = super().create_downsample_file(data_file, resample_rate, ...)
        with h5py.File(data_file, 'r') as orig_file, h5py.File(ds_file, 'r+') as new_file:
        # add/copy other metadata from the original source file

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
                aligned_arrays = []
                for name in self.aligned_arrays:
                    if isinstance(name, str) and name in h5file.keys():
                        aligned_arrays.append(name)
                    elif name[0] in h5file.keys():
                        aligned_arrays.append(name)
                # Map the raw data source using all channels (leave default electrode_channels=None)
                datasource = MappedSource.from_hdf_sources(h5file, self.data_array, aligned_arrays=aligned_arrays,
                                                           units_scale=self.scale_to_uv, transpose=self.transpose_array)
                print('Mirroring at 1 / {} rate'.format(downsamp_ratio))
                downsamp = datasource.mirror(new_rate_ratio=downsamp_ratio, mapped=True, channel_compatible=True,
                                             filename=downsamp_file)
                # TODO: this seems buried too deep -- maybe mirror should return the file name if requested
                ds_filename = downsamp.data_buffer.filename
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
            downsamp.data_buffer.close_source()
        self.transpose_array = False
        return ds_filename

    def map_raw_data(self, data_file, open_mode, electrode_chans, ground_chans, ref_chans, downsample_ratio):
        """
        Map (or load) from the raw data file. This method is called in two scenarios. The simple scenario is to
        memory-map electrode and other channels from a source HDF5 file and return those maps. A second scenario
        takes place when a new, downsampled dataset is not to be mapped nor is the downsample source to be saved. In
        this case, downsample to in-memory data sources directly after mapping the raw source.

        For some recording systems, it would be more efficient to handle the loading scenarios separately rather than
        sequentially. See `open_ephys.OpenEphysLoader.map_raw_data` for an example.

        Parameters
        ----------
        data_file: str
            Data path
        open_mode: str
            File mode (e.g. 'r', 'r+', ...)
        electrode_chans: sequence
            List of channels that correspond to electrodes
        ground_chans: sequence
            List of channels that are grounded
        ref_chans: sequence
            List of channels that are reference
        downsample_ratio: int or None
            If not None, then immediately promote the mapped data sources to downsampled arrays in memory

        Returns
        -------
        datasource: ElectrodeArraySource
            Datasource for electrodes.
        ground_chans: ElectrodeArraySource
            Datasource for ground channels. May be an empty list
        ref_chans: ElectrodeArraySource:
            Datasource for reference channels. May be an empty list

        """
        h5file = h5py.File(str(data_file), open_mode)
        if isinstance(data_file, TempFilePool):
            # need to register this to close since it will be left open
            data_file.register_to_close(h5file)
        print('Opening source file {} in mode {}'.format(data_file, open_mode))
        print('Creating mapped sources: downsample ratio {}'.format(downsample_ratio))
        # Only map aligned arrays *that exist in the file!*
        aligned_arrays = []
        for name in self.aligned_arrays:
            if isinstance(name, str) and name in h5file.keys():
                aligned_arrays.append(name)
            elif name[0] in h5file.keys():
                aligned_arrays.append(name)
        datasource = MappedSource.from_hdf_sources(h5file, self.data_array, electrode_channels=electrode_chans,
                                                   aligned_arrays=aligned_arrays, units_scale=self.units_scale,
                                                   transpose=self.transpose_array)
        if ground_chans:
            ground_chans = MappedSource.from_hdf_sources(h5file, self.data_array, electrode_channels=ground_chans,
                                                         units_scale=self.units_scale, transpose=self.transpose_array)
        if ref_chans:
            ref_chans = MappedSource.from_hdf_sources(h5file, self.data_array, electrode_channels=ref_chans,
                                                      units_scale=self.units_scale, transpose=self.transpose_array)
        if downsample_ratio > 1:
            print('Downsampling straight to memory')
            datasource = downsample_and_load(datasource, downsample_ratio, aggregate_aligned=True, verbose=True)
            if ground_chans:
                ground_chans = downsample_and_load(ground_chans, downsample_ratio)
            if ref_chans:
                ref_chans = downsample_and_load(ref_chans, downsample_ratio)
        return datasource, ground_chans, ref_chans

    def create_dataset(self):
        """
        Maps or loads raw data at the specified sampling rate and with the specified filtering applied.
        The sequence of steps follows a general logic with loading and transformation methods that can be delegated
        to subtypes.

        The final dataset can be either memory-mapped or not and downsampled or not. To avoid unnecessary
        file/memory copies, datasets are created along this path:

        This object has a "data_file" to begin dataset creation with. If downsampling, then a new file must be created
        in the case of mapping and/or saving the results. That new file source is created in
        `create_downsample_file` (candidate for overloading), and supercedes the source "data_file".

        Source file channels are organized into electrode channels and possible grounded input and reference channels.

        The prevailing "data_file" (primary or downsampled) is mapped or loaded in `map_raw_data`. If mapped,
        then MappedSource types are returned, else PlainArraySource types are retured. If downsampling is
        still pending (because the created dataset is neither mapped nor is the downsample saved), then memory
        loading is implied. This is handled by making the downsample conversion directly to memory within
        the `map_raw_data` method (another candidate for overloading).

        Check if read/write access is required for filtering, or because of self.ensure_writeable. If the data
        sources at this point are not writeable (e.g. mapped primary sources), then mirror to writeable files. If the
        dataset is not to be mapped, then promote to memory if necessary.

        Do filtering if necessary.

        Do timing extraction if necessary via `find_trigger_signals` (system specific).

        Returns
        -------
        dataset: Bunch
            Bunch containing ".data" (a DataSource), ".chan_map" (a ChannelMap), and many other metadata attributes.

        """

        channel_map, electrode_chans, ground_chans, ref_chans = self.make_channel_map()

        data_file = self.data_file
        file_is_temp = False
        # "new_downsamp_file" is not None if there was no pre-computed downsample. There are three possibilities:
        # 1. downsample to memory only (not mapped and not saving)
        # 2. downsample to a temp file (mapped and not saving)
        # 3. downsample to a named file (saving -- maybe be eventually mapped or not)
        needs_downsamp = self.new_downsamp_file is not None
        needs_file = needs_downsamp and (self.save_downsamp or self.mapped)
        if needs_file:
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
            downsample_ratio = 1
        elif needs_downsamp:
            # The "else" case now is that the master electrode source (and ref and ground channels)
            # needs downsampling to PlainArraySources
            downsample_ratio = self.raw_sample_rate() / self.resample_rate
        else:
            downsample_ratio = 1

        open_mode = 'r+' if file_is_temp else 'r'

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

        # Setting up sources should be out-sourced to a method subject to overloading. For example open-ephys data are
        # stored
        # file-per-channel. In the case of loading to memory a full sampling rate recording, the original logic would
        # require packing to HDF5 before loading (inefficient).
        datasource, ground_chans, ref_chans = self.map_raw_data(data_file, open_mode, electrode_chans, ground_chans,
                                                                ref_chans, downsample_ratio)

        # Promote to a writeable and possibly RAM-loaded array here if either the final source should be loaded,
        # or if the mapped source is not writeable.
        needs_load = isinstance(datasource, MappedSource) and not self.mapped
        filtering = bool(self.bandpass) or bool(self.notches)
        needs_writeable = (self.ensure_writeable or filtering) and not datasource.writeable
        if needs_load or needs_writeable:
            # Need to make writeable copies of these data sources. If the final source is to be loaded, then mirror
            # to memory here. Copy everything to memory if not mapped, otherwise copy only aligned arrays.
            if not self.mapped or not filtering:
                # Load data if not mapped. If mapped as writeable but not filtering, then copy to new file
                copy_mode = 'all'
            else:
                copy_mode = 'aligned'
            print('Creating writeable mirrored sources with copy mode: {}'.format(copy_mode))
            datasource_w = datasource.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if ground_chans:
                ground_chans_w = ground_chans.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if ref_chans:
                ref_chans_w = ref_chans.mirror(mapped=self.mapped, writeable=True, copy=copy_mode)
            if not self.mapped or not filtering:
                # swap handles of these objects
                datasource = datasource_w; datasource_w = None
                if ground_chans:
                    ground_chans = ground_chans_w; ground_chans_w = None
                if ref_chans:
                    ref_chans = ref_chans_w; ref_chans_w = None
        elif filtering:
            # in this case the datasource was already writeable/loaded
            datasource_w = ground_chans_w = ref_chans_w = None

        # For the filter blocks...
        # If mapped, then datasource and datasource_w will be identical (after filter_array call)
        # If loaded, then datasource_w is None and datasource is filtered in-place
        if self.bandpass:
            # TODO: should filter in two stages for stabilitys
            # filter inplace if the "writeable" source is set to None
            filter_kwargs = dict(ftype='butterworth',
                                 inplace=datasource_w is None,
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
            notch_kwargs = dict(inplace=datasource_w is None,
                                lines=self.notches, filt_kwargs=dict(filtfilt=True))
            if self.mapped:
                notch_kwargs['filt_kwargs']['verbose'] = True
            datasource = datasource.notch_filter(Fs, out=datasource_w, **notch_kwargs)
            if ground_chans:
                ground_chans = ground_chans.notch_filter(Fs, out=ground_chans_w, **notch_kwargs)
            if ref_chans:
                ref_chans = ref_chans.notch_filter(Fs, out=ref_chans_w, **notch_kwargs)

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
