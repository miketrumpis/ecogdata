import os
import h5py
import json

from .file2data import FileLoader


__all__ = ['load_rhd']


def load_rhd(experiment_path, test, electrode, load_channels=None, units='uV', bandpass=(), notches=(),
             resample_rate=None, trigger_idx=(), mapped=True, save_downsamp=True, use_stored=True, store_path=None,
             raise_on_glitches=False, **loader_kwargs):
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
    resample_rate: float
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
    loader_kwargs: dict
        Other loader kwargs (for backwards compatibility with older argument convetions)

    Returns
    -------
    dataset: Bunch
        A attribute-full dictionary with an ElectrodeDataSource and ChannelMap and other metadata.

    """

    loader = RHDLoader(experiment_path, test, electrode,
                       bandpass=bandpass,
                       notches=notches,
                       units=units,
                       load_channels=load_channels,
                       trigger_idx=trigger_idx,
                       mapped=mapped,
                       resample_rate=resample_rate,
                       use_stored=use_stored,
                       save_downsamp=save_downsamp,
                       store_path=store_path,
                       raise_on_glitch=raise_on_glitches,
                       **loader_kwargs)
    return loader.create_dataset()


class RHDLoader(FileLoader):
    scale_to_uv = 0.195
    data_array = 'amplifier_data'
    trigger_array = 'board_adc_data'
    aligned_arrays = ('board_adc_data',)
    transpose_array = False

    def raw_sample_rate(self):
        """
        Return full sampling rate (or -1 if there is no raw data file)
        """

        if os.path.exists(self.primary_data_file):
            with h5py.File(self.primary_data_file, 'r') as h5file:
                header = json.loads(h5file.attrs['JSON_header'])
                samp_rate = header['sample_rate']
        else:
            samp_rate = -1
        return samp_rate

    def create_downsample_file(self, data_file, resample_rate, downsamp_file, **kwargs):
        new_file = super(RHDLoader, self).create_downsample_file(data_file, resample_rate, downsamp_file, **kwargs)
        with h5py.File(data_file, 'r') as h5file:
            header = json.loads(h5file.attrs['JSON_header'])
        with h5py.File(new_file, 'r+') as h5file:
            print('Putting header and closing file')
            header['sample_rate'] = resample_rate
            h5file.attrs['JSON_header'] = json.dumps(header)
        return new_file
