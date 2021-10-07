import numpy as np
from scipy.signal import filtfilt

from ecogdata.parallel.tests import with_start_methods
from ecogdata.filt.time.design import butter_bp
from ecogdata.filt.time.proc import filter_array


@with_start_methods
def test_parfiltfilt():
    import ecogdata.parallel.sharedmem as shm
    r = np.random.randn(20, 2000)

    design_kwargs = dict(lo=30, hi=100, Fs=1000)
    filt_kwargs = dict(filtfilt=True)
    b, a = butter_bp(**design_kwargs)

    f1 = filtfilt(b, a, r, axis=1, padtype=None)
    # not inplace and serial
    f2 = filter_array(r, inplace=False, block_filter='serial', design_kwargs=design_kwargs, filt_kwargs=filt_kwargs)
    assert np.array_equal(f1, f2), 'serial filter with copy failed'
    # inplace and serial
    f2 = r.copy()
    f3 = filter_array(f2, inplace=True, block_filter='serial', design_kwargs=design_kwargs, filt_kwargs=filt_kwargs)
    assert np.array_equal(f1, f2), 'serial filter inplace failed'
    # not inplace and parallel
    rs = shm.shared_copy(r)
    f2 = filter_array(rs, inplace=False, block_filter='parallel', design_kwargs=design_kwargs, filt_kwargs=filt_kwargs)
    assert np.array_equal(f1, f2), 'parallel filter with copy failed'
    # inplace and serial
    f2 = shm.shared_copy(r)
    f3 = filter_array(f2, inplace=True, block_filter='parallel', design_kwargs=design_kwargs, filt_kwargs=filt_kwargs)
    assert np.array_equal(f1, f2), 'parallel filter inplace failed'
