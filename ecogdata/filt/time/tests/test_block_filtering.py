from nose.tools import assert_true
 
import numpy as np
from scipy.signal import lfilter, lfilter_zi, filtfilt

from ecogdata.parallel.tests import with_start_methods
from ecogdata.filt.time.design import butter_bp
from ecogdata.filt.time.blocked_filter import bfilter


def test_filt_1d():
    r = np.random.randn(2000)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)
    zi = lfilter_zi(b, a)

    f1, _ = lfilter(b, a, r, zi=zi*r[0])
    f2 = r.copy()
    # test w/o blocking
    bfilter(b, a, f2)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = r.copy()
    bfilter(b, a, f2, bsize=234)
    assert_true((f1 == f2).all())


def test_filt_nd():
    r = np.random.randn(2000, 3, 2)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)
    zi = lfilter_zi(b, a)

    f1, _ = lfilter(b, a, r, axis=0, zi=zi[:,None,None]*r[0])
    f2 = r.copy()
    # test w/o blocking
    bfilter(b, a, f2, axis=0)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = r.copy()
    bfilter(b, a, f2, bsize=234, axis=0)
    assert_true((f1 == f2).all())


def test_filtfilt_1d():
    r = np.random.randn(2000)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)

    f1 = filtfilt(b, a, r, padtype=None)
    f2 = r.copy()
    # test w/o blocking
    bfilter(b, a, f2, filtfilt=True)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = r.copy()
    bfilter(b, a, f2, bsize=234, filtfilt=True)
    assert_true((f1 == f2).all())


def test_filtfilt_nd():
    r = np.random.randn(2000, 3, 2)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)

    f1 = filtfilt(b, a, r, axis=0, padtype=None)
    f2 = r.copy()
    # test w/o blocking
    bfilter(b, a, f2, axis=0, filtfilt=True)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = r.copy()
    bfilter(b, a, f2, bsize=234, axis=0, filtfilt=True)
    assert_true((f1 == f2).all())


@with_start_methods
def test_parfilt():
    from ecogdata.parallel.split_methods import bfilter as bfilter_p
    import ecogdata.parallel.sharedmem as shm
    r = np.random.randn(20, 2000)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)
    zi = lfilter_zi(b, a)

    f1, _ = lfilter(b, a, r, axis=1, zi=zi*r[:,:1])
    f2 = shm.shared_copy(r)
    # test w/o blocking
    bfilter_p(b, a, f2, axis=1)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = shm.shared_copy(r)
    bfilter_p(b, a, f2, bsize=234, axis=1)
    assert_true((f1 == f2).all())


@with_start_methods
def test_parfilt2():
    from ecogdata.parallel.split_methods import bfilter as bfilter_p
    import ecogdata.parallel.sharedmem as shm
    r = np.random.randn(20, 2000)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)
    zi = lfilter_zi(b, a)

    f1, _ = lfilter(b, a, r, axis=1, zi=zi*r[:,:1])
    f2 = shm.shared_copy(r)
    f3 = shm.shared_ndarray(f2.shape, f2.dtype.char)
    # test w/o blocking
    bfilter_p(b, a, f2, out=f3, axis=1)
    assert_true((f1 == f3).all())
    # test w/ blocking
    f2 = shm.shared_copy(r)
    bfilter_p(b, a, f2, out=f3, bsize=234, axis=1)
    assert_true((f1 == f3).all())


@with_start_methods
def test_parfiltfilt():
    from ecogdata.parallel.split_methods import filtfilt as filtfilt_p
    import ecogdata.parallel.sharedmem as shm
    r = np.random.randn(20, 2000)
    b, a = butter_bp(lo=30, hi=100, Fs=1000)

    f1 = filtfilt(b, a, r, axis=1, padtype=None)
    f2 = shm.shared_copy(r)
    # test w/o blocking
    filtfilt_p(f2, b, a, bsize=0)
    assert_true((f1 == f2).all())
    # test w/ blocking
    f2 = shm.shared_copy(r)
    filtfilt_p(f2, b, a, bsize=234)
    assert_true((f1 == f2).all())

