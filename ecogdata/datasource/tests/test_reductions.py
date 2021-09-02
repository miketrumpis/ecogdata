import pytest
import numpy as np
from ecogdata.expconfig import OVERRIDE
from ecogdata.datasource.memmap import MappedSource

from .test_array_abstractions import _create_hdf5


methods = ('min', 'max', 'sum', 'mean', 'var', 'std')
axes = (0, 1, None)
keepdims = (True, False)


@pytest.mark.parametrize('reduction', methods, ids=methods)
@pytest.mark.parametrize('axis', axes, ids=['Axis {}'.format(a) for a in axes])
@pytest.mark.parametrize('keepdims', keepdims, ids=['Keep dim {}'.format(b) for b in keepdims])
def test_reductions(reduction, axis, keepdims):
    f = _create_hdf5(rand=True, dtype='f')
    arr = f['data'][()]
    data = MappedSource.from_hdf_sources(f, 'data')
    # make this just small enough to force iteration
    OVERRIDE['memory_limit'] = int(arr.shape[0] * arr.shape[1] * 0.8) * arr.dtype.itemsize
    r_method = getattr(data, reduction)
    val1 = r_method(axis=axis, keepdims=keepdims)
    val2 = getattr(arr, reduction)(axis=axis, keepdims=keepdims)
    try:
        assert np.allclose(val1, val2)
    finally:
        del OVERRIDE['memory_limit']
