import sys
import pytest
from ecogdata.parallel.mproc import parallel_context

__all__ = ['skip_win', 'with_start_methods']


skip_win = pytest.mark.skipif(sys.platform == 'win32', reason='Windows does not fork')


def with_start_methods(test):
    """
    A decorator to run a multiprocessing test under different process modes
    supported by ``ecogdata.parallel.mproc.parallel_context``

    """
    @pytest.mark.parametrize('context', ('spawn', 'fork'))
    def test_in_context(context):
        if context == 'fork' and sys.platform == 'win32':
            pytest.skip('Windows does not fork')
        with parallel_context.switch_context(context):
            test()
    return test_in_context
