import pytest
import sys
import multiprocessing.context as mpc
from ecogdata.parallel.mproc import parallel_context


test_items = [('spawn', mpc.SpawnContext)]
ids = ('spawn',)
if sys.platform != 'win32':
    test_items.extend([('fork', mpc.ForkContext),
                       ('forkserver', mpc.ForkServerContext)])
    ids = ids + ('fork', 'forkserver')

@pytest.mark.parametrize('info', tuple(test_items), ids=ids)
def test_contexts(info):
    context, ctype = info
    with parallel_context.switch_context(context):
        assert isinstance(parallel_context.ctx, ctype), 'Wrong context type'