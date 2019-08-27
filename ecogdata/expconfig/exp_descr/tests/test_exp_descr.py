import itertools
from tempfile import NamedTemporaryFile
import os

from nose.tools import assert_equal
from nose.tools import assert_true
from numpy.testing import assert_array_equal
import numpy as np

from ecogdata.expconfig.exp_descr import StimulatedExperiment, ordered_epochs

def _gen_exp(n_cond=2, rep=(), rdims=()):
    x = np.arange(100)
    e_values = [ np.random.choice(x, (i+2)*2, replace=False)
                 for i in range(n_cond) ]
    if len(rep):
        for rx, rd in zip(rep, rdims):
            v = e_values[rd]
            e_values[rd] = np.repeat(v, rx)
        
    e_names = 'abcdefghijklmnop'[:n_cond]
    combos = list(itertools.product(*e_values))
    tables = np.array( list(zip(*combos)) )
    e_tables = dict( zip(e_names, tables) )
    exp = StimulatedExperiment(
        np.cumsum(np.random.rand(len(combos))), 
        event_tables=e_tables
        )
    return exp, combos, tables

def test_rolled_shape2():
    exp, _, t = _gen_exp(n_cond=2)
    t_sizes = tuple( [len(np.unique(x)) for x in t] )
    exp.set_enum_tables( ('a', 'b') )
    assert_equal( exp.rolled_conditions_shape(), t_sizes )
    exp.set_enum_tables( ('b', 'a') )
    assert_equal( exp.rolled_conditions_shape(), t_sizes[::-1] )

def test_rolled_shape3():
    exp, _, t = _gen_exp(n_cond=3)
    # caution! this sorting only works because I know the setup a priori
    ev = sorted(exp.event_names)
    ts = tuple( [len(np.unique(x)) for x in t] )
    for i, j, k in itertools.permutations([0, 1, 2]):
        exp.set_enum_tables( (ev[i], ev[j], ev[k]) )
        assert_equal( exp.rolled_conditions_shape(), (ts[i], ts[j], ts[k]) )
    
def test_stim_exp_iterator():
    exp, combos, t = _gen_exp(n_cond=3)
    ts = tuple( [len(np.unique(x)) for x in t] )
    # caution! this sorting only works because I know the setup a priori
    ev = sorted(exp.event_names)
    # the tables are generated in alphabetical order, no matter what
    # the enum_tables setting is
    exp.set_enum_tables( ('a', 'b', 'c') )

    # test single dimension iterator
    all_mask = np.zeros( len(exp), dtype='i' )
    for n, (mask, sl) in enumerate(exp.iterate_for('a', c_slice=True)):
        assert_equal(sl, [n, slice(None), slice(None)])
        all_mask += mask.astype('i')
    assert_true( (all_mask==1).all() )

    # test double dimension iterator
    all_mask = np.zeros( len(exp), dtype='i' )
    for n, (mask, sl) in enumerate(exp.iterate_for( ('b', 'c'), c_slice=True)):
        i = n // ts[-1]
        j = n % ts[-1]
        assert_equal(sl, [slice(None), i, j])
        all_mask += mask.astype('i')
    assert_true( (all_mask==1).all() )

    # test double dimension iterator with tranposition (This definitely fails!)
    ## all_mask = np.zeros( len(exp), dtype='i' )
    ## for n, (mask, sl) in enumerate(exp.iterate_for( ('c', 'b'), c_slice=True)):
    ##     i = n // ts[-1]
    ##     j = n % ts[-1]
    ##     assert_equal(sl, [slice(None), j, i])
    ##     all_mask += mask.astype('i')
    ## assert_true( (all_mask==1).all() )
    
    # test 3/3 dimension iterator 
    # (actually don't know how this works out of order)
    ## all_mask = np.zeros( len(exp), dtype='i' )
    ## for n, (mask, sl) in enumerate(
    ##         exp.iterate_for( ('b', 'c', 'a'), c_slice=True)
    ##         ):
    ##     k = n % ts[2]
    ##     j = (n // ts[2]) % ts[1]
    ##     i = n // (ts[1] * ts[2])
    ##     assert_equal(sl, [i, j, k])
    ##     all_mask += mask.astype('i')
    ## assert_true( (all_mask==1).all() )

def test_ordered_epochs():
    exp, combos, t = _gen_exp(n_cond=2, rep=(2,), rdims=(0,))
    exp.set_enum_tables( ('a', 'b') )

    ua = exp.ua
    ub = exp.ub
    
    ix, gs = ordered_epochs(exp, [ ('a', ua[0]) ], group_sizes=2)
    # Ordered_epochs is supposed to return an index into the events
    # that puts the "free" variable (in this case "b") into order.
    # When the feature combinations are repeated, then the common
    # free values should be grouped.
    assert_array_equal( exp.b[ix], np.repeat( np.sort(ub), 2 ) )
    # The group sizes should be accurate
    assert_equal(gs, [2] * len(ub))

def test_cyclic_constructor():
    cycle_a = np.random.rand(4)
    cycle_b = np.random.rand(5)
    cycle_c = 2
    times = np.arange(20)

    exp = StimulatedExperiment.from_repeating_sequences(
        times, dict(A=cycle_a, B=cycle_b, C=cycle_c)
        )

    # test preservation of cycle values
    assert_array_equal(np.sort(cycle_a), exp.uA)
    assert_array_equal(np.sort(cycle_b), exp.uB)
    assert_array_equal(np.array( [2] ), exp.uC)

    # test preservation of cycle order
    assert_array_equal( exp.A[:4], cycle_a )
    assert_array_equal( exp.B[:5], cycle_b )
    assert_array_equal( exp.C[0], cycle_c )

    # test correct length
    assert_true( len(exp) == len(times) )

    
def test_cyclic_from_text():
    cycle_a = np.random.randint(10, size=4)
    f = NamedTemporaryFile(delete=False)
    print(f.name)
    np.savetxt(f, cycle_a, fmt='%d')
    f.close()
    try:
        exp = StimulatedExperiment.from_repeating_sequences(np.arange(10), dict(A=f.name))
        assert_array_equal(exp.A[:4], cycle_a)
    finally:
        os.unlink(f.name)

    
def test_enumerate_condition():
    # test 1-, 2-, 3-conditions
    
    exp, combos, tables = _gen_exp(n_cond=1)
    exp.set_enum_tables( ('a',) )
    seq, conds = exp.enumerate_conditions()
    assert_true( (np.take(conds.a, seq) == exp.a).all(),
                 'Enumeration order wrong 1d')

    exp, combos, tables = _gen_exp(n_cond=2)
    exp.set_enum_tables( ('a', 'b') )
    seq, conds = exp.enumerate_conditions()
    assert_true( (np.take(conds.a, seq) == exp.a).all(),
                 'Enumeration order wrong 2d')
    assert_true( (np.take(conds.b, seq) == exp.b).all(),
                 'Enumeration order wrong 2d')

    exp, combos, tables = _gen_exp(n_cond=3)
    exp.set_enum_tables( ('a', 'b', 'c') )
    seq, conds = exp.enumerate_conditions()
    assert_true( (np.take(conds.a, seq) == exp.a).all(),
                 'Enumeration order wrong 3d')
    assert_true( (np.take(conds.b, seq) == exp.b).all(),
                 'Enumeration order wrong 3d')
    assert_true( (np.take(conds.c, seq) == exp.c).all(),
                 'Enumeration order wrong 3d')
