from __future__ import division

from builtins import map
from builtins import zip
from builtins import range
from builtins import object
import numpy as np
import matplotlib as mpl
import itertools
from collections import namedtuple
from ecogdata.util import Bunch
from functools import reduce

__all__ = ['StimulatedExperiment', 'ordered_epochs', 'join_experiments']

class StimulatedExperiment(object):
    """
    The StimulatedExperiment associates event times during a recording
    with event labels (i.e. labels or levels for stimulation conditions).

    Parameters
    ----------
    time_stamps : sequence
        The timing of events given by indices into a timeeseries vector

    event_tables : dict
        Encoding of each stimulus feature's value, keyed by the feature name
    attrib : keyword args
        Any extra information regarding the stimulation events
    
    """
    
    enum_tables = ()
    
    def __init__(
            self, time_stamps=(), event_tables=dict(),
            condition_order=(), **attrib
            ):
        if time_stamps is None:
            self.time_stamps = ()
        else:
            self.time_stamps = time_stamps
        self._fill_tables(**event_tables)
        self.stim_props = Bunch(**attrib)
        if condition_order:
            self.set_enum_tables(condition_order)

    def set_enum_tables(self, table_names):
        """
        The enum_tables field dictates both the flat ("unrolled") 
        enumeration of condition features and, conversely, the 
        "rolled" shape of conditions. For example, 2 brightnesses
        and 3 contrasts can be enumerated by counting across rows
        in a 2x3 table [table_names=('brightness', 'contrast')] 
        or a 3x2 table [table_names=('contrast', 'brightness')].
        """
        if isinstance(table_names, str):
            table_names = (table_names,)
        good_tabs = [x for x in table_names if x in self.event_names]
        if len(good_tabs) < len(table_names):
            raise ValueError('some table names not found in this exp')
        self.enum_tables = table_names

    def enumerate_conditions(self):
        """
        Return the map of condition labels (counting numbers, beginning
        at 1), as well as tables to decode the labels into multiple
        stimulation parameters.

        Returns
        -------

        conditions : ndarray, len(experiment)
            The condition label (0 <= c < n_conditions) at each stim event

        cond_table : Bunch
            This Bunch contains an entry for every stimulation parameter.
            The entries are lookup tables for the parameter values.

        """
        
        if not len(self.time_stamps):
            return (), Bunch()
        tab_len = len(self.time_stamps)
        if not self.enum_tables:
            return np.ones(tab_len, 'i'), Bunch()

        all_uvals = []
        conditions = np.zeros(len(self.time_stamps), 'i')
        for name in self.enum_tables:
            tab = self.__dict__[name][:tab_len]
            uvals = np.unique(tab)
            all_uvals.append(uvals)
            conditions *= len(uvals)
            for n, val in enumerate(uvals):
                conditions[ tab==val ] += n

        n_vals = list(map(len, all_uvals))
        for n in range(len(all_uvals)):
            # first tile the current table of values to
            # match the preceeding "most-significant" values,
            # then repeat the tiled set to match to following
            # "least-significant" values
            utab = all_uvals[n]
            if all_uvals[n+1:]:
                rep = reduce(np.multiply, n_vals[n+1:])
                utab = np.repeat(utab, rep)
            if n > 0:
                tiles = reduce(np.multiply, n_vals[:n])
                utab = np.tile(utab, tiles)
            all_uvals[n] = utab
        
        return conditions, Bunch(**dict(zip(self.enum_tables, all_uvals)))

    def rolled_conditions_shape(self):
        """
        Return the shape of the "rolled" condition (hyper)table.
        """
        _, ctab = self.enumerate_conditions()
        return tuple(
            [len(np.unique(ctab[tab])) for tab in self.enum_tables]
            )

    def iterate_for(self, tables, c_slice=False):
        """
        Yield trial masks to step through 
        ( table A, [ table B, [[ table C, ... ]] ] )
        """
        if isinstance(tables, str):
            tables = (tables,)
        t_vals = [ np.unique(getattr(self, t)) for t in tables ]
        if c_slice:
            conds, tabs = self.enumerate_conditions()
            t_idx = [ self.enum_tables.index(t) for t in tables ]
            slices = [slice(None)] * len(self.enum_tables)
        for combo in itertools.product(*t_vals):
            mask = [ getattr(self, t) == combo[i] 
                     for i, t in enumerate(tables) ]
            mask = np.row_stack(mask)
            if c_slice:
                sl = slices[:]
                for i in range(len(combo)):
                    sl[ t_idx[i] ] = t_vals[i].searchsorted(combo[i])
                yield mask.all(axis=0), sl
            else:
                yield mask.all(axis=0)

    def order_for(self, table):
        """
        Return the permuted index to order trials on the given table.
        """

        idx = []
        for mask in self.iterate_for(table):
            idx.append( mask.nonzero()[0] )
        return np.concatenate( idx )
                
    def stim_str(self, n, mpl_text=False):
        if mpl_text:
            return mpl.text.Text(text='')
        return ''

    def _fill_tables(self, **tables):
        self.__dict__.update(tables)
        self.event_names = list(tables.keys())
        for k, v in list(tables.items()):
            setattr(self, 'u'+k, np.unique(v))
    
    def __getitem__(self, slicing):
        sub_tables = dict()
        if len(self.time_stamps):
            sub_trigs = self.time_stamps[slicing].copy()
        else:
            sub_trigs = None
        for name in self.event_names:
            table = self.__dict__[name]
            try:
                sub_tables[name] = table[slicing].copy()
            except:
                sub_tables[name] = table
        return type(self)(
            time_stamps=sub_trigs, event_tables=sub_tables, 
            condition_order=self.enum_tables, **self.stim_props
            )

    def __len__(self):
        return len(self.time_stamps)

    def subexp(self, indices):
        if hasattr(indices, 'dtype') and indices.dtype.char == '?':
            indices = np.where(indices)[0]
        # take advantage of fancy indexing?
        return self.__getitem__(indices)

    def extend(self, experiment, offset):
        if type(self) != type(experiment):
            raise TypeError('Can only join experiments of the same type')
        first_trigs = self.time_stamps
        second_trigs = experiment.time_stamps + offset
        trigs = np.r_[first_trigs, second_trigs]

        new_tables = dict()
        for name in self.event_names:
            tab1 = eval('self.%s'%name)
            tab2 = eval('experiment.%s'%name)
            new_tables[name] = np.r_[tab1, tab2]

        new_props = self.stim_props # ??? 

        return type(self)(
            time_stamps=trigs, event_tables=new_tables, 
            condition_order=self.enum_tables, **new_props
            )

    @classmethod
    def from_repeating_sequences(
            cls, time_stamps, sequences, condition_order=(), **kwargs
            ):
        """
        Generate a StimulatedExperiment from a set of condition
        sequences that cycle as time stamps count on.

        time_stamps : sequence 
            Event indices.
          
        sequences : dictionary
            The names and values of cycling condition sequences.
            Alternatively, the value can be the name of a file
            containing the values (one value per line).

        kwargs : other condition notes
        """
        
        events = dict()
        n = len(time_stamps)
        for name, cycle in list(sequences.items()):
            if isinstance(cycle, str):
                cycle = np.loadtxt(cycle)
            if not np.iterable(cycle):
                cycle = [cycle]
            # add a cycle but always truncate to n values
            # (this is a catch for partially compelete cycles)
            n_rep = n // len(cycle) + 1
            full_cycle = np.tile(cycle, n_rep)[:n]
            events[name] = full_cycle

        return cls(
            time_stamps=time_stamps, event_tables=events, 
            condition_order=condition_order, **kwargs
            )

def join_experiments(exps, offsets):
    if len(exps) < 1 or not len(offsets):
        raise ValueError('Empty experiment or offset list')
    if len(exps) < 2:
        return exps[0]
    if len(offsets) < len(exps) - 1:
        raise ValueError('Not enough offset points given to join experiments')
    new_exp = exps[0].extend(exps[1], offsets[0])
    for n in range(2, len(exps)):
        new_exp = new_exp.extend(exps[n], offsets[n-1])
    return new_exp

def ordered_epochs(exptab, fixed_vals=(), group_sizes=False):
    """
    Returns an index into the StimulatedExperiment tables that fixes
    all but one parameter, and returns the floating parameter in sorted
    order.

    Parameters
    ----------

    exptab : StimulatedExperiment

    fixed_vals : sequence
        A sequence of parameter names and fixed values:
        ( (param1, val1), (param2, val2), ... )

    group_sizes : bool
        Indicate whether to return the number of events found for
        each value of the floating parameter.
    
    """

    condition = namedtuple('condition', exptab.enum_tables)
    u_vals = []
    # get any unique condition label
    for tab in exptab.enum_tables:
        u_vals.append( getattr( exptab, 'u'+tab ) )

    # get all POSSIBLE combinations
    combos = []
    for pt in itertools.product( *u_vals ):
        combos.append( condition(*pt) )

    # prune list for fixed values
    for pair in fixed_vals:
        tab, val = pair
        keep_combos = [ c for c in combos if getattr(c, tab) == val ]
        combos = keep_combos[:]

    events = np.zeros( len(exptab), 'i' )
    tabs = [getattr(exptab, tab) for tab in exptab.enum_tables]

    # two hash tables: 1 to count occurances, 2 to bag indices
    counts = dict( [ (c, 0) for c in combos ] )
    trials = dict( [ (c, []) for c in combos ] )

    for n, pt in enumerate(zip(*tabs)):
        cond = condition(*pt)
        if cond in trials:
            counts[ cond ] += 1
            trials[ cond ].append(n)
    
    sizes = [ counts.get(c, 0) for c in combos ]
    events = np.concatenate( [ trials[c] for c in combos ] ).astype('i')

    if group_sizes:
        return events, sizes

    return events
   

    
