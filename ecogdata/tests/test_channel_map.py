import numpy as np

from ecogdata.channel_map import ChannelMap, map_intersection


def get_chan_map(geometry, scrambled=False, col_major=False, pitch=1.0):
    idx = list(range(geometry[0] * geometry[1]))
    if scrambled:
        idx = np.random.permutation(idx)
    return ChannelMap(idx, geometry, col_major=col_major, pitch=pitch)


def test_intersection():
    cm = get_chan_map((5, 5), pitch=0.5)
    cm2 = get_chan_map((5, 5), pitch=0.5, scrambled=True)[:10]

    ix = map_intersection((cm, cm2))
    subset = cm2.embed(np.ones(len(cm2), dtype='?'), fill=False)
    assert (subset == ix).all()


def test_slicing():
    cm = get_chan_map((5, 5), pitch=0.5)
    # should be 2nd row
    cm_sliced = cm[5:10]
    assert list(cm_sliced) == list(range(5, 10))
    assert cm_sliced.pitch == cm.pitch
    assert cm_sliced.rlookup(0) == (1, 0)    
    assert cm_sliced.rlookup(3) == (1, 3)


def test_subset_pitch():
    cm = get_chan_map((5, 5), pitch=0.5, scrambled=True)
    sub_list = [1, 4, 10, 3, 18]
    assert cm.subset(sub_list).pitch == cm.pitch
    

def test_intersection_subset():
    cm = get_chan_map((5, 5), pitch=0.5)

    cm2 = get_chan_map((5, 5), pitch=0.5, scrambled=True)[:10]

    cm_sub = cm.subset(cm2)

    assert len(cm_sub) == len(cm2)
    assert cm_sub.pitch == cm.pitch

    cm2_sites = list(zip(*cm2.to_mat()))
    sub_sites = list(zip(*cm_sub.to_mat()))
    assert set(cm2_sites) == set(sub_sites)
    

def test_1D_subset():
    cm = get_chan_map((5, 5), pitch=0.5, scrambled=True)

    sub_list = [1, 4, 10, 3, 18]

    targets = [cm[i] for i in sub_list]

    # ordered subset selection with different sequence types
    cm_sub = cm.subset(sub_list)
    assert list(cm_sub) == targets
    cm_sub = cm.subset(tuple(sub_list))
    assert list(cm_sub) == targets
    cm_sub = cm.subset(np.array(sub_list))
    assert list(cm_sub) == targets
    
    # now with masks
    sub_list = [1, 3, 4, 10, 18]
    targets = [cm[i] for i in sub_list]

    mask = np.zeros(25, '?')
    mask[sub_list] = True
    cm_sub = cm.subset(mask)
    assert list(cm_sub) == targets

    mask = np.zeros(cm.geometry, '?')
    mask[:] = True
    hot_sites = list(zip(*mask.nonzero()))
    cm_sub = cm.subset(mask)

    cm_sub_sites = list(zip(*cm_sub.to_mat()))
    assert set(cm_sub_sites) == set(hot_sites)
