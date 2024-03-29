import warnings

from ecogdata.util import ToggleState


def test_toggle_state():
    t = ToggleState(init_state=True)
    with t():
        assert not t, 'state toggle failed'
    assert t, 'state reset failed'
    with t(True):
        assert t, 'toggle with set state failed'
    # permanent over-ride
    t = ToggleState(init_state=True, permanent_state=False)
    assert not t, 'perm state not heeded'
    with warnings.catch_warnings(record=True) as w:
        with t():
            assert not t, 'perm state not heeded in context'
        assert issubclass(w[0].category, RuntimeWarning), 'did not warn about override'
    # false to true
    t = ToggleState(init_state=False)
    assert not t
    with t():
        assert t
