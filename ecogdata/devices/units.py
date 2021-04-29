import numpy as np


_nice_units = dict([
    ('fa', 'fA'), ('fc', 'fC'), ('fv', 'fV'), ('ff', 'fF'),
    ('pa', 'pA'), ('pc', 'pC'), ('pv', 'pV'), ('pf', 'pF'),
    ('na', 'nA'), ('nc', 'nC'), ('nv', 'nV'), ('nf', 'nF'),
    ('ua', u'\u03BCA'), ('uv', u'\u03BCV'), ('uc', u'\u03BCC'), ('uf', u'\u03BCF'),
    ('ma', 'mA'), ('mc', 'mC'), ('mv', 'mV'), ('mf', 'mF'),
    ('sigma', u'\u03C3')
])


def nice_unit_text(unit):
    return _nice_units.get(unit.lower(), unit)


_default_dtype = dict([(type(1), 'i'), (type(1.0), 'd'), (type(1j), 'D')])


def convert_dyn_range(x, r_in, r_out, out=None):
    if not np.iterable(r_in):
        zero = type(r_in)(0)
        r_in = (zero, r_in) if r_in > 0 else (r_in, zero)
    if not np.iterable(r_out):
        zero = type(r_out)(0)
        r_out = (zero, r_out) if r_out > 0 else (r_out, zero)

    if out is None:
        out = np.empty(x.shape, dtype=_default_dtype[type(r_out[0])])
        out[:] = x

    out -= r_in[0]
    out *= ((r_out[1] - r_out[0]) / (r_in[1] - r_in[0]))
    out += r_out[0]
    return out


def convert_scale(x, old, new, inplace=True):
    if old == new:
        return x

    y = x if inplace else x.copy()

    old = old.lower()[0] if len(old) == 2 else ' '
    new = new.lower()[0] if len(new) == 2 else ' '

    num_scale = {'f': 1e15, 'p': 1e12, 'n': 1e9, 'u': 1e6, 'm': 1e3, ' ': 1.0}

    scale = num_scale[new] / num_scale[old]

    y *= scale
    return y


scale_up = dict([('f', 'p'), ('p', 'n'), ('n', 'u'), ('u', 'm'), ('m', '')])
scale_dn = dict([(hi, lo) for (lo, hi) in scale_up.items()])


def best_scaling_step(x, unit_scale, allow_up=False):
    """
    Find best step size for characteristic size x of given unit scale.
    Unit scale is provided as a units string: e.g. 'nA', or 'uV'
    This method returns

    step_size : the integer step size
    scaling : triple order of magnitude scaling between x and step_size
    step_scale : units string of the step_size

    best_scaling_step(0.01, 'mv') --> (5, 1e3, 'uv')

    """

    if len(unit_scale) < 2:
        mag_scale = ''
        unit = unit_scale
    else:
        mag_scale = unit_scale[0]
        unit = unit_scale[1:]

    scaling = 1
    if allow_up:
        while x > 100:
            try:
                x /= 1e3
                scaling /= 1e3
                mag_scale = scale_up[mag_scale]
            except KeyError:
                pass
    while x < 1:
        try:
            x *= 1e3
            scaling *= 1e3
            mag_scale = scale_dn[mag_scale]
        except KeyError:
            print('scale smaller than femto')
            return (x, scaling, mag_scale)

    scale_range = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500])
    if np.any(x == scale_range):
        step_size = int(x)
    else:
        step_size = scale_range[scale_range.searchsorted(x) - 1]

    return step_size, scaling, mag_scale + unit
