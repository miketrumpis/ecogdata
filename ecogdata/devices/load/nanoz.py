import numpy as np


__all__ = ['import_nanoz', 'nanoz_impedance']


def import_nanoz(fname, magphs=False, mag_scale=1e6):
    "Import nano-Z electrode impedance measurements"
    # fname is the text file of the nanoz report -- assuming it always
    # comes in the same flavor
    with open(fname, 'rb') as fp:
        arr = np.genfromtxt(
            fp, delimiter='\t',
            skip_header=3, skip_footer=1,
            missing_values=['', '\t']
            )
    arr = arr[:, 1:]
    if arr.shape[1] % 2:
        arr = arr[:, :-1]
    n_freq = arr.shape[1] // 2
    arr = np.ma.masked_invalid(arr)
    mag = arr[:,0::2] * mag_scale
    phs = arr[:,1::2]
    with open(fname, 'rb') as fp:
        t = fp.readline()
        while not t.startswith(b'Site'):
            t = fp.readline()
    freqs = t.split(b'\t')[1::2]
    fx = list()
    for f in freqs[:n_freq]:
        fx.append( float(f.split(b' ')[0]) )
    fx = np.array(fx) if len(fx) > 1 else fx[0]
    mag = mag.squeeze()
    phs = phs.squeeze()
    if magphs:
        return fx, mag, phs
    else:
        return fx, mag * np.exp(1j * phs * np.pi / 180.0)
        
def nanoz_impedance(chan_map, fname, reverse_cols=True, **kwargs):
    """
    Import nano-Z electrode impedance measurements and arrange
    sites according to the given ChannelMap object.
    """

    r = import_nanoz(fname, **kwargs)
    fx = r[0]

    g = chan_map.geometry
    m_arr = list()
    for m in r[1:]:

        # reshape into electrode array grid and reverse sites
        shp = m.shape
        m = np.reshape(m, g + m.shape[1:])
        if reverse_cols:
            m = m[:, ::-1]

        m = np.reshape(m, shp)
        m_arr.append( np.take(m, chan_map.as_row_major(), axis=0) )

    return (fx,) + tuple(m_arr)
