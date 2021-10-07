import numpy as np
from patsy import dmatrix
from pywt import wavedec, waverec, dwt_max_level

from ..blocks import block_apply
from ...channel_map import ChannelMap
from ...devices.electrode_pinouts import get_electrode_map


def poly_residual(field: np.ndarray, channel_map: ChannelMap, mu: bool=False, order: int=2):
    """
    Subtract a polynomial spatial trend from an array recording.

    Parameters
    ----------
    field:
    channel_map:
    mu:
    order:

    Returns
    -------
    resid: np.ndarray
        Residual (field minus trend)

    """

    ii, jj = channel_map.to_mat()

    u = np.ones_like(ii)
    X = u[:, np.newaxis]
    if order > 0:
        X = np.c_[u, ii, jj]
    if order > 1:
        X = np.c_[X, ii * jj, ii ** 2, jj ** 2]

    beta = np.linalg.lstsq(X, field, rcond=None)[0]
    if mu:
        return np.dot(X, beta)
    return field - np.dot(X, beta)


def despeckle_fields(field: np.ndarray, channel_map: ChannelMap, pin_code: np.ndarray, trend_order: int=2,
                     board_code: np.ndarray=None, return_design: bool=False):
    """
    Subtract offsets common to pinout groups that can appear as "speckling" in array images (depending on how the
    pins map to electrode sites). The pin groups may also be nested in a second level of grouping that
    corresponds to multiple acquisition boards.

    Parameters
    ----------
    field :
    channel_map :
    pin_code :
    trend_order :
    board_code :
    return_design :

    Returns
    -------

    """

    pin_code = np.asarray(pin_code)
    factors = dict(pins=pin_code > 0)
    if board_code is not None:
        board_code = np.asarray(board_code)
        factors['board'] = board_code
        dm = dmatrix('~0 + C(pins):C(board)', factors)
    else:
        dm = dmatrix('~0 + C(pins)', factors)
    if return_design:
        return dm
    p_resid = poly_residual(field, channel_map, order=trend_order)
    poly = field - p_resid

    rho = np.linalg.lstsq(dm, p_resid, rcond=None)[0]
    p_resid -= np.dot(dm, rho)
    return poly + p_resid


def despeckle_recording(field: np.ndarray, channel_map: ChannelMap,
                        map_lookup: tuple=(), pin_info: tuple=(),
                        wavelet: str='haar', block_size: int=10000, trend_order: int=2):

    # method 1) lookup the pinout codes from channel map name and other info
    if map_lookup:
        map_name, map_mask, map_connectors = map_lookup
        pin_code, board_code = get_electrode_map(map_name, connectors=map_connectors, pin_codes=True)[-2:]
        if map_mask is not None:
            pin_code = np.asarray(pin_code)[map_mask]
            if board_code is not None:
                board_code = np.asarray(board_code)[map_mask]
    elif pin_info:
        pin_code, board_code = pin_info
    else:
        raise ValueError('Need either map info or pin info to determine pin coding')

    def despeckle_block(block):
        n = block.shape[1]
        if wavelet:
            levels = dwt_max_level(n, wavelet)
            block_coefs = wavedec(block, wavelet, level=levels - 3)
        else:
            block_coefs = [block]
        corrected = list()
        for b in block_coefs:
            c = despeckle_fields(b, channel_map, pin_code, board_code=board_code, trend_order=trend_order)
            corrected.append(c)
        if wavelet:
            corrected = waverec(corrected, wavelet)
        else:
            corrected = corrected[0]
        return corrected

    return block_apply(despeckle_block, block_size, (field,))