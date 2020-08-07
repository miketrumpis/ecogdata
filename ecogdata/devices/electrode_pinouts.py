import re
from enum import Enum
import numpy as np
from ecogdata.channel_map import ChannelMap, CoordinateChannelMap
from ecogdata.util import mat_to_flat


NonSignalChannels = Enum('NonSignalChannels', ['grounded', 'reference', 'other'])
GND = NonSignalChannels.grounded
REF = NonSignalChannels.reference
OTHER = NonSignalChannels.other
# Turn this into a set to support membership logic ("x in NonSignalChannels")
NonSignalChannels = set(NonSignalChannels)


def _rev(n, coords):
    """
    Utility to "flip" array coordinates (reverse grid columns)

    Parameters
    ----------
    n: int
        Column dimension
    coords: sequence
        Grid column indices

    Returns
    -------
    revlist: list
        Reversed index sequence

    """
    return [c if c in NonSignalChannels else (n - c - 1) for c in coords]


def _shift_indices(idx, shift=-1):
    """
    Shift indices (e.g. 1-based to 0-based)

    """
    return [i if i in NonSignalChannels else i + shift for i in idx]


def undo_gray_code(pinout, starting=0):
    # with Gray code, row order should be
    # 0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8
    _rc = np.c_[ pinout['rows'], pinout['cols'] ]
    _rc.shape = (-1, 16-starting, 2)
    # unpermute
    _gray = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8]
    _gray = _gray[starting:]
    _gray_unmix = [_gray.index(i+starting) for i in range(len(_gray))]
    _rc_unmix = np.take(_rc, _gray_unmix, axis=1)
    _rc_unmix.shape = (-1, 2)
    pinout_gray = dict(
        geometry = (8, 8),
        pitch=0.406,
        rows = _rc_unmix[:,0].copy(),
        cols = _rc_unmix[:,1].copy()
        )
    return pinout_gray


def unzip_encoded(coord_list, shift_index=True, skipchars='ioqs'):
    """
    Transform a comma-separated list of electrode coordinates from "A3, H1, C8, ..."
    to row-column coordinates [(0, 2), (7, 0), (2, 7), ...]. Coordinates named "ref"
    or "gnd" will be marked as such. This method should accommodate multiple-letter
    codes like "AA1", "AAA1", ...

    Parameters
    ----------
    coord_list: str
        "Encoded" coordinates
    shift_index: bool
        Whether to shift to zero-based indexing
    skipchars: str
        Do not enumerate these reserved letters (remove them from the "alphabet" of location codes)

    Returns
    -------
    rows: list
    cols: list
    """

    coord_list = [c.strip() for c in coord_list.strip().split(',')]
    row_set = set()
    rows = []
    cols = []
    # step 1 get the column integers and bag the row encoding into a set
    for coord in coord_list:
        if coord.lower() == 'ref':
            cols.append(REF)
            rows.append(REF)
            continue
        if coord.lower() == 'gnd':
            cols.append(GND)
            rows.append(GND)
            continue
        m = re.search(r'^[A-Za-z]+', coord)
        if not m:
            cols.append(OTHER)
            rows.append(OTHER)
            continue
        row_set.add(m.group())
        rows.append(m.group().lower())
        m = re.search(r'\d+$', coord)
        if not m:
            print('weird coordinate:', coord)
            cols.append(OTHER)
            continue
        # subtract 1 from the column number if shifting to zero-based
        cols.append(int(m.group()) - int(shift_index))

    az = []
    # step 2 form a alpha-to-enumeration table
    rep = 1
    while len(row_set) > len(az):
        az.extend([chr(i) * rep for i in range(97, 97 + 26)])
        for skip in skipchars:
            az.remove(skip * rep)
        rep += 1
    # add 1 to the enumeration if not shifting to zero-based
    row_base = int(not shift_index)
    az = dict([(alpha, num + row_base) for num, alpha in enumerate(az)])
    row_nums = [r if r in NonSignalChannels else az[r] for r in rows]
    return row_nums, cols


##### Try multi-stage mapping, e.g. site-to-zif merged with zif-to-digout

def connect_passive_map(
        geometry, electrode_map, daq_order,
        interconnects=(), reverse_cols=True, pitch=1.0
):
    """
    Make an end-to-end channel map. In the simplest scenario, only
    daq_order and electrode_map are used.

    interconnects is an optional list of inter-connecting pin-to-pin
    lookups between the DAQ and the electrode pins


    Parameters
    ----------
    geometry: tuple
        Electrode geometry (rows, cols)
    electrode_map: dict
        Dictionary of pin names (any pins) to grid indices (i.e. (row, col))
    daq_order: sequence
        A list of pin names in the order of the data array channels
    interconnects: sequence
        Any intermediate pin-to-pin mappings between the daq_order sequence
        and the electrode_map lookup. Intermediate maps are dictionaries. The
        first map in the sequence takes the pin names in daq_order as keys and
        returns new key names (that feed into the next interconnect, and so on).
        The final map returns pin names in electrode_map.
    reverse_cols: bool
        array columns should be flipped to anatomical (surgeon's) perspective
    pitch: float or (float, float)
        inter-electrode pitch

    Returns
    -------
    map_dict: dict
        a channel mapping dictionary

    """

    rows = []
    cols = []
    for n in range(len(daq_order)):
        if daq_order[n] is None:
            # if the DAQ readout has None on a channel,
            # then it is not connected
            rows.append(GND)
            cols.append(GND)
            continue

        pin = daq_order[n]
        for xc in interconnects:
            pin = xc[pin]

        site = electrode_map[pin]
        rows.append(site[0])
        # cols.append( nc - site[1] - 1 if reverse_cols else site[1] )
        if site[1] in NonSignalChannels or not reverse_cols:
            cols.append(site[1])
        else:
            cols.append(-site[1])

    if reverse_cols:
        mn_col = min(filter(lambda x: x not in NonSignalChannels, cols))
        cols = [c if c in NonSignalChannels else c - mn_col for c in cols]
    map_dict = {'geometry': geometry,
                'rows': rows,
                'cols': cols,
                'pitch': pitch}
    return map_dict


# New electrode-to-data tables are created by specifying
# 1) electrode-site to initial pinout lookup
# 2) optional intermediary pin-to-pin lookups for interconnects
# 3) final pin-to-data channel lookup
#
# The final electrode map dictionary is created by the following
#    ratv4_intan=connect_passive_map( (8,8), rat_v4_by_zif_lut,
#                                     zif_by_intan64, pitch=0.406 )

# Example 1: table, listed by "grid coordinate" strings
# step 1: list coordinates in ZIF order (these correspond to zif1, zif2, ...)
rat_v4_by_zif = """G1, G2, H2, H3, F1, G3, F2, F3, H4, E1, D1, C1, B1, A1, G4, F4, E2, D2, A2, B2, C2, E3, D3, A3, 
B3, C3, E4, D4, A4, B4, C4, C5, B5, A5, D5, E5, C6, B6, A6, D6, E6, C7, B7, A7, D7, E7, F5, G5, B8, C8, D8, E8, H5, 
F6, F7, G6, F8, H6, H7, G7, G8"""
# step 2: use "unzip_encoded()" to convert to rows, columns
rat_v4_by_zif_rc = zip(*unzip_encoded(rat_v4_by_zif))
# step 3: build lookup table (LUT) from ZIF pin # to (row, column) pair
rat_v4_by_zif_lut = dict(zip(range(1, 62), rat_v4_by_zif_rc))

# Example 2: Coordinate grid listed as (y, x) to mimic (row, column)
rat_varspace_by_zif_rc = zip(
    [43.2502, 43.1002, 43.0502, 43.0502, 42.9502, 42.9502, 42.8002, 42.8002,
     42.9502, 42.8002, 43.2502, 43.2502, 43.1002, 42.5002, 42.5002, 42.3502,
     42.2002, 42.0502, 42.5002, 42.5002, 42.3502, 42.0502, 42.2502, 42.2502,
     42.0502, 42.3502, 42.3502, 42.2002, 42.0502, 42.8002, 42.9502, 42.9502,
     42.8002, 42.0502, 42.2002, 42.3502, 42.3502, 42.0502, 42.2502, 42.2502,
     42.0502, 42.3502, 42.5002, 42.5002, 42.2002, 42.3502, 42.5002, 42.5002,
     43.1002, 43.2502, 43.2502, 42.8002, 42.9502, 42.8002, 42.8002, 42.9502,
     42.9502, 43.0502, 43.0502, 43.1002, 43.2502],
    [9.6846, 9.8846, 9.5346, 9.6346, 9.8846, 9.6346, 9.6846, 9.8846, 9.5346,
     9.4846, 9.4846, 9.2846, 9.2846, 9.8846, 9.6846, 9.8846, 9.8846, 9.8846,
     9.4846, 9.2846, 9.6346, 9.6846, 9.6346, 9.5346, 9.4846, 9.5346, 9.2846,
     9.2846, 9.2846, 9.2846, 9.2846, 8.9846, 8.9846, 8.9846, 8.9846, 8.9846,
     8.7346, 8.7846, 8.7346, 8.6346, 8.5846, 8.6346, 8.9846, 8.7846, 8.3846,
     8.3846, 8.5846, 8.3846, 8.9846, 8.9846, 8.7846, 8.7846, 8.7346, 8.3846,
     8.5846, 8.6346, 8.3846, 8.6346, 8.7346, 8.3846, 8.5846]
)
rat_varspace_by_zif_lut = dict(zip(range(1, 62), rat_varspace_by_zif_rc))

# for reference electrodes
rat_refelectrode_by_zif = """REF, H3, H4, G3, G1, G2, G4, F3, F1, F2, F4, E3, E1, E2, E4, D1, B1, A1, C1, D2, B2, A2, 
C2, D3, B3, A3, C3, D4, B4, A4, C4, C5, A5, B5, D5, C6, A6, B6, D6, C7, A7, B7, D7, C8, B8, D8, E5, E7, E8, E6, F5, 
F7, F8, F6, G5, G7, G8, G6, H5, H6, REF"""
rat_refelectrode_by_zif_rc = zip(*unzip_encoded(rat_refelectrode_by_zif))
rat_refelectrode_by_zif_lut = dict(zip(range(1, 62), rat_refelectrode_by_zif_rc))
# R1 and R2 are the same - reference electrodes

# Microlead (rat v5)
rat_v5_by_zif = """E4, H3, H2, H4, G1, G4, G2, F1, G3, F4, E1, F2, F3, D1, E2, E3, C1, D2, D3, B1, A1, C2, B2, A2, 
C3, B3, A3, C4, B4, A4, D4, D5, A5, B5, C5, A6, B6, C6, A7, B7, C7, B8, D6, D7, C8, E6, E7, D8, F6, F7, E8, F5, G6, 
F8, G7, G5, G8, H5, H7, H6, E5"""
# convert to a list of (row, col) coordinates
rat_v5_by_zif_rc = zip(*unzip_encoded(rat_v5_by_zif))
# create a zif-to-grid lookup
rat_v5_by_zif_lut = dict(zip(range(1, 62), rat_v5_by_zif_rc))


# These are the present set of lookups by recording system
# (these are the final pinout-to-data maps)
# Note 1)
# ZIF names are conventionally 1-based: leave them this way
#
# Note 2)
# The MUX maps are ordered by digital outs, so channel arrays potentially need
# to be permuted to enforce digital out order

zif_by_mux6 = [None, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30,
               None, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
               None, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33,
               None, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]

zif_by_mux6_15row = [zif_by_mux6[i]
                     for i in range(64) if i not in (0, 16, 32, 48)]

zif_by_stim4 = [61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, 31,
                None, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, None,
                None, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32]

zif_by_intan64 = [32, None, 34, 48, 36, 50, 38, 52, 40, 54, 42, 56, 44, 58,
                  46, 60, 47, 49, 45, 51, 43, 53, 41, 55, 39, 57, 37, 59, 35,
                  61, 33, None, 31, 1, 29, 3, 27, 5, 25, 7, 23, 9, 21,
                  11, 19, 13, 17, 15, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10,
                  26, 12, 28, 14, 30, None]


## Human 256 v2 (direct map from grid to Intan RHD channels)
h256_grid_by_channels = """J4, L4, G5, A8, D5, F7, C5, C8, F5, J8, D7, D9, H7, G8, H2, E9, H8, H4, K6, A5, E7, B5, 
H3, E5, G7, J1, L3, H5, L2, E6, B8, J3, J5, L7, J2, L6, C6, K7, B6, A7, F6, K2, A6, K4, K1, B7, K5, E8, G9, C7, L8, 
L1, B9, J7, C9, H6, K8, K3, L5, J6, F8, D6, D8, G6, K10, B10, L10, M8, M10, C10, L11, M6, K11, L9, F10, A9, M9, M2, 
E11, M1, M7, M11, H10, G11, D11, F11, G10, A11, J10, B11, E10, C11, D10, J11, A10, F3, H11, G1, F2, F4, C2, B4, E1, 
C4, D3, C3, D2, D4, D1, E2, E3, G3, H9, E4, M4, G2, J9, H1, F9, F1, K9, B3, G4, B2, M5, B1, M3, C1, F20, G19, F21, 
M20, F22, G21, B21, M18, B22, K14, E19, F14, H22, H14, E20, J14, M19, B20, C19, C20, E21, D22, D19, D21, B19, D20, 
F19, E22, G22, C22, G20, K13, C21, D13, L13, E13, H12, J13, C12, H13, B12, G12, A12, G13, F12, D12, E12, A13, M21, 
F13, M16, C13, M22, M14, A14, M13, L14, M12, B13, L12, M17, K12, M15, J12, J20, L18, J21, D15, H17, L22, J17, F15, 
D17, K15, C16, C14, J16, G14, K18, B14, L15, K20, A16, K21, B16, K22, K19, A17, K16, F17, L17, B17, L16, G17, E15, 
J19, C17, L21, G18, L20, J18, G16, H18, K17, J22, A18, E18, H20, B18, E16, H21, B15, G15, D16, H15, F16, E14, H16, 
D14, D18, J15, H19, L19, C18, C15, F18, A15, E17"""
h256_rows_cols = list(unzip_encoded(h256_grid_by_channels))
h256_rows_cols[1] = _rev(22, h256_rows_cols[1])
psv_256_rhd = {
    'rows': h256_rows_cols[0],
    'cols': h256_rows_cols[1],
    'pitch': 1.72,
    'geometry': (12, 22)
}


## PSV 244 Array
# **** MUX 1 ****
# each entry is a list of row or column coordinates, in order of
# the demultiplexed channels of inner (x.1) and outer (x.2) FCI
# connectors
psv_244_mux1 = {
    'geometry': (16, 16),
    'rows1.1': [0, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5,
                3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 6, 3, 1, 2, 0, 1],
    'cols1.1': [13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8,
                7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 7, 3, 3, 2, 2, 1],
    'rows1.2': [GND, GND, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4, 0, 2,
                1, 3, 5, 4, 0, 2, 1, 3, 7, 6, 0, 2, 1, GND],
    'cols1.2': [GND, GND, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8,
                7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 7, 6, 3, 3, 2, GND],
    ## Quadrant 2
    'rows2.1': [2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9,
                9, 10, 10, 10, 11, 11, 11, 8, 12, 12, 13, 13, 14],
    'cols2.1': [0, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 3, 1, 2,
                0, 4, 5, 3, 1, 2, 0, 4, 6, 3, 1, 2, 0, 1],
    'rows2.2': [GND, GND, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8, 8,
                8, 9, 9, 9, 10, 10, 10, 11, 11, 8, 9, 12, 12, 13, GND],
    'cols2.2': [GND, GND, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4, 0,
                2, 1, 3, 5, 4, 0, 2, 1, 3, 7, 6, 0, 2, 1, GND],
    ## Quadrant 3
    'rows3.1': [15, 14, 14, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 10,
                12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 9, 12, 14, 13,
                15, 14],
    'cols3.1': [2, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9,
                9, 10, 10, 10, 11, 11, 11, 8, 12, 12, 13, 13, 14],
    'rows3.2': [GND, GND, 15, 13, 15, 13, 14, 12, 9, 11, 15, 13, 14, 12,
                10, 11, 15, 13, 14, 12, 10, 11, 15, 13, 14, 12, 8, 9, 15,
                13, 14, GND],
    'cols3.2': [GND, GND, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8, 8,
                8, 9, 9, 9, 10, 10, 10, 11, 11, 8, 9, 12, 12, 13, GND],
    ## Quadrant 4
    'rows4.1': [13, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7,
                7, 6, 6, 6, 5, 5, 5, 4, 4, 4, 7, 3, 3, 2, 2, 1],
    'cols4.1': [15, 14, 14, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11,
                10, 12, 14, 13, 15, 11, 10, 12, 14, 13, 15, 11, 9, 12, 14,
                13, 15, 14],
    'rows4.2': [GND, GND, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8, 7,
                7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 7, 6, 3, 3, 2, GND],
    'cols4.2': [GND, GND, 15, 13, 15, 13, 14, 12, 9, 11, 15, 13, 14, 12,
                10, 11, 15, 13, 14, 12, 10, 11, 15, 13, 14, 12, 8, 9,
                15, 13, 14, GND],
    'pitch': 0.75
}

# **** MUX 3 ****
# Each entry is a list of row or column coordinates associated with
# a single MUX. For each quadrant "x", the MUXes are identified by
# the line label of an op-amp output {x.1-, x.1+, x.3-, x.3+}.
psv_244_mux3 = {
    'geometry': (16, 16),

    'rows1.1-': [GND, 1, 2, 0, 6, 7, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0],
    'rows1.1+': [GND, 0, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4],
    'rows1.3-': [GND, 1, 0, 2, 1, 3, 6, 4, 0, 2, 3, 5, 4, 0, 2, 1],
    'rows1.3+': [GND, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 3],

    'cols1.1-': [GND, 2, 3, 3, 6, 7, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7],
    'cols1.1+': [GND, 13, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8, 7],
    'cols1.3-': [GND, 1, 2, 2, 3, 3, 7, 4, 4, 4, 5, 5, 6, 6, 6, 7],
    'cols1.3+': [GND, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7],

    'rows2.1-': [GND, 13, 12, 12, 9, 8, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8],
    'rows2.1+': [GND, 2, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8],
    'rows2.3-': [GND, 14, 13, 13, 12, 12, 8, 11, 11, 11, 10, 10, 9, 9, 9, 8],
    'rows2.3+': [GND, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],

    'cols2.1-': [GND, 1, 2, 0, 6, 7, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0],
    'cols2.1+': [GND, 0, 0, 2, 0, 2, 1, 3, 6, 4, 0, 2, 1, 3, 5, 4],
    'cols2.3-': [GND, 1, 0, 2, 1, 3, 6, 4, 0, 2, 3, 5, 4, 0, 2, 1],
    'cols2.3+': [GND, 1, 1, 3, 1, 2, 0, 4, 5, 3, 1, 2, 0, 4, 5, 3],

    'rows3.1-': [GND, 14, 13, 15, 9, 8, 12, 14, 13,
                 15, 11, 10, 12, 14, 13, 15],
    'rows3.1+': [GND, 15, 15, 13, 15, 13, 14, 12, 9,
                 11, 15, 13, 14, 12, 10, 11],
    'rows3.3-': [GND, 14, 15, 13, 14, 12, 9, 11, 15,
                 13, 12, 10, 11, 15, 13, 14],
    'rows3.3+': [GND, 14, 14, 12, 14, 13, 15, 11, 10,
                 12, 14, 13, 15, 11, 10, 12],

    'cols3.1-': [GND, 13, 12, 12, 9, 8, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8],
    'cols3.1+': [GND, 2, 3, 3, 4, 4, 5, 5, 7, 6, 6, 6, 7, 7, 7, 8],
    'cols3.3-': [GND, 14, 13, 13, 12, 12, 8, 11, 11, 11, 10, 10, 9, 9, 9, 8],
    'cols3.3+': [GND, 2, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 8, 8],

    'rows4.1-': [GND, 2, 3, 3, 6, 7, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7],
    'rows4.1+': [GND, 13, 12, 12, 11, 11, 10, 10, 8, 9, 9, 9, 8, 8, 8, 7],
    'rows4.3-': [GND, 1, 2, 2, 3, 3, 7, 4, 4, 4, 5, 5, 6, 6, 6, 7],
    'rows4.3+': [GND, 13, 12, 11, 11, 10, 10, 10, 9, 9, 9, 8, 8, 8, 7, 7],

    'cols4.1-': [GND, 14, 13, 15, 9, 8, 12, 14, 13,
                 15, 11, 10, 12, 14, 13, 15],
    'cols4.1+': [GND, 15, 15, 13, 15, 13, 14, 12, 9,
                 11, 15, 13, 14, 12, 10, 11],
    'cols4.3-': [GND, 14, 15, 13, 14, 12, 9, 11, 15,
                 13, 12, 10, 11, 15, 13, 14],
    'cols4.3+': [GND, 14, 14, 12, 14, 13, 15, 11, 10,
                 12, 14, 13, 15, 11, 10, 12],
    'pitch': 0.75
}

psv_244_intan = dict(
    geometry=(16, 16),

    pitch=0.75,

    rows=[3, GND, 5, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3, 1, 5, 1, 6, 3, 4, 1,
          0, 2, 2, 0, 1, 2, 3, 0, 5, 0, 4, GND, 0, 1, 2, 1, 1, 2, 3, 0,
          5, 6, 4, 7, 0, 3, 2, 1, 2, 0, 1, 2, 3, 1, 5, 3, 4, 6, 0, 4, 2,
          0, 1, GND, 8, GND, 8, 5, 7, 5, 7, 5, 7, 4, 6, 4, 6, 3, 6, 2, 7,
          5, 6, 5, 6, 4, 6, 4, 7, 3, 7, 3, 7, 2, 8, GND, 8, 14, 8, 13, 9,
          12, 9, 12, 9, 9, 10, 8, 10, 11, 10, 11, 11, 13, 10, 13, 10, 12,
          10, 12, 9, 8, 9, 11, 9, 11, 8, GND, 12, GND, 10, 11, 11, 15, 15,
          13, 13, 14, 14, 12, 12, 14, 10, 14, 9, 12, 11, 14, 15, 13, 13,
          15, 14, 13, 12, 15, 10, 15, 11, GND, 15, 14, 13, 14, 14, 13, 12,
          15, 10, 9, 11, 8, 15, 12, 13, 14, 13, 15, 14, 13, 12, 14, 10,
          12, 11, 9, 15, 11, 13, 15, 14, GND, 7, GND, 7, 10, 8, 10, 8, 10,
          8, 11, 9, 11, 9, 12, 9, 13, 8, 10, 9, 10, 9, 11, 9, 11, 8, 12,
          8, 12, 8, 13, 7, GND, 7, 1, 7, 2, 6, 3, 6, 3, 6, 6, 5, 7, 5, 4,
          5, 4, 4, 2, 5, 2, 5, 3, 5, 3, 6, 7, 6, 4, 6, 4, 7, GND],

    cols=_rev(16, [7, GND, 7, 10, 8, 10, 8, 10, 8, 11, 9, 11, 9, 12, 9,
                   13, 8, 10, 9, 10, 9, 11, 9, 11, 8, 12, 8, 12, 8, 13,
                   7, GND, 7, 1, 7, 2, 6, 3, 6, 3, 6, 6, 5, 7, 5, 4, 5, 4,
                   4, 2, 5, 2, 5, 3, 5, 3, 6, 7, 6, 4, 6, 4, 7, GND, 3, GND,
                   5, 4, 4, 0, 0, 2, 2, 1, 1, 3, 3, 1, 5, 1, 6, 3, 4, 1,
                   0, 2, 2, 0, 1, 2, 3, 0, 5, 0, 4, GND, 0, 1, 2, 1, 1, 2,
                   3, 0, 5, 6, 4, 7, 0, 3, 2, 1, 2, 0, 1, 2, 3, 1, 5, 3,
                   4, 6, 0, 4, 2, 0, 1, GND, 8, GND, 8, 5, 7, 5, 7, 5, 7, 4,
                   6, 4, 6, 3, 6, 2, 7, 5, 6, 5, 6, 4, 6, 4, 7, 3, 7, 3, 7,
                   2, 8, GND, 8, 14, 8, 13, 9, 12, 9, 12, 9, 9, 10, 8, 10,
                   11, 10, 11, 11, 13, 10, 13, 10, 12, 10, 12, 9, 8, 9, 11,
                   9, 11, 8, GND, 12, GND, 10, 11, 11, 15, 15, 13, 13, 14,
                   14, 12, 12, 14, 10, 14, 9, 12, 11, 14, 15, 13, 13, 15,
                   14, 13, 12, 15, 10, 15, 11, GND, 15, 14, 13, 14, 14, 13,
                   12, 15, 10, 9, 11, 8, 15, 12, 13, 14, 13, 15, 14, 13,
                   12, 14, 10, 12, 11, 9, 15, 11, 13, 15, 14, GND])
)

psv_244_rhd = dict(
    geometry=(16, 16),

    pitch=0.75,

    rows= [3,1,4,3,2,2,1,1,0,1,4,GND,0,0,5,0,
            2,3,6,2,4,1,0,3,3,5,1,4,2,5,0,GND,
            0,0,2,6,2,7,1,3,3,0,5,2,4,1,1,2,
            1,2,1,4,GND,0,0,4,2,3,6,5,1,0,3,1,
            8,GND,8,5,7,5,7,5,7,4,6,4,6,3,6,2,
            7,5,6,5,6,4,6,4,7,3,7,3,7,2,8,GND,
            8,14,8,13,9,12,9,12,9,9,10,8,10,11,10,11,
            11,13,10,13,10,12,10,12,9,8,9,11,9,11,8,GND,
            12,GND,10,11,11,15,15,13,13,14,14,12,12,14,10,14,
            9,12,11,14,15,13,13,15,14,13,12,15,10,15,11,GND,
            15,14,13,14,14,13,12,15,10,9,11,8,15,12,13,14,
            13,15,14,13,12,14,10,12,11,9,15,11,13,15,14,GND,
            7,11,8,11,8,10,9,12,8,13,10,GND,10,12,9,13,
            12,9,8,9,9,8,9,8,10,8,10,7,11,7,11,GND,
            7,3,6,6,7,7,6,4,6,5,6,5,5,4,5,3,
            1,4,2,4,GND,4,2,6,2,5,7,5,3,6,3,7],

    cols=_rev(16, [7,11,8,11,8,10,9,12,8,13,10,GND,10,12,9,13,
                    12,9,8,9,9,8,9,8,10,8,10,7,11,7,11,GND,
                    7,3,6,6,7,7,6,4,6,5,6,5,5,4,5,3,
                    1,4,2,4,GND,4,2,6,2,5,7,5,3,6,3,7,
                    3,GND,5,4,4,0,0,2,2,1,1,3,3,1,5,1,
                    6,3,4,1,0,2,2,0,1,2,3,0,5,0,4,GND,
                    0,1,2,1,1,2,3,0,5,6,4,7,0,3,2,1,
                    2,0,1,2,3,1,5,3,4,6,0,4,2,0,1,GND,
                    8,GND,8,5,7,5,7,5,7,4,6,4,6,3,6,2,
                    7,5,6,5,6,4,6,4,7,3,7,3,7,2,8,GND,
                    8,14,8,13,9,12,9,12,9,9,10,8,10,11,10,11,
                    11,13,10,13,10,12,10,12,9,8,9,11,9,11,8,GND,
                    12,14,11,12,13,13,14,14,15,14,11,GND,15,15,10,15,
                    13,12,9,13,11,14,15,12,12,10,14,11,13,10,15,GND,
                    15,15,13,9,13,8,14,12,12,15,10,13,11,14,14,13,
                    14,13,14,11,GND,15,15,11,13,12,9,10,14,15,12,14])
)

psv_1024_rhd = dict(
    geometry=(46, 24),

    pitch=1.72,

    rows= [8,10,6,0,3,5,2,2,5,8,3,3,7,6,7,4,7,7,9,0,4,1,7,4,6,8,10,7,10,4,1,8,8,10,8,10,2,9,1,0,5,9,0,9,9,1,
            9,4,6,2,10,10,1,8,2,7,9,9,10,8,5,3,3,6,9,1,10,11,11,2,10,11,9,10,5,0,11,11,4,11,11,11,7,6,3,5,6,0,8,1,4,2,
            3,8,0,5,7,6,5,5,2,1,4,2,3,2,3,3,3,4,4,6,7,4,11,6,8,7,5,5,9,1,6,1,11,1,11,2,5,6,5,11,5,6,1,11,1,9,
            4,5,7,7,4,8,11,1,2,2,4,3,3,3,1,3,5,4,6,2,6,9,2,3,10,4,7,8,2,7,1,6,0,6,5,3,4,0,11,5,11,2,11,11,0,11,
            10,11,1,10,11,9,11,8,8,10,8,3,7,10,8,5,3,9,2,2,8,6,9,1,10,9,0,9,1,9,9,0,9,5,10,1,10,6,4,8,2,10,6,10,8,6,
            7,9,8,0,4,7,1,4,7,1,6,3,7,5,4,7,3,3,8,7,10,2,2,5,0,4,15,15,16,19,16,18,16,19,16,19,18,20,18,19,13,20,19,15,17,16,
            18,16,14,16,18,12,14,16,13,17,19,14,16,18,13,17,17,18,17,18,17,13,17,15,12,18,16,19,20,18,19,12,20,18,20,17,19,14,16,17,19,17,19,17,21,21,
            21,19,21,21,22,17,22,20,21,20,20,13,22,12,18,22,21,22,22,22,21,22,21,22,21,22,21,22,21,14,22,12,13,15,13,15,12,15,14,14,13,15,12,13,14,14,
            20,15,15,13,20,12,20,12,20,14,15,13,16,12,14,12,31,30,32,31,33,32,32,29,33,25,30,25,33,25,31,25,30,31,30,31,32,33,30,32,30,31,30,33,33,33,
            31,24,32,24,24,24,23,24,23,24,23,23,23,24,23,23,23,24,32,24,27,24,33,25,25,24,25,23,24,23,28,23,26,23,31,29,32,26,28,33,28,26,28,26,27,25,
            27,25,29,25,26,31,27,32,27,33,30,28,27,28,28,28,27,28,26,30,28,32,29,31,29,27,29,28,33,29,29,31,29,27,32,26,26,27,26,27,25,27,25,29,26,30,
            30,29,26,29,26,28,37,35,39,45,42,40,43,43,40,37,42,42,38,39,38,41,38,38,36,45,41,44,38,41,39,37,35,38,35,41,44,37,37,35,37,35,43,36,44,45,
            40,36,45,36,36,44,36,41,39,43,35,35,44,37,43,38,36,36,35,37,40,42,42,39,36,44,35,34,34,43,35,34,36,35,40,45,34,34,41,34,34,34,38,39,42,40,
            39,45,37,44,41,43,42,37,45,40,38,39,40,40,43,44,41,43,42,43,42,42,42,41,41,39,38,41,34,39,37,38,40,40,36,44,39,44,34,44,34,43,40,39,40,34,
            40,39,44,34,44,36,41,40,38,38,41,37,34,44,43,43,41,42,42,42,44,42,40,41,39,43,39,36,43,42,35,41,38,37,43,38,44,39,45,39,40,42,41,45,34,40,
            34,43,34,34,45,34,35,34,44,35,34,36,34,37,37,35,37,42,38,35,37,40,42,36,43,43,37,39,36,44,35,36,45,36,44,36,36,45,36,40,35,44,35,39,41,37,
            43,35,39,35,37,39,38,36,37,45,41,38,44,41,38,44,39,42,38,40,41,38,42,42,37,38,35,43,43,40,45,41,30,30,29,26,29,27,29,26,29,26,27,25,27,26,
            32,25,26,30,28,29,27,29,31,29,27,33,31,29,32,28,26,31,29,27,32,28,28,27,28,27,28,32,28,30,33,27,29,26,25,27,26,33,25,27,25,28,26,31,29,28,
            26,28,26,28,24,24,24,26,24,24,23,28,23,25,24,25,25,32,23,33,27,23,24,23,23,23,24,23,24,23,24,23,24,23,24,31,23,33,32,30,32,30,33,30,31,31,
            32,30,33,32,31,31,25,30,30,32,25,33,25,33,25,31,30,32,29,33,31,33,14,15,13,14,12,13,13,16,12,20,15,20,12,20,14,20,15,14,15,14,13,12,15,13,
            15,14,15,12,12,12,14,21,13,21,21,21,22,21,22,21,22,22,22,21,22,22,22,21,13,21,18,21,12,20,20,21,20,22,21,22,17,22,19,22,14,16,13,19,17,12,
            17,19,17,19,18,20,18,20,16,20,19,14,18,13,18,12,15,17,18,17,17,17,18,17,19,15,17,13,16,14,16,18,16,17,12,16,16,14,16,18,13,19,19,18,19,18,
            20,18,20,16,19,15,15,16,19,16,19,17],

    cols=_rev(24, [4,4,5,8,5,7,5,8,5,8,7,9,7,8,2,9,8,4,6,5,7,5,3,5,7,1,3,5,2,6,8,3,5,7,2,6,6,7,6,7,6,2,6,4,1,7,
                    5,8,9,7,8,1,9,7,9,6,8,3,5,6,8,6,8,6,10,10,10,8,10,10,11,6,11,9,10,9,9,2,11,1,7,11,10,11,11,11,10,11,10,11,10,11,
                    10,11,10,3,11,1,2,4,2,4,1,4,3,3,2,4,1,2,3,3,9,4,4,2,9,1,9,1,9,3,4,2,5,1,3,1,20,19,21,20,22,21,21,18,22,14,
                    19,14,22,14,20,14,19,20,19,20,21,22,19,21,19,20,19,22,22,22,20,13,21,13,13,13,12,13,12,13,12,12,12,13,12,12,12,13,21,13,16,13,22,14,14,13,
                    14,12,13,12,17,12,15,12,20,18,21,15,17,22,17,15,17,15,16,14,16,14,18,14,15,20,16,21,16,22,19,17,16,17,17,17,16,17,15,19,17,21,18,20,18,16,
                    18,17,22,18,18,20,18,16,21,15,15,16,15,16,14,16,14,18,15,19,19,18,15,18,15,17,15,13,17,23,20,18,21,21,18,15,20,20,16,17,16,19,16,16,14,23,
                    19,22,16,19,17,15,13,16,13,19,22,15,15,13,15,13,21,14,22,23,18,14,23,14,14,22,14,19,17,21,13,13,22,15,21,16,14,14,13,15,18,20,20,17,14,22,
                    13,12,12,21,13,12,14,13,18,23,12,12,19,12,12,12,16,17,20,18,17,23,15,22,19,21,20,15,23,18,16,17,18,18,21,22,19,21,20,21,20,20,20,19,19,17,
                    16,19,12,17,15,16,18,18,14,22,17,22,12,22,12,21,18,17,18,12,18,17,22,12,22,14,19,18,16,16,19,15,12,22,21,21,19,20,20,20,22,20,18,19,17,21,
                    17,14,21,20,13,19,16,15,21,16,22,17,23,17,18,20,19,23,12,18,12,21,12,12,23,12,13,12,22,13,12,14,12,15,15,13,15,20,16,13,15,18,20,14,21,21,
                    15,17,14,22,13,14,23,14,22,14,14,23,14,18,13,22,13,17,19,15,21,13,17,13,15,17,16,14,15,23,19,16,22,19,16,22,17,20,16,18,19,16,20,20,15,16,
                    13,21,21,18,23,19,19,19,18,15,18,16,18,15,18,15,16,14,16,15,21,14,15,19,17,18,16,18,20,18,16,22,20,18,21,17,15,20,18,16,21,17,17,16,17,16,
                    17,21,17,19,22,16,18,15,14,16,15,22,14,16,14,17,15,20,18,17,15,17,15,17,13,13,13,15,13,13,12,17,12,14,13,14,14,21,12,22,16,12,13,12,12,12,
                    13,12,13,12,13,12,13,12,13,20,12,22,21,19,21,19,22,19,20,20,21,19,22,21,20,20,14,19,19,21,14,22,14,22,14,20,19,21,18,22,20,22,3,4,2,3,
                    1,2,2,5,1,9,4,9,1,9,3,9,4,3,4,3,2,1,4,2,4,3,4,1,1,1,3,10,2,10,10,10,11,10,11,10,11,11,11,10,11,11,11,10,2,10,
                    7,10,1,9,9,10,9,11,10,11,6,11,8,11,3,5,2,8,6,1,6,8,6,8,7,9,7,9,5,9,8,3,7,2,7,1,4,6,7,6,6,6,7,6,8,4,
                    6,2,5,3,5,7,5,6,1,5,5,3,5,7,2,8,8,7,8,7,9,7,9,5,8,4,4,5,8,5,8,6,8,10,6,0,3,5,2,2,5,8,3,3,7,6,
                    7,4,7,7,9,0,4,1,7,4,6,8,10,7,10,4,1,8,8,10,8,10,2,9,1,0,5,9,0,9,9,1,9,4,6,2,10,10,1,8,2,7,9,9,10,8,
                    5,3,3,6,9,1,10,11,11,2,10,11,9,10,5,0,11,11,4,11,11,11,7,6,3,5,6,0,8,1,4,2,3,8,0,5,7,6,5,5,2,1,4,2,3,2,
                    3,3,3,4,4,6,7,4,11,6,8,7,5,5,9,1,6,1,11,1,11,2,5,6,5,11,5,6,1,11,1,9,4,5,7,7,4,8,11,1,2,2,4,3,3,3,
                    1,3,5,4,6,2,6,9,2,3,10,4,7,8,2,7,1,6,0,6,5,3,4,0,11,5,11,2,11,11,0,11,10,11,1,10,11,9,11,8,8,10,8,3,7,10,
                    8,5,3,9,2,2,8,6,9,1,10,9,0,9,1,9,9,0,9,5,10,1,10,6,4,8,2,10,6,10,8,6,7,9,8,0,4,7,1,4,7,1,6,3,7,5,
                    4,7,3,3,8,7,10,2,2,5,0,4])
)

psv_128_rhd = dict(
    geometry=(8, 16),

    pitch=1.33,

    rows= [6,3,5,3,6,4,6,3,
            5,2,4,2,4,3,6,2,
            2,6,4,6,4,6,6,5,
            4,5,3,5,3,5,3,5,
            5,3,5,3,5,3,5,4,
            5,4,5,4,4,4,4,3,
            2,4,2,3,2,3,2,4,
            2,4,3,4,3,5,3,5,
            0,1,0,1,0,1,0,1,
            0,2,1,2,1,1,0,2,
            1,0,1,0,0,0,1,0,
            0,0,1,0,1,0,1,7,
            0,6,7,6,7,7,7,6,
            7,7,7,6,7,7,7,6,
            2,6,1,6,2,6,2,7,
            2,7,6,7,1,7,1,7],

    cols=_rev(16, [2,5,11,6,14,10,8,2,
                    13,13,2,1,14,14,2,3,
                    15,6,1,12,0,10,4,15,
                    12,7,7,9,9,2,4,3,
                    0,0,5,1,6,13,8,8,
                    12,9,10,5,11,6,3,10,
                    7,4,9,11,2,15,0,15,
                    11,7,3,13,12,4,8,14,
                    13,0,11,3,9,4,12,7,
                    10,12,10,4,1,15,7,14,
                    5,14,14,4,5,2,12,0,
                    15,1,8,3,6,8,2,12,
                    6,7,10,9,7,14,2,15,
                    0,5,1,13,3,4,6,3,
                    6,11,11,5,8,0,5,8,
                    10,15,1,13,9,11,13,9])
)

rat_v4_kb = dict(
    geometry=(8, 8),
    rows = [3,5,4,6,3,4,2,7,
            1,6,6,7,2,7,4,6,
            5,0,5,4,3,0,1,2,
            1,3,3,1,7,0,5,7,
            2,5,3,7,0,3,4,1,
            1,0,3,4,2,6,3,5,
            6,5,7,2,0,0,6,0,
            7,1,4,4,6,2,5,1],
    cols = _rev(8, [5,5,4,5,5,7,6,5,
                    5,6,4,0,7,6,6,7,
                    7,6,4,5,6,5,6,5,
                    7,4,7,4,4,4,6,7,
                    3,1,3,3,3,0,3,0,
                    2,1,2,1,1,3,1,0,
                    0,3,1,0,7,0,1,2,
                    2,1,0,2,2,2,2,3])
)

psv_32 = dict(
    geometry=(6, 6),
    cols=_rev(6, [5, 4, 5, 4, 1, 0, 1, 0, 3, 1, 4, 2, 2, 3, 2, 3, 5,
                  4, 5, 4, 1, 0, 1, 0, 3, 1, 4, 2, 2, 3, 2, 3]),
    rows=[4, 2, 2, 0, 3, 5, 1, 3, 1, 4, 5, 0, 4, 3, 2, 5, 5,
          3, 3, 1, 2, 4, 0, 2, 0, 5, 4, 1, 5, 2, 3, 4]
)

psv_61 = dict(
    geometry=(8, 8),

    pitch=0.406,

    rows=[GND, 7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 0, 3, 0, 3, 0, GND, 7, 6, 6,
          5, 5, 4, 4, 2, 3, 0, 3, 0, 3, 0, 3, GND, 7, 7, 6, 6, 5, 5, 4,
          4, 1, 2, 1, 2, 1, 2, 1, GND, 7, 6, 6, 5, 5, 4, 4, 1, 2, 1, 2,
          1, 2, 1, 2],

    cols=_rev(8, [GND, 3, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 2, 2, 3, 3, GND,
                  6, 5, 6, 5, 6, 5, 6, 7, 7, 6, 6, 5, 5, 4, 4, GND, 5,
                  4, 7, 4, 7, 4, 7, 4, 7, 6, 6, 5, 5, 4, 4, GND, 2, 1,
                  2, 1, 2, 1, 2, 0, 0, 1, 1, 2, 2, 3, 3])

)

psv_61_gray = undo_gray_code(psv_61)

psv_16_gerbil = dict(
    geometry=(3, 6),
    rows=[GND, GND, GND, 1, 0, GND, GND, GND, 1, 1, 0, 2, 2, 1, 0, GND, 0,
          2, 2, 1, 0, GND, GND, GND, GND, GND, GND, 0, GND, GND, 1, GND],
    cols=_rev(6, [GND, GND, GND, 0, 1, GND, GND, GND, 2, 3, 3, 3, 4, 4, 5,
                  GND, 2, 2, 1, 1, 0, GND, GND, GND, GND, GND, GND, 4, GND,
                  GND, 5, GND])
)

psv_61_stim1 = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=_shift_indices([3, 2, 3, 2, 3, 2, 3, GND, 1, 4, 1, 4, 1, 4,
                         1, GND, 4, 1, 4, 1, 4, 1, 4, GND, 2, 3, 2, 3,
                         2, 3, 2, GND]),
    cols=_rev(8, _shift_indices([4, 3, 3, 2, 2, 1, 1, GND, 4, 5, 5,
                                 6, 6, 7, 7, GND, 4, 3, 3, 2, 2, 1, 1,
                                 GND, 4, 5, 5, 6, 6, 7, 7, GND]))
)

psv_61_stim64 = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=_shift_indices([GND, 8, 7, 7, 6, 6, 5, 5, 3, 4, 1, 4, 1, 4, 1, 4, GND,
                         8, 8, 7, 7, 6, 6, 5, 5, 2, 3, 2, 3, 2, 3, 2, GND,
                         8, 7, 7, 6, 6, 5, 5, 2, 3, 2, 3, 2, 3, 2, 3, GND,
                         8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 1, 4, 1, 4, 1]),
    cols=_rev(8, _shift_indices([GND, 7, 6, 7, 6, 7, 6, 7, 8, 8, 7, 7, 6, 6,
                                 5, 5, GND, 6, 5, 8, 5, 8, 5, 8, 5, 8, 7, 7,
                                 6, 6, 5, 5, GND, 3, 2, 3, 2, 3, 2, 3, 1, 1,
                                 2, 2, 3, 3, 4, 4, GND, 4, 2, 4, 1, 4, 1, 4,
                                 1, 1, 2, 2, 3, 3, 4, 4]))
)

psv_61_stim64_15row = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=_shift_indices([8, 7, 7, 6, 6, 5, 5, 3, 4, 1, 4, 1, 4, 1, 4,
                         8, 8, 7, 7, 6, 6, 5, 5, 2, 3, 2, 3, 2, 3, 2,
                         8, 7, 7, 6, 6, 5, 5, 2, 3, 2, 3, 2, 3, 2, 3,
                         8, 8, 7, 7, 6, 6, 5, 5, 4, 4, 1, 4, 1, 4, 1]),
    cols=_rev(8, _shift_indices([7, 6, 7, 6, 7, 6, 7, 8, 8, 7, 7, 6, 6,
                                 5, 5, 6, 5, 8, 5, 8, 5, 8, 5, 8, 7, 7,
                                 6, 6, 5, 5, 3, 2, 3, 2, 3, 2, 3, 1, 1,
                                 2, 2, 3, 3, 4, 4, 4, 2, 4, 1, 4, 1, 4,
                                 1, 1, 2, 2, 3, 3, 4, 4]))
)

psv_61_ddc = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=[0, 3, 2, 1, 0, 0, 4, 5, 6, 7, 6, 7, 1, 4, 4, GND, 0,
          3, 1, 2, 3, 3, 4, 5, 6, 5, 6, 7, 2, 1, 5, 2, GND, 4,
          4, 1, 7, 6, 7, 6, 5, 4, 3, 3, 1, 2, 0, 3, GND, 5, 1,
          2, 7, 6, 5, 6, 5, 4, 2, 0, 2, 1, 0, 3],

    cols=_rev(8, [2, 2, 2, 2, 1, 0, 0, 0, 0, 1, 2, 2, 0, 2, 1, GND, 3,
                  3, 3, 1, 1, 0, 3, 3, 3, 1, 1, 3, 0, 1, 2, 3, GND, 7,
                  4, 7, 4, 4, 6, 6, 6, 6, 7, 6, 5, 4, 5, 5, GND, 4, 6,
                  6, 5, 7, 7, 5, 5, 5, 7, 6, 5, 4, 4, 4])
)

# same as above, but without disconnected channels
psv_61_15row = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=[7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 0, 3, 0, 3, 0, 7, 6, 6,
          5, 5, 4, 4, 2, 3, 0, 3, 0, 3, 0, 3, 7, 7, 6, 6, 5, 5, 4,
          4, 1, 2, 1, 2, 1, 2, 1, 7, 6, 6, 5, 5, 4, 4, 1, 2, 1, 2,
          1, 2, 1, 2],

    cols=_rev(8, [3, 1, 3, 0, 3, 0, 3, 0, 0, 1, 1, 2, 2, 3, 3,
                  6, 5, 6, 5, 6, 5, 6, 7, 7, 6, 6, 5, 5, 4, 4, 5,
                  4, 7, 4, 7, 4, 7, 4, 7, 6, 6, 5, 5, 4, 4, 2, 1,
                  2, 1, 2, 1, 2, 0, 0, 1, 1, 2, 2, 3, 3])

)

psv_61_15row_gray = undo_gray_code(psv_61_15row, starting=1)

# Note: Use "~RC" to indicate a grounded (non-data) channel
psv_61_intan_encoded = """D5, GND, A5, E7, D6, E6, A6, F7, D7, F6, A7, G7, D8, G6, C8, H7, E5, E8, B8, F5, C7, F8, B7, 
G5, C6, G8, B6, H5, C5, H6, B5, GND, C4, H4, B4, H3, C3, G2, B3, G3, C2, F2, B2, F3, C1, E2, B1, E3, D1, H2, A1, G4, 
D2, G1, A2, F4, D3, F1, A3, E4, D4, E1, A4, GND"""

psv_61_intan_rev_encoded = """GND, B5, H6, C5, H5, B6, G8, C6, G5, B7, F8, C7, F5, B8, E8, E5, D1, H2, A1, G4, D2, G1, 
A2, F4, D3, F1, A3, E4, D4, E1, A4, GND, D5, GND, A5, E7, D6, E6, A6, F7, D7, F6, A7, G7, D8, G6, C8, H7, E3, B1, E2, 
C1, F3, B2, F2, C2, G3, B3, G2, C3, H3, B4, H4, C4"""

psv_61_intan2_encoded = """C5, GND, A5, G5, E5, C8, B6, E8, D6, F6, C7, G6, A7, H6, E7, G7, F5, B8, D7, D8, B7, H5, 
E6, F7, A6, F8, C6, H7, D5, G8, B5, GND, C4, G1, A4, H2, E4, F1, B3, F2, D3, H4, C2, D1, A2, B1, E2, G4, F4, G2, D2, 
H3, B2, G3, E3, F3, A3, E1, C3, C1, D4, A1, B4, GND"""

psv_61_intan2_rev_encoded = """GND, B5, G8, D5, H7, C6, F8, A6, F7, E6, H5, B7, D8, D7, B8, F5, F4, G2, D2, H3, B2, 
G3, E3, F3, A3, E1, C3, C1, D4, A1, B4, GND, C5, GND, A5, G5, E5, C8, B6, E8, D6, F6, C7, G6, A7, H6, E7, G7, G4, E2, 
B1, A2, D1, C2, H4, D3, F2, B3, F1, E4, H2, A4, G1, C4"""

psv_61_afe_encoded = """D5, A5, D6, A6, D7, A7, D8, C8, E7, E6, F7, F6, G7, G6, H7, G8, H6, H5, G5, F8, F5, E8, E5, 
B8, C7, B7, C6, B6, C5, B5, GND, GND, GND, C4, B4, C3, B3, C2, B2, C1, B1, E3, E2, F3, F2, G3, H3, H4, G2, H2, G4, 
G1, F4, F1, E4, E1, D1, ~A1, D2, A2, D3, A3, D4, A4"""

psv_61_afe = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=unzip_encoded(psv_61_afe_encoded)[0],
    cols=_rev(8, unzip_encoded(psv_61_afe_encoded)[1])
)

psv_61_intan = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=unzip_encoded(psv_61_intan_encoded)[0],
    cols=_rev(8, unzip_encoded(psv_61_intan_encoded)[1])
)

psv_61_intan_rev = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=unzip_encoded(psv_61_intan_rev_encoded)[0],
    cols=_rev(8, unzip_encoded(psv_61_intan_rev_encoded)[1])
)

psv_61_intan2 = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=unzip_encoded(psv_61_intan2_encoded)[0],
    cols=_rev(8, unzip_encoded(psv_61_intan2_encoded)[1])
)

psv_61_intan2_rev = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=unzip_encoded(psv_61_intan2_rev_encoded)[0],
    cols=_rev(8, unzip_encoded(psv_61_intan2_rev_encoded)[1])
)

psv_61_omnetix = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=[6, 7, 5, 6, 5, 4, 4, -2, -2, 4, 4, 5, 6, 5, 7, 6, 2, 3, 0, 3,
          0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 7, 7, 6, 6, 5, 5, 4, -2,
          4, 4, 5, 5, 6, 6, 7, 7, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
          1, 2, 1, 4],

    cols=_rev(8, [0, 1, 3, 3, 0, 3, 0, -2, -2, 6, 5, 6, 5, 5, 6, 6, 7,
                  7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 4, 5, 7,
                  4, 7, 4, 7, -2, 2, 1, 2, 1, 2, 1, 3, 2, 0, 0, 1, 1, 2,
                  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 4])
)

# this specifically maps to the recording with this note:
#     "Connections: Bank A connected backwards (text not matched up),
#      on top omnetics connector (same side as ZIF connection).
#      Bank B connected correctly (text matched up)."
#      -- froemke session 2013-11-27
psv_61_omnetix_2014_11_27 = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=[2, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 0, 3, 6, 7, 5, 6, 5,
          4, 4, -2, -2, 4, 4, 5, 6, 5, 7, 6, 7, 7, 6, 6, 5, 5, 4, -2, 4,
          4, 5, 5, 6, 6, 7, 7, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2,
          1, 4],
    cols=_rev(8, [7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 0, 1,
                  3, 3, 0, 3, 0, -2, -2, 6, 5, 6, 5, 5, 6, 6, 4, 5, 7,
                  4, 7, 4, 7, -2, 2, 1, 2, 1, 2, 1, 3, 2, 0, 0, 1, 1, 2,
                  2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 4])
)

psv_61_wireless_sub = dict(
    geometry=(8, 8),
    pitch=0.406,
    rows=[6, 6, 3, 0, 0, 3, 3, 7],
    cols=[4, 7, 6, 6, 3, 2, 0, 1],
)

# This is the lookup from mux3 channel to ZIF pin..
# ZIF pin counts go in zig-zag zipper order, so approximate this
# by a (2,32) "array" shape
_mux3_to_zif = [GND, 1, 2, 4, 6, 8, 10, 12, 14, 16, 20, 22, 24, 26, 28, 30, GND, 60, 58, 56, 54, 52, 50, 48, 46, 44,
                42, 40, 38, 36, 34, 32, GND, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33, GND, 3, 5,
                7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31]
_mux3_rows = list()
_mux3_cols = list()
for i in _mux3_to_zif:
    if i in NonSignalChannels:
        _mux3_rows.append(i)
        _mux3_cols.append(i)
    else:
        _mux3_rows.append((i - 1) % 2)
        _mux3_cols.append((i - 1) // 2)

mux3_to_zif = dict(
    geometry=(2, 32),
    rows=_mux3_rows,
    cols=_mux3_cols
)

ratv4_mux6_encoded = """G1, G2, H3, G3, F3, E1, C1, A1, F4, B2, E3, A3, C3, D4, B4, G7, H6, G6, F6, E8, C8, G5, E7, 
A7, C7, D6, B6, E5, A5, C5, G8, H7, F8, F7, H5, D8, B8, F5, D7, B7, E6, A6, C6, D5, B5, H2, F1, F2, H4, D1, B1, G4, 
E2, A2, C2, D3, B3, E4, A4, C4"""

ratv4_mux6 = dict(
    pitch=0.406,
    geometry=(8, 8),
    rows=unzip_encoded(ratv4_mux6_encoded)[0],
    cols=_rev(8, unzip_encoded(ratv4_mux6_encoded)[1])
)

aro_puzzle = dict(
    geometry=(16, 16),
    rows=_shift_indices(
        [9, 8, 9, 8, 16, 8, 9, 8, 15, 16, 15, 15, 15, 16, 15, 16,
         14, 16, 16, 16, 10, 12, 15, 16, 14, 7, 13, 11, 14, 8, 14, 7,
         14, 7, 15, 8, 7, 7, 1, 1, 3, 5, 2, 4, 4, 6, 5, 2, 2, 4, 6, 3,
         4, 2, 6, 1, 1, 5, 5, 3, 9, 8, 9, 8, 12, 11, 7, 11, 12, 12, 6,
         11, 12, 13, 13, 12, 12, 13, GND, 13, 15, GND, 13, 13, 14, 14,
         14, 13, 9, 11, 10, 10, 10, 10, 9, 10, 10, 10, 9, 11, 1, 5,
         5, 6, 3, 6, 4, 2, 3, 6, 4, 1, 3, 3, 4, 4, 5, 2, 5, 1, 2, 2,
         3, 6, 7, 11, 12, 11, 14, 14, 14, 14, 10, 11, 14, 13, 10, 13,
         10, 10, 9, 11, 10, 10, 10, 11, 12, 11, 11, 11, 12, 11, 16, 13,
         12, 12, 14, 12, 7, 13, 12, 13, 13, 12, 2, 1, 15, 6, 4, 3, 5,
         2, 4, 5, 3, 3, 4, 1, 3, 6, 2, 4, 6, 6, 5, 2, 1, 5, 14, 13, 14,
         13, 9, 10, 9, 16, 7, 7, 9, 7, 7, 8, 7, 7, 6, 7, 7, 8, 9, 8, 9,
         8, 9, 9, 15, 8, 15, 8, 15, 16, 15, 16, 15, 16, 15, 16, 15, 16,
         5, 1, 1, 5, 3, 6, 2, 4, 3, 6, 2, 4, 5, 2, 4, 3, 4, 2, 6, 3,
         11, 1, 1, 5, 8, 16, 8, 12]),

    cols=_shift_indices(
        [8, 2, 2, 5, 16, 7, 4, 3, 13, 12, 11, 10, 15, 14, 12,
         15, 9, 13, 9, 11, 9, 9, 9, 10, 14, 7, 9, 9, 16, 8, 12,
         5, 10, 1, 16, 1, 6, 8, 15, 14, 15, 15, 15, 15, 16, 15,
         16, 14, 12, 14, 13, 14, 13, 13, 14, 13, 12, 14, 13,
         13, 3, 6, 1, 4, 1, 3, 3, 4, 11, 2, 1, 6, 13, 15, 16,
         10, 12, 13, GND, 14, 14, GND, 12, 10, 13, 15, 11, 11, 6,
         2, 2, 4, 3, 6, 5, 5, 1, 7, 7, 1, 11, 11, 12, 10, 11,
         12, 11, 11, 12, 11, 12, 10, 16, 10, 9, 10, 9, 9, 10,
         9, 16, 10, 9, 9, 4, 5, 3, 7, 2, 3, 5, 1, 14, 16, 7,
         5, 16, 7, 12, 11, 10, 14, 13, 10, 15, 15, 15, 10, 11,
         13, 14, 12, 8, 1, 16, 5, 8, 7, 2, 3, 4, 2, 8, 6, 1, 8,
         8, 8, 7, 8, 8, 8, 8, 7, 1, 7, 5, 7, 6, 6, 6, 6, 5, 7,
         5, 7, 6, 6, 4, 4, 6, 6, 15, 8, 11, 6, 13, 11, 13, 9,
         15, 16, 12, 16, 16, 10, 14, 9, 14, 14, 12, 15, 16, 9,
         5, 13, 1, 11, 7, 2, 3, 4, 6, 1, 2, 5, 4, 3, 4, 4, 5,
         3, 4, 4, 4, 4, 5, 3, 5, 3, 1, 3, 1, 3, 2, 2, 2, 2, 8,
         3, 2, 2, 12, 7, 10, 8]),
)


# For June 5 2017 tests: single intan 64-channel recorded arm1 or arm2
# from each of three puzzle pieces
def _aro_intan_to_zif(zif_to_electrode):
    # Needs to return a 64-long list in acquisition order of rows/cols.
    # Given: the zif-to-site mapping encoded list of [A-Z][0-9] sites
    # this is the ZIF-51 list in order of 64 acquired channels
    intan_zif51 = [26, GND, 24, 10, 22, 8, 20, 6, 18, 4, 16, 2, 14,
                   GND, 12, GND, 11, 9, 13, 7, 15, 5, 17, 3, 19, 1,
                   21, GND, 23, GND, 25, GND, GND, GND, 27, GND, 29, 51,
                   31, 49, 33, 47, 35, 45, 37, 43, 39, 41, 44, GND,
                   42, GND, 40, GND, 38, GND, 36, 50, 34, 48, 32, 46, 30, 28]
    zrow, zcol = unzip_encoded(zif_to_electrode)

    irow = []
    icol = []
    for i, z in enumerate(intan_zif51):
        if z is GND:
            irow.append(GND)
            icol.append(GND)
        else:
            irow.append(zrow[z - 1])
            icol.append(zcol[z - 1])
    return irow, icol


_aro_left1 = """REF, U9, T9, R9, P9, N9, M9, L9, K9, U8, U7, U6, U5, U4, U3, T8, T7, T6, T5, T4, T3, T2, R8, R7, R6, 
R5, R4, R3, R2, R1, P8, P7, P6, P5, P4, P3, P2, P1, N8, N7, N6, N5, N4, N3, N2, N1, M8, M7, M6, M5, ~Float"""
_aro_left2 = """~Float, M4, M3, M2, M1, L8, L7, L6, L5, L4, L3, L2, L1, K8, K7, K6, K5, K4, K3, K2, K1, J8, J7, J6, 
J5, J4, J3, J2, J1, J9, H8, H7, H6, H5, H4, H3, H1, H2, H9, G9, G8, G7, G6, G1, G2, G5, G4, G3, F2, F1, ~Float"""
_aro_center1 = """~Float, C1, D1, E1, B2, C2, D2, E2, A3, B3, C3, D3, E3, F3, A4, B4, C4, D4, E4, F4, A5, B5, C5, D5, 
E5, F5, A6, B6, C6, D6, E6, F6, A7, B7, C7, D7, E7, F7, F8, A8, B8, C8, D8, E8, F9, E9, A9, B9, C9, D9, ~Float"""
_aro_center2 = """~Float, D10, C10, B10, A10, E10, F10, E11, D11, C11, B11, A11, F11, F12, E12, D12, C12, B12, A12, 
F13, E13, D13, C13, B13, A13, F14, E14, D14, C14, B14, A14, F15, E15, D15, C15, B15, A15, F16, E16, D16, C16, B16, 
A16, E17, D17, C17, B17, E18, D18, C18, ~Float"""
_aro_right1 = """~Float, F18, F17, G16, G15, G14, G17, G18, G13, G12, G11, G10, H10, H17, H18, H16, H15, H14, H13, 
H12, H11, J10, J18, J17, J16, J15, J14, J13, J12, J11, K18, K17, K16, K15, K14, K13, K12, K11, L18, L17, L16, L15, 
L14, L13, L12, L11, M18, M17, M16, M15, ~Float"""
_aro_right2 = """~Float, M14, M13, M12, M11, N18, N17, N16, N15, N14, N13, N12, N11, P18, P17, P16, P15, P14, P13, 
P12, P11, R18, R17, R16, R15, R14, R13, R12, R11, T17, T16, T15, T14, T13, T12, T11, U16, U15, U14, U13, U12, U11, 
K10, L10, M10, N10, P10, R10, T10, U10, REF"""

aro_puzzle_pieces = {
    'geometry': (17, 18),
    'rowsleft1': _aro_intan_to_zif(_aro_left1)[0],
    # 'colsleft1' : _rev(16, _aro_intan_to_zif(_aro_left1)[1]),
    'colsleft1': _aro_intan_to_zif(_aro_left1)[1],
    'rowsleft2': _aro_intan_to_zif(_aro_left2)[0],
    # 'colsleft2' : _rev(16, _aro_intan_to_zif(_aro_left2)[1]),
    'colsleft2': _aro_intan_to_zif(_aro_left2)[1],
    'rowscenter1': _aro_intan_to_zif(_aro_center1)[0],
    # 'colscenter1' : _rev(16, _aro_intan_to_zif(_aro_center1)[1]),
    'colscenter1': _aro_intan_to_zif(_aro_center1)[1],
    'rowscenter2': _aro_intan_to_zif(_aro_center2)[0],
    # 'colscenter2' : _rev(16, _aro_intan_to_zif(_aro_center2)[1]),
    'colscenter2': _aro_intan_to_zif(_aro_center2)[1],
    'rowsright1': _aro_intan_to_zif(_aro_right1)[0],
    # 'colsright1' : _rev(16, _aro_intan_to_zif(_aro_right1)[1]),
    'colsright1': _aro_intan_to_zif(_aro_right1)[1],
    'rowsright2': _aro_intan_to_zif(_aro_right2)[0],
    # 'colsright2' : _rev(16, _aro_intan_to_zif(_aro_right2)[1]),
    'colsright2': _aro_intan_to_zif(_aro_right2)[1],
}

##### A couple of dummy maps to debug analog multiplexing headstages
mux_15_passthru = {
    'geometry': (15, 4),
    'rows': [i % 15 for i in range(60)],
    'cols': [i / 15 for i in range(60)],
}
mux_16_passthru = {
    'geometry': (16, 4),
    'rows': [(i % 16 - 1 if i not in (0, 16, 32, 48) else GND) for i in range(64)],
    'cols': [(i / 16 if i not in (0, 16, 32, 48) else GND) for i in range(64)],
}

rat_v3_by_zif = """H4, H2, H3, G4, G2, G1, G3, F4, F2, F1, F3, E4, E2, E1, E3, D1, B1, A1, C1, D2, B2, A2, C2, D3, 
B3, A3, C3, D4, B4, A4, C4, D5, B5, A5, C5, D6, B6, A6, C6, D7, B7, A7, C7, D8, B8, C8, E5, E7, E8, E6, F5, F7, F8, 
F6, G5, G7, G8, G6, H5, H7, H6"""
rat_v3_by_zif_rc = zip(*unzip_encoded(rat_v3_by_zif))
rat_v3_by_zif_lut = dict(zip(range(1, 62), rat_v3_by_zif_rc))

electrode_maps = dict(
    # new passive map construction
    ratv5_intan=connect_passive_map((8, 8), rat_v5_by_zif_lut,
                                    zif_by_intan64, pitch=0.420),
    ratv4_stim4=connect_passive_map((8, 8), rat_v4_by_zif_lut,
                                    zif_by_stim4, pitch=0.420),
    ratv4_mux6=connect_passive_map((8, 8), rat_v4_by_zif_lut,
                                   zif_by_mux6, pitch=0.406),
    ratv4_mux6_15row=connect_passive_map((8, 8), rat_v4_by_zif_lut,
                                         zif_by_mux6_15row, pitch=0.420),
    ratv4_intan=connect_passive_map((8, 8), rat_v4_by_zif_lut,
                                    zif_by_intan64, pitch=0.420),

    ratv3_stim4=connect_passive_map((8, 8), rat_v3_by_zif_lut,
                                    zif_by_stim4, pitch=0.420),
    ratv3_mux6=connect_passive_map((8, 8), rat_v3_by_zif_lut,
                                   zif_by_mux6, pitch=0.420),
    ratv3_mux6_15row=connect_passive_map((8, 8), rat_v3_by_zif_lut,
                                         zif_by_mux6_15row, pitch=0.420),
    ratv3_intan=connect_passive_map((8, 8), rat_v3_by_zif_lut,
                                    zif_by_intan64, pitch=0.420),
    rat_varspace_intan=connect_passive_map(
        'auto', rat_varspace_by_zif_lut, zif_by_intan64
    ),
    rat_refelectrode_intan=connect_passive_map((8, 8), rat_refelectrode_by_zif_lut,
                                               zif_by_intan64, pitch=0.406),

    # dummy maps
    mux_15_passthru=mux_15_passthru,
    mux_16_passthru=mux_16_passthru,

    # old names
    psv_61_intan=connect_passive_map((8, 8), rat_v3_by_zif_lut,
                                     zif_by_intan64, pitch=0.420),
    psv_61_intan2=connect_passive_map((8, 8), rat_v4_by_zif_lut,
                                      zif_by_intan64, pitch=0.420),
    # direct maps
    psv_256_rhd=psv_256_rhd,
    psv_1024_rhd=psv_1024_rhd,
    psv_128_rhd=psv_128_rhd,
    rat_v4_kb=rat_v4_kb,
    psv_244_rhd=psv_244_rhd,
    psv_244_mux1=psv_244_mux1,
    psv_32=psv_32,
    psv_61=psv_61,
    psv_61_afe=psv_61_afe,
    psv_61_omnetix=psv_61_omnetix,
    psv_61_omnetix_2014_11_27=psv_61_omnetix_2014_11_27,
    psv_244_mux3=psv_244_mux3,
    mux3_to_zif=mux3_to_zif,
    psv_61_wireless_sub=psv_61_wireless_sub,
    psv_61_15row=psv_61_15row,
    psv_16_gerbil=psv_16_gerbil,
    psv_61_stim1=psv_61_stim1,
    psv_61_stim64=psv_61_stim64,
    psv_61_ddc=psv_61_ddc,
    psv_61_15row_gray=psv_61_15row_gray,
    psv_61_gray=psv_61_gray,
    psv_61_stim64_15row=psv_61_stim64_15row,
    ## psv_61_intan=psv_61_intan,
    psv_61_intan_rev=psv_61_intan_rev,
    ## psv_61_intan2=psv_61_intan2,
    psv_61_intan2_rev=psv_61_intan2_rev,
    psv_244_intan=psv_244_intan,
    aro_puzzle=aro_puzzle,
    aro_puzzle_pieces=aro_puzzle_pieces,
)


def get_electrode_map(name, connectors=()):
    try:
        pinouts = electrode_maps[name]
    except KeyError:
        raise ValueError('electrode name not found: ' + name)

    if connectors:
        if not isinstance(connectors, (list, tuple)):
            connectors = [connectors]
        if isinstance(connectors[0], float):
            connectors = [str(c) for c in connectors]
        row_spec = ['rows' + cnx for cnx in connectors]
        col_spec = ['cols' + cnx for cnx in connectors]
    else:
        # you're getting the connectors in alphanumeric order
        keys = pinouts.keys()
        connectors = set([k[4:] for k in keys
                          if k not in ('geometry', 'pitch')])
        connectors = sorted(connectors)
        row_spec = ['rows' + con for con in connectors]
        col_spec = ['cols' + con for con in connectors]
        # row_spec = ('rows',)
        # col_spec = ('cols',)

    rows = list();
    cols = list()
    for rkey, ckey in zip(row_spec, col_spec):
        rows.extend(pinouts[rkey])
        cols.extend(pinouts[ckey])

    sig_rows = []
    sig_cols = []
    no_connection = []
    reference = []
    for n in range(len(rows)):
        if rows[n] is REF:
            reference.append(n)
        elif rows[n] in NonSignalChannels:
            no_connection.append(n)
        else:
            sig_rows.append(rows[n])
            sig_cols.append(cols[n])

    geometry = pinouts['geometry']
    pitch = pinouts.get('pitch', 1.0)
    sig_rows = np.array(sig_rows)
    sig_cols = np.array(sig_cols)
    if (sig_rows.astype('i') == sig_rows).all():
        # appears to be a regular grid map
        sig_rows = sig_rows.astype('i')
        sig_cols = sig_cols.astype('i')
        flat_idx = mat_to_flat(geometry, sig_rows, sig_cols, col_major=False)
        chan_map = ChannelMap(flat_idx, geometry, col_major=False, pitch=pitch)
    else:
        # use literal coordinates
        chan_map = CoordinateChannelMap(zip(sig_rows, sig_cols), geometry=geometry,
                                        pitch=pitch)
    return chan_map, no_connection, reference

