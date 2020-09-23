import numpy as np
import itertools
from matplotlib.colors import BoundaryNorm, Normalize
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
from matplotlib.tri import Triangulation, LinearTriInterpolator, CubicTriInterpolator

from .util import Bunch, flat_to_mat, flat_to_flat, mat_to_flat


class ChannelMapError(Exception):
    pass


def map_intersection(maps):
    """Return a binary grid of intersecting electrode sites between all maps with the same geometry."""
    geometries = set([m.geometry for m in maps])
    if len(geometries) > 1:
        raise ChannelMapError('cannot intersect maps with different geometries')
    bin_map = maps[0].embed(np.ones(len(maps[0])), fill=0)
    for m in maps[1:]:
        bin_map *= m.embed(np.ones(len(m)), fill=0)
    return bin_map.astype('?')


class ChannelMap(list):
    """A map of sample vector(s) to a matrix representing 2D sampling space."""

    def __init__(self, chan_map, geometry, col_major=False, pitch=1.0, pads_up=False):
        """
        A ChannelMap is a vector of grid locations in flat indexing that correspond to the electrode location of data
        channels in counting order. E.g. a channel map of [2, 0, 1] for 3 channels [a, b, c] with geometry (2,
        2) maps to

        [[b, a],
        [[c, -]] <-- (col-major)

        [[b, c],
        [[a, -]] <-- (row-major)

        Parameters
        ----------
        chan_map: sequence
            Vector of flat indices into a grid with given geometry
        geo: 2-tuple
            Rows, columns of the array
        col_major: bool
            Flat indexing is for column-major addressing. Default is False for row-major.
        pitch: float or tuple
            The electrode pitch, either as a single distance or as (dy, dx)
        pads_up: bool
            If the map positions the electrode pads "face up", set this to True.
            Typically face down is used to match anatomical coordinates.

        """
        list.__init__(self)
        self[:] = chan_map
        self.col_major = col_major
        self.geometry = geometry
        self.pitch = pitch
        self._combs = None
        if np.iterable(pitch):
            dy, dx = pitch
        else:
            dy = dx = pitch
        r, c = geometry
        self.boundary = (-dx / 2.0, (c - 0.5) * dx, -dy / 2.0, (r - 0.5) * dy)
        # use a hash table for lookups
        rows, cols = flat_to_mat(geometry, np.array(chan_map), col_major=col_major)
        # self._chan2site = dict([(x, (r, c)) for x, r, c in zip(chan_map, rows, cols)])
        self._chan2site = dict(enumerate(zip(rows, cols)))
        self._site2chan = dict([(v, k) for k, v in self._chan2site.items()])
        self._pads_up = pads_up

    @property
    def pads_up(self):
        return self._pads_up

    @classmethod
    def from_index(cls, ij, shape, col_major=True, pitch=1.0, pads_up=False):
        """Return a ChannelMap from a list of matrix index pairs (e.g. [(0, 3), (2, 1), ...])
        and a matrix shape (e.g. (5, 5)).
        """

        i, j = zip(*ij)
        map = mat_to_flat(shape, i, j, col_major=col_major)
        return cls(map, shape, col_major=col_major, pitch=pitch, pads_up=pads_up)

    @classmethod
    def from_mask(cls, mask, col_major=True, pitch=1.0, pads_up=pads_up):
        """Create a ChannelMap from a binary grid. Note: the data channels must be aligned
        with the column-major or row-major raster order of this binary mask
        """

        i, j = mask.nonzero()
        ij = zip(i, j)
        geo = mask.shape
        return cls.from_index(ij, geo, col_major=col_major, pitch=pitch, pads_up=pads_up)

    @property
    def site_combinations(self):
        if self._combs is None and len(self) > 1:
            self._combs = channel_combinations(self, scale=self.pitch)
        return self._combs

    def flip_face(self):
        rows, cols = self.to_mat()
        num_cols = self.geometry[1]
        cols = num_cols - 1 - cols
        pads_up = not self.pads_up
        return ChannelMap.from_index(list(zip(rows, cols)), self.geometry,
                                     col_major=self.col_major, pitch=self.pitch, pads_up=pads_up)

    def to_pads_up(self):
        if not self.pads_up:
            return self.flip_face()
        else:
            return self

    def to_pads_down(self):
        if self.pads_up:
            return self.flip_face()
        else:
            return self

    def as_row_major(self):
        if self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:]),
                self.geometry, col_major=False, pitch=self.pitch
            )
        return self

    def as_col_major(self):
        if not self.col_major:
            return ChannelMap(
                flat_to_flat(self.geometry, self[:], col_major=False),
                self.geometry, col_major=True, pitch=self.pitch
            )
        return self

    def to_mat(self):
        return flat_to_mat(self.geometry, self, col_major=self.col_major)

    def lookup(self, i, j):
        if not np.iterable(i):
            i, j = map(int, (i, j))
            try:
                return self._site2chan[(i, j)]
            except KeyError:
                raise ChannelMapError('Site {}, {} is not mapped'.format(i, j))
        else:
            try:
                return [self._site2chan[(row, col)] for row, col in zip(i, j)]
            except KeyError:
                raise ChannelMapError('One of the sites is not mapped')

    def rlookup(self, c):
        if not np.iterable(c):
            try:
                return self._chan2site[c]
            except KeyError:
                raise ChannelMapError('Channel {} is not mapped'.format(c))
        try:
            return [self._chan2site[chan] for chan in c]
        except KeyError:
            raise ChannelMapError('One of the channels is not mapped')

    def subset(self, sub, as_mask=False, map_intersect=False):
        """
        Behavior depends on the type of "sub":

        Most commonly, sub is a sequence (list, tuple, array) of subset
        indices.

        ChannelMap: return the subset map for the intersecting sites

        ndarray: if NOT subset indices (i.e. a binary mask), then the
        mask is converted to indices. If the array is a 2D binary mask,
        then site-lookup is used.

        """
        if isinstance(sub, type(self)):
            # check that it's a submap
            submap = map_intersection([self, sub])
            if submap.sum() < len(sub):
                raise ChannelMapError('The given channel map is not a subset of this map')
            # get the channels/indices of the subset of sites
            sub = self.lookup(*submap.nonzero())
        elif isinstance(sub, np.ndarray):
            if sub.ndim == 2:
                # Get the channels/indices of the subset of sites.
                # The channel lookups need to be sorted to get the subset of
                # channels in sequence
                if map_intersect:
                    # AND this map with the input binary map to cover missing sites
                    this_mask = self.embed(np.ones(len(self), dtype='?'), fill=False)
                    sub *= this_mask
                # If this looks up missing sites, then raise
                sites = self.lookup(*sub.nonzero())
                sub = np.sort(sites)
            elif sub.ndim == 1:
                if sub.dtype.kind in ('b',):
                    sub = sub.nonzero()[0]
            else:
                raise ValueError('Cannot interpret subset array')
        elif not isinstance(sub, (list, tuple)):
            raise ValueError('Unknown subset type')

        if as_mask:
            mask = np.zeros((len(self),), dtype='?')
            mask[sub] = True
            return mask

        cls = type(self)
        return cls(
            [self[i] for i in sub], self.geometry,
            col_major=self.col_major, pitch=self.pitch
        )

    def __getitem__(self, index):
        if not isinstance(index, slice):
            return super(ChannelMap, self).__getitem__(index)
        cls = type(self)
        new_map = cls(
            super(ChannelMap, self).__getitem__(index),
            self.geometry, col_major=self.col_major, pitch=self.pitch
        )
        # Keep the pre-computed combinations IFF the entire map is copied
        i = index.start if index.start else 0
        j = index.stop if index.stop else len(self)
        if i == 0 and j == len(self) and self._combs is not None:
            new_map._combs = self._combs.copy()
        return new_map

    def embed(self, data, axis=0, fill=np.nan):
        """
        Embed the data in electrode array geometry, mapping channels on the given axis.
        For example, if data is (channels, time), then embed(data, axis=0) yields the
        shape (rows, cols, time). Similarly (time, channels) with axis=1 -> (time, rows, cols)

        Parameters
        ----------
        data: ndarray
            Electrode signal in channel order
        axis: int
            Axis with channel signal in data
        fill: scalar
            Fill the non-mapped channels with this value. *Caution* the default value of NaN
            will cause the output array to be floating point, potentially up-casting the input
            array. To retain an integer type output, you must use an integer fill value that
            does not overflow.

        Returns
        -------
        grid_data: ndarray
            Embedded array matrix

        """
        data = np.atleast_1d(data)
        shape = list(data.shape)
        if shape[axis] != len(self):
            raise ValueError('Data array does not have the correct number of channels')
        shape.pop(axis)
        shape.insert(axis, self.geometry[0] * self.geometry[1])
        array = np.empty(shape, dtype=np.result_type(data, fill))
        if not isinstance(fill, str):
            array.fill(fill)
        slicing = [slice(None)] * len(shape)
        slicing[axis] = self.as_row_major()[:]
        array[tuple(slicing)] = data
        shape.pop(axis)
        shape.insert(axis, self.geometry[1])
        shape.insert(axis, self.geometry[0])
        array.shape = shape
        if isinstance(fill, str):
            return self.interpolated(array, axis=axis)
        return array

    def as_channels(self, matrix, axis=0):
        """
        Take the elements of a matrix into the "natural" channel ordering.
        """
        m_shape = matrix.shape
        m_flat = m_shape[axis] * m_shape[axis + 1]
        c_dims = m_shape[:axis] + (m_flat,) + m_shape[axis + 2:]
        matrix = matrix.reshape(c_dims)
        return np.take(matrix, self, axis=axis)

    def inpainted(self, image, axis=0, **kwargs):
        pass

    def interpolated(self, image, axis=0, method='median'):
        # acts in-place
        mask = self.embed(np.zeros(len(self), dtype='?'), fill=1)
        missing = np.where(mask)
        g = self.geometry

        def _slice(i, j, w):
            before = [slice(None)] * axis
            after = [slice(None)] * (image.ndim - axis - 2)
            if w:
                isl = slice(max(0, i - w), min(g[0], i + w + 1))
                jsl = slice(max(0, j - w), min(g[1], j + w + 1))
            else:
                isl = i
                jsl = j
            before.extend([isl, jsl])
            before.extend(after)
            return tuple(before)

        # first pass, tag all missing sites with nan
        for i, j in zip(*missing):
            image[_slice(i, j, 0)] = np.nan
        for i, j in zip(*missing):
            # do a +/- 2 neighborhoods (8 neighbors)
            patch = image[_slice(i, j, 1)].copy()
            s = list(patch.shape)
            s = s[:axis] + [s[axis] * s[axis + 1]] + s[axis + 2:]
            patch.shape = s
            fill = np.nanmedian(patch, axis=axis)
            image[_slice(i, j, 0)] = fill
        return image

    def image(self, arr=None, cbar=True, nan='//', fill=np.nan, ax=None, show_channels=False, **kwargs):
        """
        Plot an array-space image of a data-space vector, or a binary map of channels.

        Parameters
        ----------
        arr: ndarray
            If given, embed and image this vector. If arr.shape == self.geometry then plot
            the array as is. If arr == None, then plot a binary grid of mapped / not-mapped channels
        cbar: bool
            By default, plot a colorbar (cbar == True)
        nan: str
            Unmapped grid locations are coded as NaN and plotted as white. Use a "hatch" style to
            visually indicate these sites (e.g. '//' for diagonal hatching, '+' for cross-hatching)
        fill: float
            Fill value for unmapped sites (NaN is conventional to mask plotting)
        ax: Axes
            Plot into an existing figure using this Axes object
        show_channels: bool
            Write channel number text into grid locations
        kwargs: dict
            Extra arguments for the Axes.imshow method (e.g. cmap, clim, ...)

        Returns
        -------
        f: Figure
            figure
        cb: Colorbar
            colorbar object (if cbar == True)

        """
        kwargs.setdefault('origin', 'upper')
        if ax is None:
            import matplotlib.pyplot as pp
            f = pp.figure()
            ax = f.add_subplot(111)
        else:
            f = ax.figure

        if arr is None:
            # image self -- reset these defaults
            arr = self.embed(np.ones(len(self), 'd'), fill=fill)
            kwargs.setdefault('clim', (0, 1))
            kwargs.setdefault('norm', BoundaryNorm([0, 0.5, 1], 256))
            kwargs.setdefault('cmap', cm.binary)
            show_channels = True
            self_map = True
        else:
            self_map = False

        if arr.shape != self.geometry:
            arr = self.embed(arr, fill=fill)

        nans = list(zip(*np.isnan(arr).nonzero()))
        im = ax.imshow(arr, **kwargs)

        ext = kwargs.pop('extent', ax.get_xlim() + ax.get_ylim())
        dx = abs(float(ext[1] - ext[0])) / arr.shape[1]
        dy = abs(float(ext[3] - ext[2])) / arr.shape[0]
        x0 = min(ext[:2])
        y0 = min(ext[2:])

        def s(x):
            return (x[0] * dy + y0, x[1] * dx + x0)
        if len(nan):
            for x in nans:
                r = Rectangle(s(x)[::-1], dx, dy, hatch=nan, fill=False)
                ax.add_patch(r)
        # ax.set_ylim(ext[2:][::-1])
        if cbar:
            cb = f.colorbar(im, ax=ax, use_gridspec=True)
            cb.solids.set_edgecolor('face')
            if self_map:
                cb.set_ticks([0.25, 0.75])
                cb.ax.set_yticklabels(['not mapped', 'mapped'], rotation=90, va='center')
        # need to make channel text after colorbar in case pixel size changes
        f.tight_layout()
        if show_channels:
            max_tx_wd = int(np.log2(len(self))) + 1
            pix_size_pts = ax.transData.get_matrix()[0, 0]
            # fontsize will accomodate either 75% height or 75% width (assume 60% W/H aspect ratio)
            max_width = 0.75 * pix_size_pts / max_tx_wd / 0.6
            max_height = 0.75 * pix_size_pts
            fontsize = min(max_width, max_height)
            im_rgb = im.get_cmap()(im.norm(im.get_array()))[..., :3]
            # "linearize" rgb values
            im_rgb_lin = im_rgb ** 2.2
            im_luminance = np.sum(im_rgb_lin * np.array([0.2126, 0.7152, 0.0722]), axis=2)
            rgb_max = im_rgb_lin.max(axis=2)
            rgb_min = im_rgb_lin.min(axis=2)
            im_saturation = rgb_max - rgb_min
            rgb_max[rgb_max == 0] = 1
            im_saturation /= rgb_max
            use_blk = (im_saturation < 0.3) & (im_luminance >= 0.5)
            use_wht = (im_saturation < 0.3) & (im_luminance < 0.5)
            # use_rgb = im_saturation >= 0.3
            # set tx_colors to the default rotated RGB value
            tx_colors = im_rgb + 0.5
            tx_colors[tx_colors > 1] = tx_colors[tx_colors > 1] - 1
            # set black and white on low saturation colors
            tx_colors[use_blk] = 0
            tx_colors[use_wht] = 1
            # tx_colors = 1 - im_colors
            for n, (i, j) in enumerate(zip(*self.to_mat())):
                ax.text(j, i, str(n), horizontalalignment='center', verticalalignment='center',
                        fontsize=fontsize, color=tx_colors[i, j])
        if cbar:
            return f, cb
        return f


class CoordinateChannelMap(ChannelMap):
    "A map of sample vector(s) to a coordinate space."

    def __init__(self, coordinates, geometry='auto', pitch=1.0, col_major=False):
        """
        Parameters
        ----------
        coordinates : sequence
            sequence of (y, x) values
        geometry : pair (optional)
            Geometry is determined from coordinate range if set to 'auto'.

        """
        list.__init__(self)
        self[:] = coordinates
        yy, xx = zip(*self)
        self.boundary = (min(xx), max(xx), min(yy), max(yy))
        self.pitch = pitch
        if len(self) > 1:
            self._combs = channel_combinations(self, scale=pitch)
            self.min_pitch = self._combs.dist.min()
        else:
            self.min_pitch = pitch
        if isinstance(geometry, str) and geometry.lower() == 'auto':
            y_gap = max(yy) - min(yy)
            x_gap = max(xx) - min(xx)
            self.geometry = int(np.round(y_gap / self.min_pitch)), int(np.round(x_gap / self.min_pitch))
        else:
            self.geometry = geometry
        # this is nonsense, but to satisfy parent class
        self.col_major = col_major

    def to_mat(self):
        return list(map(np.array, list(zip(*self))))

    def lookup(self, y, x):
        coords = np.array(self)
        sites = np.c_[y, x]
        chans = []
        for s in sites:
            dist = np.apply_along_axis(
                np.linalg.norm, 1, coords - s
            )
            chans.append(np.argmin(dist))
        return np.array(chans).squeeze()

    def rlookup(self, c):
        return self[c]

    def subset(self, sub, as_mask=False):
        """
        Works mainly as ChannelMap.subset, with these exceptions:

        * sub may not be another ChannelMask type
        * sub may not be a 2D binary mask

        """

        if isinstance(sub, np.ndarray):
            if sub.ndim == 2:
                raise ValueError('No binary maps allowed')
        elif isinstance(sub, ChannelMap) or not isinstance(sub, (tuple, list)):
            raise ValueError("Can't interpret subset type: {0}".format(type(sub)))
        return super(CoordinateChannelMap, self).subset(sub, as_mask=as_mask)

    def image(
            self, arr=None, cbar=True, ax=None, interpolate=None,
            grid_pts=None, norm=None, clim=None, cmap='viridis',
            scatter_kw={}, contour_kw={}, **passthru
    ):
        y, x = self.to_mat()
        if ax is None:
            import matplotlib.pyplot as pp
            f = pp.figure()
            ax = f.add_subplot(111)
        else:
            f = ax.figure
        if arr is None:
            arr = np.ones_like(y)
            clim = (0, 1)
            norm = BoundaryNorm([0, 0.5, 1.0], 256)
            cmap = 'binary_r'
            interpolate = False

        if not clim:
            clim = arr.min(), arr.max()
        if not norm:
            norm = Normalize(*clim)

        if interpolate:
            arrg, coords = self.embed(arr, interpolate=interpolate,
                                      grid_pts=grid_pts, grid_coords=True)
            xg, yg = coords
            CS = ax.contourf(xg, yg, arrg, 10, vmin=clim[0], vmax=clim[1],
                             cmap=cmap, norm=norm, **contour_kw)
            if cbar:
                cb = f.colorbar(CS, ax=ax, use_gridspec=True)
                cb.solids.set_edgecolor('face')

        scatter_kw.setdefault('edgecolors', 'k')
        scatter_kw.setdefault('linewidths', 1.0)
        # set default point size to be 90% of the minimum pitch (and square it to get correct area size)
        # (DOES NOT WORK WELL IF AXES GETS RESIZED AT DRAW TIME)
        # pts_per_pitch = ax.transData.transform((0, self.min_pitch))[1] - ax.transData.transform((0, 0))[1]
        # scatter_kw.setdefault('s', (0.9 * pts_per_pitch) ** 2)
        scatter_kw.setdefault('s', 100)
        if 'c' in scatter_kw:
            # someone is pretty sure what the colors should be, rather than simply mapping the given array
            sct = ax.scatter(x, y, **scatter_kw)
        else:
            sct = ax.scatter(x, y, c=arr, norm=norm, cmap=cmap, **scatter_kw)
        if cbar:
            if not interpolate:
                cb = f.colorbar(sct, ax=ax, use_gridspec=True)
                cb.solids.set_edgecolor('face')
            return f, cb
        return f

    def embed(
            self, data, axis=0, interpolate='linear', grid_pts=None,
            grid_coords=False
    ):
        """
        Interpolates sample vector(s) in data onto a grid using Delauney
        triangulation. Interpolation modes may be "linear" or "cubic"
        """
        y, x = self.to_mat()
        triang = Triangulation(x, y)
        g = self.boundary
        if grid_pts is None:
            grid_pts = self.geometry
        elif not np.iterable(grid_pts):
            grid_pts = (grid_pts, grid_pts)
        else:
            pass
        yg = np.linspace(g[2], g[3], grid_pts[0])
        xg = np.linspace(g[0], g[1], grid_pts[1])
        xg, yg = np.meshgrid(xg, yg, indexing='xy')

        def f(x, interp_mode):
            xgr = xg.ravel()
            ygr = yg.ravel()
            if interp_mode == 'linear':
                interp = LinearTriInterpolator(triang, x)
            else:
                interp = CubicTriInterpolator(triang, x)
            return interp(xgr, ygr).reshape(grid_pts)
        arrg = np.apply_along_axis(f, axis, data, interpolate)
        return (arrg, (xg, yg)) if grid_coords else arrg

    # many methods no longer make sense with coordinates
    def as_col_major(self):
        raise NotImplementedError

    def as_row_major(self):
        raise NotImplementedError

    @classmethod
    def from_mask(cls, *args, **kwargs):
        raise NotImplementedError

    def as_channels(self, *args, **kwargs):
        raise NotImplementedError

    def interpolated(self, *args, **kwargs):
        raise NotImplementedError


def channel_combinations(chan_map, scale=1.0, precision=4):
    """Compute tables identifying channel-channel pairs.

    Parameters
    ----------
    chan_map : ChannelMap
    scale : float or pair
        The constant pitch or the (dy, dx) pitch between electrodes
        precision : number of decimals for distance calculation (it seems
        some distances are not uniquely determined in floating point).

    Returns
    -------
    chan_combs : Bunch
        Lists of channel # and grid location of electrode pairs and
        distance between each pair.
    """

    combs = np.array(list(itertools.combinations(range(len(chan_map)), 2)))
    combs = combs[combs[:, 1] > combs[:, 0]]
    chan_combs = Bunch()
    npair = len(combs)
    chan_combs.p1 = np.empty(npair, 'i')
    chan_combs.p2 = np.empty(npair, 'i')
    chan_combs.idx1 = np.empty((npair, 2), 'd')
    chan_combs.idx2 = np.empty((npair, 2), 'd')
    chan_combs.dist = np.empty(npair)
    ii, jj = chan_map.to_mat()
    chan_combs.p1 = combs[:, 0]
    chan_combs.p2 = combs[:, 1]
    chan_combs.idx1 = np.c_[np.take(ii, combs[:, 0]), np.take(jj, combs[:, 0])]
    chan_combs.idx2 = np.c_[np.take(ii, combs[:, 1]), np.take(jj, combs[:, 1])]
    # Distances are measured between grid locations (i1,j1) to (i2,j2)
    # Define a (s1,s2) scaling to multiply these distances
    if np.iterable(scale):
        s_ = np.array(scale)
    else:
        s_ = np.array([scale, scale])

    d = np.abs(chan_combs.idx1 - chan_combs.idx2) * s_
    dist = (d ** 2).sum(1) ** 0.5
    chan_combs.dist = np.round(dist, decimals=precision)
    idx1 = chan_combs.idx1.astype('i')
    if (idx1 == chan_combs.idx1).all():
        chan_combs.idx1 = idx1
    idx2 = chan_combs.idx2.astype('i')
    if (idx2 == chan_combs.idx2).all():
        chan_combs.idx2 = idx2
    return chan_combs
