"""
This module provides an object with 48-pixel-manta SPAD number definition,
related utilities and rich HTML representations in Jupyter Notebook.
"""

from __future__ import division
import numpy as np
from ipythonblocks import BlockGrid, colors
from gridfit import get_grid_coords


def mirror(group):
    mask = np.zeros((4, 12), dtype=bool)
    rslice, cslice1 = group
    cslice2 = slice(cslice1.start, cslice1.stop, cslice1.step)
    mask[rslice, cslice1] = True
    mask[rslice, cslice2] = True
    return mask


class PixelGroup(object):
    def __init__(self, manta, slices=None, mask=None):
        if slices is None and mask is None:
            print('ERROR: You need to specify either `mask` or `slices`.')
            raise ValueError
        self._slices = slices
        if mask is not None:
            self._slices = None
            self._mask = mask
        self.manta = manta

    @property
    def index(self):
        """Object to index the SPAD array and get pixels in group.
        """
        if self._slices is not None:
            return self._slices
        else:
            return self.mask

    @property
    def mask(self):
        """Boolean mask to index the SPAD array and get pixels in group.
        """
        if not hasattr(self, '_mask'):
            self._mask = np.zeros(self.manta.shape, dtype=bool)
            self._mask[self._slices] = True
        return self._mask

    @property
    def shape(self):
        """Group shape in the case is specified through slices.
        """
        if self._slices is not None:
            rowslice, colslice = self._slices
            rowshape = (rowslice.stop - rowslice.start)
            colshape = (colslice.stop - colslice.start)
            return (rowshape, colshape)
        else:
            raise NotImplementedError

    @property
    def ispad(self):
        """SPADs in group (index, start from 0)."""
        return self.manta.ispad[self.index]

    @property
    def spad(self):
        """SPADs in group (number, starts from 1)."""
        return self.ispad + 1

    def coords(self, pitch, xcenter, ycenter, rotation=0):
        """Coordinates of pixels in the group.
        """
        X, Y = get_grid_coords(shape=self.shape, pitch=pitch,
                               x0=xcenter, y0=ycenter, rotation=rotation)
        return X, Y


class Manta48(object):
    longdim = 12
    shortdim = 4
    size = longdim*shortdim

    # Index of first two "lines" of SPADs
    ilines1 = np.arange(size/2).reshape(longdim, shortdim//2)[::-1]
    # Index of last two "lines" of SPADs
    ilines2 = size/2 + np.arange(longdim*shortdim//2).reshape(
        longdim, shortdim//2)

    def __init__(self, index=False, lcos_coord=True, horiz=False,
                 mirrorx=False):
        """Object representing 48-pixel manta SPAD numbering.

        Arguments:
            index (bool): if True, start numbering SPADs from 0.
                Otherwise start from 1. Default False.
            lcos_coord (bool): if True, uses the LCOS coordinates system,
                otherwise uses the SPAD coordinate system. The difference
                is a mirroring/flipping of axis direction. Default True.
            horiz (bool): if True use an horizontal layout. Default False.
            mirrorx (bool): if True the SPAD ordering is inverted in along
                the "x" direction. Default False.
        """
        self.lcos_coord = lcos_coord
        self.index = index
        self.horiz = horiz

        if horiz:
            self.nrows, self.ncols = self.shortdim, self.longdim
            self.ispad = np.vstack([self.ilines1.T, self.ilines2.T])
        else:
            self.nrows, self.ncols = self.longdim, self.shortdim
            self.ispad = np.hstack([self.ilines1, self.ilines2])

        if mirrorx:
            self.ispad = self.ispad[:, ::-1]

        if not lcos_coord:
            self.ispad = self.ispad[::-1]

        self.nspad = self.ispad + 1                         # SPAD's number
        if index:
            self.spad = self.ispad
            self.title = 'index'
        else:
            self.spad = self.nspad
            self.title = 'number'

        self.shape = self.spad.shape
        if horiz:
            self.left_half = self.spad[:, :int(self.nrows/2)]
            self.right_half = self.spad[:, int(self.nrows/2):]
        else:
            self.top_half = self.spad[:int(self.nrows/2)]
            self.bot_half = self.spad[int(self.nrows/2):]


    def slicegroup(self, slicerows, slicecols):
        if type(slicerows) is tuple:
            slicerows = slice(*slicerows)
        if type(slicecols) is tuple:
            slicecols = slice(*slicecols)
        return PixelGroup(self, slices=(slicerows, slicecols))

    def centergroup(self, nrows, ncols):
        start_row = self.nrows//2 - nrows//2
        stop_row = start_row + nrows
        start_col = self.ncols//2 - ncols//2
        stop_col = start_col + ncols
        return PixelGroup(self, slices=(slice(start_row, stop_row),
                                        slice(start_col, stop_col)))

    def dualgroup(self, group):
        """Get corresponding pixels on the "other" side of the detector.

        `group` is either a boolean mask or a 2-element tuple of slice objects.

        Returns
            "group" of pixel for the other side of the detector.
        """
        half = self.longdim//2
        if hasattr(group, '__array__'):
            # Boolean mask
            if self.horiz:
                group = group.T
            side1 = group[slice(None, half)]
            side2 = group[slice(half, None)]
            dualgroup = np.vstack([side2, side1])
            if self.horiz:
                dualgroup = dualgroup.T
        else:
            rowslice, colslice = group
            swapslice = colslice if self.horiz else rowslice

            start, stop = swapslice.start + half, swapslice.stop + half
            if start >= self.longdim:
                start, stop = start - self.longdim, stop - self.longdim
            newslice = slice(start, stop)
            if self.horiz:
                dualgroup = (rowslice, newslice)
            else:
                dualgroup = (newslice, colslice)
        return dualgroup

    def sidegroup(self, nrows, ncols, other=False):
        if self.horiz:
            rowfraction, colfraction = 2, 4
        else:
            rowfraction, colfraction = 4, 2
        rowstart = self.nrows//rowfraction - nrows//2
        rowstop = rowstart + nrows
        colstart = self.ncols//colfraction - ncols//2
        colstop = colstart + ncols
        group = (slice(rowstart, rowstop), slice(colstart, colstop))
        if other:
            group = self.dualgroup(group)
        return PixelGroup(self, slices=group)

    def doublegroup(self, nrows, ncols):
        mask = np.zeros((self.nrows, self.ncols), dtype=bool)
        sidegroup1 = self.sidegroup(nrows, ncols)
        sidegroup2 = self.sidegroup(nrows, ncols, other=True)
        mask[sidegroup1.index] = True
        mask[sidegroup2.index] = True
        return PixelGroup(self, mask=mask)

    def spad_coord(self, pixel_num):
        """Get the (row, column) coordinates of SPAD `pixel_num`."""
        return np.where(self.spad == pixel_num)

    def ispad_coord(self, index):
        """Get the (row, column) coordinates of SPAD `index`."""
        return np.where(self.ispad == index)

    def distance(self, pixel1, pixel2):
        """Return the euclidean distance between SPADs `pixel1` and `pixel2`.
        """
        vector_diff = np.array(self.spad_coord(pixel1)) - \
                      np.array(self.spad_coord(pixel2))
        return np.linalg.norm(vector_diff)

    def show_selection(self, s_block):
        """Show a colored table highlighting the selection"""
        grid = BlockGrid(self.ncols, self.nrows, fill=colors['LightGray'],
                         block_size=15)
        if hasattr(s_block, '__array__'):
            # ipython blocks does not support boolean indexing
            rows, cols = np.nonzero(s_block)
            for row, col in zip(rows, cols):
                grid[int(row), int(col)] = colors['LightGreen']
        else:
            grid[s_block] = colors['LightGreen']
        return grid

    def __repr__(self):
        s = 'Vert.\n'
        s += "    ^  \n"
        for ir, r in enumerate(self.spad):
            s += ' %2d | %s\n' % (ir, r)
        s += "    '--------------> \n"
        s += ('     ' + (' %2d'*self.spad.shape[1]) + '  Horiz.') %\
                tuple(range(self.spad.shape[1]))
        return s

    def _svg_arrow(self, x1, y1, x2, y2, width, height):
        s = '<svg width="%d" height="%d">' % (width, height)
        s += ('<defs><marker id="arrow" markerWidth="13" markerHeight="13" '
              'refx="2" refy="6" orient="auto">'
              '    <path d="M2,2 L2,11 L10,6 L2,2" style="fill:black;" />'
              '</marker></defs>')
        s += '    <path d="M%d,%d L%d,%d"' % (x1, y1, x2, y2)
        s += '          style="stroke:black; stroke-width: 1.25px; fill: none;'
        s += '                 marker-end: url(#arrow);" />'
        s += '</svg>'
        return s

    def _html_table_vert_(self):
        s = '<table>'# style="width:300px">'
        row0 = ''.join(['<td><b>%d</b></td>' % i for i in range(self.ncols)])
        s += '<tr><td> </td> %s </tr>' % row0
        for ir, r in enumerate(self.spad):
            style = ''
            if ir == 5 or ir == 6:
               style = ' style="background-color:#E7FFDE;"'
            row = ''.join(['<td %s>%d</td>' % (style, i) for i in r])
            row_style = ''
            if ir == 6:
                row_style = 'style="border-top-width:3px;"'
            s += '<tr %s><td><b>%d</b></td> %s </tr>' % (row_style, ir, row)
        s += '</table>'
        return s

    def _html_table_horiz_(self):
        s = '<table>'# style="width:300px">'
        row0 = ''.join(['<td><b>%d</b></td>' % i for i in range(self.ncols)])
        s += '<tr><td> </td> %s </tr>' % row0

        style_plain = ''
        emph = 'background-color:#E7FFDE;'
        rbord = 'border-right-width:3px;'
        def_style = [style_plain]*self.ncols
        def_style[5] = 'style="%s%s"' % (emph, rbord)
        def_style[6] = 'style="%s"' % emph
        for ir, r in enumerate(self.spad):
            style = def_style[:]
            if ir in [1, 2]:
                for ic in [2, 3, 8, 9]:
                    style[ic] = 'style="%s"' % emph
            row = ''.join(['<td %s>%d</td>' % (st, ic) for ic, st in
                           zip(r, style)])
            row_style = ''
            s += '<tr %s><td><b>%d</b></td> %s </tr>' % (row_style, ir, row)
        s += '</table>'
        return s

    def _repr_html_(self):
        ahlength = 100
        avlength = 100
        if self.horiz:
            table = self._html_table_horiz_()
        else:
            table = self._html_table_vert_()
        coord = 'LCOS' if self.lcos_coord else 'real'
        s = "<h3> 48-pixel Manta SPAD %s (%s coords)</h3>" % \
                (self.title, coord)
        s += '<div>LCOS Vertical </div>'
        s += '<div style="float:left;">'
        if self.lcos_coord or self.horiz:
            s += self._svg_arrow(5, 20+avlength, 5, 20, width=40, height=200)
        else:
            s += self._svg_arrow(5, 20, 5, 20+avlength, width=40, height=200)
        s += '</div>'
        s += '<div>'
        s += table
        s += '</div>'
        s += self._svg_arrow(40, 10, 40+ahlength, 10, width=200, height=30)
        s += 'LCOS Horizontal'
        return s

class Manta48_NI_top(Manta48):
    # Index of first two "lines" of SPADs
    ilines1 = np.arange(24).reshape(2, 12).T
    # Index of last two "lines" of SPADs
    ilines2 = np.arange(24, 48).reshape(2, 12).T

class Manta48_NI_bot(Manta48):
    # Index of first two "lines" of SPADs
    ilines1 = 48 + np.arange(24).reshape(2, 12).T
    # Index of last two "lines" of SPADs
    ilines2 = 48 + np.arange(24, 48).reshape(2, 12).T
