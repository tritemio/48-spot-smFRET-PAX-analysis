# -*- coding: utf-8 -*-
"""
Functions to generate a square grid of points and to fit a set of experimental
coordinates to a grid (pitch, offset, rotation).
"""

import numpy as np
from scipy.optimize import leastsq, minimize

def get_grid_coords(shape, pitch, x0, y0, rotation=0):
    """Return coordinates of a grid points with given geometrical properties.

    Parameters
    ----------
    `shape` : tuple (length 2)
        the shape of the grid (num. of y points, num. of x points)
    `pitch` : float
        pitch of the grid in x and y directions
    `x0`, `y0` : floats
        center position of the grid
    `rotation` : float
        grid rotation in degree

    Returns
    -------
    `X`, `Y` : arrays of shape `shape`
        coordinates grid points (as returned by `meshgrid`)
    """
    x_spots, y_spots = np.meshgrid(
             (np.arange(shape[1]) - (shape[1]-1)/2.)*pitch,
             (np.arange(shape[0]) - (shape[0]-1)/2.)*pitch)
    theta = rotation/180.*np.pi
    x_spots = x_spots*np.cos(theta) - y_spots*np.sin(theta) + x0
    y_spots = x_spots*np.sin(theta) + y_spots*np.cos(theta) + y0
    return x_spots, y_spots

def get_grid_coords_xy(shape, x0, y0, pitch_x, pitch_y, rotation=0):
    """Return coordinates of a grid points with given geometrical properties.

    Parameters
    ----------
    `shape` : tuple (length 2)
        the shape of the grid (num. of y points, num. of x points)
    `x0`, `y0` : floats
        center position of the grid
    `pitch_x`, `pitch_y` : floats
        pitch of the grid in x and y directions.
    `rotation` : float
        grid rotation in degree

    Returns
    -------
    `X`, `Y` : arrays of shape `shape`
        coordinates grid points (as returned by `meshgrid`)
    """
    x_spots, y_spots = np.meshgrid(
             (np.arange(shape[1]) - (shape[1]-1)/2.)*pitch_x,
             (np.arange(shape[0]) - (shape[0]-1)/2.)*pitch_y)
    theta = rotation/180.*np.pi
    x_spots = x_spots*np.cos(theta) - y_spots*np.sin(theta) + x0
    y_spots = x_spots*np.sin(theta) + y_spots*np.cos(theta) + y0
    return x_spots, y_spots

def get_residuals(pitch, x0, y0, xp, yp, rotation=0, shape=None):
    """Return array of distances from regular grid for each point in (xp, yp)
    """
    x_grid, y_grid = get_grid_coords(xp.shape, pitch, x0, y0, rotation)
    return np.sqrt((x_grid - xp)**2 + (y_grid - yp)**2)

def _get_residuals_flat(grid_params, xp, yp, shape, validmask=None):
    """
    Return array of euclidean distances from regular grid for each point in (xp, yp)
    Note: `xp` and `yp` must be flattened array to work well with `leastsq`.
    """
    if validmask is None:
        validmask = slice(None)
    x_grid, y_grid = get_grid_coords(shape, *grid_params)
    x_grid, y_grid = x_grid.ravel(), y_grid.ravel()
    return np.sqrt((x_grid[validmask] - xp[validmask])**2 +
                   (y_grid[validmask] - yp[validmask])**2)

def _get_residuals_flat_xy(grid_params, xp, yp, shape, validmask=None):
    """
    Return array of euclidean distances from regular grid for each point in (xp, yp)
    Note: `xp` and `yp` must be flattened array to work well with `leastsq`.
    """
    if validmask is None:
        validmask = slice(None)
    x_grid, y_grid = get_grid_coords_xy(shape, *grid_params)
    x_grid, y_grid = x_grid.ravel(), y_grid.ravel()
    return np.sqrt((x_grid[validmask] - xp[validmask])**2 +
                   (y_grid[validmask] - yp[validmask])**2)


def _get_squared_distances_mean(grid_params, xp, yp):
    """Return the **mean** distance from regular grid for points in (xp, yp)
    """
    return (_get_residuals_flat(
                grid_params, xp.ravel(), yp.ravel(), xp.shape)**2).mean()

def _get_grid_dict(grid_params):
    if len(grid_params) == 4:
        grid_params = list(grid_params) + [0]  # set rotation to zero
    param_names = ('shape', 'pitch', 'x0', 'y0', 'rotation')
    return {k: v for k, v in zip(param_names, grid_params)}

def _get_grid_dict_xy(grid_params):
    if len(grid_params) == 5:
        grid_params = list(grid_params) + [0]  # set rotation to zero
    param_names = ('shape', 'x0', 'y0', 'pitch_x', 'pitch_y', 'rotation')
    return {k: v for k, v in zip(param_names, grid_params)}

def fit_grid_leastsq(xp, yp, rotate=True,
                     pitch0=1, center_x0=0, center_y0=0, rotation0=0):
    """Fit the optimal grid using scipy `leastsq`.
    """
    parameters = [pitch0, center_x0, center_y0]
    if rotate:
        parameters.append(rotation0)
    validmask = ~(np.isnan(xp) + np.isnan(yp))
    res = leastsq(_get_residuals_flat, x0=parameters,
                  args=(xp.ravel(), yp.ravel(), xp.shape, validmask.ravel()))
    return  _get_grid_dict([xp.shape]+list(res[0]))

def fit_grid_leastsq_xy(xp, yp, pitch_x0=20, pitch_y0=20,
                        center_x0=0, center_y0=0, rotation0=None):
    """Fit the optimal grid using scipy `leastsq`.
    """
    parameters = [center_x0, center_y0, pitch_x0, pitch_y0]
    if rotation0 is not None:
        parameters.append(rotation0)
    validmask = ~(np.isnan(xp) + np.isnan(yp))
    res = leastsq(_get_residuals_flat_xy, x0=parameters,
                  args=(xp.ravel(), yp.ravel(), xp.shape, validmask.ravel()))
    return  _get_grid_dict_xy([xp.shape]+list(res[0]))


def fit_grid_minimize(xp, yp, pitch0=0, center_x0=0, center_y0=0, rotation0=0):
    """Fit the optimal grid using scipy `minimize`.
    """
    res = minimize(_get_squared_distances_mean,
                   x0=(pitch0, center_x0, center_y0, rotation0), args=(xp, yp))
    return _get_grid_dict([xp.shape]+list(res.x))

def print_grid_dict(grid_dict):
    print('Center:    %6.2f H,  %6.2f V' % (grid_dict['x0'], grid_dict['y0']))
    print('Pitch:     %6.2f' % grid_dict['pitch'])
    print('Rotation:  %6.2f degree' % grid_dict['rotation'])
    print('Shape:     (%d, %d)' % (grid_dict['shape'][0], grid_dict['shape'][1]))

if __name__ == '__main__':
    xe = np.array([ -23.31,  -4.01,  15.44,  34.71,
                    -23.39,  -4.10,  15.28,  34.60,
                    -23.75,  -4.38,  15.07,  34.34,
                    -23.91,  -4.53,  14.82,  34.15]).reshape(4, 4)
    ye = np.array([-16.00, -15.81, -15.72, -15.49,
                     3.29,   3.51,   3.90,   4.02,
                     22.75,  22.93,  23.18,  23.43,
                     42.19,  42.35,  42.69,  42.87]).reshape(4, 4)
    print('[minimize]:\n')
    print_grid_dict(fit_grid_minimize(xe, ye))
    print('\n[leastsq]:\n')
    print_grid_dict(fit_grid_leastsq(xe, ye))

    grid_dict = fit_grid_leastsq(xe, ye)

    print('\nResiduals:', get_residuals(xp=xe, yp=ye, **grid_dict))
    print('Residuals mean: %.2f' % get_residuals(xp=xe, yp=ye, **grid_dict).mean())
    print('Residuals max: %.2f' % get_residuals(xp=xe, yp=ye, **grid_dict).max())

    x_block, y_block = get_grid_coords(**grid_dict)
