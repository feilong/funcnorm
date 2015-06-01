"""
% FUNCTION computeSphericalFromCartesian
% Returns spher as (phi, theta) pair

% This file is part of the Functional Normalization Toolbox, (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""

import numpy as np

def compute_spherical_from_cartesian(cart, coord_maps):
    """
    Parameters
    ----------
    cart : array, 3 x N
    coord_maps : list or int
        If it's an int, its value must be 1, 2, or 3.
        If it's a list, each element must be 1, 2, or 3.

    Returns
    -------
    spher : array, 2 x N
    """
    N = cart.shape[1]
    spher = np.zeros((2, N), dtype=cart.dtype)

    mags = np.sqrt((cart**2).sum(axis=0))

    if str(cart.dtype).startswith('complex'):
        raise ValueError('compute_spherical_from_cartesian: '\
                         'cart array contains imaginary components.')

    xs = cart[0, :]
    ys = cart[1, :]
    zs = cart[2, :]

    if isinstance(coord_maps, int):
        coord_maps = [coord_maps] * N
    coord_maps = np.array(coord_maps)

    num_found = 0

    # Find coordinate maps equal to 1 (phi measured from x-axis; theta
    # measured from z-axis)
    idx = (coord_maps == 1)
    count = np.sum(idx)
    if count:
        num_found += count
        spher[0, idx] = np.arccos(xs[idx]/mags[idx])
        spher[1, idx] = np.arctan2(ys[idx], zs[idx])

    # Find coordinate maps equal to 2 (phi measured from y-axis; theta
    # measured from x-axis)
    idx = (coord_maps == 2)
    count = np.sum(idx)
    if count:
        num_found += count
        spher[0, idx] = np.arccos(ys[idx]/mags[idx])
        spher[1, idx] = np.arctan2(zs[idx], xs[idx])

    # Find coordinate maps equal to 3 (phi measured from z-axis; theta
    # measured from x-axis)
    idx = (coord_maps == 3)
    count = np.sum(idx)
    if count:
        num_found += count
        spher[0, idx] = np.arccos(zs[idx]/mags[idx])
        spher[1, idx] = np.arctan2(ys[idx], xs[idx])

    if num_found != N:
        raise ValueError('compute_spherical_from_cartesian: '
                         'Improper coordinate map.')

    return spher
