"""
% The objective of this function is to find an appropriate coordinate
% mapping of the sphere for each voxel such that the voxel's original
% coordinate location and current warped location are not located close to
% the poles (phi = 0, phi = pi)

% This file is part of the Functional Normalization Toolbox,
% (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""

import numpy as np


def compute_coordinate_maps(cart_coords, warp_cart_coords):
    """
    The objective of this function is to find an appropriate coordinate
    mapping of the sphere for each voxel such that the voxel's original
    coordinate location and current warped location are not located close to
    the poles (phi = 0, phi = pi)

    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    warp_cart_coords : (3, n_nodes) array

    Returns
    -------
    coord_maps : list
       Each element of the list is either 1, 2, or 3.
    """
    n_nodes = cart_coords.shape[1]
    coord_maps = []
    for j in range(n_nodes):
        cs = []
        for i in range(3):
            c = max(abs(cart_coords[i, j]), abs(warp_cart_coords[i, j]))
            cs.append(c)
        coord_maps.append(int(np.argmin(cs)) + 1)

    return coord_maps
