"""
% FUNCTION warp_zero = computeZeroCorrection(cartCoords, warps)
% cartCoords should be a 3 x N matrix of cartesian coordinates for each
% node
% warps should be a cell array of warps for each subject
% i.e., warps{1} is a 3 x N matrix

% This file is part of the Functional Normalization Toolbox,
 (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""

import numpy as np


def compute_zero_correlation(cart_coords, warps):
    """
    Parameters
    ----------
    cart_coords : array, 3 x N
        Cartesian coordinates for each node
    warps : list
        Each element is an 3 x N array

    Returns
    -------
    warp_zero : list
        Each element is an 3 x N array
    """
    N = warps[0].shape[1]
    if cart_coords.shape[1] != N:
        raise ValueError('compute_zero_correlation: '
                         'cart_coords does not match size of warp fields')

    nsubj = len(warps)

    warp_cart_coords = []
    for j in range(nsubj):
        warp_cart_coords.append(cart_coords + warps[j])
    avg_warp = np.array(warp_cart_coords).mean(axis=0)

    min_proj = 1
    for j in range(N):
        # Project average warp back to the sphere
        avg_warp[:, j] /= np.sqrt((avg_warp[:, j]**2).sum())

        if avg_warp[:, [j]].T.dot(cart_coords[:, [j]]) == 1:
            # Too close to rotate
            continue

        # Compute cross product
        c = np.cross(avg_warp[:, [j]], cart_coords[:, [j]], axisa=0, axisb=0)
        c /= np.sqrt(np.sum(c**2))
        # Now rotate c onto z-axis (rotate about z-axis followed by
        # rotation about y-axis)
        theta = np.arctan2(c[0, 1], c[0, 0])
        phi = np.arccos(c[0, 2])
        R1 = np.array([[np.cos(theta), np.sin(theta), 0],
                       [-np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])
        R1 = np.dot(np.array([[np.cos(phi), 0, -np.sin(phi)],
                              [0, 1, 0],
                              [np.sin(phi), 0, np.cos(phi)]]),
                    R1)

        theta = np.arccos(avg_warp[:, [j]].T.dot(cart_coords[:, [j]]))
        R2 = np.array([[np.cos(theta), -np.sin(theta), 0],
                       [np.sin(theta), np.cos(theta), 0],
                       [0, 0, 1]])

        # R = R1^-1*R2*R1;
        R = np.linalg.inv(R1).dot(R2).dot(R1)
        proj = cart_coords[:, [j]].T.dot(R).dot(avg_warp[:, [j]])

        # Use the rotation matrix to rotate each subject
        for k in range(nsubj):
            w = warp_cart_coords[k]
            w[:, [j]] = R.dot(w[:, [j]])
            warp_cart_coords[k] = w

    # Now get warp_zero
    warp_zero = []
    for j in range(nsubj):
        warp_zero.append(warp_cart_coords[j] - cart_coords)

    return warp_zero
