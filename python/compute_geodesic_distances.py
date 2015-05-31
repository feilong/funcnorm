"""
% FUNCTION gds = computeGeodesicDistances(coord1, coords2, rho)
%   coord1 are the coordinates for one voxel
%   If coord1 is a 3 x 1 vector, then coordinates are Cartesian (x, y, z)
%   If coord1 is a 2 x 1 vector, then coordinates are spherical (phi, theta)
%   coord2 are the coordinates of N voxels 
%   coord2 must  be in the same coordinate representation as coord1, meaning
%   size(coord1, 1) == size(coord2, 1)
%   So coord2 is either 2 x N or 3 x N, depending on coord1
%
%   This function returns a 1 x N vector representing the spherical geodesic
%   distance of coord1 to each of the voxels in coord2
%
% *** OR ***
%
% FUNCTION gds = computeGeodesicDistances(coord1, coord2, rho)
%   coord1 are the coordinates for N voxels
%   If coord1 is 3 x N vector, then coordinates are Cartesian (x, y, z)
%   If coord1 is 2 x N vector, then coordinates are spherical (phi, theta)
%   coord2 are the coordinates of N voxels as well
%   coord2 must be in the same coordinate representation as coord1, meaning
%   size(coord1, 1) == size(coord2, 1)
%   
%   This function returns a 1 x N vector representing the spherical
%   geodesic distances between the columns of coord1 with the cooresponding
%   column in coord2
%
%
% This file is part of the Functional Normalization Toolbox, (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""

import numpy as np

def compute_geodesic_distances(coord1, coord2, rho=1.0):
    """ Compute geodesic distances between voxels based on coordinates.

    If `coord1` only has one column/voxel (single sub-mode), its geodesic 
    distances with the columns/voxels of `coord2` are calculated.

    If `coord1` has multiple columns/voxels (multi sub-mode), the geodesic
    distances between corresponding columns of `coord1` and `coord2` are
    calculated.

    The number of rows of `coord1` and `coord2` could be either 3 (Cartesian
    coordinates) or 2 (spherical coordinates), but this number must be
    the same for `coord1` and `coord2`.

    Therefore, valid dimension combinations include:

    ======== ======== ========== ======== ======
    `coord1` `coord2` coordinate sub-mode `gds`
    ======== ======== ========== ======== ======
    (3, 1)   (3, N)   Cartesian  single   (1, N)
    (3, N)   (3, N)   Cartesian  multi    (1, N)
    (2, 1)   (2, N)   spherical  single   (1, N)
    (2, N)   (2, N)   spherical  multi    (1, N)
    ======== ======== ========== ======== ======

    Parameters
    ----------
    coord1, coord2 : numpy arrays
        If there're three rows, then coordinates are Cartesian (x, y, z).
        If there're two rows, then coordinates are spherical (phi, theta).
        If `coord1` have multiple columns, its number of columns must be
        the same as `coord2`.
    rho : float
        The radius for spherical coordinates.

    Returns
    -------
    gds : (1 x N) array
        In single sub-mode, it's the geodesic distances of the column in
        `coord1` to each of the columns in `coord2`.
        In multi sub-mode, it's the geodesic distances between the columns
        of `coord1` with the cooresponding column in `coord2`.

    Notes
    -----
    A straightforward approach to computing geodesic distances would
    be through the arccos of the dot product.
    However, arccos is numerically unstable for small geodesic distances
    So we use formulas with arcsin instead of arccos

    The Matlab version separate conditions of single/multi sub-mode, but the
    code here should work for both
    """
    if coord1.shape[0] != coord2.shape[0]:
        msg = 'compute_geodesic_distances: coord1 and coord2 must be in'\
              ' the same coordinate representation.'
        raise ValueError(msg)
    if coord1.shape[0] == 3:
        mode = 'cartesian'
    elif coord1.shape[0] == 2:
        mode = 'spherical'
    else:
        msg = 'compute_geodesic_distances: coord1 must have 2 or 3 rows.'
        raise ValueError(msg)

    if coord1.shape[1] == 1:
        sub_mode = 'single'
        coord1 = np.tile(coord1, (1, coord2.shape[1]))
    else:
        sub_mode = 'multi'
        if coord1.shape[1] != coord2.shape[1]:
            msg = 'compute_geodesic_distances: coord1 and coord2 must '\
                  'have the same number of columns in multi sub-mode.'
            raise ValueError(msg)

    N = coord2.shape[1]
    gds = np.zeros((1, N), dtype=coord1.dtype)

    if mode == 'spherical':
        phi1, theta1 = coord1[0, :], coord1[1, :]
        phi2, theta2 = coord2[0, :], coord2[1, :]
        gds = 2 * rho * np.arcsin(np.sqrt(
            np.sin(phi1)*np.sin(phi2)*(np.sin((theta1-theta2)/2))**2
            + (np.sin((phi1 - phi2)/2))**2
        ))
    elif mode == 'cartesian':
        gds = rho * np.arcsin(
            np.sqrt(
                (np.cross(coord1, coord2, axisa=0, axisb=0).T**2).sum(axis=0)
            ))

    gds = gds.reshape(1, -1)

    return gds


def create_test_data():
    from scipy.io import savemat
    np.random.seed(0)
    for i in range(20):
        a = np.random.random((3, 20)) * .5
        b = np.random.random((3, 20)) * .5
        savemat('tests/test_compute_geodesic_distances_data/'\
                'cartesian-multi-%03d.mat' % i,
                {'a':a, 'b':b})
        savemat('tests/test_compute_geodesic_distances_data/'\
                'cartesian-single-%03d.mat' % i,
                {'a':a[:,[0]], 'b':b})
    for i in range(20):
        a = np.random.random((2, 20))
        b = np.random.random((2, 20))
        a[0,:] *= np.pi
        b[0,:] *= np.pi
        a[1,:] *= np.pi * 2.0
        b[1,:] *= np.pi * 2.0
        savemat('tests/test_compute_geodesic_distances_data/'\
                'spherical-multi-%03d.mat' % i,
                {'a':a, 'b':b})
        savemat('tests/test_compute_geodesic_distances_data/'\
                'spherical-single-%03d.mat' % i,
                {'a':a[:,[0]], 'b':b})


if __name__ == '__main__':
    create_test_data()
