import os
import numpy as np
from scipy.io import loadmat
from nose.tools import assert_true, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_allclose

from ..regularization import compute_initial_oriented_areas, \
    compute_partials_cartesian

DIR = os.path.dirname(os.path.realpath(__file__))


def test_compute_partials_cartesian():
    fname = os.path.join(DIR, 'data_for_test_compute_partials_cartesian.mat')
    data = loadmat(fname)
    cart_coords = data['cart_coords']
    coord_maps = data['coord_maps'].ravel()
    dp_dphi, dp_dtheta = compute_partials_cartesian(cart_coords, coord_maps)
    assert_allclose(dp_dphi, data['dp_dphi'])
    assert_allclose(dp_dtheta, data['dp_dtheta'])


def test_compute_initial_oriented_areas():
    # First a very simple case (right triangle)
    triangles = np.array([[0, 1, 2]]).T
    cart_coords = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    tri_areas, oriented_normals = compute_initial_oriented_areas(
        triangles, cart_coords)
    assert_equal(tri_areas, np.array([0.5]))
    # should be either [1, 0, 0] or [-1, 0, 0]
    assert_array_equal(np.abs(oriented_normals), np.array([[1, 0, 0]]).T)

    # Then a lot of randomly generated data
    np.random.seed(0)
    n_triangles = 100
    cart_coords = np.random.randn(3, 3 * n_triangles)
    triangles = np.random.choice(
        range(3 * n_triangles), (3, n_triangles), replace=False)
    tri_areas, oriented_normals = compute_initial_oriented_areas(
        triangles, cart_coords)

    assert_equal(tri_areas.shape, (n_triangles, ))
    assert_equal(oriented_normals.shape, (3, n_triangles))

    assert_true(np.all(tri_areas >= 0))

    for j in range(n_triangles):
        p0 = cart_coords[:, triangles[0, j]]
        p1 = cart_coords[:, triangles[1, j]]
        p2 = cart_coords[:, triangles[2, j]]
        assert_almost_equal(np.dot(p1-p0, oriented_normals[:, j]), 0.0)
        assert_almost_equal(np.dot(p2-p0, oriented_normals[:, j]), 0.0)
        assert_almost_equal(np.dot(p1-p2, oriented_normals[:, j]), 0.0)

        assert_almost_equal(np.sum(oriented_normals[:, j]**2), 1.0)
