import os
from scipy.io import loadmat
import numpy as np
from nose.tools import assert_true, assert_almost_equal, assert_equal
from numpy.testing import assert_allclose, assert_array_equal

from ..Folding import _dcart_dspher, _calc_areal_terms, _calc_oriented_areas
from ..Coordinates import _normalize, _calc_cart_coords, _calc_spher_coords

DIR = os.path.dirname(os.path.realpath(__file__))


def test_calc_oriented_areas():
    # First a very simple case (right triangle)
    triangles = np.array([[0, 1, 2]])
    cart = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    tri_areas, oriented_normals = _calc_oriented_areas(triangles, cart)
    assert_equal(tri_areas, np.array([0.5]))
    # should be either [1, 0, 0] or [-1, 0, 0]
    assert_array_equal(np.abs(oriented_normals), np.array([[1, 0, 0]]))

    # Then a lot of randomly generated data
    n_triangles = 100
    cart = _normalize(np.random.random((3 * n_triangles, 3)))
    triangles = np.random.choice(
        range(3 * n_triangles), (n_triangles, 3), replace=False)
    tri_areas, oriented_normals = _calc_oriented_areas(triangles, cart)

    assert_equal(tri_areas.shape, (n_triangles, ))
    assert_equal(oriented_normals.shape, (n_triangles, 3))
    assert_true(np.all(tri_areas >= 0))
    assert_allclose(np.sum(oriented_normals**2, axis=1),
                    np.ones(n_triangles, ))
    for j in range(n_triangles):
        p0 = cart[triangles[j, 0], :]
        p1 = cart[triangles[j, 1], :]
        p2 = cart[triangles[j, 2], :]
        assert_almost_equal(np.dot(p1-p0, oriented_normals[j, :]), 0.0)
        assert_almost_equal(np.dot(p2-p0, oriented_normals[j, :]), 0.0)
        assert_almost_equal(np.dot(p1-p2, oriented_normals[j, :]), 0.0)


def test_dcart_dspher():
    n_nodes = 1000
    cart = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)
    dp_dphi, dp_dtheta = _dcart_dspher(cart, maps)
    spher = _calc_spher_coords(cart, maps)
    delta = 1e-8
    atol, rtol = 1e-6, 1e-6
    spher2 = spher.copy()
    spher2[:, 0] += delta
    cart2 = _calc_cart_coords(spher2, maps)
    assert_allclose(dp_dphi, (cart2 - cart) / delta, atol=atol, rtol=rtol)
    spher2 = spher.copy()
    spher2[:, 1] += delta
    cart2 = _calc_cart_coords(spher2, maps)
    assert_allclose(dp_dtheta, (cart2 - cart) / delta, atol=atol, rtol=rtol)


def test_compute_areal_terms():
    fname = os.path.join(DIR, 'data_for_test_compute_areal_terms.mat')
    data = loadmat(fname)
    tri_area, dareal_dphi, dareal_dtheta = _calc_areal_terms(
        data['triangles'].T, data['cart_coords'].T,
        data['coord_maps'].ravel() - 1,
        data['orig_tri_areas'].ravel(), data['tri_normals'].T)
    assert_allclose(tri_area, data['tri_area'].ravel())
    assert_allclose(dareal_dphi, data['dareal_dphi'].ravel())
    assert_allclose(dareal_dtheta, data['dareal_dtheta'].ravel())


def test_areal_terms():
    n_triangles = 300
    n_nodes = 3 * n_triangles
    cart = _normalize(np.random.random((n_nodes, 3)))
    triangles = np.random.choice(
        range(n_nodes), (n_triangles, 3), replace=False)
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher = _calc_spher_coords(cart, maps)

    cart2 = _normalize(np.random.random((n_nodes, 3)))
    tri_areas, tri_normals = _calc_oriented_areas(triangles, cart2)

    areal, dareal_dphi, dareal_dtheta = _calc_areal_terms(
        triangles, cart, maps, tri_areas, tri_normals)

    dareal_dphi2, dareal_dtheta2 = np.zeros((n_nodes, )), np.zeros((n_nodes, ))
    spher2 = spher.copy()
    # cart2 = _calc_cart_coords(spher2, maps)
    # assert_allclose(cart, cart2)
    delta = 1e-8
    for i in range(n_nodes):
        spher2[i, 0] += delta
        cart2 = _calc_cart_coords(spher2, maps)
        areal2 = _calc_areal_terms(
            triangles, cart2, maps, tri_areas, tri_normals, False)
        dareal_dphi2[i] = (areal2 - areal) / delta
        spher2[i, 0] = spher[i, 0]
        spher2[i, 1] += delta
        cart2 = _calc_cart_coords(spher2, maps)
        areal2 = _calc_areal_terms(
            triangles, cart2, maps, tri_areas, tri_normals, False)
        dareal_dtheta2[i] = (areal2 - areal) / delta
        spher2[i, 1] = spher[i, 1]
    atol, rtol = 1e-6, 1e-5
    assert_allclose(dareal_dphi, dareal_dphi2, atol=atol, rtol=rtol)
    assert_allclose(dareal_dtheta, dareal_dtheta2, atol=atol, rtol=rtol)
