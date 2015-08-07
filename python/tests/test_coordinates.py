import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
from nose.tools import assert_true

from ..Coordinates import _normalize, _calc_spher_coords, _calc_cart_coords, \
    _calc_cart_warped_from_spher_warp, _calc_spher_warp_from_cart_warp, \
    _calc_nbrs, _parse_surface_file, surf_from_file, Surface

DIR = os.path.dirname(os.path.realpath(__file__))


def test_normalize():
    n_nodes = 3000
    cart = np.random.random((n_nodes, 3))
    normalized = _normalize(cart)
    assert_allclose(np.linalg.norm(normalized, axis=1), np.ones((n_nodes, )))
    assert_allclose(np.sqrt(np.sum(normalized**2, axis=1)),
                    np.ones((n_nodes, )))


def test_coordinates():
    n_nodes = 3000
    cart = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher = _calc_spher_coords(cart, maps)
    cart2 = _calc_cart_coords(spher, maps)

    assert_allclose(cart, cart2)


def test_warps():
    n_nodes = 3000
    cart = _normalize(np.random.random((n_nodes, 3)))
    cart_warped = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)

    cart_warp = cart_warped - cart
    spher = _calc_spher_coords(cart, maps)
    spher_warp = _calc_spher_warp_from_cart_warp(cart, cart_warp, maps)
    cart_warped2 = _calc_cart_warped_from_spher_warp(cart, spher_warp, maps)
    assert_allclose(cart_warped, cart_warped2)
    spher_warped = spher + spher_warp
    assert_allclose(spher_warped, _calc_spher_coords(cart_warped, maps))


def test_nbrs():
    n_triangles = 300
    n_nodes = 100
    all_nodes = range(n_nodes)
    triangles = np.zeros((n_triangles, 3), dtype='int')
    for i in range(n_triangles):
        triangles[i, :] = np.random.choice(all_nodes, (3, ), False)
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    assert_array_equal(num_nbrs, np.sum(nbrs!=-99, axis=1))
    assert_equal(nbrs.shape[0], n_nodes)
    assert_equal(nbrs.shape[1], np.max(num_nbrs))


def test_parse_surface_file():
    surf_file = os.path.join(DIR, '..', 'data', 'lh.sphere.reg.asc')
    cart, nbrs, triangles = _parse_surface_file(surf_file)
    n_nodes = cart.shape[0]
    assert_equal(n_nodes, 149063)
    assert_equal(cart.shape, (n_nodes, 3))
    assert_equal(triangles.shape, (298122, 3))
    assert_true(triangles.min() >= 0)
    assert_true(triangles.max() < n_nodes)
    assert_equal(nbrs.shape[0], n_nodes)
    assert_equal(nbrs.shape, (149063, 19))


def test_surf_from_file():
    surf_file = os.path.join(DIR, '..', 'data', 'lh.sphere.reg.asc')
    surf = surf_from_file(surf_file)
    assert_true(isinstance(surf, Surface))
    n_nodes = surf.cart.shape[0]
    assert_equal(surf.n_nodes, 149063)
    assert_equal(surf.cart.shape, (n_nodes, 3))
    assert_equal(surf.triangles.shape, (298122, 3))
    assert_true(surf.triangles.min() >= 0)
    assert_true(surf.triangles.max() < n_nodes)
    assert_equal(surf.nbrs.shape[0], n_nodes)
    assert_equal(surf.nbrs.shape, (149063, 19))
