import os
import numpy as np
from numpy.testing import assert_allclose

from ..Coordinates import _normalize, _calc_spher_coords, _calc_cart_coords, \
    _calc_cart_warped_from_spher_warp, _calc_spher_warp_from_cart_warp

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
