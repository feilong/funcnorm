import numpy as np
from numpy.testing import assert_allclose

from ..utils import _calc_geodesic_dist
from ..Coordinates import _normalize, _calc_spher_coords


def test_geodesic_dist():
    n_nodes = 3000
    cart1 = _normalize(np.random.random((n_nodes, 3)))
    cart2 = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher1 = _calc_spher_coords(cart1, maps)
    spher2 = _calc_spher_coords(cart2, maps)
    assert_allclose(
        _calc_geodesic_dist(cart1, cart2),
        _calc_geodesic_dist(spher1, spher2))
    spher1 = _calc_spher_coords(cart1, np.ones((n_nodes, )))
    spher2 = _calc_spher_coords(cart2, np.ones((n_nodes, )))
    assert_allclose(
        _calc_geodesic_dist(cart1[[0], :], cart2),
        _calc_geodesic_dist(spher1[[0], :], spher2))
