import os
import numpy as np
from numpy.testing import assert_allclose
import logging

from ..utils import _calc_geodesic_dist
from ..Coordinates import _normalize, _calc_spher_coords
from ..Surface import surf_from_file

DIR = os.path.dirname(os.path.realpath(__file__))
logger = logging.getLogger('funcnorm')


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


def test_average_geodesic_dist():
    fname = os.path.join(DIR, os.pardir, os.pardir, 'results',
                         'standard2mm_sphere.reg.asc')
    if not os.path.exists(fname):
        print 'Surface file not found. Skipping test...'
    surf = surf_from_file(fname)
    surf.normalize_cart()
    all_gds = []
    for i in range(surf.n_nodes):
        nbrs = surf.nbrs[i, :surf.num_nbrs[i]]
        gds = _calc_geodesic_dist(surf.cart[[i], :], surf.cart[nbrs, :])
        all_gds.append(gds)
    all_gds = np.hstack(all_gds)
    print all_gds.shape
    print all_gds.min(), all_gds.mean(), all_gds.max()
