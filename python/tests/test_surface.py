import os
import numpy as np
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_array_equal

from ..Coordinates import _normalize
from ..Surface import _calc_nbrs, _calc_nbr_res, _update_nbr_res

DIR = os.path.dirname(os.path.realpath(__file__))


def test_calc_nbr_res():
    n_triangles, max_res = 300, 6
    n_nodes = n_triangles * 3
    triangles = np.random.choice(range(n_nodes), (n_triangles, 3), False)
    nbrs0, num_nbrs0 = _calc_nbrs(triangles, n_nodes)

    nbrs, res_nbr_sizes, num_nbrs = _calc_nbr_res(nbrs0, max_res)
    assert_array_equal(res_nbr_sizes.sum(axis=1), num_nbrs)
    assert_equal(nbrs.shape[0], n_nodes)
    assert_true(nbrs.shape[1] == nbrs0.shape[1] + 1)
    assert_true(np.all(num_nbrs == num_nbrs0 + 1))


def test_update_nbr_res():
    n_triangles, max_res = 300, 6
    n_nodes = n_triangles * 3
    cart = _normalize(np.random.random((n_nodes, 3)))
    cart_warped = _normalize(np.random.random((n_nodes, 3)))
    triangles = np.random.choice(range(n_nodes), (n_triangles, 3), False)
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    nbrs, res_nbr_sizes, num_nbrs = _calc_nbr_res(nbrs, max_res)
    upd_nbrs, upd_res_nbr_sizes, upd_num_nbrs = (
        nbrs.copy(), res_nbr_sizes.copy(), num_nbrs.copy())
    _update_nbr_res(cart, cart_warped, nbrs, res_nbr_sizes, num_nbrs,
                    upd_nbrs, upd_res_nbr_sizes, upd_num_nbrs)
    nn = np.zeros((n_nodes, ), 'int')
    for i in range(n_nodes):
        nn[i] = nbrs[i, np.argmax(cart[nbrs[i, :], :].dot(cart_warped[i, :]))]
    assert_array_equal(nn, upd_nbrs[:, 0])
    # # It will fail if you're extremely unlucky.
    # assert_true(np.any(nbrs != upd_nbrs))
