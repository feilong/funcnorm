import os
import numpy as np
from numpy.testing import assert_allclose

from ..Interpolation import _calc_correlation_cost, _interp_time_series
from ..Coordinates import _normalize, _calc_spher_coords
from ..Surface import _calc_nbrs, surf_from_file

DIR = os.path.dirname(os.path.realpath(__file__))


def test_interp_time_series():
    fname = os.path.join(DIR, os.pardir, os.pardir, 'results',
                         'standard2mm_sphere.reg.asc')
    if not os.path.exists(fname):
        print 'Surface file not found. Skipping test...'
    surf = surf_from_file(fname)
    surf.normalize_cart()
    T = np.random.random((100, surf.n_nodes))
    surf.cart_warped = surf.cart
    surf.orig_nbrs = surf.nbrs
    TW = _interp_time_series(T, surf, True)
    assert_allclose(T, TW)
    _interp_time_series(T, surf, False)


def test_calc_correlation_cost():
    n_triangles = 300
    n_nodes = 1000
    n_timepoints = 100
    triangles = np.zeros((n_triangles, 3), 'int')
    all_nodes = range(n_nodes)
    for i in range(n_triangles):
        triangles[i, :] = np.random.choice(all_nodes, (3, ), False)
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    cart = _normalize(np.random.random((n_nodes, 3)))
    cart_warped = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher_warped = _calc_spher_coords(cart_warped, maps)
    coords_list = [_calc_spher_coords(cart, np.ones((n_nodes, )) * _)
                   for _ in range(3)]
    ds1 = np.random.random((n_timepoints, n_nodes))
    norm = np.linalg.norm(ds1, axis=0)[np.newaxis, :]
    ds1 /= np.tile(norm, (n_timepoints, 1))
    ds2 = np.random.random((n_timepoints, n_nodes))
    norm = np.linalg.norm(ds2, axis=0)[np.newaxis, :]
    ds2 /= np.tile(norm, (n_timepoints, 1))
    # smoke test
    S, dS_dphi, dS_dtheta = _calc_correlation_cost(
        ds1, ds2, coords_list, maps, spher_warped, nbrs, num_nbrs, 10, 1e-12)


def test_corr_cost_derivatives():
    n_triangles = 1000
    n_nodes = 300
    n_timepoints = 100
    res = 10
    triangles = np.zeros((n_triangles, 3), 'int')
    all_nodes = range(n_nodes)
    for i in range(n_triangles):
        triangles[i, :] = np.random.choice(all_nodes, (3, ), False)
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    cart = _normalize(np.random.random((n_nodes, 3)))
    cart_warped = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher_warped = _calc_spher_coords(cart_warped, maps)
    coords_list = [_calc_spher_coords(cart, np.ones((n_nodes, )) * _)
                   for _ in range(3)]
    ds1 = np.random.random((n_timepoints, n_nodes))
    norm = np.linalg.norm(ds1, axis=0)[np.newaxis, :]
    ds1 /= np.tile(norm, (n_timepoints, 1))
    ds2 = np.random.random((n_timepoints, n_nodes))
    norm = np.linalg.norm(ds2, axis=0)[np.newaxis, :]
    ds2 /= np.tile(norm, (n_timepoints, 1))

    S, dS_dphi, dS_dtheta = _calc_correlation_cost(
        ds1, ds2, coords_list, maps, spher_warped, nbrs, num_nbrs, res)

    dS_dphi2, dS_dtheta2 = np.zeros((n_nodes, )), np.zeros((n_nodes, ))
    spher_warped2 = spher_warped.copy()
    delta = 1e-8
    for i in range(n_nodes):
        spher_warped2[i, 0] += delta
        S2 = _calc_correlation_cost(
            ds1, ds2, coords_list, maps, spher_warped2, nbrs, num_nbrs, res,
            compute_derivatives=False)
        dS_dphi2[i] = (S2 - S) / delta
        spher_warped2[i, 0] = spher_warped[i, 0]
        spher_warped2[i, 1] += delta
        S2 = _calc_correlation_cost(
            ds1, ds2, coords_list, maps, spher_warped2, nbrs, num_nbrs, res,
            compute_derivatives=False)
        dS_dtheta2[i] = (S2 - S) / delta
        spher_warped2[i, 1] = spher_warped[i, 1]
    atol, rtol = 1e-5, 1e-5
    assert_allclose(dS_dphi, dS_dphi2, atol=atol, rtol=rtol)
    assert_allclose(dS_dtheta, dS_dtheta2, atol=atol, rtol=rtol)
