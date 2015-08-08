import numpy as np
from numpy.testing import assert_allclose

from ..Metric import _calc_metric_terms, _calc_metric_dist, _dgds_dspher
from ..Coordinates import _normalize, _calc_spher_coords, _calc_cart_coords, \
    _calc_nbrs
from ..utils import _calc_geodesic_dist

import os
from scipy.io import loadmat
DIR = os.path.dirname(os.path.realpath(__file__))


def _test_metric():
    n_nodes = 1000
    max_nbrs = 6
    nbrs = -99 * np.ones((n_nodes, max_nbrs), dtype='int')
    num_nbrs = np.zeros((n_nodes, ))
    for j in range(n_nodes):
        n_nbrs = np.random.randint(1, max_nbrs+1)
        nbrs[j, :n_nbrs] = np.random.choice(range(n_nodes), (n_nbrs, ), False)
        num_nbrs[j] = n_nbrs
    cart = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)

    orig_md = _calc_metric_dist(cart, nbrs, num_nbrs)
    M = _calc_metric_terms(nbrs, cart, maps, orig_md, False)
    np.testing.assert_equal(M, 0)


def test_gds_derivatives():
    n_nodes = 100
    spher1 = _calc_spher_coords(np.random.random((n_nodes, 3)),
                                np.zeros((n_nodes, )))
    spher2 = _calc_spher_coords(np.random.random((n_nodes, 3)),
                                np.zeros((n_nodes, )))
    gds = _calc_geodesic_dist(spher1, spher2)
    dg_dphi, dg_dtheta = _dgds_dspher(spher1, spher2, gds)

    sph = spher1.copy()
    dg_dphi2, dg_dtheta2 = np.zeros((n_nodes, )), np.zeros((n_nodes, ))
    delta = 1e-8
    for i in range(n_nodes):
        sph[i, 0] += delta
        dg_dphi2[i] = (_calc_geodesic_dist(sph, spher2)[i] - gds[i]) / delta
        sph[i, 0] = spher1[i, 0]
        sph[i, 1] += delta
        dg_dtheta2[i] = (_calc_geodesic_dist(sph, spher2)[i] - gds[i]) / delta
        sph[i, 1] = spher1[i, 1]
    atol, rtol = 1e-6, 1e-6
    assert_allclose(dg_dphi, dg_dphi2, atol=atol, rtol=rtol)
    assert_allclose(dg_dtheta, dg_dtheta2, atol=atol, rtol=rtol)


def test_gds_derivatives2():
    fname = os.path.join(DIR, 'data_for_test_derivatives.mat')
    mat = loadmat(fname)
    coords1, coords2 = mat['spher_coords_1'].T, mat['spher_coords_2'].T
    gds = mat['gds'].ravel()

    dg_dphi, dg_dtheta = _dgds_dspher(coords1, coords2, gds)

    assert_allclose(dg_dphi, mat['dg_dphi'].ravel())
    assert_allclose(dg_dtheta, mat['dg_dtheta'].ravel())


def _test_compute_metric_terms():
    fname = os.path.join(DIR, 'data_for_test_compute_metric_terms.mat')
    mat = loadmat(fname)
    md_diff, dmd_diffs_dphi, dmd_diffs_dtheta = _calc_metric_terms(
        mat['orig_nbrs'].T, mat['cart_coords'].T, mat['coord_maps'].ravel()-1,
        mat['orig_metric_distances'].T, compute_derivatives=True)
    # assert_almost_equal(md_diff, float(mat['md_diff']))
    assert_allclose(dmd_diffs_dphi, mat['dmd_diffs_dphi'].ravel())
    assert_allclose(dmd_diffs_dtheta, mat['dmd_diffs_dtheta'].ravel())


def test_metric_derivatives():
    print ''
    np.random.seed(0)
    n_triangles = 300
    n_nodes = n_triangles * 3
    triangles = np.random.choice(
        range(3 * n_triangles), (n_triangles, 3), replace=False)
    nbrs, num_nbrs = _calc_nbrs(triangles, n_nodes)
    cart = _normalize(np.random.random((n_nodes, 3)))
    maps = np.random.choice(range(3), (n_nodes, ), True)

    cart2 = _normalize(np.random.random((n_nodes, 3)))
    orig_md = _calc_metric_dist(cart2, nbrs, num_nbrs)

    M, dM_dphi, dM_dtheta = _calc_metric_terms(nbrs, cart, maps, orig_md)

    delta = 1e-8
    spher = _calc_spher_coords(cart, maps)
    cart2 = _calc_cart_coords(spher, maps)
    assert_allclose(cart, cart2)
    spher2 = spher.copy()

    dM_dphi2, dM_dtheta2 = np.zeros((n_nodes, )), np.zeros((n_nodes, ))
    for i in range(n_nodes):
        spher2[i, 0] += delta
        cart2 = _calc_cart_coords(spher2, maps)
        md = _calc_metric_dist(cart2, nbrs, num_nbrs)
        M2 = np.sum((md - orig_md)**2)
        dM_dphi2[i] = (M2 - M) / delta
        spher2[i, 0] = spher[i, 0]
        spher2[i, 1] += delta
        cart2 = _calc_cart_coords(spher2, maps)
        M2 = _calc_metric_terms(nbrs, cart2, maps, orig_md, False)
        dM_dtheta2[i] = (M2 - M) / delta
        spher2[i, 1] = spher[i, 1]
    atol, rtol = 1e-5, 1e-5
    assert_allclose(dM_dphi, dM_dphi2, atol=atol, rtol=rtol)
    assert_allclose(dM_dtheta, dM_dtheta2, atol=atol, rtol=rtol)
