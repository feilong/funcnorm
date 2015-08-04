from nose.tools import assert_true
from numpy.testing import assert_allclose, assert_almost_equal
import numpy as np

from ..blur_dataset import blur_dataset_no_svd
from ..compute_correlation_cost import compute_correlation_cost
from ..compute_spherical_from_cartesian import compute_spherical_from_cartesian

from .test_utils import random_dataset, random_nbrs, random_coords


def test_compute_correlation_cost():
    n_nodes = 300
    n_timepoints = 100
    ds1 = random_dataset(0, n_nodes, n_timepoints)
    ds2 = random_dataset(1, n_nodes, n_timepoints)
    nbrs, total_nbrs = random_nbrs(0, n_nodes)
    cart_coords, spher_coords, coord_maps = random_coords(0, n_nodes)
    coords_list = [compute_spherical_from_cartesian(cart_coords, i+1)
                   for i in range(3)]

    U, s, Vt = np.linalg.svd(ds1, full_matrices=False)
    V1ST = np.tile(s[:, np.newaxis], (1, n_nodes)) * Vt
    W2TU1 = ds2.T.dot(U)
    assert_allclose(W2TU1.dot(V1ST), ds2.T.dot(ds1))
    S = compute_correlation_cost(
        V1ST, W2TU1, coords_list, coord_maps, spher_coords,
        nbrs, total_nbrs, 2, compute_derivatives=False)
    assert_true(0 < S < n_nodes)
    Q = blur_dataset_no_svd(Vt.T, s, coords_list[0], nbrs, total_nbrs, 2)
    ds3 = U.dot(Q)
    idx = np.where(np.linalg.norm(ds3, axis=0))[0]
    S2 = 1 - np.sum((ds3 * ds2)[:, idx], axis=0)
    assert_almost_equal(S2.sum(), S)


def test_compute_correlation_cost_derivatives():
    for seed in range(10):
        print seed,
        _test_compute_correlation_cost_derivatives(seed)
    print 'Finished testing compute_correlation_cost'


def _test_compute_correlation_cost_derivatives(seed=0, atol=1e-5, rtol=1e-4):
    n_nodes = 300
    n_timepoints = 100
    ds1 = random_dataset(seed, n_nodes, n_timepoints)
    ds2 = random_dataset(1, n_nodes, n_timepoints)
    nbrs, total_nbrs = random_nbrs(seed, n_nodes, n_timepoints)
    cart_coords, spher_coords, coord_maps = random_coords(0, n_nodes)
    coords_list = [compute_spherical_from_cartesian(cart_coords, i+1)
                   for i in range(3)]

    U, s, Vt = np.linalg.svd(ds1, full_matrices=False)
    V1ST = np.tile(s[:, np.newaxis], (1, n_nodes)) * Vt
    W2TU1 = ds2.T.dot(U)
    S, dS_dphi, dS_dtheta = compute_correlation_cost(
        V1ST, W2TU1, coords_list, coord_maps, spher_coords,
        nbrs, total_nbrs, 2)

    dS_dphi2 = np.zeros(dS_dphi.shape)
    dS_dtheta2 = np.zeros(dS_dtheta.shape)

    delta = 1e-8
    coords2 = spher_coords.copy()
    for j in range(n_nodes):
        coords2[0, j] += delta
        S2 = compute_correlation_cost(
            V1ST, W2TU1, coords_list, coord_maps, coords2, nbrs, total_nbrs, 2,
            compute_derivatives=False)
        dS_dphi2[j] = (S2 - S) / delta
        coords2[0, j] = spher_coords[0, j]

        coords2[1, j] += delta
        S2 = compute_correlation_cost(
            V1ST, W2TU1, coords_list, coord_maps, coords2, nbrs, total_nbrs, 2,
            compute_derivatives=False)
        dS_dtheta2[j] = (S2 - S) / delta
        coords2[1, j] = spher_coords[1, j]

    # Q = blur_dataset_no_svd(Vt.T, s, coords, nbrs, total_nbrs, 2)
    # ds3 = U.dot(Q)
    # idx = np.where(np.linalg.norm(ds3, axis=0))[0]
    # for j in idx:
    #     print '%5s,%4d,%10f,%10f' % ('phi', j, dS_dphi[j], dS_dphi2[j])
    #     print '%5s,%4d,%10f,%10f' % ('theta', j, dS_dtheta[j], dS_dtheta2[j])
    assert_allclose(dS_dphi, dS_dphi2, atol=atol, rtol=rtol)
    assert_allclose(dS_dtheta, dS_dtheta2, atol=atol, rtol=rtol)
