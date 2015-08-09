import numpy as np
from nose.tools import assert_equal
from numpy.testing import assert_allclose, assert_array_equal

from .utils import random_dataset, random_cart, random_triangles

from ..Cost_Function import cost_func
from ..Coordinates import _calc_spher_coords, _normalize, _calc_cart_warped_from_spher_warp
from ..Surface import Surface, _calc_nbrs
from ..Metric import _calc_metric_dist
from ..Folding import _calc_oriented_areas

from ..Interpolation import _calc_correlation_cost
from ..Metric import _calc_metric_terms
from ..Folding import _calc_areal_terms


def test_cost_func():
    n_triangles = 3000
    n_timepoints = 100
    n_nodes = 300
    res = 2
    cart = random_cart(n_nodes)
    triangles, nbrs, num_nbrs = random_triangles(n_nodes, n_triangles)
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher = _calc_spher_coords(cart, maps)
    cart_warped = random_cart(n_nodes)
    spher_warped = _calc_spher_coords(cart_warped, maps)
    spher_warp = spher_warped - spher
    surf = Surface(cart, nbrs, triangles)
    surf.calc_coord_maps()
    surf._calc_spher_coords()
    surf.calc_nbr_res(res)
    surf.init_upd_nbr_res()
    ds1 = random_dataset(n_timepoints, n_nodes)
    ds2 = random_dataset(n_timepoints, n_nodes)

    orig_md = _calc_metric_dist(cart, surf.nbrs, num_nbrs)
    tri_areas, tri_normals = _calc_oriented_areas(triangles, cart)
    f, g = cost_func(spher_warp, surf, res, 1.0, 1.0,
                     orig_md, tri_areas, tri_normals,
                     ds1, ds2)


def test_cost_func_derivatives():
    n_triangles = 300
    n_timepoints = 30
    n_nodes = 100
    res = 6
    lambda_areal = 0.0
    lambda_metric = 0.0
    cart = random_cart(n_nodes)
    triangles, nbrs, num_nbrs = random_triangles(n_nodes, n_triangles)
    maps = np.random.choice(range(3), (n_nodes, ), True)
    spher = _calc_spher_coords(cart, maps)
    cart_warped = random_cart(n_nodes)
    spher_warped = _calc_spher_coords(cart_warped, maps)
    spher_warp = spher_warped - spher
    surf = Surface(cart, nbrs, triangles)
    surf.calc_coord_maps()
    surf._calc_spher_coords()
    surf.calc_nbr_res(res)
    surf.init_upd_nbr_res()
    ds1 = random_dataset(n_timepoints, n_nodes)
    ds2 = random_dataset(n_timepoints, n_nodes)
    orig_md = _calc_metric_dist(cart, surf.nbrs, num_nbrs)
    tri_areas, tri_normals = _calc_oriented_areas(triangles, cart)

    # surf.cart_warped = _calc_cart_warped_from_spher_warp(
    #     surf.cart, spher_warp, surf.maps, surf.spher)
    # surf.calc_coord_maps()
    # surf.update_nbr_res()
    # surf.spher_warped = _calc_spher_coords(surf.cart_warped, surf.maps)
    # surf.calc_coords_list()

    # fc, g_phi, g_theta =_calc_correlation_cost(
    #         ds1, ds2, surf.coords_list, surf.maps, surf.spher_warped,
    #         surf.nbrs, surf.num_nbrs, res)
    maps = surf.maps.copy()
    f, g = cost_func(spher_warp, surf, res, lambda_metric, lambda_areal,
                     orig_md, tri_areas, tri_normals,
                     ds1, ds2)

    g2 = np.zeros((n_nodes, 2))
    spher_warp2 = spher_warp.copy()
    delta = 1e-8
    for i in range(n_nodes):
        for j in range(2):
            surf.maps = maps
            spher_warp2[i, j] += delta
            f2 = cost_func(spher_warp2, surf, res, lambda_metric, lambda_areal,
                           orig_md, tri_areas, tri_normals,
                           ds1, ds2, compute_g=False)
            g2[i, j] = (f2 - f) / delta
            spher_warp2[i, j] = spher_warp[i, j]
    atol, rtol = 1e-5, 1e-5
    idx = np.where(maps==surf.maps)[0]
    idx2 = np.where(maps!=surf.maps)[0]
    print idx
    print idx.shape
    print g[idx, :].shape
    assert_allclose(g[idx, :], g2[idx, :], atol=atol, rtol=rtol)
    assert_allclose(g[idx2, :], g2[idx2, :], atol=atol, rtol=rtol)


    # coords_list = surf.coords_list
    # maps = surf.maps
    # spher_warped = surf.spher_warped
    # nbrs = surf.nbrs
    # num_nbrs = surf.num_nbrs

    # S, dS_dphi, dS_dtheta = _calc_correlation_cost(
    #     ds1, ds2, coords_list, maps, spher_warped, nbrs, num_nbrs, res)

    # dS_dphi2, dS_dtheta2 = np.zeros((n_nodes, )), np.zeros((n_nodes, ))
    # spher_warped2 = spher_warped.copy()
    # delta = 1e-8
    # for i in range(n_nodes):
    #     spher_warped2[i, 0] += delta
    #     S2 = _calc_correlation_cost(
    #         ds1, ds2, coords_list, maps, spher_warped2, nbrs, num_nbrs, res,
    #         compute_derivatives=False)
    #     dS_dphi2[i] = (S2 - S) / delta
    #     spher_warped2[i, 0] = spher_warped[i, 0]
    #     spher_warped2[i, 1] += delta
    #     S2 = _calc_correlation_cost(
    #         ds1, ds2, coords_list, maps, spher_warped2, nbrs, num_nbrs, res,
    #         compute_derivatives=False)
    #     dS_dtheta2[i] = (S2 - S) / delta
    #     spher_warped2[i, 1] = spher_warped[i, 1]
    # atol, rtol = 1e-5, 1e-5
    # assert_allclose(dS_dphi, dS_dphi2, atol=atol, rtol=rtol)
    # assert_allclose(dS_dtheta, dS_dtheta2, atol=atol, rtol=rtol)
