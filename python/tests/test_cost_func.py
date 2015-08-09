import numpy as np
from numpy.testing import assert_allclose

from .utils import random_dataset, random_cart, random_triangles

from ..Cost_Function import cost_func
from ..Coordinates import _calc_spher_coords
from ..Surface import Surface
from ..Metric import _calc_metric_dist
from ..Folding import _calc_oriented_areas


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
    """
    Notes
    -----
    This part is tricky. Coordinate maps would change during the computation,
    and the derivatives are calculated based on the new `maps`.  However,
    derivatives of the old `maps` are obtained by approximation.  Here I only
    use the nodes whose Coordinate map didn't change during the computation.
    That would be roughly 1/3 of the nodes, and those nodes are enough to test
    if the function is working properly.
    """
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
    idx = np.where(maps == surf.maps)[0]
    assert_allclose(g[idx, :], g2[idx, :], atol=atol, rtol=rtol)
