import numpy as np

from .Coordinates import _calc_cart_warped_from_spher_warp, _calc_spher_coords
from .Interpolation import _calc_correlation_cost
from .Metric import _calc_metric_terms
from .Folding import _calc_areal_terms


def cost_func(spher_warp, surf, res,
              lambda_metric, lambda_areal,
              ds1, ds2, compute_g=True, dtype='float'):
    spher_warp = spher_warp.reshape((surf.n_nodes, 2)).astype(dtype)
    surf.cart_warped = _calc_cart_warped_from_spher_warp(
        surf.cart, spher_warp, surf.maps, surf.spher)
    surf.calc_coord_maps()
    surf.update_nbr_res()
    surf.spher_warped = _calc_spher_coords(surf.cart_warped, surf.maps)
    surf.calc_coords_list()
    if compute_g:
        g = np.zeros((surf.n_nodes, 2), dtype=dtype)

    multi_intersubj = 1.0
    returns = _calc_correlation_cost(
        ds1, ds2, surf.coords_list, surf.maps, surf.spher_warped,
        surf.nbrs, surf.num_nbrs, res, compute_derivatives=compute_g)
    if compute_g:
        S, dS_dphi, dS_dtheta = returns
        g[:, 0] += multi_intersubj * dS_dphi
        g[:, 1] += multi_intersubj * dS_dtheta
    else:
        S = returns
    f = multi_intersubj * S

    if lambda_metric > 0:
        returns = _calc_metric_terms(
            surf.nbrs, surf.cart_warped, surf.maps,
            surf.orig_md, compute_derivatives=compute_g)
        if compute_g:
            M, dM_dphi, dM_dtheta = returns
            g[:, 0] += lambda_metric * dM_dphi
            g[:, 1] += lambda_metric * dM_dtheta
        else:
            M = returns
        f += lambda_metric * M

    if lambda_areal > 0:
        returns = _calc_areal_terms(
            surf.triangles, surf.cart_warped, surf.maps,
            surf.tri_areas, surf.tri_normals, compute_derivatives=compute_g)
        if compute_g:
            areal, dareal_dphi, dareal_dtheta = returns
            g[:, 0] += lambda_areal * dareal_dphi
            g[:, 1] += lambda_areal * dareal_dtheta
        else:
            areal = returns
        f += lambda_areal * areal

    if compute_g:
        return f, g.ravel()
    return f
