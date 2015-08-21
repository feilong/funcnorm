import numpy as np
import logging

from .Coordinates import _calc_cart_warped_from_spher_warp, _calc_spher_coords
from .Interpolation import _calc_correlation_cost
from .Metric import _calc_metric_terms
from .Folding import _calc_areal_terms

logger = logging.getLogger('funcnorm')


def cost_func(spher_warp, surf, res,
              lambda_metric, lambda_areal,
              ds1, ds2, compute_g=True, dtype='float'):
    logger.debug("Calculating warped coordinates.")
    spher_warp = spher_warp.reshape((surf.n_nodes, 2)).astype(dtype)
    surf.cart_warped = _calc_cart_warped_from_spher_warp(
        surf.cart, spher_warp, surf.maps, surf.spher)
    surf.calc_coord_maps()
    surf.update_nbr_res()
    surf.spher_warped = _calc_spher_coords(surf.cart_warped, surf.maps)
    surf.calc_coords_list()
    logger.debug("Finished calculating warped coordinates.")

    if compute_g:
        g = np.zeros((surf.n_nodes, 2), dtype=dtype)

    logger.debug("Calculating correlations across subjects.")
    multi_intersubj = 1.0
    returns = _calc_correlation_cost(
        ds1, ds2, surf.coords_list, surf.maps, surf.spher_warped,
        surf.nbrs, surf.num_nbrs, res, compute_derivatives=compute_g)
    if compute_g:
        S, corrs, dS_dphi, dS_dtheta = returns
        g[:, 0] += multi_intersubj * dS_dphi
        g[:, 1] += multi_intersubj * dS_dtheta
    else:
        S, corrs = returns
    f = multi_intersubj * S
    logger.debug("Finished calculating correlations across subjects.")

    logger.debug("Calculating metric distortion terms.")
    if lambda_metric > 0:
        returns = _calc_metric_terms(
            surf.orig_nbrs, surf.cart_warped, surf.maps,
            surf.orig_md, compute_derivatives=compute_g)
        if compute_g:
            M, dM_dphi, dM_dtheta = returns
            g[:, 0] += lambda_metric * dM_dphi
            g[:, 1] += lambda_metric * dM_dtheta
        else:
            M = returns
        f += lambda_metric * M
    logger.debug("Finished calculating metric distortion terms.")

    logger.debug("Calculating folding (areal) terms.")
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
    logger.debug("Finished calculating folding (areal) terms.")

    logger.info("Avg Corr: %7.5f, Avg R^2: %7.5f, Corr Cost: %5.3f, "
                "Metric: %5.3f, Areal: %5.3f, Total Cost: %6.3f" % (
                    np.mean(corrs), np.mean(np.array(corrs)**2), S * multi_intersubj,
                    M * lambda_metric, areal * lambda_areal, f))

    if compute_g:
        return f, g.ravel()
    return f
