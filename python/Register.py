# import numpy as np
import logging
from scipy.optimize import minimize

from .Cost_Function import cost_func
from .Interpolation import _blur_dataset_full

logger = logging.getLogger('funcnorm')


def funcnorm_register(warp_ds, template_ds, surf, cart_warp,
                      lambda_metric, lambda_areal, max_res):
    n_timepoints, n_nodes = warp_ds.shape
    # K = min(n_timepoints, n_nodes)

    surf.normalize_cart()
    surf.calc_coords_list()
    surf.orig_nbrs, surf.orig_num_nbrs = surf.nbrs.copy(), surf.num_nbrs.copy()
    surf.calc_nbr_res(max_res)

    if lambda_metric > 0:
        lambda_metric /= 4.0 * surf.n_nodes
        surf.init_metric(warp_ds.dtype)

    if lambda_areal > 0:
        lambda_areal /= 2.0 * surf.n_triangles
        surf.init_areal()

    surf.init_upd_nbr_res()
    # surf.cart_warp = cart_warp
    surf.cart_warped = surf.cart + cart_warp
    surf.calc_coord_maps()
    # spher_warp = _calc_spher_warp_from_cart_warp(
    #     surf.cart, cart_warp, surf.maps, surf.spher)
    spher_warp = surf.calc_spher_warp(cart_warp)

    for res in range(max_res, 0, -1):
        logger.info('Beginning pass at resolution # %d' % res)
        if res >= 4:
            tol = 1e-1
        elif res == 3:
            tol = 1e-2
        elif res in [2, 1]:
            tol = 1e-3
        ds2 = _blur_dataset_full(template_ds, surf.cart, surf.nbrs,
                                 surf.num_nbrs, res)
        logger.info('Completed blurring the template dataset at resolution #%d'
                    % res)
        # surf.cart_warped = _calc_cart_warped_from_spher_warp(
        #     surf.cart, spher_warp, surf.maps, surf.spher)
        surf.calc_cart_warped(spher_warp)

        method = 'BFGS'
        maxiter = 300
        callback_fn = None  # TODO: iteration number
        # max_fun_evals = 500
        output = minimize(cost_func, x0=spher_warp.ravel(), method=method,
                          jac=True, tol=tol, callback=callback_fn,
                          args=(surf, res, lambda_metric, lambda_areal,
                                warp_ds, ds2),
                          options={'maxiter': maxiter, 'disp': True})
        logger.info("Results of optimization at resolution #%d" % res)
        logger.info("Number of iterations: %d" % output.nit)
        logger.info("Number of function evaluations: %d" % output.nfev)
        logger.info("Algorithm used: %s" % method)
        logger.info("Success: %s" % output.success)
        if not output.success:
            logger.warn("Error Message: %s" % output.message)

        surf.trim_nbrs()

        spher_warp = output.x.reshape((surf.n_nodes, 2))

    # surf.cart_warped = _calc_cart_warped_from_spher_warp(
    #     surf.cart, spher_warp, surf.maps, surf.spher)
    surf.calc_cart_warped(spher_warp)
    cart_warp = surf.cart_warped - surf.cart

    return cart_warp
