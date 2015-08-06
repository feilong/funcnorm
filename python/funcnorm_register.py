#!/usr/bin/env python

import numpy as np
import logging
from scipy.optimize import minimize

from .normalize_cartesian_coordinates import normalize_cartesian_coordinates
from .compute_spherical_from_cartesian import compute_spherical_from_cartesian
from .compute_neighbor_resolutions import compute_neighbor_resolutions
from .compute_geodesic_distances import compute_geodesic_distances
from .blur_dataset import blur_dataset_no_svd
from .convert_cartesian_warp_to_spherical_warp import \
    convert_cartesian_warp_to_spherical_warp
from .compute_warp_coords_from_spherical_warp import \
    compute_warp_coords_from_spherical_warp
from .compute_coordinate_maps import compute_coordinate_maps
from .regularization import compute_initial_oriented_areas
from .compute_objective import compute_objective

logger = logging.getLogger('funcnorm')


def funcnorm_register(warp_ds, template_ds, coords, x, regularization,
                      max_resolution):
    """
    Parameters
    ----------
    warp_ds : (n_timepoints, n_nodes) array
        The dataset to be warped.
    template_ds : (n_timepoints, n_nodes) array
        The template dataset.
    coords : dict
        coords['cart_coords'] : (3, n_nodes) array, each column is (x, y, z)
        coords['neighbors'] : (max_nbrs, n_nodes) array, each column is the
            closest neighbors to a node
        coords['triangles'] : (3, n_triangles) array
    x : (3, n_nodes) array
        Initial warp field. The j-th column specifying the (x, y, z) warp for
        the j-th cortical node.
    regularization : dict
        regularization = {'mode': 'metric_and_areal_distortion',
                          'lambda_metric': 1, 'lambda_areal': 1}
        regularization = {'mode': 'none'}
    max_resolution : int?
        The starting resolution of the algorithm (using multi-resolution
        approach).

    Returns
    -------
    warp : array, 3 x n_nodes
        The optimized warp field for `template_ds` (for `warp_ds`?)

    Notes
    -----
    I removed the logfile parameter, because the python logging library should
    be a better alternative.
    n_timepoints was M, n_nodes was N in the Matlab version.

    For SVD, Matlab `U, S, V' = svd(X, 'econ')` (Note the transpose!)
    Python `U, S, V = numpy.linalg.svd(a, full_matrices=False)`
    http://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.svd.html
    http://www.mathworks.com/help/matlab/ref/svd.html

    """
    rho = 1.0  # radius of the sphere

    check_input(warp_ds, template_ds, coords, regularization)

    logger.info('Parsing the datasets...')
    n_timepoints, n_nodes = warp_ds.shape
    # K = min(M, N), should be used in Q's shape instead of M
    K = min(n_timepoints, n_nodes)

    U1, s1, V1t = np.linalg.svd(warp_ds, full_matrix=False)
    del warp_ds
    U2, s2, V2t = np.linalg.svd(template_ds, full_matrix=False)
    del template_ds
    V1, V2 = V1t.T, V2t.T
    dtype = U1.dtype

    # U: (M, K), s: (K,), V: (N, K); k = min(M, N)
    V1S = V1 * np.tile(s1[np.newaxis, :], (n_nodes, 1))
    del V1, s1  # we still keep `U1`
    V2_orig, s2_orig, U2_orig = (V2, s2, U2)
    logger.info('Completed parsing the datasets.')

    logger.info('Parsing the coordinates and neighbors structures...')
    nbrs = coords['neighbors']
    triangles = getattr(coords, 'triangles', None)
    cart_coords = normalize_cartesian_coordinates(coords['cart_coords'], rho)
    del coords
    spher_coords_list = [compute_spherical_from_cartesian(cart_coords, _+1)
                         for _ in range(3)]

    orig_nbrs, orig_num_nbrs = nbrs, np.sum(nbrs != -99, axis=0)
    [nbrs, res_nbr_sizes, total_nbrs] = compute_neighbor_resolutions(
        nbrs, max_resolution)
    max_nbrs = max(total_nbrs)
    logger.info('Completed parsing the coordinates and neighbors structures')

    # Initial metric distances and oriented area terms
    if regularization['mode'] == 'metric_and_areal_distortion':
        logger.info('Using metric and areal distortion terms for '
                    'regularization...')
        if regularization['lambda_metric'] > 0:
            regularization['lambda_metric'] /= 4.0 * n_nodes
            # re-use some code from `compute_metric_terms`?
            logger.info('Computing initial metric distances...')
            md = -99 * np.ones(nbrs.shape, dtype=dtype)
            for j in range(n_nodes):
                curr_cart_coords = cart_coords[:, [j]]
                n_nbrs = orig_num_nbrs[j]
                curr_nbrs = orig_nbrs[:n_nbrs, j]
                nbr_cart_coords = cart_coords[:, curr_nbrs]
                md[:n_nbrs, j] = compute_geodesic_distances(
                    curr_cart_coords, nbr_cart_coords)
            regularization['metric_distances'] = md
            logger.info('Completed computing initial metric distances!')

        if regularization['lambda_areal'] > 0:
            logger.info('Computing initial oriented area terms...')
            regularization['triangles'] = triangles
            n_triangles = regularization['triangles'].shape[1]
            regularization['lambda_areal'] /= 2.0 * n_triangles
            (regularization['oriented_areas'],
             regularization['oriented_normals']) = \
                compute_initial_oriented_areas(
                    regularization['triangles'], cart_coords)
            logger.info('Completed computing initial oriented area terms.')
    elif regularization['mode'] == 'none':
        logger.info('No regularization used.')
    del orig_nbrs, orig_num_nbrs

    # opt_method = 'steepest_descent'
    # logger.info('Using %s as optimization technique.' % opt_method)

    upd_nbrs = nbrs.copy()
    upd_res_nbr_sizes = res_nbr_sizes.copy()
    upd_total_nbrs = total_nbrs.copy()

    warp_cart_coords = cart_coords + x
    coord_maps = compute_coordinate_maps(cart_coords, warp_cart_coords)
    x = convert_cartesian_warp_to_spherical_warp(cart_coords, x, coord_maps)
    x = x.ravel()

    for res in range(max_resolution, 0, -1):
        logger.info('Beginning pass at resolution # %d' % res)
        # Q = V1S.T
        V1ST = V1S.T

        if res >= 4:
            tol = 1e-1
        elif res == 3:
            tol = 1e-2
        elif res in [2, 1]:
            tol = 1e-3

        W2T = blur_dataset_no_svd(V2_orig, s2_orig, U2_orig, cart_coords,
                                  nbrs, total_nbrs, res)
        logger.info('Completed blurring the template dataset at resolution #%d'
                    % res)
        W2TU1 = W2T.dot(U1)
        warp_cart_coords = compute_warp_coords_from_spherical_warp(
            cart_coords, x.reshape((2, n_nodes)), coord_maps)
        coord_maps = compute_coordinate_maps(cart_coords, warp_cart_coords)
        x = x.ravel()

        # "fminunc" here.
        method = 'BFGS'
        max_iter = 300
        callback_fn = None  # TODO: iteration number
        # max_fun_evals = 500
        output = minimize(compute_objective, x0=x, method=method, jac=True,
                          tol=tol, callback=callback_fn,
                          args=(cart_coords, coord_maps, spher_coords_list,
                                nbrs, res_nbr_sizes, total_nbrs,
                                upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs,
                                K, res, regularization, V1ST, W2TU1),
                          options={'max_iter': max_iter})
        logger.info("Results of optimization at resolution #%d" % res)
        logger.info("Number of iterations: %d" % output.nit)
        logger.info("Number of function evaluations: %d" % output.nfev)
        logger.info("Algorithm used: %s" % method)
        logger.info("Success: %s" % output.success)
        if not output.success:
            logger.warn("Error Message: %s" % output.message)

        total_nbrs -= res_nbr_sizes[-1, :]
        res_nbr_sizes = res_nbr_sizes[:-1, :]
        upd_total_nbrs -= upd_res_nbr_sizes[-1, :]
        upd_res_nbr_sizes = upd_res_nbr_sizes[:-1, :]
        max_nbrs = max(total_nbrs.max(), upd_total_nbrs.max())
        nbrs = nbrs[:max_nbrs, :]
        upd_nbrs = upd_nbrs[:max_nbrs, :]

    x = x.reshape((2, n_nodes))
    warp_cart_coords = compute_warp_coords_from_spherical_warp(
        cart_coords, x, coord_maps)
    warp = warp_cart_coords - cart_coords
    return warp


def check_input(warp_ds, template_ds, coords, regularization):
    if warp_ds.shape != template_ds.shape:
        msg = 'The two datasets must have the same shape. Currently warp_ds is %s, '\
              'and template_ds is %s' % (warp_ds.shape, template_ds.shape)
        raise ValueError(msg)
    n_nodes = warp_ds.shape[1]

    for key in ['neighbors', 'cart_coords']:
        if key not in coords:
            raise KeyError("Key %r not found in dict coords." % key)
    if coords['cart_coords'].shape != (3, n_nodes):
        msg = "Improper shape of coords['cart_coords'], shape is %s, should be %s" % \
              (coords['cart_coords'].shape, (3, n_nodes))
        raise ValueError(msg)
    has_triangles = ('triangles' in coords)

    if regularization['mode'] not in ['metric_and_areal_distortion', 'none']:
        msg = "regularization['mode'] must be in " \
              "['metric_and_areal_distortion', 'none']"
        raise ValueError(msg)
    if regularization['mode'] == 'metric_and_areal_distortion':
        if 'lambda_metric' not in regularization:
            raise KeyError("regularization must contains key 'lambda_metric'.")
        if 'lambda_areal' not in regularization:
            raise KeyError("regularization must contains key 'lambda_areal'.")
        if regularization['lambda_areal'] > 0 and not has_triangles:
            raise KeyError("In order to compute areal distortion terms, "
                           "`coords` must have key 'triangles'.")
