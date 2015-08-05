#!/usr/bin/env python

import numpy as np
from scipy import sparse

from compute_warp_coords_from_spherical_warp import \
    compute_warp_coords_from_spherical_warp
from compute_coordinate_maps import compute_coordinate_maps
from compute_spherical_from_cartesian import compute_spherical_from_cartesian
# from compute_geodesic_distances import compute_geodesic_distances
from update_neighbor_resolutions import update_neighbor_resolutions
# from derivatives import f_derivatives, gds_derivatives
# from interp_f import gds_to_interp_vals
from compute_correlation_cost import compute_correlation_cost
from regularization import compute_areal_terms, compute_metric_terms


def compute_objective(x, cart_coords, coord_maps, spher_coords_list,
                      nbrs, res_nbr_sizes, total_nbrs,
                      upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs,
                      K, res, regularization,
                      V1ST, W2TU1, rho=1.0,
                      dtype='float', compute_g=True, compute_H=False):
    """
    Parameters
    ----------
    x : array
        Will be reshaped into (2, n_nodes), specifies the current
        (dphi, dtheta) warp for each voxel.

    Returns
    -------
    f : float
    g : (2 x n_nodes) array
    H : (2 x n_nodes, 2 x n_nodes) sparse matrix

    Notes
    -----
    - `orig_num_nbrs` removed from arguments.
    - use `rho` in some functions?

    - Apply the current warp to `cart_coords` to get `warp_cart_coords`.
    - Calculate the optimized `coord_maps` based on old and new coordinates.
    - Find neighbors of the nearest neighbor of each node.
    """
    x = x.reshape((2, -1)).astype(dtype)
    n_nodes = x.shape[1]

    warp_cart_coords = compute_warp_coords_from_spherical_warp(
        cart_coords, x, coord_maps)
    coord_maps = compute_coordinate_maps(cart_coords, warp_cart_coords)
    upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs = \
        update_neighbor_resolutions(
            cart_coords, warp_cart_coords, nbrs, res_nbr_sizes, total_nbrs,
            upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs)
    warp_spher_coords = compute_spherical_from_cartesian(
        warp_cart_coords, coord_maps)

    if compute_g:
        g = np.zeros((2 * n_nodes, ), dtype=dtype)  # was (2*n_nodes, 1)
    if compute_H:  # No further computations for H
        H = sparse.identity(2 * n_nodes)

    multi_intersubj = 1.0

    returns = compute_correlation_cost(
        V1ST, W2TU1, spher_coords_list, coord_maps, warp_spher_coords,
        upd_nbrs, upd_total_nbrs, res, thr=1e-10,
        compute_derivatives=compute_g)
    if compute_g:
        S, dS_dphi, dS_dtheta = returns
        g[::2] += multi_intersubj * dS_dphi
        g[1::2] += multi_intersubj * dS_dtheta
    else:
        S = returns
    f = multi_intersubj * S

    if (regularization['mode'] == 'metric_and_areal_distortion'
            and regularization['lambda_metric'] > 0):
        returns = compute_metric_terms(
            nbrs, warp_cart_coords, coord_maps,
            regularization['metric_distances'], res, rho,
            compute_g)
        if compute_g:
            md_diff, dmd_diffs_dphi, dmd_diffs_dtheta = returns
            factor = 4 * regularization['lambda_metric']
            g[::2] += factor * dmd_diffs_dphi
            g[1::2] += factor * dmd_diffs_dtheta
        else:
            md_diff = returns
        f += regularization['lambda_metric'] * md_diff

    if (regularization['mode'] == 'metric_and_areal_distortion'
            and regularization['lambda_areal'] > 0):
        returns = compute_areal_terms(
            regularization['triangles'], warp_cart_coords, coord_maps,
            regularization['oriented_areas'],
            regularization['oriented_normals'], compute_g)
        if compute_g:
            tri_area, dareal_dphi, dareal_dtheta = returns
            g[::2] += 2 * regularization['lambda_areal'] * dareal_dphi
            g[1::2] += 2 * regularization['lambda_areal'] * dareal_dtheta
        else:
            tri_area = returns
        f += regularization['lambda_areal'] * tri_area

    if compute_g:
        if compute_H:
            return f, g, H
        return f, g
    return f

    # Old code for computing correlation cost as part of the cost function.
    # Hasn't been tested yet.

    # Q = np.zeros((K, n_nodes), dtype=dtype)

    # if compute_g:
    #     dAD_dphi = np.zeros(nbrs.shape, dtype=dtype)
    #     dAD_dtheta = np.zeros(nbrs.shape, dtype=dtype)
    # alocs = -99 * np.ones(nbrs.shape, dtype='int')
    # locs_length = np.zeros((n_nodes, ), dtype='int')  # was (n_nodes, 1)

    # for j in range(n_nodes):
    #     curr_coord_map = coord_maps[j]
    #     spher_coords = spher_coords_list[curr_coord_map]
    #     curr_warp_spher_coords = warp_spher_coords[:, j]
    #     curr_nbrs = upd_nbrs[:upd_total_nbrs[j], j]
    #     nbr_spher_coords = spher_coords[:, curr_nbrs + 1]

    #     gds = compute_geodesic_distances(
    #         curr_warp_spher_coords, nbr_spher_coords)
    #     A, non_zero_locs = gds_to_interp_vals(gds, res, dtype='float')
    #     A = A[non_zero_locs]
    #     curr_nbrs = curr_nbrs(non_zero_locs)
    #     nbr_spher_coords = nbr_spher_coords[:, non_zero_locs]
    #     gds = gds[non_zero_locs]

    #     curr_length = len(curr_nbrs)
    #     locs_length[j] = curr_length
    #     alocs[:curr_length, j] = curr_nbrs

    #     Q[:, j] = V1ST[:, curr_nbrs].dot(A)
    #     qnorm = np.linalg.norm(Q[:, j])
    #     D = 1.0 / qnorm if qnorm > 1e-10 else 0
    #     Q[:, j] *= D

    #     if compute_g:
    #         dA_dphi, dA_dtheta = f_derivatives(
    #             curr_warp_spher_coords, nbr_spher_coords, res, gds)

    #         Q_V1S = Q[:, j].dot(V1ST[:, curr_nbrs])

    #         dD_dphi = D**2 * Q_V1S.dot(dA_dphi)
    #         dD_dtheta = D**2 * Q_V1S.dot(dA_dtheta)

    #         dAD_dphi[:curr_length, j] = D * dA_dphi - dD_dphi.dot(A)
    #         dAD_dtheta[:curr_length, j] = D * dA_dtheta - dD_dtheta.dot(A)

    # for j in range(n_nodes):
    #     f += multi_intersubj * (1 - W2TU1[j, :].dot(Q[:, j]))
    # if compute_g:
    #     count = 0
    #     for p in range(n_nodes):
    #         loc_length = locs_length[p]
    #         locs = alocs[:loc_length, p]

    #         dS_dAD = W2TU1[p, :].dot(V1ST[:, locs])
    #         g[count] -= multi_intersubj * \
    #                     dS_dAD.dot(dAD_dphi[:loc_length, p])
    #         g[count+1] -= multi_intersubj * \
    #             dS_dAD.dot(dAD_dtheta[:loc_length, p])
    #         count += 2

    #         Old metric_and_areal_distortion part

    #         Why compute curr_warp_spher_coords and nbr_warp_spher_coords in
    #         different ways?

    #         count = 0
    #         for j in range(n_nodes):
    #             # curr_warp_cart_coords = warp_cart_coords[:, j]
    #             curr_num_nbrs = orig_num_nbrs[j]
    #             curr_nbrs = nbrs[:curr_num_nbrs, j]
    #             nbr_warp_cart_coords = warp_cart_coords[:, curr_nbrs]

    #             curr_init_MD = \
    #                 regularization['metric_distances'][:curr_num_nbrs, j]

    #             curr_coord_map = coord_maps[:, j]
    #             spher_coords = spher_coords_list[curr_coord_map]
    #             curr_warp_spher_coords = spher_coords[:, j] + x[:, j]
    #             nbr_warp_spher_coords = compute_spherical_from_cartesian(
    #                 nbr_warp_cart_coords, curr_coord_map)
    #             curr_MD = compute_geodesic_distances(curr_warp_spher_coords,
    #                                                  nbr_warp_spher_coords)
    #             md_diffs = curr_MD - curr_init_MD

    #             f += regularization['lambda_metric'] * (md_diffs**2).sum()

    #             if compute_g:
    #                 dg_dphi, dg_dtheta = gds_derivatives(
    #                     curr_warp_spher_coords, nbr_warp_spher_coords, res,
    #                     gds)
    #                 g[count] += 4 * regularization['lambda_metric'] * \
    #                     dg_dphi.dot(md_diffs)
    #                 g[count + 1] += 4 * regularization['lambda_metric'] * \
    #                     dg_dtheta.dot(md_diffs)
    #                 count += 2
