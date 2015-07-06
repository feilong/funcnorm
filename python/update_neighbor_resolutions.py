"""
% FUNCTION [retNbrs, retResNbrSizes, retTotalNbrs] =
%     updateNeighborResolutions(cartCoords, warpCartCoords, nbrs,
%         resNbrSizes, totalNbrs, updNbrs, updResNbrSizes, updTotalNbrs)
% This function is compatible with nodes that lie on multiple hemispheres

% This file is part of the Functional Normalization Toolbox,
% (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""
import numpy as np


def update_neighbor_resolutions(
        cart_coords, warp_cart_coords,
        nbrs, res_nbr_sizes, total_nbrs,
        upd_nbrs, upd_res_nbr_sizes, upd_total_nbrs):
    """
    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    warp_cart_coords : (3, n_nodes) array
    nbrs : (max_nbrs, n_nodes) array
    res_nbr_sizes : (n_res, n_nodes) array
    total_nbrs : (n_nodes, ) array
    upd_nbrs : (max_nbrs, n_nodes) array
    upd_res_nbr_sizes : (n_res, n_nodes) array
    upd_total_nbrs : (n_nodes, ) array

    Returns
    -------
    ret_nbrs : (max_nbrs, n_nodes) array
        Similar to `nbrs`, but column j stores neighbors of the nearest
        neighbor of node j, rather than neighbors of node j.
    ret_res_nbr_sizes : (n_res, n_nodes) array
    ret_total_nbrs : (n_nodes, ) array

    Notes
    -----
    My understanding:
    Assume all nodes are connected together. If node k is not the nearest
    neighbor of node j, then one of node k's neighbors is nearer to node j than
    node k. Therefore local minimal is global minimal.

    Since numpy arrays are mutable, probably we don't need to return anything,
    just modify the array "in place".
    """
    n_nodes = cart_coords.shape[1]

    ret_nbrs = -99 * np.ones(nbrs.shape, dtype='int')
    ret_res_nbr_sizes = np.zeros(res_nbr_sizes.shape, dtype='int')
    ret_total_nbrs = np.zeros(total_nbrs.shape, dtype='int')

    for j in range(n_nodes):
        """
        - Get neighbors of node j from `upd_nbrs`.
        - Label the current nearest neighbor as the first one.
        - In each loop:
            - Calculate nearest neighbor (node k) using projection
            - Label neighbors of node k as neighbors of node j
            - Break the loop if k doesn't change any more

        Are we just trying to find the nearest neighbor of each node?

        ret_* stores information of the nearest neighbor of that node,
        rather than information of the node itself.
        """
        try:  # Why do we need try/catch here?
            ret_nbrs[:, j] = upd_nbrs[:, j]
        except Exception:
            print j, ret_nbrs.shape, upd_nbrs.shape
            raise

        ret_res_nbr_sizes[:, j] = upd_res_nbr_sizes[:, j]
        ret_total_nbrs[j] = upd_total_nbrs[j]

        closest_nbr_pre = ret_nbrs[0, j]

        while True:
            curr_nbrs = ret_nbrs[:ret_total_nbrs[j], j]
            nbrs_cart_coords = cart_coords[:, curr_nbrs]

            projections = warp_cart_coords[:, [j]].T.dot(nbrs_cart_coords)
            I = np.argmax(projections)

            closest_nbr = curr_nbrs[I]
            ret_nbrs[:, j] = nbrs[:, closest_nbr]
            ret_res_nbr_sizes[:, j] = res_nbr_sizes[:, closest_nbr]
            ret_total_nbrs[j] = total_nbrs[closest_nbr]

            if closest_nbr_pre == closest_nbr:
                break
            closest_nbr_pre = closest_nbr

    return ret_nbrs, ret_res_nbr_sizes, ret_total_nbrs
