"""
% FUNCTION computeNeighborResolutions(nearestNbrs, maxRes)
% This function is compatible with nodes that lie on multiple hemispheres

% This file is part of the Functional Normalization Toolbox,
% (c) 2008 by the authors.
% Please see AUTHORS and LICENSE file in the project root directory
"""
import numpy as np


def compute_neighbor_resolutions(nearest_nbrs, max_res):
    """
    Parameters
    ----------
    nearest_nbrs : (max_nbrs, n_nodes) array
        Each column specifies the node numbers of the closest neighbors
        to a node. Node numbers start with 0.
    max_res : int
        The number of iterations for adding new neighbors.
        If `max_res == 1`, `nbrs` only contains the node itself and its
        original neighbors; If `max_res == 2`, it also contains neighbors'
        neighbors, etc.

    Returns
    -------
    nbrs : (max_nbrs, n_nodes) array
        Similar to `nearest_nbrs`, but contains new neighbors, and `max_nbrs`
        might be larger.
    res_nbr_sizes : (max_res, n_nodes) array
        The number of neighbors added in each resolution.
        `res_nbr_sizes[0, j]` counts the node itself and its original
        neighbors; `res_nbr_sizes[1, j]` counts new neigbhors added by
        including neighbors' neighbors, etc.
    total_nbrs : (n_nodes, ) array
        The total number of neighbors (over all resolutions) for each node.
    """
    max_nbrs, n_nodes = nearest_nbrs.shape

    nbrs = -99 * np.ones((min(200, max_nbrs+1), n_nodes), dtype='int')
    res_nbr_sizes = np.zeros((max_res, n_nodes), dtype='int')
    total_nbrs = np.zeros((n_nodes, ), dtype='int')

    # The start and end indices for the neighbors added in the previous
    # resolution.
    # I.e., the indices are `res_ptrs[0, j]:res_ptrs[1, j]`
    res_ptrs = np.zeros((2, n_nodes), dtype='int')

    # If itself is not a neighbor yet, set it as the first neighbor (0th).
    # What if itself is a neighbor but not the first one?
    for j in range(n_nodes):
        n_nbrs = np.sum(nearest_nbrs[:, j] != -99)
        if nearest_nbrs[0, j] != j:
            n_nbrs += 1
            nbrs[:n_nbrs, j] = [j] + nearest_nbrs[:(n_nbrs-1), j].tolist()
        else:
            nbrs[:n_nbrs, j] = nearest_nbrs[:n_nbrs, j]
        res_nbr_sizes[0, j] = n_nbrs
        res_ptrs[:, j] = [0, n_nbrs]
        total_nbrs[j] += n_nbrs

    for res in range(1, max_res):  # 2:maxRes in matlab
        for j in range(n_nodes):
            prev_start, prev_end = res_ptrs[:, j]
            # Account for the fact that the first element is always itself
            if prev_start == 0:
                prev_start = 1
            prev_nbrs = nbrs[prev_start:prev_end, j]

            # Neighbors of neighbors
            curr_nbrs = nearest_nbrs[:, prev_nbrs]
            # Unique neighbors w/o dummy values
            curr_nbrs = np.setdiff1d(curr_nbrs, [-99])
            # New neighbors
            curr_nbrs = np.setdiff1d(curr_nbrs, nbrs[:prev_end, j])

            new_size = len(curr_nbrs)
            res_nbr_sizes[res, j] = new_size
            total_nbrs[j] += new_size

            new_end = prev_end + new_size
            res_ptrs[:, j] = [prev_end, new_end]

            if nbrs.shape[0] < new_end:
                shape = (new_end-nbrs.shape[0], nbrs.shape[1])
                nbrs = np.vstack([
                    nbrs,
                    -99 * np.ones(shape, 'int')])
            nbrs[prev_end:new_end, j] = curr_nbrs

    max_nbrs = total_nbrs.max()

    nbrs = nbrs[:max_nbrs, :]

    return nbrs, res_nbr_sizes, total_nbrs
