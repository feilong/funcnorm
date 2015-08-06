import numpy as np
import logging
from compute_geodesic_distances import compute_geodesic_distances
from derivatives import f_derivatives
from interp_f import gds_to_interp_vals

logger = logging.getLogger('funcnorm')


def compute_correlation_cost(V1ST, W2TU1,
                             coords_list, coord_maps, warp_coords,
                             nbrs, total_nbrs, resolution,
                             thr=1e-8, dtype='float',
                             compute_derivatives=True):
    """
    Parameters
    ----------
    V1ST: (K, n_nodes) array
    W2TU1: (n_nodes, K) array
    coords_list : list of three (2, n_nodes) array
    coord_maps : (n_nodes, ) array or alike
    warp_coords : (2, n_nodes) array
    nbrs : (max_nbrs, n_nodes) array
    total_nbrs : (n_nodes, ) array
    resolution : float?
    thr : float

    Returns
    -------
    S : float
    dS_dphi : (n_nodes, ) array
    dS_dtheta : (n_nodes, ) array

    """
    n_nodes = V1ST.shape[1]
    S = 0.0
    if compute_derivatives:
        dS_dphi = np.zeros((n_nodes, ))
        dS_dtheta = np.zeros((n_nodes, ))
    # Q = np.zeros(V1ST.shape, dtype=dtype)
    # if compute_derivatives:
    #     dAD_dphi = np.zeros(nbrs.shape, dtype=dtype)
    #     dAD_dtheta = np.zeros(nbrs.shape, dtype=dtype)

    corrs = []
    for j in range(n_nodes):
        curr_coords = warp_coords[:, [j]]
        curr_nbrs = nbrs[:total_nbrs[j], j]
        nbr_coords = coords_list[coord_maps[j]-1][:, curr_nbrs]

        gds = compute_geodesic_distances(curr_coords, nbr_coords)
        A, non_zero_locs = gds_to_interp_vals(gds, resolution)
        A = A[non_zero_locs]
        curr_nbrs = curr_nbrs[non_zero_locs]
        nbr_coords = nbr_coords[:, non_zero_locs]
        gds = gds[non_zero_locs]
        Q = V1ST[:, curr_nbrs].dot(A)
        qnorm = np.linalg.norm(Q)

        if qnorm < thr:
            continue  # ignore node j

        Q /= qnorm
        D = 1.0 / qnorm
        corr = W2TU1[j, :].dot(Q)
        corrs.append(corr)
        S += 1.0 - corr

        if not compute_derivatives:
            continue

        dA_dphi, dA_dtheta = f_derivatives(
            curr_coords, nbr_coords, resolution, gds)
        Q_V1S = Q.dot(V1ST[:, curr_nbrs])
        # dA_dphi: (n_nbrs, ); Q_V1S: (n_nbrs, ); A: (n_nbrs, );
        # dAD_dphi: (n_nbrs, )
        dAD_dphi = D * (dA_dphi - D * Q_V1S.dot(dA_dphi) * A)
        dAD_dtheta = D * (dA_dtheta - D * Q_V1S.dot(dA_dtheta) * A)
        dS_dAD = -W2TU1[j, :].dot(V1ST[:, curr_nbrs])  # (n_nbrs, )
        dS_dphi[j] = dS_dAD.dot(dAD_dphi)
        dS_dtheta[j] = dS_dAD.dot(dAD_dtheta)

    logger.info("Average inter-subject correlation: %f" % np.mean(corrs))

    if compute_derivatives:
        return S, dS_dphi, dS_dtheta
    return S
