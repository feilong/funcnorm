import numpy as np
import logging

from .utils import _calc_geodesic_dist, gds_const

logger = logging.getLogger('funcnorm')


def _calc_interp_weights(spher1, spher2, res, dtype='float', zm=[],
                         compute_derivatives=False):
    h = gds_const * res
    if spher1.shape[0] == 1:
        spher1 = np.tile(spher1, (spher2.shape[0], 1))
    gds = _calc_geodesic_dist(spher1, spher2)
    non_zero = np.where(gds < 2 * np.arcsin(h/2))[0]
    if len(zm) > 0:
        non_zero = np.setdiff1d(non_zero, zm)
    s = 2 * np.sin(gds[non_zero]/2) / h
    weights = (1 - s)**4 * (4*s + 1)
    if not compute_derivatives:
        return weights, non_zero
    dA_dphi = np.zeros(gds.shape, dtype=gds.dtype)
    dA_dtheta = np.zeros(gds.shape, dtype=gds.dtype)
    idx = non_zero
    phi1, theta1 = spher1[idx, 0], spher1[idx, 1]
    phi2, theta2 = spher2[idx, 0], spher2[idx, 1]
    gds = gds[idx]
    tmp = -20 * (1 - 2 * np.sin(gds/2) / h)**3 / h**2
    diff = theta1 - theta2
    dA_dphi[idx] = (np.sin(phi1) * np.cos(phi2) -
                    np.cos(phi1) * np.sin(phi2) * np.cos(diff)) * tmp
    dA_dtheta[idx] = np.sin(phi1) * np.sin(phi2) * np.sin(diff) * tmp
    return weights, non_zero, dA_dphi[idx], dA_dtheta[idx]


def _gds_to_interp_weights(gds, res):
    h = gds_const * res
    non_zero = np.where(gds < 2 * np.arcsin(h/2))[0]
    s = 2 * np.sin(gds[non_zero]/2) / h
    weights = (1 - s)**4 * (4*s + 1)
    return weights, non_zero


def _blur_dataset_full(ds, cart, nbrs, num_nbrs, res, thr=1e-8):
    Q = np.zeros(ds.shape, ds.dtype)
    for i in range(ds.shape[1]):
        curr_cart = cart[[i], :]
        curr_nbrs = nbrs[i, :num_nbrs[i]]
        nbr_cart = cart[curr_nbrs, :]
        gds = _calc_geodesic_dist(curr_cart, nbr_cart)
        A, non_zero = _gds_to_interp_weights(gds, res)
        # A = A[non_zero]
        curr_nbrs = curr_nbrs[non_zero]
        Q[:, i] = ds[:, curr_nbrs].dot(A)
        qnorm = np.linalg.norm(Q[:, i])
        if qnorm > thr:
            Q[:, i] /= qnorm
        else:
            Q[:, i] = 0
    return Q


def _interp_time_series(T, surf, nn=False):
    n_nodes = T.shape[1]
    TW = np.zeros(T.shape)
    nbrs = np.hstack([np.arange(n_nodes)[:, np.newaxis], surf.orig_nbrs])
    num_nbrs = np.sum(nbrs != -99, axis=1)
    for j in range(n_nodes):
        curr_cart = surf.cart_warped[j, :]
        curr_nbrs = nbrs[j, :num_nbrs[j]]
        nbrs_cart = surf.cart[curr_nbrs, :]
        prev_nbr = j - 1
        while True:
            projections = nbrs_cart.dot(curr_cart)
            closest_nbr = curr_nbrs[np.argmax(projections)]
            curr_nbrs = nbrs[closest_nbr, :num_nbrs[closest_nbr]]
            nbrs_cart = surf.cart[curr_nbrs, :]
            if prev_nbr == closest_nbr:
                break
            prev_nbr = closest_nbr
        I = np.argsort(nbrs_cart.dot(curr_cart))[::-1]
        if nn is True:
            tri_nbrs = [curr_nbrs[I[0]]]
        else:
            tri_nbrs = curr_nbrs[I[:3]]
        tri_cart = surf.cart[tri_nbrs, :]
        gds = _calc_geodesic_dist(curr_cart[np.newaxis, :], tri_cart)
        weights, non_zero = _gds_to_interp_weights(gds, 1)
        if len(non_zero) == 0:
            logger.error("Geodesic distances: %r" % gds)
            raise ValueError("No neighbors found for node #%d %r after "
                             "warping." % (j, surf.cart_warped[j, :]))
        elif len(non_zero) == 1:
            TW[:, j] = T[:, [tri_nbrs[non_zero]]].dot(weights).ravel()
        else:
            TW[:, j] = T[:, tri_nbrs[non_zero]].dot(weights).ravel()
    return TW


def _calc_correlation_cost(ds1, ds2, coords_list, maps, spher_warped,
                           nbrs, num_nbrs, res, thr=1e-8, dtype='float',
                           compute_derivatives=True):
    n_nodes = ds1.shape[1]
    S = 0.0
    if compute_derivatives:
        dS_dphi = np.zeros((n_nodes, ), dtype=dtype)
        dS_dtheta = np.zeros((n_nodes, ), dtype=dtype)
    corrs = []
    for j in range(n_nodes):
        curr_coords = spher_warped[[j], :]
        curr_nbrs = nbrs[j, :num_nbrs[j]]
        nbr_coords = coords_list[maps[j]][curr_nbrs, :]
        returns = _calc_interp_weights(
            curr_coords, nbr_coords, res,
            compute_derivatives=compute_derivatives)
        A, non_zero = returns[:2]
        curr_nbrs = curr_nbrs[non_zero]
        Q = ds1[:, curr_nbrs].dot(A)
        qnorm = np.linalg.norm(Q)
        # D = 0.0 if qnorm < thr else 1.0/qnorm
        # if qnorm < thr:
        #     D = 0
        # D = 1.0 / qnorm
        if qnorm < thr:
            S += 1.0
            corrs.append(0.0)
        else:
            D = 1.0/qnorm
            Q *= D
            corr = ds2[:, j].dot(Q)
            corrs.append(corr)
            S += 1.0 - corr
        if not compute_derivatives or qnorm < thr:
            continue
        dA_dphi, dA_dtheta = returns[2:]
        Q_ds1 = Q.dot(ds1[:, curr_nbrs])
        # print non_zero, Q_ds1.shape, dA_dphi.shape
        dAD_dphi = D * (dA_dphi - D * Q_ds1.dot(dA_dphi) * A)
        dAD_dtheta = D * (dA_dtheta - D * Q_ds1.dot(dA_dtheta) * A)
        dS_dAD = -ds2[:, j].dot(ds1[:, curr_nbrs])
        dS_dphi[j] = dS_dAD.dot(dAD_dphi)
        dS_dtheta[j] = dS_dAD.dot(dAD_dtheta)
    if compute_derivatives:
        return S, corrs, dS_dphi, dS_dtheta
    return S, corrs


# """
# ds1: (n_timepoints, n_nodes)
# ds2.T.dot(ds1): (n_nodes, n_timepoints) x (n_timepoints, n_nodes)
#     -> n_nodes**2 * n_timepoints
# U1: (n_timepoints, K), s: (K, ), V1t: (K, n_nodes)
# W2TU1: (n_nodes, n_timepoints)
# Q: (n_timepoints, n_nodes)
# """
