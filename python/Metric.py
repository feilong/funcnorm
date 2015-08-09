import numpy as np

from .utils import _calc_geodesic_dist
from .Coordinates import _calc_spher_coords


def _calc_metric_terms(nbrs, cart, maps, orig_md, compute_derivatives=True):
    n_nodes, max_nbrs = nbrs.shape

    full_nbrs = nbrs.copy()
    locs = np.where(nbrs == -99)
    full_nbrs[locs] = locs[0]

    cc = np.repeat(cart, max_nbrs, axis=0)
    nc = cart[full_nbrs.ravel(), :]

    gds = _calc_geodesic_dist(cc, nc)
    curr_md = gds.reshape((n_nodes, max_nbrs))

    M_mat = curr_md - orig_md
    M = np.sum(M_mat**2)
    if not compute_derivatives:
        return M

    cm = np.repeat(maps, max_nbrs, axis=0)
    csc = _calc_spher_coords(cc, cm)
    nsc = _calc_spher_coords(nc, cm)
    dg_dphi, dg_dtheta = _dgds_dspher(csc, nsc, gds)
    dg_dphi = dg_dphi.reshape((n_nodes, max_nbrs))
    dg_dtheta = dg_dtheta.reshape((n_nodes, max_nbrs))

    dM_dphi = 4.0 * np.sum(dg_dphi * M_mat, axis=1)
    dM_dtheta = 4.0 * np.sum(dg_dtheta * M_mat, axis=1)

    return M, dM_dphi, dM_dtheta


def _calc_metric_dist(cart, nbrs, num_nbrs, dtype='float'):
    n_nodes, max_nbrs = nbrs.shape
    full_nbrs = nbrs.copy()
    locs = np.where(nbrs == -99)
    full_nbrs[locs] = locs[0]

    cc = np.repeat(cart, max_nbrs, axis=0)
    nc = cart[full_nbrs.ravel(), :]
    gds = _calc_geodesic_dist(cc, nc)
    md = gds.reshape((n_nodes, max_nbrs))
    # n_nodes = cart.shape[0]
    # md = np.zeros(nbrs.shape, dtype)
    # for i in range(n_nodes):
    #     curr_cart = cart[[i], :]
    #     n_nbrs = num_nbrs[i]
    #     curr_nbrs = nbrs[i, :n_nbrs]
    #     nbr_cart = cart[curr_nbrs, :]
    #     md[i, :n_nbrs] = _calc_geodesic_dist(curr_cart, nbr_cart)
    return md


def _dgds_dspher(spher1, spher2, gds, thr=1e-5):
    phi1, theta1 = spher1[:, 0], spher1[:, 1]
    phi2, theta2 = spher2[:, 0], spher2[:, 1]

    idx = np.where(gds > thr)[0]  # nonZeroLocs
    dg_dphi = np.zeros(gds.shape, dtype=gds.dtype)
    dg_dtheta = np.zeros(gds.shape, dtype=gds.dtype)

    diff = theta1[idx] - theta2[idx]
    dg_dphi[idx] = (np.sin(phi1[idx]) * np.cos(phi2[idx]) -
                    np.cos(phi1[idx]) * np.sin(phi2[idx]) * np.cos(diff)) / \
        np.sin(gds[idx])
    dg_dtheta[idx] = (np.sin(phi1[idx]) * np.sin(phi2[idx]) * np.sin(diff)) / \
        np.sin(gds[idx])

    return dg_dphi, dg_dtheta
