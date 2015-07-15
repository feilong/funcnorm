import numpy as np

from compute_geodesic_distances import compute_geodesic_distances


def gds_derivatives(spher_coords_1, spher_coords_2, resolution, gds=None):
    """Compute derivative of geodesic distance between `spher_coord_1` and
    `spher_coord_2` with respect to phi spherical coordinates of
    `spher_coord_1`.

    Parameters
    ----------
    spher_coords_1 : (2, n_nodes) or (2, ) or (2, 1) array
    spher_coords_2 : (2, n_nodes) array
    resolution : int or float?
    gds : None or (n_nodes, ) array, optional

    Returns
    -------
    dg_dphi : (n_nodes, ) array
    dg_dtheta : (n_nodes, ) array

    """
    if spher_coords_1.shape == (2, ):
        spher_coords_1 = spher_coords_1[:, np.newaxis]
    if spher_coords_1.shape[1] == 1:
        spher_coords_1 = np.tile(spher_coords_1,
                                 (1, spher_coords_2.shape[1]))

    if gds is None:
        gds = compute_geodesic_distances(spher_coords_1, spher_coords_2)

    # h = 0.0201 * resolution

    phi1, theta1 = spher_coords_1[0, :], spher_coords_1[1, :]
    phi2, theta2 = spher_coords_2[0, :], spher_coords_2[1, :]

    idx = np.where(gds > 1e-5)[0]  # nonZeroLocs
    dg_dphi = np.zeros(gds.shape, dtype=gds.dtype)
    dg_dtheta = np.zeros(gds.shape, dtype=gds.dtype)

    diff = theta1[idx] - theta2[idx]
    dg_dphi[idx] = (np.sin(phi1[idx]) * np.cos(phi2[idx]) -
                    np.cos(phi1[idx]) * np.sin(phi2[idx]) * np.cos(diff)) / \
        np.sin(gds[idx])
    dg_dtheta[idx] = (np.sin(phi1[idx]) * np.sin(phi2[idx]) * np.sin(diff)) / \
        np.sin(gds[idx])

    return dg_dphi, dg_dtheta


def f_derivatives(spher_coords_1, spher_coords_2, resolution, gds=None):
    """
    """
    if spher_coords_1.shape == (2, ):
        spher_coords_1 = spher_coords_1[:, np.newaxis]
    if spher_coords_1.shape[1] == 1:
        spher_coords_1 = np.tile(spher_coords_1,
                                 (1, spher_coords_2.shape[1]))

    if gds is None:
        gds = compute_geodesic_distances(spher_coords_1, spher_coords_2)

    h = 0.0201 * resolution

    phi1, theta1 = spher_coords_1[0, :], spher_coords_1[1, :]
    phi2, theta2 = spher_coords_2[0, :], spher_coords_2[1, :]

    idx = np.where(gds < 2 * np.arcsin(h/2))  # nonZeroLocs
    df_dphi = np.zeros(gds.shape, dtype=gds.dtype)
    df_dtheta = np.zeros(gds.shape, dtype=gds.dtype)

    tmp = -20 * (1 - 2 * np.sin(gds[idx]/2) / h)**3 / h**2
    diff = theta1[idx] - theta2[idx]
    df_dphi[idx] = (np.sin(phi1[idx]) * np.cos(phi2[idx]) -
                    np.cos(phi1[idx]) * np.sin(phi2[idx]) * np.cos(diff)) * tmp
    df_dtheta[idx] = np.sin(phi1[idx]) * np.sin(phi2[idx]) * np.sin(diff) * tmp

    return df_dphi, df_dtheta
