import numpy as np

from compute_geodesic_distances import compute_geodesic_distances


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
