import numpy as np

from compute_geodesic_distances import compute_geodesic_distances


def compute_metric_distances(cart_coords, nbrs, num_nbrs, dtype='float'):
    """
    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    nbrs : (max_nbrs, n_nodes) array
    num_nbrs : (n_nodes, ) array

    Returns
    -------
    metric_distances : (max_nbrs, n_nodes) array
    """
    n_nodes = cart_coords.shape[1]
    # lambda_metric /= (4.0 * n_nodes)
    # metric_distances = -99 * np.ones((6, n_nodes), dtype)
    metric_distances = -99 * np.ones(nbrs.shape, dtype)
    for j in range(n_nodes):
        curr_cart_coords = cart_coords[:, [j]]
        n_nbrs = num_nbrs[j]
        curr_nbrs = nbrs[:n_nbrs, j]
        nbr_cart_coords = cart_coords[:, curr_nbrs]
        metric_distances[:n_nbrs, j] = compute_geodesic_distances(
            curr_cart_coords, nbr_cart_coords)
    return metric_distances


def compute_oriented_areas(triangles):
    """
    """
    pass


def compute_initial_oriented_areas(triangles, cart_coords, rho=1.0):
    """
    Parameters
    ----------
    triangles : (3, n_triangles) array
    cart_coords : (3, n_nodes) array

    Returns
    -------
    tri_areas : (n_triangles, ) array
        Areas of the triangles.
    oriented_normals : (3, n_triangles) array
        `n_triangles` vectors, should be orthogonal to triangle surface
        and its length should be 1.
    """
    n_triangles = triangles.shape[1]

    p0s = cart_coords[:, triangles[0, :]]
    p1s = cart_coords[:, triangles[1, :]]
    p2s = cart_coords[:, triangles[2, :]]

    As = rho * (p1s - p0s)
    Bs = rho * (p2s - p0s)

    ns = np.cross(As.T, Bs.T).T
    n_norms = np.sqrt(np.sum(ns**2, axis=0))
    tri_areas = n_norms / 2

    oriented_normals = ns / np.tile(n_norms.reshape(1, n_triangles), (3, 1))
    # In case out of memory
    try:
        oriented_normals = \
            ns / np.tile(n_norms.reshape(1, n_triangles), (3, 1))
    except Exception:
        oriented_normals = np.zeros((3, n_triangles), dtype=ns.dtype)
        for j in range(n_triangles):
            oriented_normals[:, j] = ns[:, j] / n_norms[j]

    return tri_areas, oriented_normals


def triangles_to_vectors(triangles, cart_coords, rho):
    """
    Returns
    -------
    As : (3, n_triangles) array
        Each column is one of the triangle's edge as a vector.
    Bs : (3, n_triangles) array
    """
    p0s = cart_coords[:, triangles[0, :]]
    p1s = cart_coords[:, triangles[1, :]]
    p2s = cart_coords[:, triangles[2, :]]

    As = rho * (p1s - p0s)
    Bs = rho * (p2s - p1s)

    return As, Bs


def t_cross(As, Bs):
    cross_prod = np.cross(As.T, Bs.T).T
    return cross_prod


def compute_cross_products(triangles, cart_coords, rho):
    """
    Returns
    -------
    cross_prod : (3, n_triangles) array
    """
    As, Bs = triangles_to_vectors(triangles, cart_coords, rho)
    cross_prod = t_cross(As, Bs)
    return cross_prod


def compute_partials_cartesian(cart_coords, coord_maps):
    """
    Returns
    -------
    dp_dphi : (3, n_nodes) array
    dp_dtheta : (3, n_nodes) array

    Notes
    -----
    Replaces `computePartialCartesian_dtheta` and
    `computePartialCartesian_dphi`.
    """
    n_nodes = cart_coords.shape[1]
    dtype = cart_coords.dtype

    if isinstance(coord_maps, int):
        coord_maps = coord_maps * np.ones((n_nodes, ), dtype='int')

    dp_dphi = np.zeros((3, n_nodes), dtype=dtype)
    dp_dtheta = np.zeros((3, n_nodes), dtype=dtype)

    mags = np.sqrt((cart_coords**2).sum(axis=0))

    mapping = {
        1: (0, 1, 2),  # x, y, z
        2: (1, 2, 0),  # y, z, x
        3: (2, 1, 0),  # z, y, x
    }

    for j in range(n_nodes):
        idx = mapping[coord_maps[j]]
        xx, yy, zz = cart_coords[idx, j]
        mag = mags[j]
        phi = np.arccos(xx/mag)
        theta = np.arctan2(yy, zz)
        dp_dphi_vals = (-np.sin(phi),
                        np.cos(phi) * np.sin(theta),
                        np.cos(phi) * np.cos(theta))
        dp_dtheta_vals = (0,
                          np.sin(phi) * np.cos(theta),
                          -np.sin(phi) * np.sin(theta))
        dp_dphi[idx, j] = dp_dphi_vals
        dp_dtheta[idx, j] = dp_dtheta_vals

        # x, y, z = cart_coords[:, j]
        # mag = mags[j]

        # if coord_maps[j] == 1:
        #     phi = np.arccos(x/mag)
        #     theta = np.arctan2(y, z)
        #     dp_dphi[:, j] = (-np.sin(phi),
        #                      np.cos(phi) * np.sin(theta),
        #                      np.cos(phi) * np.cos(theta))
        #     dp_dtheta[:, j] = (0,
        #                        np.sin(phi) * np.cos(theta),
        #                        -np.sin(phi) * np.sin(theta))
        # elif coord_maps[j] == 2:
        #     phi = np.arccos(y/mag)
        #     theta = np.arctan2(z, x)
        #     dp_dphi[:, j] = (np.cos(phi) * np.cos(theta),
        #                      -np.sin(phi),
        #                      np.cos(phi) * np.sin(theta))
        #     dp_dtheta[:, j] = (-np.sin(phi) * np.sin(theta),
        #                        0,
        #                        np.sin(phi) * np.cos(theta))
        # elif coord_maps[j] == 3:
        #     phi = np.arccos(z/mag)
        #     theta = np.arctan2(y, x)
        #     dp_dphi[:, j] = (np.cos(phi) * np.cos(theta),
        #                      np.cos(phi) * np.sin(theta),
        #                      -np.sin(phi))
        #     dp_dtheta[:, j] = (-np.sin(phi) * np.sin(theta),
        #                        np.sin(phi) * np.cos(theta),
        #                        0)
    return dp_dphi, dp_dtheta
