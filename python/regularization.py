import numpy as np

from compute_geodesic_distances import compute_geodesic_distances
from compute_spherical_from_cartesian import compute_spherical_from_cartesian


def compute_metric_terms(orig_nbrs, cart_coords, coord_maps,
                         orig_metric_distances, res, rho=1.0,
                         compute_derivatives=False):
    """
    Parameters
    ----------
    orig_nbrs : (max_nbrs, n_nodes) array
    cart_coords : (3, n_nodes) array
    coord_maps : (n_nodes, ) array or list
    orig_metric_distances : (max_nbrs, n_nodes) array
    res : int or float?

    Returns
    -------
    md_diff : float
    dmd_diffs_dphi : (n_nodes, ) array
    dmd_diffs_dtheta : (n_nodes, ) array

    Notes
    -----
    We don't need `orig_num_nbrs` as a parameter as the Matlab version.

    """
    n_nodes = cart_coords.shape[1]
    max_num_nbrs = orig_nbrs.shape[0]
    # dtype = cart_coords.dtype

    locs = np.where(orig_nbrs == -99)

    full_orig_nbrs = orig_nbrs.copy()  # Do we need deep copy here?
    full_orig_nbrs[locs] = locs[1]  # locs[1] == which column
    cc = np.repeat(cart_coords, max_num_nbrs, axis=1)
    nc = cart_coords[:, full_orig_nbrs.T.ravel()]

    gds = compute_geodesic_distances(cc, nc, rho)
    curr_md = gds.reshape((n_nodes, max_num_nbrs)).T

    md_diff_mat = curr_md - orig_metric_distances
    md_diff = np.sum(md_diff_mat**2)

    if not compute_derivatives:
        return md_diff

    cm = np.repeat(coord_maps, max_num_nbrs, axis=0)

    csc = compute_spherical_from_cartesian(cc, cm)
    nsc = compute_spherical_from_cartesian(nc, cm)

    dg_dphi, dg_dtheta = gds_derivatives(csc, nsc, res, gds)
    dg_dphi = dg_dphi.reshape((n_nodes, max_num_nbrs)).T
    dg_dtheta = dg_dtheta.reshape((n_nodes, max_num_nbrs)).T

    # (max_nbrs, n_nodes) * (max_nbrs, n_nodes) -> sum -> (n_nodes, )
    dmd_diffs_dphi = np.sum(dg_dphi * md_diff_mat, axis=0)
    dmd_diffs_dtheta = np.sum(dg_dtheta * md_diff_mat, axis=0)

    return md_diff, dmd_diffs_dphi, dmd_diffs_dtheta


def compute_areal_terms(triangles, cart_coords, coord_maps,
                        orig_tri_areas, tri_normals, rho=1.0,
                        compute_derivatives=True):
    """
    Parameters
    ----------
    triangles : (3, n_triangles) array
    cart_coords : (3, n_nodes) array
    coord_maps : int or (n_nodes, ) array-like
    orig_tri_areas : (n_triangles, ) array
    tri_normals : (3, n_triangles) array

    Returns
    -------
    tri_area : float
    dareal_dphi : (n_triangles, ) array
    dareal_dtheta : (n_triangles, ) array
    """
    n_nodes = cart_coords.shape[1]
    dtype = cart_coords.dtype
    # triArea = cast(0, class(cartCoords));

    As, Bs = triangles_to_vectors(triangles, cart_coords, rho)
    abcrosses = t_cross(As, Bs)

    # The area of the triangle formed by `abcross` and `tri_normals`.
    new_areas = np.sum(abcrosses * tri_normals, axis=0) / 2

    # Less than 0 indicates folding of the cortex
    locs = np.where(new_areas <= 0)[0]
    a_terms = new_areas - orig_tri_areas
    # sum(diff_areas^2) for areas w/ negative values (foldings)
    # Seems to be something used for regularization.
    tri_area = np.sum(a_terms[locs]**2)

    if not compute_derivatives:
        return tri_area

    dareal_dphi = np.zeros((n_nodes, ), dtype=dtype)
    dareal_dtheta = np.zeros((n_nodes, ), dtype=dtype)
    dp_dphi, dp_dtheta = compute_partials_cartesian(cart_coords, coord_maps)

    bcns = t_cross(Bs, tri_normals)  # B cross N, (3, n_triangles)
    ncas = t_cross(tri_normals, As)  # N cross A

    for loc in locs:
        tmp1 = a_terms[loc] * bcns[:, loc]
        tmp2 = a_terms[loc] * ncas[:, loc]
        tmps = [-tmp1 - tmp2, tmp1, tmp2]
        for node in range(3):
            cI = triangles[node, loc]
            tmp = tmps[node]  # (3, )
            dareal_dphi[cI] += np.sum(tmp * dp_dphi[:, cI])
            dareal_dtheta[cI] += np.sum(tmp * dp_dtheta[:, cI])

        # cI = triangles[0, loc]
        # dp0_dphi = dp_dphi[:, cI]
        # dp0_dtheta = dp_dtheta[:, cI]
        # tmp = -a_terms[loc] * (bcns[:, loc] + ncas[:, loc]).T
        # dareal_dphi[cI] += np.sum(tmp * dp0_dphi)
        # dareal_dtheta[cI] += np.sum(tmp * dp0_dtheta)

        # cI = triangles[1, loc]
        # dp1_dphi = dp_dphi[:, cI]
        # dp1_dtheta = dp_dtheta[:, cI]
        # tmp = a_terms[loc] * bcns[:, loc].T
        # dareal_dphi[cI] += np.sum(tmp * dp1_dphi)
        # dareal_dtheta[cI] += np.sum(tmp * dp1_dtheta)

        # cI = triangles[2, loc]
        # dp2_dphi = dp_dphi[:, cI]
        # dp2_dtheta = dp_dtheta[:, cI]
        # tmp = a_terms[loc] * ncas[:, loc].T
        # dareal_dphi[cI] += np.sum(tmp * dp2_dphi)
        # dareal_dtheta[cI] += np.sum(tmp * dp2_dtheta)

    return tri_area, dareal_dphi, dareal_dtheta


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
    Bs = rho * (p2s - p0s)

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

    return dp_dphi, dp_dtheta


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
