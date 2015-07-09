import numpy as np

from compute_geodesic_distances import compute_geodesic_distances


def interp_f(coord1, coord2, resolution, gds=None, dtype='float', zM=[]):
    """
    Parameters
    ----------
    coord1, coord2 : array
        `coord1` and `coord2` can be either spherical (2 rows) or Cartesian
        (3 rows).
    resolution : int or float?
    gds : None or (n_nodes, ) array
        If gds is not None, you can use function `gds_to_interp_vals` instead.
    zm : optional
        zero-mask (indices to ignore in `coord2`)

    Returns
    -------
    interp_vals : (n_nodes, ) array
    non_zero_locs : ndarray
        1D array of non-zero locations.
    """
    if gds is None:
        gds = compute_geodesic_distances(coord1, coord2)

    interp_vals, non_zero_locs = gds_to_interp_vals(gds, resolution, dtype, zM)

    return interp_vals, non_zero_locs


def gds_to_interp_vals(gds, resolution, dtype='float', zM=[]):
    """
    Parameters
    ----------
    gds : (n_nodes, ) array
        Output of `compute_geodesic_distances`.
    resolution : int or float?
    zM : optional
        zero-mask (indices to ignore in gds)

    Returns
    -------
    interp_vals : (n_nodes, ) array
    non_zero_locs : ndarray
        1D array of non-zero locations.
    """
    h = 0.0201 * resolution

    non_zero_locs = np.where(gds < 2 * np.arcsin(h/2))[0]
    if len(zM) > 0:
        non_zero_locs = np.setdiff1d(non_zero_locs, zM)

    interp_vals = np.zeros(gds.shape, dtype=dtype)
    s = 2 * np.sin(gds[non_zero_locs]/2) / h
    interp_vals[non_zero_locs] = (1 - s)**4 * (4*s + 1)

    return interp_vals, non_zero_locs
