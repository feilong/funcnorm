import numpy as np


def normalize_cartesian_coordinates(cart_coords, rho=1.0):
    """
    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    rho : float

    Returns
    -------
    cart_coords : (3, n_nodes) array
        Cartesian coordinates after normalization.

    Notes
    -----
    In case out of memory, do it row-by-row instead.
    See `regularization.compute_initial_oriented_areas` for similar code.

    """
    n_nodes = cart_coords.shape[1]

    norms = np.sqrt(np.sum(cart_coords**2, axis=0))

    cart_coords = rho * cart_coords / \
        np.tile(norms.reshape(1, n_nodes), (3, 1))

    return cart_coords
