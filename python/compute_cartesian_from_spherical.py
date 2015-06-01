import numpy as np

def compute_cartesian_from_spherical(spher, coord_maps):
    """
    Parameters
    ----------
    spher : array, 2 x N
    coord_maps : list or int
        If it's an int, its value must be 1, 2, or 3.
        If it's a list, each element must be 1, 2, or 3.

    Returns
    -------
    cart : array, 3 x N
    """

    N = spher.shape[1]
    cart = np.zeros((3, N), dtype=spher.dtype)

    if str(spher.dtype).startswith('complex'):
        raise ValueError('compute_cartesian_from_spherical: '\
                         'spher array contains imaginary components.')

    phis = spher[0, :]
    thetas = spher[1, :]

    if isinstance(coord_maps, int):
        coord_maps = [coord_maps] * N
    coord_maps = np.array(coord_maps)

    num_found = 0

    # Find coordinate maps equal to 1 (phi measured from x-axis; theta
    # measured from z-axis)
    idx = (coord_maps == 1)
    count = np.sum(idx)
    if count:
        num_found += count
        cart[0, idx] = np.cos(phis[idx])
        cart[1, idx] = np.sin(phis[idx]) * np.sin(thetas[idx])
        cart[2, idx] = np.sin(phis[idx]) * np.cos(thetas[idx])
    # Find coordinate maps equal to 2 (phi measured from y-axis; theta
    # measured from x-axis)
    idx = (coord_maps == 2)
    count = np.sum(idx)
    if count:
        num_found += count
        cart[0, idx] = np.sin(phis[idx]) * np.cos(thetas[idx])
        cart[1, idx] = np.cos(phis[idx])
        cart[2, idx] = np.sin(phis[idx]) * np.sin(thetas[idx])

    # Find coordinate maps equal to 3 (phi measured from z-axis; theta
    # measured from x-axis)
    idx = (coord_maps == 3)
    count = np.sum(idx)
    if count:
        num_found += count
        cart[0, idx] = np.sin(phis[idx]) * np.cos(thetas[idx])
        cart[1, idx] = np.sin(phis[idx]) * np.sin(thetas[idx])
        cart[2, idx] = np.cos(phis[idx])

    if num_found != N:
        raise ValueError('compute_cartesian_from_spherical: '
                         'Improper coordinate map.')

    return cart
