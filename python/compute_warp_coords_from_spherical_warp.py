from compute_spherical_from_cartesian import compute_spherical_from_cartesian
from compute_cartesian_from_spherical import compute_cartesian_from_spherical


def compute_warp_coords_from_spherical_warp(
        cart_coords, spher_warp, coord_maps):
    """ Warp Cartesian coordinates with `spher_warp`.
    Calculations are done with spherical coordinates.

    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    spher_warp : (2, n_nodes) array
    coord_maps : int or list
        If it's an int, it must be either 1, 2, or 3.
        If it's a list, each element must be either 1, 2, or 3

    Returns
    -------
    warp_cart_coords : (3, n_nodes) array
    """
    spher_coords = compute_spherical_from_cartesian(cart_coords, coord_maps)
    warp_spher_coords = spher_coords + spher_warp
    warp_cart_coords = compute_cartesian_from_spherical(
        warp_spher_coords, coord_maps)

    return warp_cart_coords
