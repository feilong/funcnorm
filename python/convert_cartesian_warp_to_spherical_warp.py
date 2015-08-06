from .compute_spherical_from_cartesian import compute_spherical_from_cartesian


def convert_cartesian_warp_to_spherical_warp(
        cart_coords, cart_warp, coord_maps):
    """
    Parameters
    ----------
    cart_coords : (3, n_nodes) array
    cart_warp : (3, n_nodes) array
    coord_maps : (n_nodes, ) array

    Returns
    -------
    spher_warp : (2, n_nodes) array
    """
    spher_coords = compute_spherical_from_cartesian(cart_coords, coord_maps)
    warp_spher_coords = compute_spherical_from_cartesian(
        cart_coords + cart_warp, coord_maps)
    spher_warp = warp_spher_coords - spher_coords
    return spher_warp
