import numpy as np
from numpy.testing import assert_allclose

from ..convert_cartesian_warp_to_spherical_warp import \
    convert_cartesian_warp_to_spherical_warp
from ..compute_spherical_from_cartesian import compute_spherical_from_cartesian


def test_convert_cartesian_warp_to_spherical_warp():
    n_nodes = 1000
    cart_coords = np.random.random((3, n_nodes))
    cart_coords /= np.linalg.norm(cart_coords, axis=0)
    new_coords = np.random.random((3, n_nodes))
    new_coords /= np.linalg.norm(new_coords, axis=0)
    cart_warp = new_coords - cart_coords
    coord_maps = np.random.choice([1, 2, 3], (n_nodes, ), True)

    spher_warp = convert_cartesian_warp_to_spherical_warp(
        cart_coords, cart_warp, coord_maps)
    spher_coords = compute_spherical_from_cartesian(cart_coords, coord_maps)
    new_spher_coords = spher_coords + spher_warp
    new_spher_coords2 = compute_spherical_from_cartesian(
        new_coords, coord_maps)
    assert_allclose(new_spher_coords, new_spher_coords2)
