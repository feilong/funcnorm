import numpy as np

from ..compute_spherical_from_cartesian import compute_spherical_from_cartesian


def random_dataset(seed, n_nodes, n_timepoints):
    """Normalized random dataset."""
    np.random.seed(seed)
    ds = np.random.random((n_timepoints, n_nodes))
    ds /= np.linalg.norm(ds, axis=0)
    return ds


def random_nbrs(seed, n_nodes, max_nbrs=6):
    """Random neighbors."""
    np.random.seed(seed)
    nbrs = -99 * np.ones((max_nbrs, n_nodes), dtype='int')
    total_nbrs = np.zeros((n_nodes, ))
    for j in range(n_nodes):
        n_nbrs = np.random.randint(1, max_nbrs+1)
        nbrs[:n_nbrs, j] = np.random.choice(range(n_nodes), (n_nbrs, ), False)
        total_nbrs[j] = n_nbrs
    return nbrs, total_nbrs


def random_coords(seed, n_nodes):
    """Random cart_coords, spher_coords and coord_maps."""
    np.random.seed(seed)
    cart_coords = np.random.random((3, n_nodes))
    coord_maps = np.random.choice([1, 2, 3], (n_nodes, ), True)
    spher_coords = compute_spherical_from_cartesian(cart_coords, coord_maps)
    return cart_coords, spher_coords, coord_maps
