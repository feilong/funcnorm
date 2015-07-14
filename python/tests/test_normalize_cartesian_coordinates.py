import numpy as np
from numpy.testing import assert_allclose

from ..normalize_cartesian_coordinates import normalize_cartesian_coordinates


def test_normalize_cartesian_coordinates():
    n_nodes = 5000
    np.random.seed(0)
    cart_coords = np.random.random((3, n_nodes))
    cart_coords = normalize_cartesian_coordinates(cart_coords)
    norms = np.sqrt(np.sum(cart_coords**2, axis=0))
    assert_allclose(norms, np.ones((n_nodes, )))
