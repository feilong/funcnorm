import numpy as np
from numpy.testing import assert_allclose

from ..compute_spherical_from_cartesian import compute_spherical_from_cartesian
from ..compute_cartesian_from_spherical import compute_cartesian_from_spherical

def test_cartesian_spherical_cartesian():
    np.random.seed(0)
    N = 500
    cart = np.random.random((3, N))
    mags = np.sqrt((cart**2).sum(axis=0))
    cart /= mags
    coord_maps = np.random.choice([1, 2, 3], (N,)).tolist()
    spher = compute_spherical_from_cartesian(cart, coord_maps)
    cart2 = compute_cartesian_from_spherical(spher, coord_maps)
    assert_allclose(cart, cart2)

def test_spherical_cartesian_spherical():
    np.random.seed(0)
    N = 500
    spher = np.random.random((2, N)) * np.pi
    spher[1, :] = spher[1, :] * 2 - np.pi
    coord_maps = np.random.choice([1, 2, 3], (N,)).tolist()
    cart = compute_cartesian_from_spherical(spher, coord_maps)
    spher2 = compute_spherical_from_cartesian(cart, coord_maps)
    assert_allclose(spher, spher2)
