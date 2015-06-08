import os
import numpy as np
from scipy.io import loadmat
from nose.tools import assert_equal, assert_true
from numpy.testing import assert_allclose, assert_almost_equal

from ..compute_geodesic_distances import compute_geodesic_distances

DIR = os.path.dirname(os.path.realpath(__file__))


def test_compute_geodesic_distances():
    coord1 = np.random.random((3, 10))
    gds = compute_geodesic_distances(coord1, coord1)
    # test results size and type are correct
    assert_equal(gds.shape, (1, 10))
    assert_true(isinstance(gds, np.ndarray))
    # test distance with self is 0
    assert_allclose(gds, np.zeros((1, 10)))

    coord2 = np.random.random((3, 10)) * .5
    gds2 = compute_geodesic_distances(coord1, coord2)
    # test distance within range
    assert_true(np.all(gds2 < 1.0))

    # Test cases for special conditions (e.g., opposite points on a sphere)
    assert_equal(
        compute_geodesic_distances(np.array([[np.pi], [0.5]]),
                                   np.array([[0], [0.5]])),
        np.pi)
    assert_equal(
        compute_geodesic_distances(np.array([[np.pi*0.5], [0.5+np.pi]]),
                                   np.array([[np.pi*0.5], [0.5]])),
        np.pi)
    assert_almost_equal(
        compute_geodesic_distances(np.array([[np.pi], [0.5+np.pi]]),
                                   np.array([[np.pi], [0.5]])),
        0)
    assert_almost_equal(
        compute_geodesic_distances(np.array([[0], [0.5+np.pi]]),
                                   np.array([[0], [0.5]])),
        0)


def test_compare_with_matlab():
    """
    Test if output of compute_geodesic_distance is exactly the same
    as corresponding Matlab function.
    """
    for coord in ['cartesian', 'spherical']:
        for submode in ['single', 'multi']:
            for i in range(20):
                in_file = os.path.join(
                    DIR, 'test_compute_geodesic_distances_data',
                    '%s-%s-%03d.mat' % (coord, submode, i))
                data = loadmat(in_file)
                a, b = data['a'], data['b']
                out_file = os.path.join(
                    DIR, 'test_compute_geodesic_distances_data',
                    '%s-%s-%03d-out.mat' % (coord, submode, i))
                res = loadmat(out_file)['res']
                res2 = compute_geodesic_distances(a, b)
                assert_allclose(res, res2)
