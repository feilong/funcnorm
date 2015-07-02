from numpy.testing import assert_array_equal
from nose.tools import assert_equal
import numpy as np

from ..compute_neighbor_resolutions import compute_neighbor_resolutions


def _sample_nbrs():
    # 4 nodes, 0 - 1 - 2 - 3
    nbrs = np.array([[1, -99],
                     [0, 2],
                     [1, 3],
                     [2, -99]], dtype='int').T
    return nbrs


def test_compute_neighbor_resolutions():
    nbrs = _sample_nbrs()
    nbrs, res_nbr_sizes, total_nbrs = compute_neighbor_resolutions(nbrs, 1)
    assert_equal(nbrs.shape, (3, 4))
    assert_array_equal(
        nbrs,
        np.array([[0, 1, -99], [1, 0, 2], [2, 1, 3], [3, 2, -99]],
                 dtype='int').T
    )
    assert_array_equal(
        res_nbr_sizes,
        np.array([2, 3, 3, 2], dtype='int').reshape(1, 4)
    )
    assert_array_equal(res_nbr_sizes.ravel(), total_nbrs)

    nbrs = _sample_nbrs()
    nbrs, res_nbr_sizes, total_nbrs = compute_neighbor_resolutions(nbrs, 2)
    assert_equal(nbrs.shape, (4, 4))
    assert_array_equal(
        res_nbr_sizes,
        np.array([[2, 3, 3, 2], [1, 1, 1, 1]], dtype='int')
    )

    nbrs = _sample_nbrs()
    nbrs, res_nbr_sizes, total_nbrs = compute_neighbor_resolutions(nbrs, 3)
    assert_equal(nbrs.shape, (4, 4))
    assert_array_equal(
        res_nbr_sizes,
        np.array([[2, 3, 3, 2], [1, 1, 1, 1], [1, 0, 0, 1]], dtype='int')
    )

    nbrs = _sample_nbrs()
    nbrs, res_nbr_sizes, total_nbrs = compute_neighbor_resolutions(nbrs, 4)
    assert_equal(nbrs.shape, (4, 4))
    assert_array_equal(
        res_nbr_sizes,
        np.array([[2, 3, 3, 2], [1, 1, 1, 1], [1, 0, 0, 1], [0, 0, 0, 0]],
                 dtype='int')
    )
