import numpy as np
from numpy.testing import assert_allclose

from ..normalize_dataset import normalize_dataset


def test_normalize_dataset():
    n_timepoints = 1000
    n_nodes = 5000

    np.random.seed(0)
    T = np.random.random((n_timepoints, n_nodes))
    T = normalize_dataset(T)
    assert_allclose(T.mean(), np.zeros((n_nodes, )), atol=1e-15)
    mag = np.sqrt(np.sum(T**2, axis=0)).reshape((1, n_nodes))
    # It should either be 1 or 0
    mag = np.vstack([mag, 1-mag]).max(axis=0)
    assert_allclose(mag, np.ones((n_nodes, )))

    # Test `thr` (threshold) parameter
    T = np.random.random((n_timepoints, n_nodes)) * 1e-10
    T = normalize_dataset(T, thr=1e-8)
    assert_allclose(T.mean(), np.zeros((n_nodes, )), atol=1e-15)
    mag = np.sqrt(np.sum(T**2, axis=0))
    assert_allclose(mag, np.zeros((n_nodes, )))
