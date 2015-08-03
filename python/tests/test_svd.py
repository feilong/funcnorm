import os
from scipy.io import loadmat
from numpy.testing import assert_allclose
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def test_svd():
    fname = os.path.join(DIR, 'data_for_test_svd.mat')
    mat = loadmat(fname)
    U, s, Vt = np.linalg.svd(mat['dataset'], full_matrices=False)
    rtol = 1e-5
    assert_allclose(U, mat['U'], rtol=rtol)
    assert_allclose(s, mat['s'].ravel(), rtol=rtol)
    assert_allclose(Vt, mat['V'].T, rtol=rtol)

    M, N = 300, 100
    ds = np.random.random((M, N))
    U, s, Vt = np.linalg.svd(ds, full_matrices=False)
    assert_allclose(U.dot(np.diag(s)).dot(Vt), ds)
    assert_allclose(U.dot(np.tile(s[:, np.newaxis], (1, N)) * Vt), ds)


def test_svd_norm():
    """test_svd_norm
    Test if the norm of SVt is the same as that of USVt.
    """
    for i in range(20):
        M, N = 300, 100
        ds = np.random.random((M, N))
        U, s, Vt = np.linalg.svd(ds, full_matrices=False)
        K = s.shape[0]
        SVt = np.tile(s.reshape((K, 1)), (1, N)) * Vt
        n1 = np.linalg.norm(SVt, axis=0)
        ds1 = U.dot(SVt / np.tile(n1.reshape((1, N)), (K, 1)))
        n2 = np.linalg.norm(ds, axis=0)
        ds2 = ds / np.tile(n2.reshape((1, N)), (M, 1))
        assert_allclose(n1, n2)
        assert_allclose(ds1, ds2)
        assert_allclose(np.linalg.norm(ds2, axis=0), np.ones((N, )))
        assert_allclose(np.linalg.norm(ds1, axis=0), np.ones((N, )))
