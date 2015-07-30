import os
from scipy.io import loadmat
from numpy.testing import assert_allclose
import numpy as np

DIR = os.path.dirname(os.path.realpath(__file__))


def test_svd():
    fname = os.path.join(DIR, 'data_for_test_svd.mat')
    mat = loadmat(fname)
    U, s, V = np.linalg.svd(mat['dataset'], full_matrices=False)
    rtol = 1e-5
    assert_allclose(U, mat['U'], rtol=rtol)
    assert_allclose(s, mat['s'].ravel(), rtol=rtol)
    assert_allclose(V, mat['V'].T, rtol=rtol)
