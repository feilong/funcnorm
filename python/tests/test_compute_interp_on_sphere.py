import os
from scipy.io import loadmat
from numpy.testing import assert_allclose

from ..compute_interp_on_sphere import compute_interp_on_sphere

DIR = os.path.dirname(os.path.realpath(__file__))


def test_compute_interp_on_sphere():
    for i in range(2):
        fname = os.path.join(
            DIR, 'data_for_test_compute_interp_on_sphere_%d.mat' % i)
        mat = loadmat(fname)
        TW = compute_interp_on_sphere(mat['T'], mat['cart_coords'],
                                      mat['neighbors'], mat['warp'], mat['nn'])
        assert_allclose(TW, mat['TW'])
