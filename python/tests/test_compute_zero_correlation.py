import os
from scipy.io import loadmat
from glob import glob

from numpy.testing import assert_allclose

from ..compute_zero_correlation import compute_zero_correlation

DIR = os.path.dirname(os.path.realpath(__file__))

def test_compute_zero_correlation():
    """ test_compute_zero_correlation
    Test if compute_zero_correlation gives the same results
    as the Matlab version.
    """
    files = glob(os.path.join(DIR, 'test_compute_zero_correlation_data/*.mat'))
    for i, fname in enumerate(sorted(files)):
        data = loadmat(fname)
        warps = data['warps'].flatten().tolist()
        cart_coords = data['cart_coords']
        res = data['res'].flatten().tolist()
        res2 = compute_zero_correlation(cart_coords, warps)
        if len(res) != len(res2):
            raise ValueError('Results shape incorrect')
        for j in range(len(res)):
            assert_allclose(res[j], res2[j])
