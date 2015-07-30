import os
from scipy.io import loadmat
from numpy.testing import assert_allclose

from ..blur_dataset import blur_dataset, blur_dataset_no_svd

DIR = os.path.dirname(os.path.realpath(__file__))


def test_blur_dataset_no_svd():
    fname = os.path.join(DIR, 'data_for_test_blur_dataset')
    mat = loadmat(fname)

    Q = blur_dataset_no_svd(
        V=mat['V'], s=mat['s'], cart_coords=mat['cart_coords'],
        nbrs=mat['nbrs'], total_nbrs=mat['total_nbrs'].ravel(),
        resolution=float(mat['resolution']))
    rtol = 1e-7
    assert_allclose(Q, mat['Q'], rtol=rtol)


def test_blur_dataset():
    fname = os.path.join(DIR, 'data_for_test_blur_dataset')
    mat = loadmat(fname)

    V2, s2 = blur_dataset(
        V=mat['V'], s=mat['s'], cart_coords=mat['cart_coords'],
        nbrs=mat['nbrs'], total_nbrs=mat['total_nbrs'].ravel(),
        resolution=float(mat['resolution']))
    rtol = 1e-7
    assert_allclose(V2, mat['V2'], rtol=rtol)
    assert_allclose(s2, mat['s2'].ravel(), rtol=rtol)

    V3, s3, U3 = blur_dataset(
        V=mat['V'], s=mat['s'], U=mat['U'], cart_coords=mat['cart_coords'],
        nbrs=mat['nbrs'], total_nbrs=mat['total_nbrs'].ravel(),
        resolution=float(mat['resolution']))
    assert_allclose(V3, mat['V3'], rtol=rtol)
    assert_allclose(s3, mat['s3'].ravel(), rtol=rtol)
    # SVD in Matlab and Python will yield slightly different numbers
    # Here we use rtol of 1e-6 rather than the default 1e-7
    assert_allclose(U3, mat['U3'], rtol=1e-6)
