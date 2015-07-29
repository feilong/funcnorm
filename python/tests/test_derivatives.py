import os
from scipy.io import loadmat
from numpy.testing import assert_allclose

from ..derivatives import f_derivatives

DIR = os.path.dirname(os.path.realpath(__file__))


def test_f_derivatives():
    fname = os.path.join(DIR, 'data_for_test_derivatives.mat')
    mat = loadmat(fname)
    coords1, coords2 = mat['spher_coords_1'], mat['spher_coords_2']
    resolution, gds = float(mat['resolution']), mat['gds'].ravel()

    df_dphi, df_dtheta = f_derivatives(coords1, coords2, resolution, gds)
    assert_allclose(df_dphi, mat['df_dphi_vals'].ravel())
    assert_allclose(df_dtheta, mat['df_dtheta_vals'].ravel())
