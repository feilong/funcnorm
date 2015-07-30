import os
from scipy.io import loadmat
from numpy.testing import assert_allclose

from ..interp_f import interp_f, gds_to_interp_vals

DIR = os.path.dirname(os.path.realpath(__file__))


def test_gds_to_interp_vals():
    for coord in ['cartesian', 'spherical']:
        for submode in ['single', 'multi']:
            for i in range(20):
                in_file = os.path.join(
                    DIR, 'test_compute_geodesic_distances_data',
                    'interp_f-%s-%s-%03d.mat' % (coord, submode, i))
                data = loadmat(in_file)
                interp_vals, non_zero_locs = gds_to_interp_vals(
                    gds=data['gds'].ravel(), resolution=2, zM=data['zM']-1)
                assert_allclose(interp_vals, data['interp_vals'].ravel())
                assert_allclose(non_zero_locs, data['non_zero_locs'].ravel())

    fname = os.path.join(DIR, 'data_for_test_interp_f.mat')
    mat = loadmat(fname)
    interp_vals, non_zero_locs = gds_to_interp_vals(
        mat['gds'].ravel(), float(mat['resolution']))
    assert_allclose(interp_vals, mat['interp_vals'].ravel())
    assert_allclose(non_zero_locs, mat['non_zero_locs'].ravel()-1)


def test_interp_f():
    for coord in ['cartesian', 'spherical']:
        for submode in ['single', 'multi']:
            for i in range(20):
                in_file = os.path.join(
                    DIR, 'test_compute_geodesic_distances_data',
                    'interp_f-%s-%s-%03d.mat' % (coord, submode, i))
                data = loadmat(in_file)
                interp_vals, non_zero_locs = interp_f(
                    data['data']['a'], data['data']['b'], resolution=2,
                    gds=data['gds'].ravel(), zM=data['zM']-1)
                assert_allclose(interp_vals, data['interp_vals'].ravel())
                assert_allclose(non_zero_locs, data['non_zero_locs'].ravel())

    fname = os.path.join(DIR, 'data_for_test_interp_f.mat')
    mat = loadmat(fname)
    interp_vals, non_zero_locs = interp_f(
        mat['coords1'], mat['coords2'], float(mat['resolution']))
    assert_allclose(interp_vals, mat['interp_vals'].ravel())
    assert_allclose(non_zero_locs, mat['non_zero_locs'].ravel()-1)
